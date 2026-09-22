// Native image/audio IO adapters for the tensorplay.vision / tensorplay.audio
// load paths.
//
// Image decoding runs through the system codec libraries and lands directly
// in the destination tensor layout: decoders emit per-scanline output that is
// split into contiguous channel planes on the fly, so the interleaved
// (H, W, C) staging buffer and the separate transpose pass both disappear.
// On CUDA the JPEG path goes through the hardware decoder, whose planar
// output format is pointed straight at the channel planes of one contiguous
// device buffer — no post-processing copy at all.
//
// Encode works the same way: rows are assembled from the channel planes one at a
// time and streamed to the codec, never materializing a full HWC buffer.
//
// The audio section below keeps the numpy-to-tensor adapters: the kernels
// fuse the layout conversion (interleaved -> planar) with the integer
// normalization the loaders need.

#include "python_bindings.h"

#include "DataPtr.h"
#include "Storage.h"
#include "TensorImpl.h"
#include "Tensor.h"

#include <cstring>
#include <vector>
#include <mutex>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#ifdef TP_USE_LIBJPEG
#include <csetjmp>
#include <jpeglib.h>
#include <cstddef>
#endif

#ifdef TP_USE_LIBPNG
#include <csetjmp>
#include <png.h>
#endif

#ifdef TP_USE_NVJPEG
#include "CUDARuntime.h"
#include <cuda_runtime.h>
#include <nvjpeg.h>
#endif

namespace {

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

void require_cpu_uint8(const Tensor& t, const char* who, int want_dim, const char* shape_desc) {
    if (t.device().type() != DeviceType::CPU) {
        TP_THROW(RuntimeError, std::string(who) + ": expected a CPU tensor");
    }
    if (t.dtype() != DType::UInt8) {
        TP_THROW(TypeError, std::string(who) + ": expected a uint8 tensor");
    }
    if (t.dim() != want_dim) {
        TP_THROW(RuntimeError, std::string(who) + ": expected " + shape_desc);
    }
}

// Scatter one interleaved scanline into the channel planes of a CHW tensor.
// Reads stay sequential; the three-four plane streams are far enough apart
// that hardware prefetchers track them independently.
inline void deinterleave_row(const uint8_t* row, uint8_t* planes,
                             int64_t width, int64_t channels, int64_t plane_size, int64_t y) {
    uint8_t* dst = planes + y * width;
    if (channels == 1) {
        std::memcpy(dst, row, static_cast<size_t>(width));
        return;
    }
    if (channels == 3) {
        const uint8_t* p0 = dst;
        uint8_t* p1 = dst + plane_size;
        uint8_t* p2 = dst + 2 * plane_size;
        for (int64_t x = 0; x < width; ++x) {
            p0[x] = row[3 * x];
            p1[x] = row[3 * x + 1];
            p2[x] = row[3 * x + 2];
        }
        return;
    }
    for (int64_t c = 0; c < channels; ++c) {
        uint8_t* plane = planes + c * plane_size + y * width;
        for (int64_t x = 0; x < width; ++x) {
            plane[x] = row[x * channels + c];
        }
    }
}

// Assemble one interleaved scanline from the channel planes of a CHW tensor.
inline void interleave_row(const uint8_t* planes, uint8_t* row,
                           int64_t width, int64_t channels, int64_t plane_size, int64_t y) {
    const uint8_t* src = planes + y * width;
    if (channels == 1) {
        std::memcpy(row, src, static_cast<size_t>(width));
        return;
    }
    if (channels == 3) {
        const uint8_t* p0 = src;
        const uint8_t* p1 = src + plane_size;
        const uint8_t* p2 = src + 2 * plane_size;
        for (int64_t x = 0; x < width; ++x) {
            row[3 * x] = p0[x];
            row[3 * x + 1] = p1[x];
            row[3 * x + 2] = p2[x];
        }
        return;
    }
    for (int64_t c = 0; c < channels; ++c) {
        const uint8_t* plane = planes + c * plane_size + y * width;
        for (int64_t x = 0; x < width; ++x) {
            row[x * channels + c] = plane[x];
        }
    }
}

Tensor make_uint8_tensor(const std::vector<int64_t>& shape, Device device) {
    return Tensor::empty(shape, DType::UInt8, device);
}

#ifdef TP_USE_LIBJPEG

// ---------------------------------------------------------------------------
// JPEG (CPU) — libjpeg with SIMD-accelerated IDCT and color conversion
// ---------------------------------------------------------------------------

struct JpegErrorMgr {
    jpeg_error_mgr pub;
    jmp_buf escape;
};

void jpeg_error_escape(j_common_ptr info) {
    longjmp(reinterpret_cast<JpegErrorMgr*>(info->err)->escape, 1);
}

enum { JPEG_MODE_UNCHANGED = 0, JPEG_MODE_GRAY = 1, JPEG_MODE_RGB = 3 };

Tensor decode_jpeg_cpu(const uint8_t* data, size_t size, int64_t mode) {
    if (mode != JPEG_MODE_UNCHANGED && mode != JPEG_MODE_GRAY && mode != JPEG_MODE_RGB) {
        TP_THROW(ValueError, "decode_jpeg: JPEG carries no alpha channel; mode must be 0 (unchanged), 1 (gray) or 3 (rgb)");
    }
    jpeg_decompress_struct cinfo{};
    JpegErrorMgr jerr;
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jpeg_error_escape;
    if (setjmp(jerr.escape)) {
        jpeg_destroy_decompress(&cinfo);
        TP_THROW(RuntimeError, "decode_jpeg: corrupt or unsupported JPEG stream");
    }
    jpeg_create_decompress(&cinfo);
    jpeg_mem_src(&cinfo, const_cast<unsigned char*>(data), size);
    jpeg_read_header(&cinfo, TRUE);
    if (mode == JPEG_MODE_GRAY || (mode == JPEG_MODE_UNCHANGED && cinfo.num_components == 1)) {
        cinfo.out_color_space = JCS_GRAYSCALE;
    } else {
        // YCbCr, CMYK and YCCK all funnel through the RGB conversion.
        cinfo.out_color_space = JCS_RGB;
    }
    jpeg_start_decompress(&cinfo);
    const int64_t channels = cinfo.output_components;
    const int64_t height = cinfo.output_height;
    const int64_t width = cinfo.output_width;
    Tensor out = make_uint8_tensor({channels, height, width}, Device(DeviceType::CPU, 0));
    std::vector<uint8_t> row(static_cast<size_t>(width * channels));
    uint8_t* row_ptr = row.data();
    uint8_t* planes = out.data_ptr<uint8_t>();
    const int64_t plane_size = height * width;
    while (cinfo.output_scanline < cinfo.output_height) {
        jpeg_read_scanlines(&cinfo, &row_ptr, 1);
        deinterleave_row(row_ptr, planes, width, channels, plane_size, cinfo.output_scanline - 1);
    }
    jpeg_finish_decompress(&cinfo);
    jpeg_destroy_decompress(&cinfo);
    return out;
}

Tensor encode_jpeg_cpu(const uint8_t* planes, int64_t channels,
                       int64_t height, int64_t width, int64_t quality) {
    jpeg_compress_struct cinfo{};
    JpegErrorMgr jerr;
    cinfo.err = jpeg_std_error(&jerr.pub);
    jerr.pub.error_exit = jpeg_error_escape;
    unsigned char* out_buf = nullptr;
    unsigned long out_size = 0;
    if (setjmp(jerr.escape)) {
        jpeg_destroy_compress(&cinfo);
        if (out_buf) free(out_buf);
        TP_THROW(RuntimeError, "encode_jpeg: compression failed");
    }
    jpeg_create_compress(&cinfo);
    jpeg_mem_dest(&cinfo, &out_buf, &out_size);
    cinfo.image_width = static_cast<JDIMENSION>(width);
    cinfo.image_height = static_cast<JDIMENSION>(height);
    cinfo.input_components = static_cast<int>(channels);
    cinfo.in_color_space = channels == 1 ? JCS_GRAYSCALE : JCS_RGB;
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, static_cast<int>(std::min(std::max(quality, int64_t(0)), int64_t(100))), TRUE);
    jpeg_start_compress(&cinfo, TRUE);
    std::vector<uint8_t> row(static_cast<size_t>(width * channels));
    const int64_t plane_size = height * width;
    while (cinfo.next_scanline < cinfo.image_height) {
        interleave_row(planes, row.data(), width, channels, plane_size, cinfo.next_scanline);
        JSAMPLE* row_ptr = row.data();
        jpeg_write_scanlines(&cinfo, &row_ptr, 1);
    }
    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
    Tensor out = make_uint8_tensor({static_cast<int64_t>(out_size)}, Device(DeviceType::CPU, 0));
    std::memcpy(out.data_ptr<uint8_t>(), out_buf, out_size);
    free(out_buf);
    return out;
}

#endif // TP_USE_LIBJPEG

#ifdef TP_USE_LIBPNG

// ---------------------------------------------------------------------------
// PNG (CPU) — libpng, per-row fused plane split
// ---------------------------------------------------------------------------

struct PngSource {
    const uint8_t* data;
    size_t size;
    size_t offset;
};

void png_read_from_memory(png_structp png, png_bytep out, png_size_t n) {
    PngSource* src = reinterpret_cast<PngSource*>(png_get_io_ptr(png));
    if (src->offset + n > src->size) {
        png_error(png, "truncated PNG stream");
        return;
    }
    std::memcpy(out, src->data + src->offset, n);
    src->offset += n;
}

void png_write_to_memory(png_structp png, png_bytep data, png_size_t n) {
    auto* buf = reinterpret_cast<std::vector<uint8_t>*>(png_get_io_ptr(png));
    buf->insert(buf->end(), data, data + n);
}

void png_flush_memory(png_structp) {}

Tensor decode_png_cpu(const uint8_t* data, size_t size, int64_t mode) {
    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) TP_THROW(RuntimeError, "decode_png: failed to initialize the PNG reader");
    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, nullptr, nullptr);
        TP_THROW(RuntimeError, "decode_png: failed to initialize PNG info");
    }
    PngSource src{data, size, 0};
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_read_struct(&png, &info, nullptr);
        TP_THROW(RuntimeError, "decode_png: corrupt or unsupported PNG stream");
    }
    png_set_read_fn(png, &src, png_read_from_memory);
    png_read_info(png, info);

    const png_uint_32 width = png_get_image_width(png, info);
    const png_uint_32 height = png_get_image_height(png, info);
    png_byte bit_depth = png_get_bit_depth(png, info);
    png_byte color_type = png_get_color_type(png, info);
    const bool has_alpha = (color_type & PNG_COLOR_MASK_ALPHA) != 0 ||
                           png_get_valid(png, info, PNG_INFO_tRNS) != 0;

    // Normalize every input shape to 8-bit gray/rgb (+ optional alpha).
    if (color_type == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
        png_set_expand_gray_1_2_4_to_8(png);
    }
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (bit_depth == 16) png_set_strip_16(png);
    png_set_packing(png);

    switch (mode) {
        case 0:  // unchanged: keep the file's gray/rgb and alpha presence
            break;
        case 1:  // gray: collapse color, drop alpha
            if (color_type & PNG_COLOR_MASK_COLOR) png_set_rgb_to_gray(png, 1, -1, -1);
            png_set_strip_alpha(png);
            break;
        case 2:  // gray + alpha
            if (color_type & PNG_COLOR_MASK_COLOR) png_set_rgb_to_gray(png, 1, -1, -1);
            if (!has_alpha) png_set_add_alpha(png, 255, PNG_FILLER_AFTER);
            break;
        case 3:  // rgb
            if (!(color_type & PNG_COLOR_MASK_COLOR)) png_set_gray_to_rgb(png);
            png_set_strip_alpha(png);
            break;
        case 4:  // rgb + alpha
            if (!(color_type & PNG_COLOR_MASK_COLOR)) png_set_gray_to_rgb(png);
            if (!has_alpha) png_set_add_alpha(png, 255, PNG_FILLER_AFTER);
            break;
        default:
            png_destroy_read_struct(&png, &info, nullptr);
            TP_THROW(ValueError, "decode_png: mode must be between 0 and 4");
    }

    png_set_interlace_handling(png);
    png_read_update_info(png, info);
    const int64_t channels = png_get_channels(png, info);
    const int64_t rowbytes = png_get_rowbytes(png, info);
    Tensor out = make_uint8_tensor({channels, static_cast<int64_t>(height), static_cast<int64_t>(width)},
                                   Device(DeviceType::CPU, 0));
    std::vector<uint8_t> row(static_cast<size_t>(rowbytes));
    uint8_t* planes = out.data_ptr<uint8_t>();
    const int64_t plane_size = static_cast<int64_t>(height) * width;
    for (int64_t y = 0; y < static_cast<int64_t>(height); ++y) {
        png_read_row(png, row.data(), nullptr);
        deinterleave_row(row.data(), planes, width, channels, plane_size, y);
    }
    png_read_end(png, nullptr);
    png_destroy_read_struct(&png, &info, nullptr);
    return out;
}

Tensor encode_png_cpu(const uint8_t* planes, int64_t channels,
                      int64_t height, int64_t width, int64_t compression_level) {
    png_structp png = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png) TP_THROW(RuntimeError, "encode_png: failed to initialize the PNG writer");
    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_write_struct(&png, nullptr);
        TP_THROW(RuntimeError, "encode_png: failed to initialize PNG info");
    }
    std::vector<uint8_t> out_buf;
    if (setjmp(png_jmpbuf(png))) {
        png_destroy_write_struct(&png, &info);
        TP_THROW(RuntimeError, "encode_png: compression failed");
    }
    png_set_write_fn(png, &out_buf, png_write_to_memory, png_flush_memory);
    const png_byte color_type = channels == 1 ? PNG_COLOR_TYPE_GRAY : PNG_COLOR_TYPE_RGB;
    png_set_IHDR(png, info, static_cast<png_uint_32>(width), static_cast<png_uint_32>(height),
                 8, color_type, PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);
    compression_level = std::min(std::max(compression_level, int64_t(0)), int64_t(9));
    png_set_compression_level(png, static_cast<int>(compression_level));
    png_write_info(png, info);
    std::vector<uint8_t> row(static_cast<size_t>(width * channels));
    const int64_t plane_size = height * width;
    for (int64_t y = 0; y < height; ++y) {
        interleave_row(planes, row.data(), width, channels, plane_size, y);
        png_write_row(png, row.data());
    }
    png_write_end(png, info);
    png_destroy_write_struct(&png, &info);
    Tensor out = make_uint8_tensor({static_cast<int64_t>(out_buf.size())}, Device(DeviceType::CPU, 0));
    if (!out_buf.empty()) {
        std::memcpy(out.data_ptr<uint8_t>(), out_buf.data(), out_buf.size());
    }
    return out;
}

#endif // TP_USE_LIBPNG

#ifdef TP_USE_NVJPEG

// ---------------------------------------------------------------------------
// JPEG (CUDA) — hardware decoder, planar output pointed directly at the
// channel planes of one contiguous device buffer.
// ---------------------------------------------------------------------------

nvjpegHandle_t io_nvjpeg_handle() {
    static nvjpegHandle_t handle = nullptr;
    static std::once_flag once;
    std::call_once(once, [] {
        if (nvjpegCreateSimple(&handle) != NVJPEG_STATUS_SUCCESS) {
            handle = nullptr;
        }
    });
    return handle;
}

nvjpegJpegState_t io_nvjpeg_state(nvjpegHandle_t handle) {
    static nvjpegJpegState_t state = nullptr;
    static std::once_flag once;
    std::call_once(once, [handle] {
        if (handle) {
            nvjpegJpegStateCreate(handle, &state);
        }
    });
    return state;
}

Tensor decode_jpeg_cuda(const uint8_t* data, size_t size, int64_t mode) {
    nvjpegHandle_t handle = io_nvjpeg_handle();
    nvjpegJpegState_t state = io_nvjpeg_state(handle);
    if (!handle || !state) {
        TP_THROW(RuntimeError, "decode_jpeg: failed to initialize the hardware JPEG decoder");
    }
    // The simple decode API is not thread-safe per state.
    static std::mutex decode_mutex;
    std::lock_guard<std::mutex> lock(decode_mutex);

    int num_components = 0;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAXCOMPONENT];
    int heights[NVJPEG_MAXCOMPONENT];
    if (nvjpegGetImageInfo(handle, data, size, &num_components, &subsampling,
                           widths, heights) != NVJPEG_STATUS_SUCCESS) {
        TP_THROW(RuntimeError, "decode_jpeg: corrupt or unsupported JPEG stream");
    }
    nvjpegOutputFormat_t format;
    int64_t channels;
    if (mode == 1 || (mode == 0 && num_components == 1)) {
        format = NVJPEG_OUTPUT_Y;
        channels = 1;
    } else if (mode == 0 || mode == 3) {
        format = NVJPEG_OUTPUT_RGB;
        channels = 3;
    } else {
        TP_THROW(ValueError, "decode_jpeg: GPU decode supports modes 0 (unchanged), 1 (gray) and 3 (rgb)");
    }
    const int64_t width = widths[0];
    const int64_t height = heights[0];
    Tensor out = make_uint8_tensor({channels, height, width}, Device(DeviceType::CUDA, 0));
    uint8_t* base = out.data_ptr<uint8_t>();
    nvjpegImage_t image{};
    for (int64_t c = 0; c < channels; ++c) {
        image.channel[c] = base + c * height * width;
        image.pitch[c] = static_cast<unsigned int>(width);
    }
    cudaStream_t stream = getCurrentCUDAStream().stream();
    if (nvjpegDecode(handle, state, data, size, format, stream, &image) != NVJPEG_STATUS_SUCCESS) {
        TP_THROW(RuntimeError, "decode_jpeg: hardware decode failed");
    }
    return out;
}

#endif // TP_USE_NVJPEG

}  // namespace

// Audio loading kernel: converts an interleaved (Time, Channels) buffer into
// a channel-plane (Channels, Time) float32 tensor and folds the integer
// scaling into the same pass:
//   int16 -> x * (1/32768)   int32 -> x * (1/2^31)   uint8 -> (x - 128) * (1/128)
// Channel counts in audio are small, so each output plane is emitted as one
// contiguous stream and the interleaved input is read with a short fixed
// stride; that keeps every hot loop sequential in memory and vectorizable at
// any clip length.
namespace {

template <typename InT, typename Scale>
void audio_convert_planes(const InT* in, float* out, size_t T, size_t C, Scale scale) {