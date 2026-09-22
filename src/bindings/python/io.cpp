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
// normalization the loaders need.  WAV files additionally decode natively
// (classic RIFF/WAVE layout, no third-party dependency), so playable clips
// never touch an intermediate numpy buffer; a batch entry point parallelizes
// whole-file decode across the shared thread pool.
//
// Codec support is decided at configure time; the public entry points are
// registered only for the codecs that are actually built, so the Python
// layer can probe for them and keep a slower fallback where a codec library
// is missing.

#include "python_bindings.h"

#include "DataPtr.h"
#include "Exception.h"
#include "Parallel.h"
#include "Storage.h"
#include "Tensor.h"
#include "TensorImpl.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <mutex>
#include <tuple>
#include <utility>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

#ifdef TP_USE_LIBJPEG
#include <csetjmp>
#include <cstddef>
#include <jpeglib.h>
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
        uint8_t* p0 = dst;
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
            if (color_type & PNG_COLOR_MASK_COLOR) png_set_rgb_to_gray(png, 1, 29900, 58700);
            png_set_strip_alpha(png);
            break;
        case 2:  // gray + alpha
            if (color_type & PNG_COLOR_MASK_COLOR) png_set_rgb_to_gray(png, 1, 29900, 58700);
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
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];
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
    cudaStream_t stream = tensorplay::cuda::getCurrentCUDAStream().stream();
    if (nvjpegDecode(handle, state, data, size, format, &image, stream) != NVJPEG_STATUS_SUCCESS) {
        TP_THROW(RuntimeError, "decode_jpeg: hardware decode failed");
    }
    return out;
}

// Batched decoding: every image must be decoded to the same output format, so
// the per-image component count decides the format first; a mixed batch (e.g.
// gray and color files at mode 0) falls back to per-image single decode.
std::vector<Tensor> decode_jpeg_batch_cuda(const std::vector<Tensor>& inputs,
                                           int64_t mode, int batch_size) {
    nvjpegHandle_t handle = io_nvjpeg_handle();
    nvjpegJpegState_t state = io_nvjpeg_state(handle);
    if (!handle || !state) {
        TP_THROW(RuntimeError, "decode_jpeg: failed to initialize the hardware JPEG decoder");
    }
    if (mode != 0 && mode != 1 && mode != 3) {
        TP_THROW(ValueError, "decode_jpeg: GPU decode supports modes 0 (unchanged), 1 (gray) and 3 (rgb)");
    }

    // Resolve the output format of every image and check that they agree.
    static std::mutex info_mutex;
    std::vector<int64_t> widths(batch_size), heights(batch_size);
    std::vector<nvjpegOutputFormat_t> formats(batch_size);
    std::vector<int64_t> channels(batch_size);
    bool uniform = true;
    {
        std::lock_guard<std::mutex> lock(info_mutex);
        for (int i = 0; i < batch_size; ++i) {
            int num_components = 0;
            nvjpegChromaSubsampling_t subsampling;
            int ws[NVJPEG_MAX_COMPONENT];
            int hs[NVJPEG_MAX_COMPONENT];
            const uint8_t* data = inputs[i].data_ptr<uint8_t>();
            if (nvjpegGetImageInfo(handle, data, static_cast<size_t>(inputs[i].numel()),
                                   &num_components, &subsampling, ws, hs) != NVJPEG_STATUS_SUCCESS) {
                TP_THROW(RuntimeError, "decode_jpeg: corrupt or unsupported JPEG stream");
            }
            widths[i] = ws[0];
            heights[i] = hs[0];
            if (mode == 1 || (mode == 0 && num_components == 1)) {
                formats[i] = NVJPEG_OUTPUT_Y;
                channels[i] = 1;
            } else {
                formats[i] = NVJPEG_OUTPUT_RGB;
                channels[i] = 3;
            }
            if (i > 0 && formats[i] != formats[i - 1]) uniform = false;
        }
    }

    if (!uniform) {
        // Mixed output formats: fall back to per-image single decode.
        std::vector<Tensor> out(batch_size);
        for (int i = 0; i < batch_size; ++i) {
            out[i] = decode_jpeg_cuda(inputs[i].data_ptr<uint8_t>(),
                                      static_cast<size_t>(inputs[i].numel()), mode);
        }
        return out;
    }

    // Lazily initialized per-handle batch state, re-initialized when the
    // batch size changes (the library requires matching sizes).
    static std::once_flag batch_once;
    static nvjpegJpegState_t batch_state = nullptr;
    static int cached_batch = -1;
    std::call_once(batch_once, [handle] {
        if (handle) nvjpegJpegStateCreate(handle, &batch_state);
    });
    if (!batch_state) {
        TP_THROW(RuntimeError, "decode_jpeg: failed to initialize the hardware batch decoder");
    }
    if (cached_batch != batch_size) {
        int threads = std::max(1, std::min(tensorplay::parallel::get_num_threads(), 64));
        if (nvjpegDecodeBatchedInitialize(handle, batch_state, batch_size, threads,
                                          formats[0]) != NVJPEG_STATUS_SUCCESS) {
            TP_THROW(RuntimeError, "decode_jpeg: failed to initialize the hardware batch decoder");
        }
        cached_batch = batch_size;
    }

    std::vector<Tensor> out(batch_size);
    std::vector<nvjpegImage_t> dests(batch_size);
    std::vector<const unsigned char*> ptrs(batch_size);
    std::vector<size_t> lengths(batch_size);
    for (int i = 0; i < batch_size; ++i) {
        out[i] = make_uint8_tensor({channels[i], heights[i], widths[i]},
                                   Device(DeviceType::CUDA, 0));
        uint8_t* base = out[i].data_ptr<uint8_t>();
        nvjpegImage_t& image = dests[i];
        for (int64_t c = 0; c < channels[i]; ++c) {
            image.channel[c] = base + c * heights[i] * widths[i];
            image.pitch[c] = static_cast<unsigned int>(widths[i]);
        }
        ptrs[i] = inputs[i].data_ptr<uint8_t>();
        lengths[i] = static_cast<size_t>(inputs[i].numel());
    }
    cudaStream_t stream = tensorplay::cuda::getCurrentCUDAStream().stream();
    if (nvjpegDecodeBatched(handle, batch_state, ptrs.data(), lengths.data(),
                            dests.data(), stream) != NVJPEG_STATUS_SUCCESS) {
        TP_THROW(RuntimeError, "decode_jpeg: hardware batch decode failed");
    }
    return out;
}

#endif // TP_USE_NVJPEG

// ---------------------------------------------------------------------------
// Audio: interleaved (Time, Channels) -> planar (Channels, Time) float32,
// integer scaling folded into the same pass:
//   int16 -> x * (1/32768)   int32 -> x * (1/2^31)   uint8 -> (x - 128) * (1/128)
// Channel counts in audio are small, so each output plane is emitted as one
// contiguous stream and the interleaved input is read with a short fixed
// stride; that keeps every hot loop sequential in memory and vectorizable at
// any clip length.
// ---------------------------------------------------------------------------

template <typename InT, typename Scale>
void audio_convert_planes(const InT* in, float* out, size_t T, size_t C, Scale scale) {
    if (C == 1) {
        float* plane = out;
        for (size_t t = 0; t < T; ++t) {
            plane[t] = scale(in[t]);
        }
        return;
    }
    if (C == 2) {
        float* plane0 = out;
        float* plane1 = out + T;
        for (size_t t = 0; t < T; ++t) {
            plane0[t] = scale(in[2 * t]);
            plane1[t] = scale(in[2 * t + 1]);
        }
        return;
    }
    for (size_t c = 0; c < C; ++c) {
        float* plane = out + c * T;
        for (size_t t = 0; t < T; ++t) {
            plane[t] = scale(in[t * C + c]);
        }
    }
}

#if defined(__x86_64__)
namespace audio_simd {

inline bool cpu_has_avx2() {
    static const bool ok = __builtin_cpu_supports("avx2") != 0;
    return ok;
}

__attribute__((target("avx2")))
void convert_i16_stream(const int16_t* in, float* out, size_t n) {
    const __m256 scale = _mm256_set1_ps(1.0f / 32768.0f);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m256i raw = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(in + i));
        const __m256 lo = _mm256_cvtepi32_ps(
            _mm256_cvtepi16_epi32(_mm256_castsi256_si128(raw)));
        const __m256 hi = _mm256_cvtepi32_ps(
            _mm256_cvtepi16_epi32(_mm256_extracti128_si256(raw, 1)));
        _mm256_storeu_ps(out + i, _mm256_mul_ps(lo, scale));
        _mm256_storeu_ps(out + i + 8, _mm256_mul_ps(hi, scale));
    }
    for (; i < n; ++i) {
        out[i] = static_cast<float>(in[i]) * (1.0f / 32768.0f);
    }
}

__attribute__((target("avx2")))
void convert_i32_stream(const int32_t* in, float* out, size_t n) {
    const __m256 scale = _mm256_set1_ps(1.0f / 2147483648.0f);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256i raw = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(in + i));
        _mm256_storeu_ps(out + i,
                         _mm256_mul_ps(_mm256_cvtepi32_ps(raw), scale));
    }
    for (; i < n; ++i) {
        out[i] = static_cast<float>(in[i]) * (1.0f / 2147483648.0f);
    }
}

__attribute__((target("avx2")))
void convert_i16_stereo(const int16_t* in, float* out, size_t n) {
    // Byte masks picking the even/odd int16 lanes of each 128-bit half; only
    // the low half of the shuffled register feeds the widening convert.
    const __m128i even_mask = _mm_setr_epi8(0, 1, 4, 5, 8, 9, 12, 13,
                                            0, 0, 0, 0, 0, 0, 0, 0);
    const __m128i odd_mask = _mm_setr_epi8(2, 3, 6, 7, 10, 11, 14, 15,
                                           0, 0, 0, 0, 0, 0, 0, 0);
    const __m128 scale = _mm_set1_ps(1.0f / 32768.0f);
    float* plane0 = out;
    float* plane1 = out + n;
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const __m256i raw = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(in + 2 * i));
        const __m128i lo = _mm256_castsi256_si128(raw);
        const __m128i hi = _mm256_extracti128_si256(raw, 1);
        const __m128 even_lo = _mm_cvtepi32_ps(
            _mm_cvtepi16_epi32(_mm_shuffle_epi8(lo, even_mask)));
        const __m128 even_hi = _mm_cvtepi32_ps(
            _mm_cvtepi16_epi32(_mm_shuffle_epi8(hi, even_mask)));
        const __m128 odd_lo = _mm_cvtepi32_ps(
            _mm_cvtepi16_epi32(_mm_shuffle_epi8(lo, odd_mask)));
        const __m128 odd_hi = _mm_cvtepi32_ps(
            _mm_cvtepi16_epi32(_mm_shuffle_epi8(hi, odd_mask)));
        _mm_storeu_ps(plane0 + i, _mm_mul_ps(even_lo, scale));
        _mm_storeu_ps(plane0 + i + 4, _mm_mul_ps(even_hi, scale));
        _mm_storeu_ps(plane1 + i, _mm_mul_ps(odd_lo, scale));
        _mm_storeu_ps(plane1 + i + 4, _mm_mul_ps(odd_hi, scale));
    }
    for (; i < n; ++i) {
        plane0[i] = static_cast<float>(in[2 * i]) * (1.0f / 32768.0f);
        plane1[i] = static_cast<float>(in[2 * i + 1]) * (1.0f / 32768.0f);
    }
}

__attribute__((target("avx2")))
void convert_i32_stereo(const int32_t* in, float* out, size_t n) {
    // Lane order {0,2,4,6,1,3,5,7} gathers the even samples of four
    // interleaved frames into the low half and the odd ones into the high
    // half.
    const __m256i split = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    const __m256 scale = _mm256_set1_ps(1.0f / 2147483648.0f);
    float* plane0 = out;
    float* plane1 = out + n;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256i raw = _mm256_loadu_si256(
            reinterpret_cast<const __m256i*>(in + 2 * i));
        const __m256 planes = _mm256_mul_ps(
            _mm256_cvtepi32_ps(_mm256_permutevar8x32_epi32(raw, split)),
            scale);
        _mm_storeu_ps(plane0 + i, _mm256_castps256_ps128(planes));
        _mm_storeu_ps(plane1 + i, _mm256_extractf128_ps(planes, 1));
    }
    for (; i < n; ++i) {
        plane0[i] = static_cast<float>(in[2 * i]) * (1.0f / 2147483648.0f);
        plane1[i] = static_cast<float>(in[2 * i + 1]) * (1.0f / 2147483648.0f);
    }
}

__attribute__((target("avx2")))
void copy_f32_stereo(const float* in, float* out, size_t n) {
    const __m256i split = _mm256_setr_epi32(0, 2, 4, 6, 1, 3, 5, 7);
    float* plane0 = out;
    float* plane1 = out + n;
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        const __m256 raw = _mm256_loadu_ps(in + 2 * i);
        const __m256 planes = _mm256_permutevar8x32_ps(raw, split);
        _mm_storeu_ps(plane0 + i, _mm256_castps256_ps128(planes));
        _mm_storeu_ps(plane1 + i, _mm256_extractf128_ps(planes, 1));
    }
    for (; i < n; ++i) {
        plane0[i] = in[2 * i];
        plane1[i] = in[2 * i + 1];
    }
}

}  // namespace audio_simd
#endif

void audio_load_i16(const int16_t* in, float* out, size_t T, size_t C) {
#if defined(__x86_64__)
    if (audio_simd::cpu_has_avx2()) {
        if (C == 1) {
            audio_simd::convert_i16_stream(in, out, T);
            return;
        }
        if (C == 2) {
            audio_simd::convert_i16_stereo(in, out, T);
            return;
        }
    }
#endif
    audio_convert_planes(
        in, out, T, C,
        [](int16_t v) { return static_cast<float>(v) * (1.0f / 32768.0f); });
}

void audio_load_i32(const int32_t* in, float* out, size_t T, size_t C) {
#if defined(__x86_64__)
    if (audio_simd::cpu_has_avx2()) {
        if (C == 1) {
            audio_simd::convert_i32_stream(in, out, T);
            return;
        }
        if (C == 2) {
            audio_simd::convert_i32_stereo(in, out, T);
            return;
        }
    }
#endif
    audio_convert_planes(
        in, out, T, C,
        [](int32_t v) { return static_cast<float>(v) * (1.0f / 2147483648.0f); });
}

void audio_load_u8(const uint8_t* in, float* out, size_t T, size_t C) {
    audio_convert_planes(
        in, out, T, C,
        [](uint8_t v) { return (static_cast<float>(v) - 128.0f) * (1.0f / 128.0f); });
}

void audio_load_f32(const float* in, float* out, size_t T, size_t C) {
#if defined(__x86_64__)
    if (C == 2 && audio_simd::cpu_has_avx2()) {
        audio_simd::copy_f32_stereo(in, out, T);
        return;
    }
#endif
    if (C == 1) {
        std::memcpy(out, in, T * sizeof(float));
        return;
    }
    audio_convert_planes(in, out, T, C, [](float v) { return v; });
}

// 24-bit PCM has no native integer type; samples are sign-extended from three
// little-endian bytes on the fly.
inline int32_t read_i24_le(const uint8_t* p) {
    uint32_t u = static_cast<uint32_t>(p[0]) |
                 (static_cast<uint32_t>(p[1]) << 8) |
                 (static_cast<uint32_t>(p[2]) << 16);
    if (u & 0x800000u) u |= 0xFF000000u;
    return static_cast<int32_t>(u);
}

void audio_load_i24(const uint8_t* in, float* out, size_t T, size_t C) {
    const float scale = 1.0f / 8388608.0f;
    if (C == 1) {
        float* plane = out;
        for (size_t t = 0; t < T; ++t) {
            plane[t] = static_cast<float>(read_i24_le(in + 3 * t)) * scale;
        }
        return;
    }
    if (C == 2) {
        float* plane0 = out;
        float* plane1 = out + T;
        for (size_t t = 0; t < T; ++t) {
            plane0[t] = static_cast<float>(read_i24_le(in + 6 * t)) * scale;
            plane1[t] = static_cast<float>(read_i24_le(in + 6 * t + 3)) * scale;
        }
        return;
    }
    for (size_t c = 0; c < C; ++c) {
        float* plane = out + c * T;
        for (size_t t = 0; t < T; ++t) {
            plane[t] = static_cast<float>(read_i24_le(in + (t * C + c) * 3)) * scale;
        }
    }
}

// G.711 companding decoders (both 8-bit, unsigned within the container).
inline float alaw_to_float(uint8_t a) {
    a ^= 0x55;
    int32_t seg = (a & 0x70) >> 4;
    int32_t v = (a & 0x0F) << 4;
    switch (seg) {
        case 0: v += 8; break;
        case 1: v += 0x108; break;
        default: v += 0x108; v <<= seg - 1; break;
    }
    const float s = static_cast<float>(v) * (1.0f / 32768.0f);
    return (a & 0x80) ? s : -s;
}

inline float ulaw_to_float(uint8_t u) {
    u = static_cast<uint8_t>(~u);
    int32_t t = ((u & 0x0F) << 3) + 0x84;
    t <<= (u & 0x70) >> 4;
    const float s = static_cast<float>(t - 0x84) * (1.0f / 32768.0f);
    return (u & 0x80) ? -s : s;
}

// ---------------------------------------------------------------------------
// WAV — classic RIFF/WAVE container, decoded natively.
// Supported encodings: PCM 8/16/24/32-bit, IEEE float 32/64-bit, and the two
// G.711 companded 8-bit forms.  frame_offset seeks by byte arithmetic, so a
// partial read never decodes the skipped samples.
// ---------------------------------------------------------------------------

inline uint16_t read_le16(const uint8_t* p) {
    return static_cast<uint16_t>(static_cast<uint16_t>(p[0]) |
                                 (static_cast<uint16_t>(p[1]) << 8));
}

inline uint32_t read_le32(const uint8_t* p) {
    return static_cast<uint32_t>(p[0]) |
           (static_cast<uint32_t>(p[1]) << 8) |
           (static_cast<uint32_t>(p[2]) << 16) |
           (static_cast<uint32_t>(p[3]) << 24);
}

inline void write_le16(uint8_t* p, uint16_t v) {
    p[0] = static_cast<uint8_t>(v & 0xFF);
    p[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
}

inline void write_le32(uint8_t* p, uint32_t v) {
    p[0] = static_cast<uint8_t>(v & 0xFF);
    p[1] = static_cast<uint8_t>((v >> 8) & 0xFF);
    p[2] = static_cast<uint8_t>((v >> 16) & 0xFF);
    p[3] = static_cast<uint8_t>((v >> 24) & 0xFF);
}

struct WavInfo {
    uint16_t encoding;
    uint16_t channels;
    uint32_t sample_rate;
    uint16_t bits;
    uint16_t block_align;
    const uint8_t* data;
    size_t data_size;
    uint64_t frames;
};

void parse_wav(const uint8_t* p, size_t n, WavInfo* out) {
    if (n < 12 || std::memcmp(p, "RIFF", 4) != 0 || std::memcmp(p + 8, "WAVE", 4) != 0) {
        TP_THROW(RuntimeError, "decode_wav: not a RIFF/WAVE stream");
    }
    bool have_fmt = false;
    bool have_data = false;
    uint16_t encoding = 0, channels = 0, bits = 0, block_align = 0;
    uint32_t sample_rate = 0;
    const uint8_t* data_ptr = nullptr;
    size_t data_size = 0;
    size_t off = 12;
    while (off + 8 <= n) {
        const bool is_data = std::memcmp(p + off, "data", 4) == 0;
        uint32_t chunk_size = read_le32(p + off + 4);
        const uint8_t* chunk = p + off + 8;
        size_t avail = n - (off + 8);
        if (chunk_size > avail) {
            // A streaming data chunk may declare an unbounded size; clamp to
            // what is actually present and stop.
            if (is_data) {
                data_ptr = chunk;
                data_size = avail;
                have_data = true;
            }
            break;
        }
        if (std::memcmp(p + off, "fmt ", 4) == 0) {
            if (chunk_size < 16) {
                TP_THROW(RuntimeError, "decode_wav: malformed fmt chunk");
            }
            encoding = read_le16(chunk);
            channels = read_le16(chunk + 2);
            sample_rate = read_le32(chunk + 4);
            block_align = read_le16(chunk + 12);
            bits = read_le16(chunk + 14);
            if (encoding == 0xFFFE && chunk_size >= 40) {
                // WAVE_FORMAT_EXTENSIBLE: the sub-format GUID starts with the
                // underlying encoding tag.
                encoding = read_le16(chunk + 24);
            }
            have_fmt = true;
        } else if (is_data) {
            data_ptr = chunk;
            data_size = chunk_size;
            have_data = true;
        }
        off += 8 + chunk_size + (chunk_size & 1u);
    }
    if (!have_fmt) {
        TP_THROW(RuntimeError, "decode_wav: missing fmt chunk");
    }
    if (!have_data) {
        TP_THROW(RuntimeError, "decode_wav: missing data chunk");
    }
    if (channels == 0) {
        TP_THROW(RuntimeError, "decode_wav: zero channels");
    }
    if (sample_rate == 0) {
        TP_THROW(RuntimeError, "decode_wav: zero sample rate");
    }
    switch (encoding) {
        case 1:
            if (bits != 8 && bits != 16 && bits != 24 && bits != 32) {
                TP_THROW(NotImplementedError, "decode_wav: unsupported PCM depth");
            }
            break;
        case 3:
            if (bits != 32 && bits != 64) {
                TP_THROW(NotImplementedError, "decode_wav: unsupported float depth");
            }
            break;
        case 6:
        case 7:
            if (bits != 8) {
                TP_THROW(NotImplementedError, "decode_wav: unsupported companding depth");
            }
            break;
        default:
            TP_THROW(NotImplementedError, "decode_wav: unsupported WAV encoding");
    }
    const size_t bytes_per_sample = static_cast<size_t>(bits) / 8;
    const size_t per_frame = block_align != 0
                                 ? static_cast<size_t>(block_align)
                                 : static_cast<size_t>(channels) * bytes_per_sample;
    out->encoding = encoding;
    out->channels = channels;
    out->sample_rate = sample_rate;
    out->bits = bits;
    out->block_align = static_cast<uint16_t>(per_frame);
    out->data = data_ptr;
    out->data_size = data_size;
    out->frames = per_frame != 0 ? static_cast<uint64_t>(data_size / per_frame) : 0;
}

Tensor decode_wav_impl(const uint8_t* p, size_t n, int64_t frame_offset, int64_t num_frames) {
    if (frame_offset < 0) {
        TP_THROW(ValueError, "decode_wav: frame_offset must be >= 0");
    }
    WavInfo info;
    parse_wav(p, n, &info);
    const int64_t total = static_cast<int64_t>(info.frames);
    int64_t start = frame_offset;
    int64_t avail = total - start;
    if (avail < 0) avail = 0;
    int64_t count = num_frames < 0 ? avail : std::min<int64_t>(num_frames, avail);
    const int64_t C = info.channels;
    Tensor out = Tensor::empty({C, count}, DType::Float32, Device(DeviceType::CPU, 0));
    if (count == 0) return out;
    const uint8_t* src = info.data + static_cast<size_t>(start) * info.block_align;
    float* dst = out.data_ptr<float>();
    const size_t T = static_cast<size_t>(count);
    switch (info.encoding) {
        case 1:
            if (info.bits == 8) {
                audio_load_u8(src, dst, T, C);
            } else if (info.bits == 16) {
                audio_load_i16(reinterpret_cast<const int16_t*>(src), dst, T, C);
            } else if (info.bits == 32) {
                audio_load_i32(reinterpret_cast<const int32_t*>(src), dst, T, C);
            } else {
                audio_load_i24(src, dst, T, C);
            }
            break;
        case 3:
            if (info.bits == 32) {
                audio_load_f32(reinterpret_cast<const float*>(src), dst, T, C);
            } else {
                audio_convert_planes(reinterpret_cast<const double*>(src), dst, T, C,
                                     [](double v) { return static_cast<float>(v); });
            }
            break;
        case 6:
            audio_convert_planes(src, dst, T, C,
                                 [](uint8_t v) { return alaw_to_float(v); });
            break;
        case 7:
            audio_convert_planes(src, dst, T, C,
                                 [](uint8_t v) { return ulaw_to_float(v); });
            break;
    }
    return out;
}

// Sample interleaving helper used in the encoder below: every sample is
// written little-endian after clamping to the target integer grid. NaN is
// treated as silence (0) so corrupted input cannot poison a file.
template <typename WriteFn>
void write_interleaved(const float* planes, int64_t C, int64_t T, WriteFn write) {
    for (int64_t t = 0; t < T; ++t) {
        for (int64_t c = 0; c < C; ++c) {
            float v = planes[c * T + t];
            if (std::isnan(v)) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            if (v < -1.0f) v = -1.0f;
            write(v);
        }
    }
}

Tensor encode_wav_impl(const float* planes, int64_t C, int64_t T,
                       uint32_t sample_rate, int64_t bits) {
    const size_t bytes_per_sample = static_cast<size_t>(bits) / 8;
    const size_t block_align = static_cast<size_t>(C) * bytes_per_sample;
    const uint64_t data_size = static_cast<uint64_t>(T) * block_align;
    const uint64_t total = 44 + data_size;
    Tensor out = make_uint8_tensor({static_cast<int64_t>(total)}, Device(DeviceType::CPU, 0));
    uint8_t* dst = out.data_ptr<uint8_t>();
    std::memcpy(dst, "RIFF", 4);
    write_le32(dst + 4, static_cast<uint32_t>(total - 8));
    std::memcpy(dst + 8, "WAVE", 4);
    std::memcpy(dst + 12, "fmt ", 4);
    write_le32(dst + 16, 16);
    write_le16(dst + 20, 1);  // PCM
    write_le16(dst + 22, static_cast<uint16_t>(C));
    write_le32(dst + 24, sample_rate);
    write_le32(dst + 28, sample_rate * static_cast<uint32_t>(block_align));
    write_le16(dst + 32, static_cast<uint16_t>(block_align));
    write_le16(dst + 34, static_cast<uint16_t>(bits));
    std::memcpy(dst + 36, "data", 4);
    write_le32(dst + 40, static_cast<uint32_t>(data_size));
    uint8_t* pay = dst + 44;
    if (bits == 8) {
        write_interleaved(planes, C, T, [&](float v) {
            long s = std::lrint(v * 128.0f);
            if (s > 127) s = 127;
            if (s < -128) s = -128;
            *pay++ = static_cast<uint8_t>(s + 128);
        });
    } else if (bits == 16) {
        write_interleaved(planes, C, T, [&](float v) {
            long s = std::lrint(v * 32768.0f);
            if (s > 32767) s = 32767;
            if (s < -32768) s = -32768;
            write_le16(pay, static_cast<uint16_t>(s & 0xFFFF));
            pay += 2;
        });
    } else if (bits == 24) {
        write_interleaved(planes, C, T, [&](float v) {
            long s = std::lrint(v * 8388608.0f);
            if (s > 8388607) s = 8388607;
            if (s < -8388608) s = -8388608;
            uint32_t u = static_cast<uint32_t>(s) & 0xFFFFFFu;
            pay[0] = static_cast<uint8_t>(u & 0xFF);
            pay[1] = static_cast<uint8_t>((u >> 8) & 0xFF);
            pay[2] = static_cast<uint8_t>((u >> 16) & 0xFF);
            pay += 3;
        });
    } else {  // bits == 32
        write_interleaved(planes, C, T, [&](float v) {
            long long s = static_cast<long long>(std::lrint(v * 2147483648.0f));
            if (s > 2147483647LL) s = 2147483647LL;
            if (s < -2147483648LL) s = -2147483648LL;
            write_le32(pay, static_cast<uint32_t>(s) & 0xFFFFFFFFu);
            pay += 4;
        });
    }
    return out;
}

}  // namespace

// ---------------------------------------------------------------------------
// Public entry points (registered from the module init; see python_bindings.h)
// ---------------------------------------------------------------------------

#ifdef TP_USE_LIBJPEG

Tensor decode_jpeg(Tensor data, int64_t mode) {
    require_cpu_uint8(data, "decode_jpeg", 1, "a 1-D uint8 tensor");
    return decode_jpeg_cpu(data.data_ptr<uint8_t>(), static_cast<size_t>(data.numel()), mode);
}

Tensor encode_jpeg(Tensor data, int64_t quality) {
    if (data.device().type() != DeviceType::CPU) {
        TP_THROW(RuntimeError, "encode_jpeg: expected a CPU tensor");
    }
    if (data.dtype() != DType::UInt8) {
        TP_THROW(TypeError, "encode_jpeg: expected a uint8 tensor");
    }
    if (data.dim() != 3) {
        TP_THROW(RuntimeError, "encode_jpeg: expected a CHW (channels, height, width) tensor");
    }
    const int64_t channels = data.size(0);
    const int64_t height = data.size(1);
    const int64_t width = data.size(2);
    if (channels != 1 && channels != 3) {
        TP_THROW(ValueError, "encode_jpeg: expected 1 (grayscale) or 3 (rgb) channels");
    }
    if (height == 0 || width == 0) {
        TP_THROW(ValueError, "encode_jpeg: empty image");
    }
    if (!data.is_contiguous()) {
        TP_THROW(TypeError, "encode_jpeg: expected a contiguous CHW tensor");
    }
    return encode_jpeg_cpu(data.data_ptr<uint8_t>(), channels, height, width, quality);
}

#endif // TP_USE_LIBJPEG

#ifdef TP_USE_LIBPNG

Tensor decode_png(Tensor data, int64_t mode) {
    require_cpu_uint8(data, "decode_png", 1, "a 1-D uint8 tensor");
    return decode_png_cpu(data.data_ptr<uint8_t>(), static_cast<size_t>(data.numel()), mode);
}

Tensor encode_png(Tensor data, int64_t compression_level) {
    if (data.device().type() != DeviceType::CPU) {
        TP_THROW(RuntimeError, "encode_png: expected a CPU tensor");
    }
    if (data.dtype() != DType::UInt8) {
        TP_THROW(TypeError, "encode_png: expected a uint8 tensor");
    }
    if (data.dim() != 3) {
        TP_THROW(RuntimeError, "encode_png: expected a CHW (channels, height, width) tensor");
    }
    const int64_t channels = data.size(0);
    const int64_t height = data.size(1);
    const int64_t width = data.size(2);
    if (channels != 1 && channels != 3) {
        TP_THROW(ValueError, "encode_png: expected 1 (grayscale) or 3 (rgb) channels");
    }
    if (height == 0 || width == 0) {
        TP_THROW(ValueError, "encode_png: empty image");
    }
    if (!data.is_contiguous()) {
        TP_THROW(TypeError, "encode_png: expected a contiguous CHW tensor");
    }
    return encode_png_cpu(data.data_ptr<uint8_t>(), channels, height, width, compression_level);
}

#endif // TP_USE_LIBPNG

#ifdef TP_USE_NVJPEG

Tensor decode_jpeg_cuda(Tensor data, int64_t mode) {
    require_cpu_uint8(data, "decode_jpeg_cuda", 1, "a 1-D uint8 tensor");
    return decode_jpeg_cuda(data.data_ptr<uint8_t>(), static_cast<size_t>(data.numel()), mode);
}

#endif // TP_USE_NVJPEG

#if defined(TP_USE_LIBJPEG) || defined(TP_USE_NVJPEG)

std::vector<Tensor> decode_jpeg_batch(std::vector<Tensor> data, int64_t mode,
                                      const std::string& device) {
    for (const Tensor& t : data) {
        require_cpu_uint8(t, "decode_jpeg_batch", 1, "1-D uint8 tensors");
    }
    const int64_t N = static_cast<int64_t>(data.size());
    if (N == 0) return {};

    if (device == "cpu") {
#ifdef TP_USE_LIBJPEG
        std::vector<Tensor> out(N);
        std::exception_ptr error;
        std::mutex error_mutex;
        tensorplay::parallel::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
            for (int64_t i = begin; i < end; ++i) {
                try {
                    out[i] = decode_jpeg_cpu(data[i].data_ptr<uint8_t>(),
                                             static_cast<size_t>(data[i].numel()), mode);
                } catch (...) {
                    std::lock_guard<std::mutex> guard(error_mutex);
                    if (!error) error = std::current_exception();
                }
            }
        });
        if (error) std::rethrow_exception(error);
        return out;
#else
        TP_THROW(NotImplementedError, "decode_jpeg_batch: CPU JPEG support is not compiled in");
#endif
    }
    if (device == "cuda") {
#ifdef TP_USE_NVJPEG
        return decode_jpeg_batch_cuda(data, mode, static_cast<int>(N));
#else
        TP_THROW(NotImplementedError, "decode_jpeg_batch: CUDA JPEG support is not compiled in");
#endif
    }
    TP_THROW(ValueError, "decode_jpeg_batch: device must be 'cpu' or 'cuda'");
}

#endif

std::pair<Tensor, int64_t> decode_wav(Tensor data, int64_t frame_offset, int64_t num_frames) {
    require_cpu_uint8(data, "decode_wav", 1, "a 1-D uint8 tensor");
    Tensor waveform = decode_wav_impl(data.data_ptr<uint8_t>(),
                                      static_cast<size_t>(data.numel()),
                                      frame_offset, num_frames);
    // Re-parse the header for the sample rate (the header pass is cheap; the
    // decode work above dominates anyway).
    WavInfo info;
    parse_wav(data.data_ptr<uint8_t>(), static_cast<size_t>(data.numel()), &info);
    return {std::move(waveform), static_cast<int64_t>(info.sample_rate)};
}

std::vector<std::pair<Tensor, int64_t>> decode_wav_batch(std::vector<Tensor> data,
                                                         int64_t frame_offset,
                                                         int64_t num_frames) {
    for (const Tensor& t : data) {
        require_cpu_uint8(t, "decode_wav_batch", 1, "1-D uint8 tensors");
    }
    const int64_t N = static_cast<int64_t>(data.size());
    std::vector<std::pair<Tensor, int64_t>> out(N);
    std::exception_ptr error;
    std::mutex error_mutex;
    tensorplay::parallel::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            try {
                out[i] = decode_wav(data[i], frame_offset, num_frames);
            } catch (...) {
                std::lock_guard<std::mutex> guard(error_mutex);
                if (!error) error = std::current_exception();
            }
        }
    });
    if (error) std::rethrow_exception(error);
    return out;
}

Tensor encode_wav(Tensor data, int64_t sample_rate, int64_t bits) {
    if (data.device().type() != DeviceType::CPU) {
        TP_THROW(RuntimeError, "encode_wav: expected a CPU tensor");
    }
    if (data.dtype() != DType::Float32) {
        TP_THROW(TypeError, "encode_wav: expected a float32 tensor");
    }
    int64_t C;
    int64_t T;
    if (data.dim() == 1) {
        C = 1;
        T = data.size(0);
    } else if (data.dim() == 2) {
        C = data.size(0);
        T = data.size(1);
    } else {
        TP_THROW(RuntimeError, "encode_wav: expected a 1-D (mono) or 2-D (channels, time) tensor");
    }
    if (C < 1 || C > 65535) {
        TP_THROW(ValueError, "encode_wav: unsupported channel count");
    }
    if (sample_rate <= 0) {
        TP_THROW(ValueError, "encode_wav: sample_rate must be positive");
    }
    if (bits != 8 && bits != 16 && bits != 24 && bits != 32) {
        TP_THROW(ValueError, "encode_wav: bits must be 8, 16, 24 or 32");
    }
    if (!data.is_contiguous()) {
        TP_THROW(TypeError, "encode_wav: expected a contiguous tensor");
    }
    return encode_wav_impl(data.data_ptr<float>(), C, T,
                           static_cast<uint32_t>(sample_rate), bits);
}

std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t> wav_info(Tensor data) {
    require_cpu_uint8(data, "wav_info", 1, "a 1-D uint8 tensor");
    WavInfo info;
    parse_wav(data.data_ptr<uint8_t>(), static_cast<size_t>(data.numel()), &info);
    return {static_cast<int64_t>(info.sample_rate),
            static_cast<int64_t>(info.frames),
            static_cast<int64_t>(info.channels),
            static_cast<int64_t>(info.bits),
            static_cast<int64_t>(info.encoding)};
}

Tensor audio_to_tensor(py::object obj) {
    // We expect a numpy array
    py::array array = py::array::ensure(obj);
    if (!array) {
        TP_THROW(TypeError, "audio_to_tensor: expected a numpy array");
    }

    // Check dimensions
    size_t ndim = array.ndim();
    if (ndim != 1 && ndim != 2) {
        TP_THROW(RuntimeError, "audio_to_tensor: input must be 1D or 2D array");
    }

    size_t time_steps = array.shape(0);
    size_t channels = (ndim == 2) ? array.shape(1) : 1;

    py::dtype dt = array.dtype();
    size_t numel = channels * time_steps;
    if (numel > 0) {
        const py::ssize_t item = static_cast<py::ssize_t>(dt.itemsize());
        bool contiguous;
        if (ndim == 1) {
            contiguous = array.strides(0) == item;
        } else {
            contiguous = array.strides(0) == static_cast<py::ssize_t>(channels) * item
                      && array.strides(1) == item;
        }
        if (!contiguous) {
            py::object np = py::module_::import("numpy");
            array = py::array::ensure(np.attr("ascontiguousarray")(array));
            if (!array) {
                TP_THROW(RuntimeError,
                         "audio_to_tensor: failed to make the input contiguous");
            }
        }
    }

    // Output shape: (Channels, Time)
    std::vector<int64_t> out_shape = {static_cast<int64_t>(channels), static_cast<int64_t>(time_steps)};
    std::vector<int64_t> out_strides = {static_cast<int64_t>(time_steps), 1}; // Contiguous CHW (here C, T)

    float* data = new float[numel];

    char kind = dt.kind();
    size_t bits = dt.itemsize() * 8;
    const void* in_raw = array.data();

    if (kind == 'i' && bits == 16) {
        audio_load_i16(static_cast<const int16_t*>(in_raw), data, time_steps, channels);
    } else if (kind == 'i' && bits == 32) {
        audio_load_i32(static_cast<const int32_t*>(in_raw), data, time_steps, channels);
    } else if (kind == 'u' && bits == 8) {
        audio_load_u8(static_cast<const uint8_t*>(in_raw), data, time_steps, channels);
    } else if (kind == 'f' && bits == 32) {
        audio_load_f32(static_cast<const float*>(in_raw), data, time_steps, channels);
    } else {
        delete[] data;
        TP_THROW(TypeError, "audio_to_tensor: unsupported input dtype. Expected int16, int32, uint8 or float32.");
    }

    auto deleter = [](void* p) { delete[] static_cast<float*>(p); };
    tensorplay::DataPtr ptr(data, deleter, Device(DeviceType::CPU, 0));
    tensorplay::Storage storage(std::move(ptr), numel * sizeof(float));

    auto impl = std::make_shared<tensorplay::TensorImpl>(storage, out_shape, out_strides, DType::Float32);
    return Tensor(impl);
}
