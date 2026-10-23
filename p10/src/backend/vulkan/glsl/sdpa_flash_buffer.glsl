#version 450 core
// clang-format off
#define PRECISION ${PRECISION}
// clang-format on

// A named constant (instead of a literal) lets the SPIR-V compiler fully
// unroll every loop below.
#define ROWS ${ROW_TILE}
#define KV 16
#define THREADS ${THREAD_COUNT}
#define DMAX 128
#define DQMAX (DMAX / 4)
#define DQPAD (DQMAX + 1)
#define KVPAD (KV + 1)
#define DOTS (ROWS * KV / THREADS)

layout(std430) buffer;

/*
 * Attention over linear storage, without a score matrix.
 *
 * A workgroup of THREADS lanes owns ROWS query rows of one (batch, head) pair
 * and walks the key sequence in chunks of KV.  Each chunk stages its key block
 * into shared memory, forms the ROWS x KV scores out of shared memory alone,
 * and folds them into the row's running maximum, exponential sum and weighted
 * value accumulator:
 *
 *   m' = max(m, max_k s_k)                pivot of the row so far
 *   w  = exp(m - m')                      rescaling of everything kept so far
 *   l  = w * l + sum_k exp(s_k - m')      exponential sum, on the new pivot
 *   O  = w * O + sum_k exp(s_k - m') V_k  value sum, on the new pivot
 *
 * and the output is O / l once the walk ends.  Only the four arrays above are
 * resident per workgroup, so the intermediate that a two-product chain would
 * write - one score per (query row, key) pair - never exists, which is what
 * makes a long prefill window affordable: its score block alone would run to
 * gigabytes.
 *
 * One staged block carries the keys while the tile's scores are formed and the
 * values while those scores are applied.  The two are the same shape, and a
 * value row is read only once every score of the chunk exists, so the slots the
 * key walk leaves behind stage the value block instead of a second array.  Both
 * halves of the chunk then read their operands from shared memory: a value row
 * staged once serves every query row of the tile, where reading it from the
 * payload would fetch the same elements again per row.
 *
 * The shared rows are four wide, since a score is a dot product of two of them
 * and a value row enters the accumulator at the same width: one step of either
 * walk carries four dimensions, and a row padded by a whole quad keeps the
 * lanes of a workgroup off one bank column.
 *
 * Entries a row may not attend carry a value far below any real score, so
 * exponentiating one against a pivot that the row does have underflows to zero
 * and it drops out of both sums.  A row with nothing to attend keeps its
 * running maximum below that sentinel, and then every weight is taken against
 * a pivot of zero: its output comes back as zeros rather than as the 0/0 the
 * plain formula would produce.
 *
 * The operands are dense in logical order, so every address is a stride sum;
 * the mask rides the same rule with zero strides on the axes it broadcasts
 * over, and an absent mask leaves its binding unused.  Grouped query heads map
 * a query head onto key/value head h / group directly.  The shared footprint
 * is (2 * ROWS + KV) * DQPAD quads plus ROWS * KVPAD + 4 * ROWS floats: a
 * wider row tile costs shared memory and saves operand traffic, because one
 * staged key and value window then serves twice the query rows.  The lane
 * count travels with the row count, so a lane owns one (row, key) score per
 * chunk.  The kernel needs no subgroup operation, no image operand and no
 * 16-bit storage.
 */
layout(set = 0, binding = 0) buffer PRECISION restrict writeonly OutBuffer {
  float data[];
}
uOutput;
layout(set = 0, binding = 1) buffer PRECISION restrict readonly QueryBuffer {
  float data[];
}
uQuery;
layout(set = 0, binding = 2) buffer PRECISION restrict readonly KeyBuffer {
  float data[];
}
uKey;
layout(set = 0, binding = 3) buffer PRECISION restrict readonly ValueBuffer {
  float data[];
}
uValue;
layout(set = 0, binding = 4) buffer PRECISION restrict readonly MaskBuffer {
  float data[];
}
uMask;
layout(set = 0, binding = 5) uniform PRECISION restrict Block {
  ivec4 q_strides;    // element steps of the query payload (batch, head, row, dim)
  ivec4 kv_strides;   // element steps shared by the key and value payloads
  ivec4 out_strides;  // element steps of the result
  ivec4 mask_strides; // element steps of the mask, zero on a broadcast axis
  int seq_q;          // Sq
  int seq_kv;         // Skv
  int dim;            // head dimension D
  int chunks;         // div_up(Skv, KV) key chunks to walk
  int d_quads;        // div_up(D, 4) staging groups per row
  int hq;             // query head count
  int group;          // query heads per key/value head (1 = plain multi-head)
  float scale;        // multiplier of every score
  int is_causal;      // 1: query row t attends the keys up to t
  int has_mask;       // 1: add the mask entry of each score
  int row_tiles;      // div_up(seq_q, ROWS) tiles of the dispatch's y axis
  int fill2;
}
uBlock;

// Below this a row has nothing it attends; see the header note on a row that
// normalizes to zeros.
const float EMPTY_PIVOT = -1.0e29f;
// The value an unattended entry carries.
const float REJECT = -1.0e30f;
// The state of a running maximum before any score has been seen.
const float NO_SCORE = -1.0e38f;

shared vec4 sQ[ROWS][DQPAD];
// Key block while the tile's scores are formed, value block once they are.
shared vec4 sKV[KV][DQPAD];
shared vec4 sO[ROWS][DQPAD];
shared float sS[ROWS][KVPAD];
shared float sM[ROWS];
shared float sL[ROWS];
shared float sW[ROWS];
shared float sC[ROWS];

layout(local_size_x_id = 0, local_size_y_id = 1, local_size_z_id = 2) in;

// Dot product of a staged query row with a staged key row.  Both hold whole
// quads, and the quad past the last dimension of the row carries zeros, so the
// walk covers ceil(dim / 4) steps and nothing more.
float dot_qk(const int r, const int kk) {
  float acc = 0.0;
  for (int dq = 0; dq < uBlock.d_quads; ++dq) {
    acc += dot(sQ[r][dq], sKV[kk][dq]);
  }
  return acc;
}

// Four consecutive elements of a payload row, or the remainder of one where the
// head dimension does not fill the quad.  The wide arm is the one every whole
// dimension takes; the other only ever runs for the last quad of a row.
vec4 payload_quad(const int src, const int d0, const int d_last) {
  if (d0 + 3 <= d_last) {
    return vec4(
        uValue.data[src],
        uValue.data[src + 1],
        uValue.data[src + 2],
        uValue.data[src + 3]);
  }
  vec4 tail = vec4(0.0);
  if (d0 <= d_last) {
    tail.x = uValue.data[src];
  }
  if (d0 + 1 <= d_last) {
    tail.y = uValue.data[src + 1];
  }
  if (d0 + 2 <= d_last) {
    tail.z = uValue.data[src + 2];
  }
  return tail;
}

vec4 query_quad(const int src, const int d0, const int d_last) {
  if (d0 + 3 <= d_last) {
    return vec4(
        uQuery.data[src],
        uQuery.data[src + 1],
        uQuery.data[src + 2],
        uQuery.data[src + 3]);
  }
  vec4 tail = vec4(0.0);
  if (d0 <= d_last) {
    tail.x = uQuery.data[src];
  }
  if (d0 + 1 <= d_last) {
    tail.y = uQuery.data[src + 1];
  }
  if (d0 + 2 <= d_last) {
    tail.z = uQuery.data[src + 2];
  }
  return tail;
}

vec4 key_quad(const int src, const int d0, const int d_last) {
  if (d0 + 3 <= d_last) {
    return vec4(
        uKey.data[src],
        uKey.data[src + 1],
        uKey.data[src + 2],
        uKey.data[src + 3]);
  }
  vec4 tail = vec4(0.0);
  if (d0 <= d_last) {
    tail.x = uKey.data[src];
  }
  if (d0 + 1 <= d_last) {
    tail.y = uKey.data[src + 1];
  }
  if (d0 + 2 <= d_last) {
    tail.z = uKey.data[src + 2];
  }
  return tail;
}

void main() {
  // The lanes of a workgroup run along x, and the two axes above it number the
  // row tile and the (batch, head) plane, so an axis holds workgroups rather
  // than lanes and keeps its whole range for the sequence it counts.
  const ivec3 pos = ivec3(gl_GlobalInvocationID);
  const int tid = pos.x;

  // A causal tile walks down to its own last row, so the tiles at the head of
  // the sequence are the shortest jobs.  They take the end of the launch, where
  // they fill the wave a long tile would otherwise leave half idle.
  const int row0 = (uBlock.row_tiles - 1 - pos.y) * ROWS;
  const int batch = pos.z / uBlock.hq;
  const int head = pos.z - batch * uBlock.hq;
  const int kv_head = head / uBlock.group;

  const int d_quads = uBlock.d_quads;
  const int d_last = uBlock.dim - 1;

  const int q_base = batch * uBlock.q_strides.x + head * uBlock.q_strides.y;
  const int kv_base =
      batch * uBlock.kv_strides.x + kv_head * uBlock.kv_strides.y;
  const int o_base = batch * uBlock.out_strides.x + head * uBlock.out_strides.y;
  const int m_base = batch * uBlock.mask_strides.x + head * uBlock.mask_strides.y;

  // Stage the query tile and clear the accumulator.  A row past the query
  // range stages nothing, which is what keeps a tile that overhangs the
  // sequence from contributing scores.
  for (int i = tid; i < ROWS * d_quads; i += THREADS) {
    const int r = i / d_quads;
    const int dq = i - r * d_quads;
    const int d0 = dq * 4;
    const int row = row0 + r;
    vec4 qv = vec4(0.0);
    if (row < uBlock.seq_q) {
      qv = query_quad(
          q_base + row * uBlock.q_strides.z + d0, d0, d_last);
    }
    sQ[r][dq] = qv;
    sO[r][dq] = vec4(0.0);
  }
  if (tid < ROWS) {
    sM[tid] = NO_SCORE;
    sL[tid] = 0.0;
  }
  barrier();

  // A causal row tile can attend no key past its own last row, so the walk
  // stops at that chunk: everything beyond it would only ever be rejected, and
  // a triangular window is half the chunks of the rectangular one it reads.
  int chunks = uBlock.chunks;
  if (uBlock.is_causal != 0) {
    const int reach = min(row0 + ROWS, uBlock.seq_q);
    chunks = min(chunks, (reach + KV - 1) / KV);
  }

  for (int chunk = 0; chunk < chunks; ++chunk) {
    const int k0 = chunk * KV;

    // One key block per chunk: read from the payload once however many rows
    // of the tile go on to use it.
    for (int i = tid; i < KV * d_quads; i += THREADS) {
      const int kk = i / d_quads;
      const int dq = i - kk * d_quads;
      const int d0 = dq * 4;
      const int key = k0 + kk;
      vec4 kv = vec4(0.0);
      if (key < uBlock.seq_kv) {
        kv = key_quad(
            kv_base + key * uBlock.kv_strides.z + d0, d0, d_last);
      }
      sKV[kk][dq] = kv;
    }
    barrier();

    // Scores of the whole tile: a lane owns DOTS (row, key) pairs.  A pair
    // outside either sequence, or past the causal window, carries the
    // rejection value instead of a score, so it needs no special handling
    // further down.
    for (int s = 0; s < DOTS; ++s) {
      const int idx = tid * DOTS + s;
      const int r = idx / KV;
      const int kk = idx - r * KV;
      const int row = row0 + r;
      const int key = k0 + kk;
      float value = REJECT;
      if (row < uBlock.seq_q && key < uBlock.seq_kv) {
        if (uBlock.is_causal == 0 || key <= row) {
          value = dot_qk(r, kk) * uBlock.scale;
          if (uBlock.has_mask != 0) {
            value += uMask.data
                [m_base + row * uBlock.mask_strides.z +
                 key * uBlock.mask_strides.w];
          }
        }
      }
      sS[r][kk] = value;
    }
    barrier();

    // Pivot and weights of the chunk, one lane per row.  The row's own score
    // slots are the only ones it touches.
    if (tid < ROWS) {
      const int r = tid;
      float m_new = sM[r];
      for (int kk = 0; kk < KV; ++kk) {
        m_new = max(m_new, sS[r][kk]);
      }
      const bool empty = m_new <= EMPTY_PIVOT;
      const float pivot = empty ? 0.0 : m_new;
      float total = 0.0;
      for (int kk = 0; kk < KV; ++kk) {
        const float e = exp(sS[r][kk] - pivot);
        sS[r][kk] = e;
        total += e;
      }
      sW[r] = sM[r] <= EMPTY_PIVOT ? 0.0 : exp(sM[r] - pivot);
      sC[r] = total;
      if (!empty) {
        sM[r] = m_new;
      }
    }

    // The key rows of this chunk are spent once every score of the tile is in
    // sS, so the same slots stage the value rows the accumulator reads.
    for (int i = tid; i < KV * d_quads; i += THREADS) {
      const int kk = i / d_quads;
      const int dq = i - kk * d_quads;
      const int d0 = dq * 4;
      const int key = k0 + kk;
      vec4 vv = vec4(0.0);
      if (key < uBlock.seq_kv) {
        vv = payload_quad(
            kv_base + key * uBlock.kv_strides.z + d0, d0, d_last);
      }
      sKV[kk][dq] = vv;
    }
    barrier();

    // Rescale what the walk kept so far and add the chunk.  A key a row may
    // not attend, or one past the sequence, carries a weight of zero, so its
    // staged quad multiplies to nothing and needs no guard of its own.
    for (int i = tid; i < ROWS * d_quads; i += THREADS) {
      const int r = i / d_quads;
      const int dq = i - r * d_quads;
      vec4 acc = sO[r][dq] * sW[r];
      for (int kk = 0; kk < KV; ++kk) {
        acc += sS[r][kk] * sKV[kk][dq];
      }
      sO[r][dq] = acc;
    }
    if (tid < ROWS) {
      sL[tid] = sL[tid] * sW[tid] + sC[tid];
    }
    barrier();
  }

  // A row whose sums stayed at zero keeps its accumulator, which is already
  // zero, by dividing nothing out of it.
  for (int i = tid; i < ROWS * d_quads; i += THREADS) {
    const int r = i / d_quads;
    const int dq = i - r * d_quads;
    const int d0 = dq * 4;
    const int row = row0 + r;
    if (row >= uBlock.seq_q) {
      continue;
    }
    const float total = sL[r];
    const float inv = total == 0.0 ? 0.0 : 1.0 / total;
    const vec4 value = sO[r][dq] * inv;
    const int dst = o_base + row * uBlock.out_strides.z + d0;
    if (d0 <= d_last) {
      uOutput.data[dst] = value.x;
    }
    if (d0 + 1 <= d_last) {
      uOutput.data[dst + 1] = value.y;
    }
    if (d0 + 2 <= d_last) {
      uOutput.data[dst + 2] = value.z;
    }
    if (d0 + 3 <= d_last) {
      uOutput.data[dst + 3] = value.w;
    }
  }
}
