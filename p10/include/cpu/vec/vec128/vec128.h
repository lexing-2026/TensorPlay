#pragma once

// On x86 the 128-bit vec layer is unused (mirrors ATen/cpu/vec/vec128/vec128.h,
// which is ARM-NEON-only). Kept so the vec.h include chain matches torch.
