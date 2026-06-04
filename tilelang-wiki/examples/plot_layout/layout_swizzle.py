from tilelang.layout import (
    make_full_bank_swizzled_layout,
    make_half_bank_swizzled_layout,
    make_quarter_bank_swizzled_layout,
)
from tilelang.tools import plot_layout
from tvm import tirx


def make_shared_buffer(rows, cols, dtype="float16"):
    return tirx.decl_buffer((rows, cols), dtype, name="A", scope="shared")


# ---- Plot the swizzle patterns ----

# 1. Quarter-bank (32B) — 1-bit XOR — 8x16
# Rows 0-3: identity; Rows 4-7: two 8-element halves swap
layout = make_quarter_bank_swizzled_layout(make_shared_buffer(8, 16))
print(f"Quarter-bank swizzle (8x16, fp16): {layout}")
plot_layout(layout, name="swizzle_quarter_8x16")

# 2. Half-bank (64B) — 2-bit XOR — 8x32
layout = make_half_bank_swizzled_layout(make_shared_buffer(8, 32))
print(f"Half-bank swizzle (8x32, fp16): {layout}")
plot_layout(layout, name="swizzle_half_8x32")

# 3. Full-bank (128B) — 3-bit XOR — 8x64
layout = make_full_bank_swizzled_layout(make_shared_buffer(8, 64))
print(f"Full-bank swizzle (8x64, fp16): {layout}")
plot_layout(layout, name="swizzle_full_8x64")

# 4. Full-bank (128B) — multi-tile: 32x64
layout = make_full_bank_swizzled_layout(make_shared_buffer(32, 64))
print(f"Full-bank swizzle (32x64, fp16): {layout}")
plot_layout(layout, name="swizzle_full_32x64")
