use cubecl::prelude::*;

#[cube(launch_unchecked)]
pub fn matmul_naive_f32(
    a: &Array<f32>,
    b: &Array<f32>,
    c: &mut Array<f32>,
    m: u32,
    k: u32,
    n: u32
) {
    let row: u32 = ABSOLUTE_POS_Y;
    let col: u32 = ABSOLUTE_POS_X;

    if row < m && col < n {
        let mut acc = 0.0f32;
        let mut i: u32 = 0;
        while i < k {
            let a_idx: u32 = row * k + i;
            let b_idx: u32 = i * n + col;
            // If Array<T> in kernels expects u32, index directly; otherwise cast once:
            acc += a[a_idx] * b[b_idx];
            i += 1;
        }
        let c_idx: u32 = row * n + col;
        c[c_idx] = acc;
    }
}
