//! CPU/MPI-independent helpers adapted from `shared/util.cxx`.
//!
//! The functions below retain the source's column-major and packed-index
//! recurrences rather than expanding tensors or adding compatibility macros.

use crate::symmetry::Symmetry::{self, NS, SH, SY};

/// Storage size retaining diagonal slots for every symmetric group.
pub fn sy_packed_size(order: usize, len: &[i64], sym: &[Symmetry]) -> i64 {
    if order == 0 {
        return 1;
    }

    let mut k = 1i64;
    let mut temporary = 1i64;
    let mut size = 1i64;
    let mut maximum = len[0];
    for i in 0..order {
        temporary = temporary * maximum / k;
        k += 1;
        maximum += 1;
        if sym[i] == NS {
            size *= temporary;
            k = 1;
            temporary = 1;
            if i + 1 < order {
                maximum = len[i + 1];
            }
        }
    }
    size * temporary
}

/// Packed size excluding repeated coordinates for AS/SH groups.
pub fn packed_size(order: usize, len: &[i64], sym: &[Symmetry]) -> i64 {
    if order == 0 {
        return 1;
    }

    let mut k = 1i64;
    let mut temporary = 1i64;
    let mut size = 1i64;
    let mut maximum = len[0];
    for i in 0..order {
        temporary = temporary * maximum / k;
        k += 1;
        if sym[i] != SY {
            maximum -= 1;
        } else {
            maximum += 1;
        }
        if sym[i] == NS {
            size *= temporary;
            k = 1;
            temporary = 1;
            if i + 1 < order {
                maximum = len[i + 1];
            }
        }
    }
    size * temporary
}

fn calc_idx_arr_impl(
    order: usize,
    lens: &[i64],
    sym: &[Symmetry],
    idx: i64,
    use_symmetric_storage: bool,
) -> Vec<i64> {
    let mut idx_rem = idx;
    let mut idx_arr = vec![0i64; order];
    for dim in (0..order).rev() {
        if idx_rem == 0 {
            break;
        }
        if dim == 0 || sym[dim - 1] == NS {
            let lda = if use_symmetric_storage {
                sy_packed_size(dim, lens, sym)
            } else {
                packed_size(dim, lens, sym)
            };
            idx_arr[dim] = idx_rem / lda;
            idx_rem -= idx_arr[dim] * lda;
        } else {
            let mut plen = lens[..=dim].to_vec();
            let mut sg = 2usize;
            let mut factorial = 2i64;
            while dim >= sg && sym[dim - sg] != NS {
                sg += 1;
                factorial *= sg as i64;
            }
            let lda = if use_symmetric_storage {
                sy_packed_size(dim + 1 - sg, lens, sym)
            } else {
                packed_size(dim + 1 - sg, lens, sym)
            };
            let scaled = idx_rem as f64 * factorial as f64 / lda as f64;
            let mut candidate = scaled.powf(1.0 / sg as f64) as i64 + sg as i64 + 1;
            let mut prefix = 0i64;
            while candidate >= 0 {
                for length in &mut plen[dim + 1 - sg..=dim] {
                    *length = candidate;
                }
                prefix = if use_symmetric_storage {
                    sy_packed_size(dim + 1, &plen, sym)
                } else {
                    packed_size(dim + 1, &plen, sym)
                };
                if prefix <= idx_rem {
                    break;
                }
                candidate -= 1;
            }
            if prefix == 0 {
                candidate = 0;
            }
            idx_arr[dim] = candidate;
            idx_rem -= prefix;
        }
    }
    assert_eq!(idx_rem, 0);
    idx_arr
}

/// Decode an index in `packed_size` column-major storage.
pub fn calc_idx_arr(order: usize, lens: &[i64], sym: &[Symmetry], idx: i64) -> Vec<i64> {
    calc_idx_arr_impl(order, lens, sym, idx, false)
}

/// Decode an index in `sy_packed_size` storage, retaining AS/SH diagonal slots.
pub fn sy_calc_idx_arr(order: usize, lens: &[i64], sym: &[Symmetry], idx: i64) -> Vec<i64> {
    calc_idx_arr_impl(order, lens, sym, idx, true)
}

/// Prime factors in the source's ascending trial-division order.
pub fn factorize(n: i64) -> Vec<i64> {
    let mut remaining = n;
    let mut factors = Vec::new();
    while remaining > 1 {
        for divisor in 2..=remaining {
            if remaining % divisor == 0 {
                factors.push(divisor);
                remaining /= divisor;
                break;
            }
        }
    }
    factors
}

pub fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 { a } else { gcd(b, a % b) }
}

pub fn lcm(a: i64, b: i64) -> i64 {
    (a * b) / gcd(a, b)
}

/// Copy a column-major submatrix between byte buffers with arbitrary leading
/// dimensions.  `el_size` is measured in bytes, as in the source helper.
pub fn lda_cpy(
    el_size: usize,
    nrow: usize,
    ncol: usize,
    lda_a: usize,
    lda_b: usize,
    a: &[u8],
    b: &mut [u8],
) {
    if lda_a == nrow && lda_b == nrow {
        let bytes = el_size * nrow * ncol;
        b[..bytes].copy_from_slice(&a[..bytes]);
    } else {
        for col in 0..ncol {
            let source = el_size * lda_a * col;
            let target = el_size * lda_b * col;
            let bytes = nrow * el_size;
            b[target..target + bytes].copy_from_slice(&a[source..source + bytes]);
        }
    }
}

/// Apply a source permutation in place: `arr[i] <- old_arr[perm[i]]`.
pub fn permute<T: Clone>(perm: &[usize], arr: &mut [T]) {
    let swap: Vec<T> = perm.iter().map(|&source| arr[source].clone()).collect();
    arr[..perm.len()].clone_from_slice(&swap);
}

/// Build contiguous sizes and offsets for an `m`-by-`n` block table.
pub fn socopy(
    m: usize,
    n: usize,
    lda_a: usize,
    lda_b: usize,
    sizes_a: &[i64],
) -> (Vec<i64>, Vec<i64>) {
    let mut sizes_b = vec![0i64; lda_b * n];
    let mut offsets_b = vec![0i64; lda_b * n];
    let mut last_offset = 0i64;
    for col in 0..n {
        for row in 0..m {
            sizes_b[lda_b * col + row] = sizes_a[lda_a * col + row];
            offsets_b[lda_b * col + row] = last_offset;
            last_offset += sizes_a[lda_a * col + row];
        }
    }
    (sizes_b, offsets_b)
}

/// Copy each source block to its destination offset.  Sizes and offsets are
/// bytes, matching the source's `char` buffers.
pub fn spcopy(
    m: usize,
    n: usize,
    lda_a: usize,
    lda_b: usize,
    sizes_a: &[i64],
    offsets_a: &[i64],
    a: &[u8],
    sizes_b: &[i64],
    offsets_b: &[i64],
    b: &mut [u8],
) {
    for col in 0..n {
        for row in 0..m {
            let source_size = sizes_a[lda_a * col + row] as usize;
            assert_eq!(sizes_b[lda_b * col + row] as usize, source_size);
            let source_offset = offsets_a[lda_a * col + row] as usize;
            let target_offset = offsets_b[lda_b * col + row] as usize;
            b[target_offset..target_offset + source_size]
                .copy_from_slice(&a[source_offset..source_offset + source_size]);
        }
    }
}

pub fn fact(n: i64) -> i64 {
    let mut result = 1i64;
    for value in 1..=n {
        result *= value;
    }
    result
}

pub fn choose(n: i64, k: i64) -> i64 {
    fact(n) / (fact(k) * fact(n - k))
}

/// Decode the `ch`th k-combination in the source's SH,...,SH,NS table.
pub fn get_choice(n: usize, k: usize, ch: i64) -> Vec<i64> {
    if k == 0 {
        return Vec::new();
    }
    if k == 1 {
        return vec![ch];
    }
    let lens = vec![n as i64; k];
    let mut sym = vec![SH; k];
    sym[k - 1] = NS;
    calc_idx_arr(k, &lens, &sym, ch)
}

pub fn chchoose(n: i64, k: i64) -> i64 {
    fact(n + k - 1) / (fact(k) * fact(n - 1))
}
