// Adapted from cc4s CTF interface/semiring.h MTTKRP.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
use crate::algebra::Semiring;

pub(crate) fn mttkrp<A: Semiring>(
    algebra: &A,
    shape: &[usize],
    phases: &[usize],
    width: usize,
    output_mode: usize,
    pairs: &[(usize, A::Element)],
    factors: &[&[A::Element]],
    output: &mut [A::Element],
) {
    assert!(shape.len() >= 2, "MTTKRP requires tensor order at least two");
    assert!(output_mode < shape.len(), "MTTKRP output mode is out of bounds");

    let order = shape.len();
    let mut inds = vec![0usize; order - 1];
    let mut buffer = vec![algebra.zero(); width];
    let mut out_buffer = if output_mode != 0 {
        vec![algebra.zero(); width]
    } else {
        Vec::new()
    };
    let mut idx = 0usize;
    while idx < pairs.len() {
        let fiber_idx = pairs[idx].0 / shape[0];
        let mut fi = fiber_idx;
        for mode in 0..order - 1 {
            inds[mode] = (fi % shape[mode + 1]) / phases[mode + 1];
            fi /= shape[mode + 1];
        }

        let mut fiber_nnz = 1usize;
        while idx + fiber_nnz < pairs.len()
            && pairs[idx + fiber_nnz].0 / shape[0] == fiber_idx
        {
            fiber_nnz += 1;
        }

        if output_mode == 0 {
            buffer.clone_from_slice(
                &factors[1][inds[0] * width..(inds[0] + 1) * width],
            );
            for mode in 1..order - 1 {
                let factor = &factors[mode + 1][inds[mode] * width..(inds[mode] + 1) * width];
                for auxiliary in 0..width {
                    buffer[auxiliary] =
                        algebra.multiply(&buffer[auxiliary], &factor[auxiliary]);
                }
            }
            for (key, value) in &pairs[idx..idx + fiber_nnz] {
                let row = (*key % shape[0]) / phases[0];
                let out = &mut output[row * width..(row + 1) * width];
                for auxiliary in 0..width {
                    let product = algebra.multiply(value, &buffer[auxiliary]);
                    out[auxiliary] = algebra.add(&out[auxiliary], &product);
                }
            }
        } else {
            if output_mode > 1 {
                buffer.clone_from_slice(
                    &factors[1][inds[0] * width..(inds[0] + 1) * width],
                );
            } else if order > 2 {
                buffer.clone_from_slice(
                    &factors[2][inds[1] * width..(inds[1] + 1) * width],
                );
            } else {
                buffer.fill(algebra.one());
            }
            for mode in 1 + usize::from(output_mode == 1)..order - 1 {
                if output_mode != mode + 1 {
                    let factor =
                        &factors[mode + 1][inds[mode] * width..(inds[mode] + 1) * width];
                    for auxiliary in 0..width {
                        buffer[auxiliary] =
                            algebra.multiply(&buffer[auxiliary], &factor[auxiliary]);
                    }
                }
            }

            let output_row = inds[output_mode - 1];
            out_buffer.fill(algebra.zero());
            for (key, value) in &pairs[idx..idx + fiber_nnz] {
                let row = (*key % shape[0]) / phases[0];
                let factor = &factors[0][row * width..(row + 1) * width];
                for auxiliary in 0..width {
                    let product = algebra.multiply(value, &factor[auxiliary]);
                    out_buffer[auxiliary] = algebra.add(&out_buffer[auxiliary], &product);
                }
            }
            for auxiliary in 0..width {
                out_buffer[auxiliary] =
                    algebra.multiply(&out_buffer[auxiliary], &buffer[auxiliary]);
            }
            let out = &mut output[output_row * width..(output_row + 1) * width];
            for auxiliary in 0..width {
                out[auxiliary] = algebra.add(&out[auxiliary], &out_buffer[auxiliary]);
            }
        }

        idx += fiber_nnz;
    }
}
