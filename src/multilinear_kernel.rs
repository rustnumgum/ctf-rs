// Adapted from cc4s CTF interface/semiring.h MTTKRP.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
pub(crate) fn mttkrp(
    shape: &[usize],
    phases: &[usize],
    width: usize,
    output_mode: usize,
    pairs: &[(usize, f64)],
    factors: &[&[f64]],
    output: &mut [f64],
) {
    assert!(shape.len() >= 2, "MTTKRP requires tensor order at least two");
    assert!(output_mode < shape.len(), "MTTKRP output mode is out of bounds");

    let order = shape.len();
    let mut inds = vec![0usize; order - 1];
    let mut buffer = vec![0.; width];
    let mut out_buffer = if output_mode != 0 { vec![0.; width] } else { Vec::new() };
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
            buffer.copy_from_slice(
                &factors[1][inds[0] * width..(inds[0] + 1) * width],
            );
            for mode in 1..order - 1 {
                let factor = &factors[mode + 1][inds[mode] * width..(inds[mode] + 1) * width];
                for auxiliary in 0..width {
                    buffer[auxiliary] *= factor[auxiliary];
                }
            }
            for &(key, value) in &pairs[idx..idx + fiber_nnz] {
                let row = (key % shape[0]) / phases[0];
                let out = &mut output[row * width..(row + 1) * width];
                for auxiliary in 0..width {
                    out[auxiliary] += value * buffer[auxiliary];
                }
            }
        } else {
            if output_mode > 1 {
                buffer.copy_from_slice(
                    &factors[1][inds[0] * width..(inds[0] + 1) * width],
                );
            } else if order > 2 {
                buffer.copy_from_slice(
                    &factors[2][inds[1] * width..(inds[1] + 1) * width],
                );
            } else {
                buffer.fill(1.0);
            }
            for mode in 1 + usize::from(output_mode == 1)..order - 1 {
                if output_mode != mode + 1 {
                    let factor =
                        &factors[mode + 1][inds[mode] * width..(inds[mode] + 1) * width];
                    for auxiliary in 0..width {
                        buffer[auxiliary] *= factor[auxiliary];
                    }
                }
            }

            let output_row = inds[output_mode - 1];
            out_buffer.fill(0.);
            for &(key, value) in &pairs[idx..idx + fiber_nnz] {
                let row = (key % shape[0]) / phases[0];
                let factor = &factors[0][row * width..(row + 1) * width];
                for auxiliary in 0..width {
                    out_buffer[auxiliary] += value * factor[auxiliary];
                }
            }
            for auxiliary in 0..width {
                out_buffer[auxiliary] *= buffer[auxiliary];
            }
            let out = &mut output[output_row * width..(output_row + 1) * width];
            for auxiliary in 0..width {
                out[auxiliary] += out_buffer[auxiliary];
            }
        }

        idx += fiber_nnz;
    }
}
