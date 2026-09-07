// Virtual-block traversal adapted from cc4s CTF contraction/spctr_tsr.cxx
// spctr_virt::run. Copyright (c) 2011, Edgar Solomonik. See LICENSE.

/// Visit every virtual-label coordinate with dimension zero varying fastest.
/// The callback receives the A/B/C virtual block offsets and the output
/// coefficient: `beta` on the first visit to that C block, then `one`.
pub fn execute<E: Clone>(
    dimensions: &[usize],
    indices: [&[usize]; 3],
    beta: &E,
    one: &E,
    mut kernel: impl FnMut([usize; 3], &E),
) {
    assert!(dimensions.iter().all(|&dimension| dimension > 0));

    let mut block_counts = [1usize; 3];
    let mut union_strides = [
        vec![0usize; dimensions.len()],
        vec![0usize; dimensions.len()],
        vec![0usize; dimensions.len()],
    ];
    for operand in 0..3 {
        for &index in indices[operand] {
            assert!(index < dimensions.len());
            let stride = block_counts[operand];
            block_counts[operand] *= dimensions[index];
            union_strides[operand][index] += stride;
        }
    }

    let mut output_visited = vec![false; block_counts[2]];
    let mut coordinates = vec![0usize; dimensions.len()];
    let mut offsets = [0usize; 3];
    loop {
        let coefficient = if output_visited[offsets[2]] {
            one
        } else {
            output_visited[offsets[2]] = true;
            beta
        };
        kernel(offsets, coefficient);

        let mut dimension = 0;
        while dimension < dimensions.len() {
            for operand in 0..3 {
                offsets[operand] -=
                    union_strides[operand][dimension] * coordinates[dimension];
            }
            coordinates[dimension] += 1;
            if coordinates[dimension] >= dimensions[dimension] {
                coordinates[dimension] = 0;
            }
            for operand in 0..3 {
                offsets[operand] +=
                    union_strides[operand][dimension] * coordinates[dimension];
            }
            if coordinates[dimension] != 0 {
                break;
            }
            dimension += 1;
        }
        if dimension == dimensions.len() {
            break;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::execute;

    #[test]
    fn gemm_order_and_beta_once_per_output_block() {
        let mut actual = Vec::new();
        execute(
            &[2, 3, 2],
            [&[0, 1], &[1, 2], &[0, 2]],
            &7,
            &1,
            |offsets, coefficient| actual.push((offsets, *coefficient)),
        );

        let mut expected = Vec::new();
        for j in 0..2 {
            for k in 0..3 {
                for i in 0..2 {
                    expected.push(([i + 2 * k, k + 3 * j, i + 2 * j], if k == 0 { 7 } else { 1 }));
                }
            }
        }
        assert_eq!(actual, expected);
    }

    #[test]
    fn repeated_output_index_visits_only_diagonal_blocks() {
        let mut output = vec![99; 9];
        let mut visited = Vec::new();
        execute(&[3], [&[0], &[0], &[0, 0]], &7, &1, |offsets, coefficient| {
            visited.push(offsets[2]);
            output[offsets[2]] = *coefficient;
        });
        assert_eq!(visited, vec![0, 4, 8]);
        assert_eq!(output, vec![7, 99, 99, 99, 7, 99, 99, 99, 7]);
    }

    #[test]
    fn scalar_calls_kernel_once() {
        let mut calls = Vec::new();
        execute(&[], [&[], &[], &[]], &7, &1, |offsets, coefficient| {
            calls.push((offsets, *coefficient));
        });
        assert_eq!(calls, vec![([0, 0, 0], 7)]);
    }

    #[test]
    fn sparse_output_buckets_keep_output_block_ownership() {
        let mut buckets = [vec![10], vec![20]];
        execute(
            &[2, 3],
            [&[0, 1], &[1], &[0]],
            &7,
            &1,
            |offsets, coefficient| {
                let bucket = &mut buckets[offsets[2]];
                if *coefficient == 7 {
                    bucket[0] *= *coefficient;
                }
                bucket.push(offsets[0] as i32);
            },
        );
        assert_eq!(buckets[0], vec![70, 0, 2, 4]);
        assert_eq!(buckets[1], vec![140, 1, 3, 5]);
    }
}
