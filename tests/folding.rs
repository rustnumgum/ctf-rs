#![cfg(feature = "native-linalg")]

use ctf::{
    folding::{Operand, Plan, Rejected},
    linalg::Native,
};

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut stride = 1;
    shape
        .iter()
        .map(|&dimension| {
            let result = stride;
            stride *= dimension;
            result
        })
        .collect()
}

fn offset(shape: &[usize], indices: &str, coordinates: &[usize; 256]) -> usize {
    strides(shape)
        .iter()
        .zip(indices.bytes())
        .map(|(&stride, label)| stride * coordinates[label as usize])
        .sum()
}

#[test]
fn high_order_multiple_k_weighted_batches_and_nontrivial_layout() {
    // a,p are k; i,x are m; y,z are n; w is a weighted batch.
    let shape_a = [2, 1, 2, 2, 2]; // i x a p w
    let shape_b = [2, 2, 2, 2, 1]; // y p w a z
    let shape_c = [1, 2, 2, 1, 2]; // z w i x y
    let a: Vec<_> = (1..=16).map(|x| x as f64).collect();
    let b: Vec<_> = (1..=16).rev().map(|x| x as f64).collect();
    let mut c: Vec<_> = (1..=8).map(|x| x as f64).collect();
    let old_c = c.clone();

    Plan::new([&shape_a, &shape_b, &shape_c], ["ixapw", "ypwaz", "zwixy"])
        .unwrap()
        .execute::<f64, Native>(&a, &b, &mut c, 2., 3.);

    let mut expected = old_c.iter().map(|value| 3. * value).collect::<Vec<_>>();
    let mut coordinates = [0; 256];
    for w in 0..2 {
        coordinates[b'w' as usize] = w;
        for y in 0..2 {
            coordinates[b'y' as usize] = y;
            for i in 0..2 {
                coordinates[b'i' as usize] = i;
                let output = offset(&shape_c, "zwixy", &coordinates);
                for p in 0..2 {
                    coordinates[b'p' as usize] = p;
                    for a_index in 0..2 {
                        coordinates[b'a' as usize] = a_index;
                        expected[output] += 2.
                            * a[offset(&shape_a, "ixapw", &coordinates)]
                            * b[offset(&shape_b, "ypwaz", &coordinates)];
                    }
                }
            }
        }
    }
    assert_eq!(c, expected);
}

#[test]
fn outer_product_with_alpha_and_beta() {
    let shape_a = [2, 2];
    let shape_b = [2, 2];
    let shape_c = [2, 2, 2, 2];
    let a = [1., 2., 3., 4.];
    let b = [5., 6., 7., 8.];
    let mut c = [1.; 16];

    Plan::new([&shape_a, &shape_b, &shape_c], ["ji", "qp", "piqj"])
        .unwrap()
        .execute::<f64, Native>(&a, &b, &mut c, 3., 2.);

    let mut coordinates = [0; 256];
    let mut expected = [0.; 16];
    for j in 0..2 {
        coordinates[b'j' as usize] = j;
        for i in 0..2 {
            coordinates[b'i' as usize] = i;
            for q in 0..2 {
                coordinates[b'q' as usize] = q;
                for p in 0..2 {
                    coordinates[b'p' as usize] = p;
                    let output = offset(&shape_c, "piqj", &coordinates);
                    expected[output] = 2.
                        + 3. * a[offset(&shape_a, "ji", &coordinates)]
                            * b[offset(&shape_b, "qp", &coordinates)];
                }
            }
        }
    }
    assert_eq!(c, expected);
}

#[test]
fn explicitly_rejects_repeated_and_one_operand_labels() {
    assert_eq!(
        Plan::new([&[2, 2], &[2], &[2]], ["ii", "i", "i"]).unwrap_err(),
        Rejected::RepeatedIndex {
            operand: Operand::A,
            label: 'i'
        }
    );
    assert_eq!(
        Plan::new([&[2, 3], &[2], &[]], ["ka", "k", ""]).unwrap_err(),
        Rejected::OneOperandLabel {
            operand: Operand::A,
            label: 'a'
        }
    );
}
