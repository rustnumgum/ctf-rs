use ctf::{algebra::Arithmetic, summation::virtualized};

#[test]
fn virtual_reduction_beta_once() {
    // A has three k-blocks, each with two rows. B is one two-row block.
    let mut output = [10, 20];
    virtualized(
        &Arithmetic::<i64>::new(),
        &[2, 1],
        &[1, 3],
        "ik",
        &[1, 2, 3, 4, 5, 6],
        &[2],
        &[1],
        "i",
        &mut output,
        &2,
        &3,
    );
    assert_eq!(output, [48, 84]);
}

#[test]
fn virtual_broadcast_and_permutation() {
    let mut output = [0; 6];
    virtualized(
        &Arithmetic::<i64>::new(),
        &[1],
        &[2],
        "i",
        &[4, 7],
        &[1, 1],
        &[3, 2],
        "ji",
        &mut output,
        &1,
        &0,
    );
    assert_eq!(output, [4, 4, 4, 7, 7, 7]);
}

#[test]
fn virtual_diagonal() {
    let mut output = [9; 4];
    virtualized(
        &Arithmetic::<i64>::new(),
        &[1],
        &[2],
        "i",
        &[2, 3],
        &[1, 1],
        &[2, 2],
        "ii",
        &mut output,
        &1,
        &0,
    );
    assert_eq!(output, [2, 9, 9, 3]);
}
