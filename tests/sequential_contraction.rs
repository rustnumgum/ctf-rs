use ctf::{
    algebra::{Arithmetic, CustomMonoid, CustomSemiring},
    contraction::{sequential, sequential_function},
};
#[test]
fn matrix_product() {
    let mut c = [10; 4];
    sequential(
        &Arithmetic::<i64>::new(),
        &[2, 3],
        "ik",
        &[1, 2, 3, 4, 5, 6],
        &[3, 2],
        "kj",
        &[7, 8, 9, 10, 11, 12],
        &[2, 2],
        "ij",
        &mut c,
        &2,
        &3,
    );
    assert_eq!(c, [182, 230, 236, 302]);
}
#[test]
fn repeated_indices_and_scalar() {
    let mut c = [0];
    sequential(
        &Arithmetic::<i64>::new(),
        &[2, 2],
        "ii",
        &[1, 2, 3, 4],
        &[2],
        "i",
        &[5, 6],
        &[],
        "",
        &mut c,
        &1,
        &0,
    );
    assert_eq!(c, [29]);
    sequential(
        &Arithmetic::<i64>::new(),
        &[],
        "",
        &[2],
        &[],
        "",
        &[3],
        &[],
        "",
        &mut c,
        &2,
        &3,
    );
    assert_eq!(c, [99]);
}
#[test]
fn empty_and_custom_function() {
    let mut c = [5, 6];
    sequential(
        &Arithmetic::<i64>::new(),
        &[2, 0],
        "ik",
        &[],
        &[0],
        "k",
        &[],
        &[2],
        "i",
        &mut c,
        &1,
        &2,
    );
    assert_eq!(c, [10, 12]);
    sequential_function(
        &Arithmetic::<i64>::new(),
        &[2],
        "i",
        &[2, 3],
        &[2],
        "i",
        &[4, 5],
        &[2],
        "i",
        &mut c,
        &2,
        &0,
        |a, b| a + b,
    );
    assert_eq!(c, [12, 16]);
}
#[test]
fn boolean_semiring() {
    let algebra = CustomSemiring {
        monoid: CustomMonoid {
            identity: false,
            addition: |a: &bool, b: &bool| *a || *b,
        },
        identity: true,
        multiplication: |a: &bool, b: &bool| *a && *b,
    };
    let mut c = [false; 4];
    sequential(
        &algebra,
        &[2, 2],
        "ik",
        &[true, false, true, true],
        &[2, 2],
        "kj",
        &[false, true, true, false],
        &[2, 2],
        "ij",
        &mut c,
        &true,
        &false,
    );
    assert_eq!(c, [true, true, true, false]);
}
