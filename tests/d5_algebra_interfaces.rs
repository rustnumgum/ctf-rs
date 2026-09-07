//! D5 exact checks for the native set/monoid/ring surface.
//!
//! The Rust algebra traits are the direct replacement for the source
//! hierarchy.  Set's raw storage/ABI duties are represented by owned COO,
//! CSR and CCSR values, while MPI reductions consume the same Monoid contract.

use ctf::{
    algebra::{Arithmetic, CustomMonoid, CustomRing, CustomSemiring, Group, Monoid, Ring, Semiring, Wire},
    context::{Context, Runtime},
    sparse_formats::Coo,
};

#[derive(Clone, Debug, PartialEq)]
struct Affine(i64, i64);

impl Wire for Affine {
    const WIDTH: usize = 16;

    fn encode(&self, output: &mut Vec<u8>) {
        self.0.encode(output);
        self.1.encode(output);
    }

    fn decode(input: &[u8]) -> Self {
        Self(i64::decode(&input[..8]), i64::decode(&input[8..16]))
    }
}

fn compose(left: &Affine, right: &Affine) -> Affine {
    Affine(left.0 * right.0, left.0 * right.1 + left.1)
}

fn sum(left: &i64, right: &i64) -> i64 {
    *left + *right
}

fn product(left: &i64, right: &i64) -> i64 {
    *left * *right
}

fn negate(value: &i64) -> i64 {
    -*value
}

fn assert_ring<A: Ring>(_: &A) {}

fn check_ring_identities() {
    macro_rules! check {
        ($ty:ty, $x:expr, $y:expr, $sum:expr, $product:expr, $negative:expr) => {{
            let algebra = Arithmetic::<$ty>::new();
            assert_ring(&algebra);
            assert_eq!(algebra.add(&algebra.zero(), &$x), $x);
            assert_eq!(algebra.multiply(&algebra.one(), &$x), $x);
            assert_eq!(algebra.multiply(&$x, &algebra.one()), $x);
            assert_eq!(algebra.add(&$x, &$y), $sum);
            assert_eq!(algebra.multiply(&$x, &$y), $product);
            assert_eq!(algebra.negate(&$x), $negative);
        }};
    }

    check!(f32, 6.0, 7.0, 13.0, 42.0, -6.0);
    check!(f64, 6.0, 7.0, 13.0, 42.0, -6.0);
    check!(i32, 6, 7, 13, 42, -6);
    check!(i64, 6, 7, 13, 42, -6);
    check!(u32, 6, 7, 13, 42, u32::MAX - 5);
    check!(u64, 6, 7, 13, 42, u64::MAX - 5);
}

fn check_custom_ring() {
    let algebra = CustomRing {
        semiring: CustomSemiring {
            monoid: CustomMonoid {
                identity: 0_i64,
                addition: sum,
            },
            identity: 1_i64,
            multiplication: product,
        },
        negation: negate,
    };

    let zero = algebra.zero();
    let one = algebra.one();
    assert_eq!(zero, 0);
    assert_eq!(one, 1);
    assert_eq!(algebra.add(&zero, &6), 6);
    assert_eq!(algebra.multiply(&one, &7), 7);
    assert_eq!(algebra.negate(&6), -6);
}

fn check_set_conversions() {
    let coo = Coo::new(
        4,
        3,
        vec![(4, 3, 9_i64), (1, 2, 2), (3, 1, 6), (1, 1, 3), (1, 2, 0)],
    );
    let csr = coo.to_csr();
    assert_eq!(csr.row_offsets(), &[1, 4, 4, 5, 6]);
    assert_eq!(csr.columns(), &[1, 2, 2, 1, 3]);
    assert_eq!(csr.values(), &[3, 2, 0, 6, 9]);

    let ccsr = coo.to_ccsr();
    assert_eq!(ccsr.row_encoding(), &[1, 3, 4]);
    assert_eq!(ccsr.row_offsets(), &[1, 4, 5, 6]);
    assert_eq!(ccsr.columns(), csr.columns());
    assert_eq!(ccsr.values(), csr.values());
    assert_eq!(ccsr.to_coo(), csr.to_coo());
    assert_eq!(csr.to_coo().entries(), &[(1, 1, 3), (1, 2, 2), (1, 2, 0), (3, 1, 6), (4, 3, 9)]);
}

fn check_monoid_production(context: &Context<'_>) {
    let algebra = CustomMonoid {
        identity: Affine(1, 0),
        addition: compose,
    };

    let mut values = [Affine(2, context.rank() as i64), Affine(1, 1)];
    context.all_reduce_monoid(&algebra, &mut values, false);
    let mut expected = Affine(1, 0);
    for rank in 0..context.size() {
        expected = compose(&expected, &Affine(2, rank as i64));
    }
    assert_eq!(values, [expected, Affine(1, context.size() as i64)]);

    let a = Coo::new(1, 1, vec![(1, 1, Affine(2, 3))]).to_csr();
    let b = Coo::new(1, 1, vec![(1, 1, Affine(5, 7))]).to_csr();
    assert_eq!(a.add(&b, &algebra).to_coo().entries(), &[(1, 1, Affine(10, 17))]);
}

fn run(context: &Context<'_>) {
    check_ring_identities();
    check_custom_ring();
    check_set_conversions();
    check_monoid_production(context);
}

fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    run(&world);

    let parity = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    run(&parity);
    parity.close();

    if world.rank() == 0 {
        println!(
            "DIGIT / PASS d5_algebra_interfaces: ring identities, custom monoid reduction, COO/CSR/CCSR conversion; world+parity"
        );
    }
    world.close();
    runtime.finalize();
}
