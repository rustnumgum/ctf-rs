use ctf::{algebra::{Arithmetic, Monoid, Semiring}, sparse_sequential::sequential};

#[test]
fn sparse_key_recursion_with_output_and_dense_input_only_axes() {
    let a = [(0,2_i64),(3,-1),(4,3)];
    let b: Vec<_> = (0..12).map(|key| key % 5 - 2).collect();
    let mut c: Vec<_> = (0..12).map(|key| key + 7).collect();
    let expected: Vec<_> = (0..12).map(|key| {
        let i = key % 2; let j = key / 2 % 2;
        let mut value = 3 * (key as i64 + 7);
        for &(ak, av) in &a { if ak % 2 == i { for q in 0..2 {
            value += 2 * av * b[ak / 2 + 3 * (q + 2*j)];
        } } }
        value
    }).collect();
    sequential(&Arithmetic::<i64>::new(), &[2,3], "ik", &a, &[3,2,2], "kqj", &b,
        &[2,2,3], "ijx", &mut c, &2, &3);
    assert_eq!(c, expected);
}

#[test]
fn empty_sparse_and_zero_extent_scale_output_once() {
    let algebra = Arithmetic::<i64>::new();
    let mut c = [4,5];
    sequential(&algebra, &[2,3], "ik", &[], &[3], "k", &[2,3,4],
        &[2], "i", &mut c, &7, &2);
    assert_eq!(c, [8,10]);
    sequential(&algebra, &[2,0], "ik", &[], &[0], "k", &[],
        &[2], "i", &mut c, &7, &3);
    assert_eq!(c, [24,30]);
}

struct Matrices;
impl Monoid for Matrices {
    type Element = [i64;4];
    fn zero(&self)->Self::Element {[0;4]}
    fn add(&self,a:&Self::Element,b:&Self::Element)->Self::Element {
        std::array::from_fn(|i|a[i]+b[i])
    }
}
impl Semiring for Matrices {
    fn one(&self)->Self::Element {[1,0,0,1]}
    fn multiply(&self,a:&Self::Element,b:&Self::Element)->Self::Element {
        [a[0]*b[0]+a[1]*b[2],a[0]*b[1]+a[1]*b[3],
         a[2]*b[0]+a[3]*b[2],a[2]*b[1]+a[3]*b[3]]
    }
}
#[test]
fn noncommutative_scalar_products() {
    let algebra = Matrices;
    let a = [1,2,0,1]; let b = [2,0,1,3];
    let alpha = [1,0,2,1]; let beta = [2,1,0,1]; let old = [3,0,1,2];
    let mut c = [old];
    sequential(&algebra, &[1], "i", &[(0,a)], &[1], "i", &[b],
        &[1], "i", &mut c, &alpha, &beta);
    let product = algebra.multiply(&algebra.multiply(&a,&b),&alpha);
    assert_eq!(c[0],algebra.add(&product,&algebra.multiply(&beta,&old)));
}
