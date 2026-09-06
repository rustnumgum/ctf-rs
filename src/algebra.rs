// Independently written Rust algebra interfaces; upstream responsibilities below.
//! Algebraic responsibilities from interface/{monoid,group,semiring,ring}.
//! Laws (associativity, identities and distributivity) are the implementor's contract.

pub trait Monoid {
    type Element: Clone + PartialEq;
    fn zero(&self) -> Self::Element;
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
}

pub trait Group: Monoid {
    fn negate(&self, a: &Self::Element) -> Self::Element;
}

pub trait Semiring: Monoid {
    fn one(&self) -> Self::Element;
    fn multiply(&self, a: &Self::Element, b: &Self::Element) -> Self::Element;
}

pub trait Ring: Group + Semiring {}
impl<A: Group + Semiring> Ring for A {}

/// User-defined associative addition with its explicit identity.
pub struct CustomMonoid<T, Add> {
    pub identity: T,
    pub addition: Add,
}
impl<T: Clone + PartialEq, Add: Fn(&T, &T) -> T> Monoid for CustomMonoid<T, Add> {
    type Element = T;
    fn zero(&self) -> T { self.identity.clone() }
    fn add(&self, a: &T, b: &T) -> T { (self.addition)(a, b) }
}

pub struct CustomSemiring<M: Monoid, Mul> {
    pub monoid: M,
    pub identity: M::Element,
    pub multiplication: Mul,
}
impl<M: Monoid, Mul> Monoid for CustomSemiring<M, Mul> {
    type Element = M::Element;
    fn zero(&self) -> Self::Element { self.monoid.zero() }
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.monoid.add(a, b)
    }
}
impl<M: Monoid, Mul: Fn(&M::Element, &M::Element) -> M::Element> Semiring for CustomSemiring<M, Mul> {
    fn one(&self) -> Self::Element { self.identity.clone() }
    fn multiply(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        (self.multiplication)(a, b)
    }
}

pub struct CustomRing<S: Semiring, Neg> {
    pub semiring: S,
    pub negation: Neg,
}
impl<S: Semiring, Neg> Monoid for CustomRing<S, Neg> {
    type Element = S::Element;
    fn zero(&self) -> Self::Element { self.semiring.zero() }
    fn add(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.semiring.add(a, b)
    }
}
impl<S: Semiring, Neg> Semiring for CustomRing<S, Neg> {
    fn one(&self) -> Self::Element { self.semiring.one() }
    fn multiply(&self, a: &Self::Element, b: &Self::Element) -> Self::Element {
        self.semiring.multiply(a, b)
    }
}
impl<S: Semiring, Neg: Fn(&S::Element) -> S::Element> Group for CustomRing<S, Neg> {
    fn negate(&self, a: &Self::Element) -> Self::Element { (self.negation)(a) }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct Arithmetic<T>(std::marker::PhantomData<T>);
impl<T> Arithmetic<T> { pub fn new() -> Self { Self(std::marker::PhantomData) } }

macro_rules! arithmetic {
    ($($t:ty),*) => {$(
        impl Monoid for Arithmetic<$t> {
            type Element = $t;
            fn zero(&self) -> $t { 0 as $t }
            fn add(&self, a: &$t, b: &$t) -> $t { *a + *b }
        }
        impl Semiring for Arithmetic<$t> {
            fn one(&self) -> $t { 1 as $t }
            fn multiply(&self, a: &$t, b: &$t) -> $t { *a * *b }
        }
        impl Group for Arithmetic<$t> {
            fn negate(&self, a: &$t) -> $t { -*a }
        }
    )*}
}
arithmetic!(f32, f64, i32, i64);

/// Explicit serialization, rather than transmitting Rust object representations.
/// Each encoded element occupies exactly WIDTH bytes on every rank.
pub trait Wire: Sized {
    const WIDTH: usize;
    fn encode(&self, output: &mut Vec<u8>);
    fn decode(input: &[u8]) -> Self;
}
macro_rules! wire_number {
    ($($t:ty),*) => {$(
        impl Wire for $t {
            const WIDTH: usize = size_of::<$t>();
            fn encode(&self, output: &mut Vec<u8>) { output.extend_from_slice(&self.to_le_bytes()); }
            fn decode(input: &[u8]) -> Self { Self::from_le_bytes(input.try_into().unwrap()) }
        }
    )*}
}
wire_number!(i32, i64, u32, u64, f32, f64);
impl Wire for bool {
    const WIDTH: usize = 1;
    fn encode(&self, output: &mut Vec<u8>) { output.push(u8::from(*self)); }
    fn decode(input: &[u8]) -> Self { assert!(input[0] <= 1); input[0] == 1 }
}
