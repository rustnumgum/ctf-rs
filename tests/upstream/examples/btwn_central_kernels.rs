// Rust equivalents of the custom kernels in
// examples/btwn_central_kernels.cxx from the pinned CTF source.

use ctf::algebra::{Monoid, Semiring, Wire};

pub const INFINITY: i32 = i32::MAX / 2;

/// glibc's default `rand` stream, used by the pinned WSL source.
pub struct GlibcRand {
    state: [u32; 31],
    front: usize,
    rear: usize,
}

impl GlibcRand {
    pub fn new(seed: u32) -> Self {
        let mut state = [0; 31];
        state[0] = if seed == 0 { 1 } else { seed };
        for index in 1..state.len() {
            state[index] = ((16807_i64 * i64::from(state[index - 1])) % 2147483647) as u32;
        }
        let mut result = Self {
            state,
            front: 3,
            rear: 0,
        };
        for _ in 0..(31 * 10) {
            result.next();
        }
        result
    }

    fn next(&mut self) -> u32 {
        let value = self.state[self.front].wrapping_add(self.state[self.rear]);
        self.state[self.front] = value;
        self.front = (self.front + 1) % self.state.len();
        self.rear = (self.rear + 1) % self.state.len();
        value >> 1
    }

    pub fn index(&mut self, bound: usize) -> usize {
        (self.next() as usize) % bound
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MPath {
    pub w: i32,
    pub m: i32,
}

impl MPath {
    pub const fn new(w: i32, m: i32) -> Self {
        Self { w, m }
    }
}

impl Wire for MPath {
    const WIDTH: usize = 8;

    fn encode(&self, output: &mut Vec<u8>) {
        self.w.encode(output);
        self.m.encode(output);
    }

    fn decode(input: &[u8]) -> Self {
        Self {
            w: i32::decode(&input[..4]),
            m: i32::decode(&input[4..8]),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CPath {
    pub w: i32,
    pub m: f32,
    pub c: f64,
}

impl CPath {
    pub const fn new(w: i32, m: f32, c: f64) -> Self {
        Self { w, m, c }
    }
}

impl Wire for CPath {
    const WIDTH: usize = 16;

    fn encode(&self, output: &mut Vec<u8>) {
        self.w.encode(output);
        self.m.encode(output);
        self.c.encode(output);
    }

    fn decode(input: &[u8]) -> Self {
        Self {
            w: i32::decode(&input[..4]),
            m: f32::decode(&input[4..8]),
            c: f64::decode(&input[8..16]),
        }
    }
}

#[derive(Clone, Copy)]
pub struct Tropical;

impl Monoid for Tropical {
    type Element = i32;

    fn zero(&self) -> Self::Element {
        INFINITY
    }

    fn add(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        (*left).min(*right)
    }
}

impl Semiring for Tropical {
    fn one(&self) -> Self::Element {
        0
    }

    fn multiply(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        *left + *right
    }
}

#[derive(Clone, Copy)]
pub struct MPathSemiring;

impl Monoid for MPathSemiring {
    type Element = MPath;

    fn zero(&self) -> Self::Element {
        MPath::new(INFINITY, 0)
    }

    fn add(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        if left.w < right.w {
            *left
        } else if right.w < left.w {
            *right
        } else {
            MPath::new(left.w, left.m + right.m)
        }
    }
}

impl Semiring for MPathSemiring {
    fn one(&self) -> Self::Element {
        MPath::new(0, 1)
    }

    fn multiply(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        MPath::new(left.w + right.w, left.m * right.m)
    }
}

#[derive(Clone, Copy)]
pub struct CPathMonoid;

impl Monoid for CPathMonoid {
    type Element = CPath;

    fn zero(&self) -> Self::Element {
        CPath::new(-INFINITY, 1.0, 0.0)
    }

    fn add(&self, left: &Self::Element, right: &Self::Element) -> Self::Element {
        if left.w > right.w {
            *left
        } else if right.w > left.w {
            *right
        } else {
            CPath::new(left.w, left.m, left.c + right.c)
        }
    }
}

/// Bellman `addw`: add one adjacency weight to a path distance.
pub fn bellman_function(weight: &i32, path: &MPath) -> MPath {
    MPath::new(path.w + *weight, path.m)
}

/// Bellman `mfunc`: retain the shorter path and add equal-distance counts.
pub fn bellman_accumulate(value: MPath, old: &mut MPath) {
    if value.w < old.w {
        *old = value;
    } else if old.w == value.w {
        old.m += value.m;
    }
}

/// Brandes `subw`: move one adjacency weight backward and carry its score.
pub fn brandes_function(weight: &i32, path: &CPath) -> CPath {
    CPath::new(path.w - *weight, path.m, path.c * path.m as f64)
}

/// Brandes `cfunc`: retain the larger reverse distance and add tied scores.
/// The source updates only `w` and `c` on a larger-distance replacement;
/// `m` is intentionally left unchanged.
pub fn brandes_accumulate(value: CPath, old: &mut CPath) {
    if value.w > old.w {
        old.w = value.w;
        old.c = value.c;
    } else if old.w == value.w {
        old.c += value.c;
    }
}

pub fn mpath_from_weight(weight: i32) -> MPath {
    MPath::new(weight, 1)
}

pub fn cpath_from_mpath_initial(path: MPath) -> CPath {
    CPath::new(path.w, 1.0 / path.m as f32, 1.0)
}

pub fn cpath_from_mpath_zero_score(path: MPath) -> CPath {
    CPath::new(path.w, 1.0 / path.m as f32, 0.0)
}

pub fn cpath_from_mpath_naive(path: MPath) -> CPath {
    CPath::new(path.w, path.m as f32, 0.0)
}
