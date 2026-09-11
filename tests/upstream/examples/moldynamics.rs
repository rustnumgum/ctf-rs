//! Native Rust form of pinned examples/moldynamics.h.

#![allow(dead_code)]

use ctf::{
    algebra::{Group, Monoid, Semiring, Wire},
    scalar_conversion::CastFromF64,
};

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Force {
    pub fx: f64,
    pub fy: f64,
}

impl Force {
    pub fn negated(self) -> Self {
        Self {
            fx: -self.fx,
            fy: -self.fy,
        }
    }
}

impl Wire for Force {
    const WIDTH: usize = 16;

    fn encode(&self, output: &mut Vec<u8>) {
        self.fx.encode(output);
        self.fy.encode(output);
    }

    fn decode(input: &[u8]) -> Self {
        Self {
            fx: f64::decode(&input[..8]),
            fy: f64::decode(&input[8..]),
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Particle {
    pub dx: f64,
    pub dy: f64,
    pub coeff: f64,
    pub id: i32,
}

impl Wire for Particle {
    const WIDTH: usize = 28;

    fn encode(&self, output: &mut Vec<u8>) {
        self.dx.encode(output);
        self.dy.encode(output);
        self.coeff.encode(output);
        self.id.encode(output);
    }

    fn decode(input: &[u8]) -> Self {
        Self {
            dx: f64::decode(&input[..8]),
            dy: f64::decode(&input[8..16]),
            coeff: f64::decode(&input[16..24]),
            id: i32::decode(&input[24..]),
        }
    }
}

#[derive(Clone, Copy)]
pub struct ParticleSet;

impl Monoid for ParticleSet {
    type Element = Particle;

    fn zero(&self) -> Particle {
        Particle::default()
    }

    fn add(&self, _old: &Particle, incoming: &Particle) -> Particle {
        *incoming
    }
}

#[derive(Clone, Copy)]
pub struct ForceAlgebra;

impl Monoid for ForceAlgebra {
    type Element = Force;

    fn zero(&self) -> Force {
        Force::default()
    }

    fn add(&self, left: &Force, right: &Force) -> Force {
        Force {
            fx: left.fx + right.fx,
            fy: left.fy + right.fy,
        }
    }
}

impl Group for ForceAlgebra {
    fn negate(&self, value: &Force) -> Force {
        value.negated()
    }
}

impl Semiring for ForceAlgebra {
    fn one(&self) -> Force {
        Force { fx: 1.0, fy: 1.0 }
    }

    fn multiply(&self, left: &Force, right: &Force) -> Force {
        Force {
            fx: left.fx * right.fx,
            fy: left.fy * right.fy,
        }
    }
}

impl CastFromF64 for ForceAlgebra {
    fn cast_f64(&self, value: f64) -> Force {
        Force {
            fx: value,
            fy: value,
        }
    }
}

pub fn acc_force(force: Force, particle: &mut Particle) {
    particle.dx += force.fx * particle.coeff;
    particle.dy += force.fy * particle.coeff;
}

pub fn get_distance(p: Particle, q: Particle) -> f64 {
    ((p.dx - q.dx) * (p.dx - q.dx) + (p.dy - q.dy) * (p.dy - q.dy)).sqrt()
}

pub fn get_force(p: Particle, q: Particle) -> Force {
    let denominator = (get_distance(p, q) + 0.01).powi(3);
    Force {
        fx: (p.dx - q.dx) / denominator,
        fy: (p.dy - q.dy) / denominator,
    }
}

/// POSIX `srand48`/`drand48`, retaining the source's rank-seeded fixture.
pub struct Drand48 {
    state: u64,
}

impl Drand48 {
    pub fn new(seed: usize) -> Self {
        Self {
            state: ((seed as u64) << 16) | 0x330e,
        }
    }

    pub fn next(&mut self) -> f64 {
        self.state =
            (0x5deece66du64.wrapping_mul(self.state).wrapping_add(0xb)) & ((1u64 << 48) - 1);
        self.state as f64 / (1u64 << 48) as f64
    }
}
