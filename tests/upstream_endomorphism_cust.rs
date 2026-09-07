//! Deterministic port of pinned CTF test/endomorphism_cust.cxx.
use ctf::{
    algebra::{CustomMonoid, Wire},
    context::{Context, Runtime},
    mapping::Distribution,
    tensor::Tensor,
};

const N: usize = 5;

#[derive(Clone, Debug, PartialEq, Eq)]
struct Name {
    bytes: [u8; 256],
    len_name: u32,
}

impl Name {
    fn empty() -> Self { Self { bytes: [0; 256], len_name: 0 } }
    fn with_length(length: usize) -> Self {
        let mut value = Self::empty();
        value.bytes[..length].fill(b'a');
        value
    }
    fn length(&self) -> usize {
        self.bytes.iter().position(|&byte| byte == 0).unwrap()
    }
}

impl Wire for Name {
    const WIDTH: usize = 260;
    fn encode(&self, output: &mut Vec<u8>) {
        output.extend_from_slice(&self.bytes);
        output.extend_from_slice(&self.len_name.to_le_bytes());
    }
    fn decode(input: &[u8]) -> Self {
        assert_eq!(input.len(), Self::WIDTH);
        let mut bytes = [0; 256];
        bytes.copy_from_slice(&input[..256]);
        let len_name = u32::from_le_bytes(input[256..].try_into().unwrap());
        Self { bytes, len_name }
    }
}

fn longest(a: &Name, b: &Name) -> Name {
    if a.length() >= b.length() { a.clone() } else { b.clone() }
}

type Names = CustomMonoid<Name, fn(&Name, &Name) -> Name>;
fn algebra() -> Names { CustomMonoid { identity: Name::empty(), addition: longest } }

fn run(context: &Context<'_>) {
    let shape = vec![N + 1, N, N + 2, N + 3];
    let distribution = Distribution::cyclic(shape, context.size());
    let length = distribution.global_len();
    let pairs: Vec<_> = (0..length)
        .filter(|&key| distribution.owner(key) == context.rank())
        .map(|key| (key, Name::with_length((key * 73 + 19) % 250)))
        .collect();
    let mut a = Tensor::new(context, distribution, algebra());
    a.write_add(&pairs);
    a.transform_indexed("ijkl", |value| value.len_name = value.length() as u32);

    let keys: Vec<_> = (0..length).collect();
    for (key, value) in a.read(&keys).into_iter().enumerate() {
        assert_eq!(value.length(), value.len_name as usize,
            "source custom endomorphism length mismatch at key {key}");
    }
}

fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    run(&world);
    let parity = world.split(Some((world.rank() % 2) as i32), world.rank() as i32).unwrap();
    run(&parity);
    parity.close();
    if world.rank() == 0 {
        println!("DIGIT / PASS upstream_endomorphism_cust: cached string lengths; exact; world+parity");
    }
    world.close();
    runtime.finalize();
}
