//! Exact tests for newly implemented local transpose and rank-shift slicing.
use ctf::{algebra::Arithmetic, context::Runtime, mapping::{Distribution, Mapping, Topology}, tensor::Tensor};

fn main() {
    let runtime = Runtime::initialize();
    let world = runtime.world();
    let topology = Topology::new(vec![world.size()]);
    let mut physical = Mapping::Unmapped;
    physical.augment_physical(&topology,0);
    physical.augment_virtual(world.size()*3);
    let distribution = Distribution::new(vec![7,5],topology,vec![physical,Mapping::Unmapped]);
    let mut tensor = Tensor::new(&world,distribution,Arithmetic::<i64>::new());
    tensor.transform(|key,value|*value = key as i64+1);
    let slice = tensor.slice(&[1..6,2..5]);
    for (key,value) in slice.local_pairs() {
        let c = slice.distribution().decode_key(key);
        assert_eq!(value, (c[0]+1 + (c[1]+2)*7 + 1) as i64);
    }
    // Read only one requested element per rank; no global gather is used.
    assert_eq!(slice.read(&[0]),vec![16]);
    let transpose = tensor.permute_axes(&[1,0]);
    for (key,value) in transpose.local_pairs() {
        let c = transpose.distribution().decode_key(key);
        assert_eq!(value,(c[1]+7*c[0]+1) as i64);
    }
    let nested = transpose.slice(&[1..4,1..3]);
    for (key,value) in nested.local_pairs() {
        let c = nested.distribution().decode_key(key);
        assert_eq!(value,(c[1]+1+7*(c[0]+1)+1) as i64);
    }
    let small = tensor.slice(&[2..3,0..1]);
    assert_eq!(small.read(&[0]),vec![3]);
    let empty = tensor.slice(&[3..3,0..5]);
    assert!(empty.local_storage().is_empty());
    assert_eq!(empty.reduce(),0);
    let mut diagonal = Tensor::new(&world,Distribution::cyclic(vec![5,5,2],world.size()),Arithmetic::<i64>::new());
    diagonal.transform(|_,value|*value=1);
    diagonal.transform_indexed("iij",|value|*value *= 7);
    for (key,value) in diagonal.local_pairs() {
        let c = diagonal.distribution().decode_key(key);
        assert_eq!(value,if c[0] == c[1] {7} else {1});
    }
    let scalar = Tensor::new(&world,Distribution::cyclic(vec![],world.size()),Arithmetic::<i64>::new());
    assert_eq!(scalar.slice(&[]).read(&[0]),vec![0]);
    drop(scalar); drop(diagonal); drop(empty); drop(small); drop(nested); drop(transpose); drop(slice); drop(tensor);
    if world.rank() == 0 { println!("DIGIT / PASS dense_views: exact slice offsets, virtual blocks, owner shifts, axis transpose, repeated labels, empty slices; ranks={}",world.size()); }
    world.close(); runtime.finalize();
}
