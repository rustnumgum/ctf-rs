use ctf::{algebra::Arithmetic,context::{Context,Runtime},mapping::{Distribution,Mapping,Topology},
    symmetry::Symmetry::{self,*},symmetric_distribution::SymmetricDistribution,symmetric_tensor::SymmetricTensor};
fn tensor<'c,'r>(c:&'c Context<'r>,shape:Vec<usize>,links:Vec<Symmetry>,replicas:bool)
    ->SymmetricTensor<'c,'r,Arithmetic<i64>> {
    let topology=Topology::new(vec![c.size()]);let mut maps=vec![Mapping::Unmapped;shape.len()];
    if !replicas && !maps.is_empty(){maps[0].augment_physical(&topology,0);}
    for map in &mut maps {map.augment_virtual(2*c.size());}
    SymmetricTensor::new(c,SymmetricDistribution::new(Distribution::new(shape,topology,maps),links),Arithmetic::new())
}
fn run(c:&Context<'_>){
    let mut a=tensor(c,vec![3,3],vec![SY,NS],false);a.transform(|key,v|*v=key as i64+1);
    let mut b=tensor(c,vec![3,3],vec![NS,NS],true);b.transform(|_,v|*v=10);
    b.sum_canonical_from("ij",&a,"ij",2,3);
    assert_eq!(b.read(&(0..9).collect::<Vec<_>>()),vec![32,30,30,38,40,30,44,46,48]);
    let mut rows=tensor(c,vec![3],vec![NS],false);rows.transform(|_,v|*v=10);
    rows.sum_canonical_from("i",&a,"ij",2,3);
    assert_eq!(rows.read(&[0,1,2]),vec![54,56,48]);
    let mut scalar=tensor(c,vec![],vec![],true);scalar.transform(|_,v|*v=10);
    scalar.sum_canonical_from("",&a,"ii",2,3);assert_eq!(scalar.read(&[0]),vec![60]);
    let mut diagonal=tensor(c,vec![3,3],vec![SY,NS],false);diagonal.transform(|_,v|*v=10);
    diagonal.sum_canonical_from("ii",&rows,"i",2,3);
    assert_eq!(diagonal.read(&[0,3,4,6,7,8]),vec![138,10,142,10,10,126]);
    let mut vector=tensor(c,vec![3],vec![NS],true);vector.transform(|key,v|*v=key as i64+1);
    let mut broadcast=tensor(c,vec![3,3],vec![AS,NS],false);
    broadcast.sum_canonical_from("ij",&vector,"i",1,0);
    assert_eq!(broadcast.read(&(0..9).collect::<Vec<_>>()),vec![0,-1,-1,1,0,-2,1,2,0]);
    // sum_tensors aligns preserved symmetric labels before the raw kernel.
    let mut transposed=tensor(c,vec![3,3],vec![SY,NS],true);
    transposed.sum_canonical_from("ij",&a,"ji",1,0);
    assert_eq!(transposed.read(&[0,3,4,6,7,8]),vec![1,4,5,7,8,9]);
    let empty=tensor(c,vec![0,0],vec![SY,NS],false);
    scalar.sum_canonical_from("",&empty,"ij",2,3);assert_eq!(scalar.read(&[0]),vec![180]);
}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let parity=world.split(Some((world.rank()%2) as i32),world.rank() as i32).unwrap();run(&parity);parity.close();
    if world.rank()==0{println!("DIGIT / PASS distributed_canonical_sum: packed tensor routing, reductions, broadcast, diagonals, replicas; exact i64; world+parity");}
    world.close();runtime.finalize();}
