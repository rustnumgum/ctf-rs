use ctf::{algebra::Arithmetic,context::{Context,Runtime},mapping::{Distribution,Mapping,Topology},
    symmetry::Symmetry::{self,*},symmetric_distribution::SymmetricDistribution,symmetric_tensor::SymmetricTensor};
fn tensor<'c,'r>(c:&'c Context<'r>,shape:Vec<usize>,links:Vec<Symmetry>,physical_axis:usize)
    ->SymmetricTensor<'c,'r,Arithmetic<i64>> {
    let topology=Topology::new(vec![c.size()]);let mut maps=vec![Mapping::Unmapped;shape.len()];
    if !maps.is_empty(){maps[physical_axis].augment_physical(&topology,0);}
    for map in &mut maps{map.augment_virtual(2*c.size());}
    SymmetricTensor::new(c,SymmetricDistribution::new(Distribution::new(shape,topology,maps),links),Arithmetic::new())
}
fn run(c:&Context<'_>) {
    let mut source=tensor(c,vec![3,3,2,2],vec![NS,NS,AS,NS],1);
    source.transform(|key,value|*value=key as i64+1);
    let (mut diagonal,labels)=source.extract_diagonal("iikl");
    assert_eq!(labels,"ikl");
    assert_eq!(diagonal.distribution().distribution().shape,vec![3,2,2]);
    assert_eq!(diagonal.distribution().links(),&[NS,AS,NS]);
    assert_eq!(diagonal.read(&[6,7,8,3,4,5]),vec![19,23,27,-19,-23,-27]);
    diagonal.transform(|_,value|*value+=100);
    let before=source.read(&(0..36).collect::<Vec<_>>());
    source.replace_diagonal("iikl",&diagonal);
    let after=source.read(&(0..36).collect::<Vec<_>>());
    for key in 0..36 {
        let i=key%3;let j=key/3%3;let k=key/9%2;let l=key/18;
        let correction=if i==j && k!=l {if k<l{100}else{-100}}else{0};
        assert_eq!(after[key],before[key]+correction);
    }
    let mut sy=tensor(c,vec![3,3],vec![SY,NS],0);
    sy.transform(|key,value|*value=key as i64+1);
    let (mut d,labels)=sy.extract_diagonal("ii");
    assert_eq!(labels,"i");assert_eq!(d.read(&[0,1,2]),vec![1,5,9]);
    d.transform(|_,value|*value*=2);sy.replace_diagonal("ii",&d);
    assert_eq!(sy.read(&[0,3,4,6,7,8]),vec![2,4,10,7,8,18]);
    let mut triple=tensor(c,vec![3,3,3],vec![NS,NS,NS],2);
    triple.transform(|key,value|*value=key as i64+1);
    let (d,labels)=triple.extract_diagonal("iii");
    assert_eq!(labels,"i");assert_eq!(d.read(&[0,1,2]),vec![1,14,27]);
    for kind in [AS,SH] {
        let mut hollow=tensor(c,vec![3,3],vec![kind,NS],0);
        hollow.transform(|key,value|*value=key as i64+1);
        let (mut d,labels)=hollow.extract_diagonal("ii");
        assert_eq!(labels,"i");assert_eq!(d.read(&[0,1,2]),vec![0;3]);
        d.transform(|_,value|*value=999);hollow.replace_diagonal("ii",&d);
        assert_eq!(hollow.read(&[0,3,4,6,7,8]),vec![0,4,0,7,8,0]);
    }
    let mut a=tensor(c,vec![3,3,2],vec![NS,NS,NS],1);
    a.transform(|key,value|*value=key as i64+1);
    let mut b=tensor(c,vec![3,3],vec![NS,NS],0);b.transform(|_,value|*value=10);
    b.sum_hollow_from("ii",&a,"iik",2,3);
    assert_eq!(b.read(&(0..9).collect::<Vec<_>>()),vec![52,10,10,10,68,10,10,10,84]);
}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let parity=world.split(Some((world.rank()%2) as i32),world.rank() as i32).unwrap();run(&parity);parity.close();
    if world.rank()==0{println!("DIGIT / PASS distributed_symmetric_diagonal: extraction/reinsertion, surviving AS links, SY diagonal, virtual mapping; exact i64; world+parity");}
    world.close();runtime.finalize();}
