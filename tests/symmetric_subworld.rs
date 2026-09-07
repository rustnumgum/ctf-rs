use ctf::{algebra::{Arithmetic,Complex,Ring,Wire},context::{Context,Runtime},mapping::{Distribution,Mapping,Topology},
    symmetric_distribution::SymmetricDistribution,symmetric_tensor::SymmetricTensor,symmetry::Symmetry::{self,NS,SY,AS,SH}};
use std::fmt::Debug;
fn layout(np:usize,n:usize,kind:Symmetry,axis:Option<usize>)->SymmetricDistribution{
    let top=Topology::new(vec![np]);let mut maps=vec![Mapping::Unmapped;2];
    if let Some(axis)=axis{maps[axis].augment_physical(&top,0);}
    for map in &mut maps{map.augment_virtual(2*np);}
    SymmetricDistribution::new(Distribution::new(vec![n,n],top,maps),vec![kind,NS])
}
fn check<A:Ring+Clone>(c:&Context<'_>,algebra:A,value:impl Fn(i64)->A::Element)
where A::Element:Wire+Debug {
    let child=c.split((c.rank()%2==0).then_some(0),-(c.rank()as i32));
    let np=(c.size()+1)/2;
    for kind in [SY,AS,SH]{for n in [0,1,3]{for replicate in [false,true]{
        let parent_dist=layout(c.size(),n,kind,if replicate{None}else{Some(0)});
        let child_dist=layout(np,n,kind,if replicate{Some(1)}else{None});
        let mut parent=SymmetricTensor::new(c,parent_dist,algebra.clone());parent.transform(|key,x|*x=value(key as i64+1));
        let before=parent.local_storage().to_vec();
        let mut target=child.as_ref().map(|child|{
            let mut t=SymmetricTensor::new(child,child_dist.clone(),algebra.clone());t.transform(|key,x|*x=value(-(key as i64)));t
        });
        let alpha=value(2);let beta=value(3);let gamma=value(4);let delta=value(5);
        let down=|key:usize|algebra.add(&algebra.multiply(&value(key as i64+1),&alpha),&algebra.multiply(&value(-(key as i64)),&beta));
        parent.add_to_subworld(target.as_mut(),&child_dist,alpha.clone(),beta.clone());
        assert_eq!(parent.local_storage(),before);
        if let Some(t)=&target{for(key,x)in t.local_pairs(){assert_eq!(x,down(key));}}
        parent.add_from_subworld(target.as_ref(),&child_dist,gamma.clone(),delta.clone());
        for(key,x)in parent.local_pairs(){assert_eq!(x,algebra.add(&algebra.multiply(&down(key),&gamma),&algebra.multiply(&value(key as i64+1),&delta)));}
        if let Some(t)=&target{for(key,x)in t.local_pairs(){assert_eq!(x,down(key));}}
    }}}
    if let Some(child)=child{child.close();}
}
fn run(c:&Context<'_>){check(c,Arithmetic::<i64>::new(),|x|x);check(c,Arithmetic::<Complex<f64>>::new(),|x|Complex::new(x as f64,(x%3)as f64));}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let child=world.split(Some((world.rank()%2)as i32),world.rank()as i32).unwrap();run(&child);child.close();
    if world.rank()==0{println!("DIGIT / PASS symmetric_subworld: exact SY/AS/SH affine accumulation, reversed noncontiguous children, virtual/replicas, empty/scalar-sized domains, integer/complex; world+parity");}
    world.close();runtime.finalize();}
