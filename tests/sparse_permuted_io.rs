use ctf::{algebra::Arithmetic,context::{Context,Runtime},mapping::Distribution,sparse::SparseTensor,
    symmetric_distribution::SymmetricDistribution,symmetric_tensor::SymmetricTensor,tensor::Tensor,symmetry::Symmetry::{NS,SY,AS,SH}};
fn fresh<'c,'r>(c:&'c Context<'r>)->SparseTensor<'c,'r,Arithmetic<i64>>{
    let mut t=SparseTensor::new(c,Distribution::cyclic(vec![3,3],c.size()),Arithmetic::<i64>::new());
    t.write_add(&if c.rank()==0{(0..9).map(|key|(key,7)).collect()}else{vec![]});t
}
fn run(c:&Context<'_>){
    let maps=vec![(0..3).map(Some).collect::<Vec<_>>();2];let child=c.split((c.rank()==0).then_some(0),0);
    let mut source=SparseTensor::new(c,Distribution::cyclic(vec![3,3],c.size()),Arithmetic::<i64>::new());
    source.write_add(&if c.rank()==0{vec![(0,2),(1,0),(3,5),(8,-1)]}else{vec![]});
    let value=|key|match key{0=>2,3=>5,8=>-1,_=>0};
    let mut dense_child=child.as_ref().map(|child|{let mut t=Tensor::new(child,Distribution::cyclic(vec![3,3],1),Arithmetic::<i64>::new());t.transform(|_,x|*x=7);t});
    source.gather_permuted_into_dense(dense_child.as_mut(),&maps,2,3);
    if let Some(t)=&dense_child{for(key,x)in t.local_pairs(){assert_eq!(x,21+2*value(key));}}
    let sparse_child=child.as_ref().map(|child|{
        let mut t=SparseTensor::new(child,Distribution::cyclic(vec![3,3],1),Arithmetic::<i64>::new());t.write_add(&[(0,0),(3,4),(7,-2)]);t
    });
    let mut dense=Tensor::new(c,Distribution::cyclic(vec![3,3],c.size()),Arithmetic::<i64>::new());dense.transform(|_,x|*x=7);
    dense.scatter_permuted_from_sparse(sparse_child.as_ref(),&maps,2,3);
    for(key,x)in dense.local_pairs(){assert_eq!(x,match key{0=>21,3=>29,7=>17,_=>7});}
    let mut dest=fresh(c);dest.scatter_permuted_from(sparse_child.as_ref(),&maps,2,3);
    for(key,x)in dest.local_pairs(){assert_eq!(x,match key{0=>21,3=>29,7=>17,_=>7});}
    if let Some(t)=&mut dense_child{t.transform(|key,x|*x=key as i64);}
    let mut dest=fresh(c);dest.scatter_permuted_from_dense(dense_child.as_ref(),&maps,2,3);
    for(key,x)in dest.local_pairs(){assert_eq!(x,if key==0{7}else{21+2*key as i64});}
    for kind in [SY,AS,SH]{
        let child_dist=SymmetricDistribution::new(Distribution::cyclic(vec![3,3],1),vec![kind,NS]);
        let mut sym_child=child.as_ref().map(|child|{let mut t=SymmetricTensor::new(child,child_dist.clone(),Arithmetic::<i64>::new());t.transform(|_,x|*x=7);t});
        source.gather_permuted_into_symmetric(sym_child.as_mut(),&maps,2,3);
        if let Some(t)=&sym_child{for(key,x)in t.local_pairs(){
            let sum:i64=(0..9).filter_map(|raw|child_dist.canonicalize(raw).filter(|&(canonical,_)|canonical==key).map(|(_,sign)|sign as i64*value(raw))).sum();
            assert_eq!(x,21+2*sum);
        }}
        if let Some(t)=&mut sym_child{t.transform(|key,x|*x=key as i64);}
        let mut dest=fresh(c);dest.scatter_permuted_from_symmetric(sym_child.as_ref(),&maps,2,3);
        for(key,x)in dest.local_pairs(){let touched=key!=0&&child_dist.canonicalize(key)==Some((key,1));assert_eq!(x,if touched{21+2*key as i64}else{7});}
        // Fully replicated parent compressed layout with phase 1 is valid.
        let parent_dist=SymmetricDistribution::new(Distribution::new(vec![3,3],ctf::mapping::Topology::new(vec![c.size()]),vec![ctf::mapping::Mapping::Unmapped;2]),vec![kind,NS]);
        let mut sym=SymmetricTensor::new(c,parent_dist,Arithmetic::<i64>::new());sym.transform(|_,x|*x=7);
        sym.scatter_permuted_from_sparse(sparse_child.as_ref(),&maps,2,3);
        for(key,x)in sym.local_pairs(){assert_eq!(x,match key{0=>21,3=>29,7=>17,_=>7});}
    }
    drop(sparse_child);drop(dense_child);if let Some(child)=child{child.close();}
}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let child=world.split(Some((world.rank()%2)as i32),world.rank()as i32).unwrap();run(&child);child.close();
    if world.rank()==0{println!("DIGIT / PASS sparse_permuted_io: all seven sparse combinations, explicit-zero versus dense-zero scatter, canonical sparse-of-packed source, orbit gather; exact world+parity");}
    world.close();runtime.finalize();}
