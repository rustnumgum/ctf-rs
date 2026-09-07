use ctf::{algebra::Arithmetic, context::{Context,Runtime}, cost::Models,
    linalg::Native, mapping::{Distribution,Topology,Mapping}, node_reordering,
    normal_mapping::Problem, partial_fold::{self,Outcome}, symmetry::Symmetry::NS,
    tensor::Tensor};

fn run(context:&Context<'_>) {
    if context.size()==4 {
        let topo=Topology::new(vec![2,2]);
        let p=Mapping::Physical{axis:0,processes:2,child:Box::new(Mapping::Unmapped)};
        let q=Mapping::Physical{axis:1,processes:2,child:Box::new(Mapping::Unmapped)};
        let mapped=[Distribution::new(vec![8],topo.clone(),vec![p.clone()]),
            Distribution::new(vec![8,2],topo.clone(),vec![p,q.clone()]),Distribution::new(vec![2],topo,vec![q])];
        let models=Models::upstream(1);
        let choice=node_reordering::select_dense(mapped.each_ref(),["i","ij","j"],&[0,1],2,8,true,&models).unwrap();
        assert_eq!(choice.intra_node_lens,vec![1,2]);
        assert_eq!(choice.inter_node_lens,vec![2,1]);
        assert_eq!((choice.original_volume,choice.selected_volume),(32.,8.));
        assert!(node_reordering::select_dense(mapped.each_ref(),["i","ij","j"],&[1,0],2,8,true,&models).is_none());
        assert!(node_reordering::select_dense(mapped.each_ref(),["i","ij","j"],&[0,1],1,8,true,&models).is_none());
        let Outcome::Selected(fold)=partial_fold::select([&[4],&[4,1],&[1]],[&[NS],&[NS,NS],&[NS]],["i","ij","j"],&models,[1;3]).unwrap() else {panic!("dot fold")};
        let mut a=Tensor::new(context,Distribution::cyclic(vec![8],4),Arithmetic::<f64>::new());
        let mut b=Tensor::new(context,Distribution::cyclic(vec![8,2],4),Arithmetic::<f64>::new());
        let mut c=Tensor::new(context,Distribution::cyclic(vec![2],4),Arithmetic::<f64>::new());
        a.transform(|key,v|*v=(key+1)as f64);b.transform(|_,v|*v=2.);c.transform(|_,v|*v=3.);
        c.contract_folded_from_mapped::<Native>("j",&a,"i",&b,"ij",mapped,&fold,Some(&choice.intra_node_lens),2.,3.);
        for(_,value)in c.local_pairs(){assert!(value.is_finite()&&(value-153.).abs()<1e-6);}
    }
    let shapes:[&[usize];3]=[&[3,5],&[5,3],&[3,3]];
    let indices=["ik","kj","ij"];
    let topology=Topology::new(if context.size()==4 {vec![2,2]} else {vec![context.size()]});
    let intra=if context.size()==4 {vec![1,2]} else {vec![context.size()]};
    let problem=Problem::new(shapes,indices).unwrap();
    let models=Models::upstream(1);
    for permutation in 0..6 {
        let mapped=problem.map_to_topology(&topology,permutation,[None;3]).unwrap();
        let blocks=mapped.each_ref().map(Distribution::block_shape);
        let links=blocks.each_ref().map(|s|vec![NS;s.len()]);
        let copies=mapped.each_ref().map(|d|d.mappings.iter().map(|m|m.phase()/m.physical_phase()).product());
        let Outcome::Selected(fold)=partial_fold::select(blocks.each_ref().map(Vec::as_slice),
            links.each_ref().map(Vec::as_slice),indices,&models,copies).unwrap() else {panic!("fold")};
        let mut a=Tensor::new(context,Distribution::cyclic(shapes[0].to_vec(),context.size()),Arithmetic::<f64>::new());
        let mut b=Tensor::new(context,Distribution::cyclic(shapes[1].to_vec(),context.size()),Arithmetic::<f64>::new());
        let mut c=Tensor::new(context,Distribution::cyclic(shapes[2].to_vec(),context.size()),Arithmetic::<f64>::new());
        a.transform(|key,v|*v=(key+1)as f64);
        b.transform(|key,v|*v=(key%7+1)as f64);
        c.transform(|key,v|*v=(key+2)as f64);
        let old=c.distribution().clone();
        c.contract_folded_from_mapped::<Native>("ij",&a,"ik",&b,"kj",mapped,&fold,Some(&intra),2.,3.);
        assert_eq!(c.distribution(),&old);
        for(key,value)in c.local_pairs(){
            let i=key%3;let j=key/3;
            let expected=2.*(0..5).map(|k|((i+3*k+1)*((k+5*j)%7+1))as f64).sum::<f64>()+3.*(key+2)as f64;
            assert!(value.is_finite()&&(value-expected).abs()<1e-6,"rank {}, key {key}: {value} != {expected}",context.rank());
        }
    }
    // Exact non-involutive source permutation, so swapped direction names cannot pass.
    let forward:Vec<_>=(0..6).map(|r|node_reordering::reorder_rank(&[2,3],&[1,3],r)).collect();
    let backward:Vec<_>=(0..6).map(|r|node_reordering::inverse_rank(&[2,3],&[1,3],r)).collect();
    assert_eq!(forward,vec![0,2,4,1,3,5]);
    assert_eq!(backward,vec![0,3,1,4,2,5]);
}
fn main(){
    let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let child=world.split(Some((world.rank()%2)as i32),world.rank()as i32).unwrap();run(&child);child.close();
    if world.rank()==0{println!("DIGIT / PASS distributed_node_fold: rank permutation exact; six raw folded maps and backmapping abs<1e-6; world+parity");}
    world.close();runtime.finalize();
}
