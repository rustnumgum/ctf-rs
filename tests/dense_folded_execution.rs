use ctf::{algebra::Arithmetic,context::{Context,Runtime},cost::Models,dense_search::{SearchCache,Options,TopologyFacts},
    linalg::Native,mapping::{Distribution,Mapping,Topology},normal_mapping::Problem,mapping_variants::VariantSpace,
    partial_fold::{self,Descriptor,Outcome},symmetry::Symmetry::NS,tensor::Tensor,topology_candidates};
fn av(key:usize)->f64{(key%7+1)as f64}
fn bv(key:usize)->f64{(key%5)as f64-2.}
fn make<'c,'r>(c:&'c Context<'r>,shape:&[usize])->Tensor<'c,'r,Arithmetic<f64>>{
    Tensor::new(c,Distribution::cyclic(shape.to_vec(),c.size()),Arithmetic::<f64>::new())
}
fn reference(shapes:[&[usize];3],indices:[&str;3],key:usize)->f64{
    let mut labels=Vec::new();let mut lengths=Vec::new();
    for operand in 0..3{for(axis,label)in indices[operand].bytes().enumerate(){
        if !labels.contains(&label){labels.push(label);lengths.push(shapes[operand][axis]);}
    }}
    let mut coordinates=vec![0;labels.len()];let mut remainder=key;
    for(axis,label)in indices[2].bytes().enumerate(){coordinates[labels.iter().position(|&l|l==label).unwrap()]=remainder%shapes[2][axis];remainder/=shapes[2][axis];}
    let reduced:Vec<_>=labels.iter().enumerate().filter_map(|(i,&l)|(!indices[2].bytes().any(|x|x==l)).then_some(i)).collect();
    let mut sum=0.;for flat in 0..reduced.iter().map(|&i|lengths[i]).product(){
        let mut n=flat;for &axis in &reduced{coordinates[axis]=n%lengths[axis];n/=lengths[axis];}
        let keys:[usize;2]=std::array::from_fn(|operand|{let mut key=0;let mut stride=1;
            for(axis,label)in indices[operand].bytes().enumerate(){key+=stride*coordinates[labels.iter().position(|&l|l==label).unwrap()];stride*=shapes[operand][axis];}key});
        sum+=av(keys[0])*bv(keys[1]);
    }sum
}
fn descriptor(mapped:&[Distribution;3],indices:[&str;3])->Descriptor{
    let shapes=mapped.each_ref().map(Distribution::block_shape);
    let links=shapes.each_ref().map(|s|vec![NS;s.len()]);
    let copies=mapped.each_ref().map(|d|d.mappings.iter().map(|m|m.phase()/m.physical_phase()).product());
    let Outcome::Selected(d)=partial_fold::select(shapes.each_ref().map(Vec::as_slice),links.each_ref().map(Vec::as_slice),indices,&Models::upstream(1),copies).unwrap()
        else{panic!("expected foldable contraction")};d
}
fn exercise(c:&Context<'_>,shapes:[&[usize];3],indices:[&str;3],mapped:[Distribution;3]){
    let d=descriptor(&mapped,indices);let mut a=make(c,shapes[0]);let mut b=make(c,shapes[1]);let mut output=make(c,shapes[2]);
    a.transform(|key,v|*v=av(key));b.transform(|key,v|*v=bv(key));output.transform(|_,v|*v=3.);
    let old=output.distribution().clone();
    output.contract_folded_from_mapped::<Native>(indices[2],&a,indices[0],&b,indices[1],mapped,&d,2.,3.);
    assert_eq!(output.distribution(),&old);
    for(key,value)in output.local_pairs(){let expected=2.*reference(shapes,indices,key)+9.;
        assert!(value.is_finite()&&(value-expected).abs()<1e-6,"{indices:?}, rank {}, key {key}: {value} != {expected}",c.rank());}
}
fn run(c:&Context<'_>){
    let shapes:[&[usize];3]=[&[3,3],&[3,2],&[3,2]];let indices=["ik","kj","ij"];
    let topology=Topology::new(if c.size()==4{vec![2,2]}else{vec![c.size()]});
    let problem=Problem::new(shapes,indices).unwrap();
    for permutation in 0..6{exercise(c,shapes,indices,problem.map_to_topology(&topology,permutation,[None;3]).unwrap());}
    let partial:[&[usize];3]=[&[2,3,3],&[3,2],&[3,2]];
    exercise(c,partial,["xik","kj","ij"],Problem::new(partial,["xik","kj","ij"]).unwrap().map_to_topology(&topology,0,[None;3]).unwrap());
    if c.size()==4{let small:[&[usize];3]=[&[1,3],&[3,1],&[1,1]];
        let variants=VariantSpace::new(small,indices,topology).unwrap();
        for variant in 3..6{exercise(c,small,indices,variants.decode(variant).unwrap().distributions);}
    }
    let topology=Topology::new(vec![c.size()]);
    let k=Mapping::Physical{axis:0,processes:c.size(),child:Box::new(Mapping::Unmapped)};
    let l=Mapping::Virtual{copies:2,child:Box::new(Mapping::Unmapped)};
    let batch:[&[usize];3]=[&[3,3,3],&[3,2,3],&[3,2,3]];
    let maps=[vec![Mapping::Unmapped,k.clone(),l.clone()],vec![k,Mapping::Unmapped,l.clone()],vec![Mapping::Unmapped,Mapping::Unmapped,l]];
    exercise(c,batch,["ikl","kjl","ijl"],std::array::from_fn(|o|Distribution::new(batch[o].to_vec(),topology.clone(),maps[o].clone())));
    let topology=Topology::new(vec![1,1,if c.size()==4{2}else{c.size()},if c.size()==4{2}else{1}]);
    let p=|axis|Mapping::Physical{axis,processes:topology.dimensions[axis],child:Box::new(Mapping::Unmapped)};
    let nested:[&[usize];3]=[&[2,1,2,3],&[2,3,1,2],&[2,1,1,2]];
    let mut maps:[Vec<Mapping>;3]=std::array::from_fn(|_|vec![p(1),p(3),p(0),p(2)]);
    if c.size()==2{maps[1][1].augment_virtual(2);}
    exercise(c,nested,["imkl","kljn","imjn"],std::array::from_fn(|o|Distribution::new(nested[o].to_vec(),topology.clone(),maps[o].clone())));

    let catalog:Vec<_>=topology_candidates::all_shapes(c.size()).into_iter().map(|topology|
        TopologyFacts{nodes_per_axis:vec![1;topology.dimensions.len()],topology}).collect();
    let models=Models::upstream(1);let mut cache=SearchCache::new(c,&catalog,&models,8,false,true,
        Options{memory_limit:1_000_000,weight:0.,allow_exhaustive:true,enable_folding:true});
    let mut a=make(c,shapes[0]);let mut b=make(c,shapes[1]);let mut output=make(c,shapes[2]);
    b.transform(|key,v|*v=bv(key));
    for factor in [1.,2.]{a.transform(|key,v|*v=factor*av(key));output.transform(|_,v|*v=3.);
        let selected=cache.prepare([a.distribution(),b.distribution(),output.distribution()],[&[1];3],indices).unwrap().unwrap();
        output.contract_folded_from_mapped::<Native>("ij",&a,"ik",&b,"kj",selected.distributions.clone(),selected.fold.as_ref().unwrap(),2.,3.);
        for(key,value)in output.local_pairs(){assert!(value.is_finite()&&(value-(2.*factor*reference(shapes,indices,key)+9.)).abs()<1e-6);}
    }
    assert_eq!(cache.stats(),ctf::planning::CacheStats{hits:1,misses:1});
}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let child=world.split(Some((world.rank()%2)as i32),world.rank()as i32).unwrap();run(&child);child.close();
    if world.rank()==0{println!("DIGIT / PASS dense_folded_execution: selected/cache fold to BLAS, 2D/nested panels, partial residuals, virtual batches, padded fragments; world+parity; abs<1e-6");}
    world.close();runtime.finalize();}
