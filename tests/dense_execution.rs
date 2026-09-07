use ctf::{algebra::Arithmetic,context::{Context,Runtime},cost::Models,
    dense_search::{SearchCache,Options,TopologyFacts},mapping::{Distribution,Mapping,Topology},
    mapping_variants::VariantSpace,normal_mapping::Problem,tensor::Tensor,topology_candidates};

fn a_value(key:usize)->f64{(key%7+1)as f64}
fn b_value(key:usize)->f64{(key%5)as f64-2.}
fn make<'c,'r>(c:&'c Context<'r>,shape:Vec<usize>)->Tensor<'c,'r,Arithmetic<f64>>{
    Tensor::new(c,Distribution::cyclic(shape,c.size()),Arithmetic::<f64>::new())
}
fn reference(shapes:[&[usize];3],indices:[&str;3],key:usize)->f64{
    let mut labels=Vec::new();let mut lengths=Vec::new();
    for operand in 0..3{for(axis,label)in indices[operand].bytes().enumerate(){
        if !labels.contains(&label){labels.push(label);lengths.push(shapes[operand][axis]);}
    }}
    let mut coordinates=vec![0;labels.len()];let mut remainder=key;
    for(axis,label)in indices[2].bytes().enumerate(){
        coordinates[labels.iter().position(|&l|l==label).unwrap()]=remainder%shapes[2][axis];remainder/=shapes[2][axis];
    }
    let reduced:Vec<_>=labels.iter().enumerate().filter_map(|(i,&label)|(!indices[2].bytes().any(|l|l==label)).then_some(i)).collect();
    let mut result=0.;
    for flat in 0..reduced.iter().map(|&i|lengths[i]).product(){
        let mut n=flat;for &axis in &reduced{coordinates[axis]=n%lengths[axis];n/=lengths[axis];}
        let keys:[usize;2]=std::array::from_fn(|operand|{
            let mut stride=1;let mut key=0;
            for(axis,label)in indices[operand].bytes().enumerate(){key+=stride*coordinates[labels.iter().position(|&l|l==label).unwrap()];stride*=shapes[operand][axis];}key
        });result+=a_value(keys[0])*b_value(keys[1]);
    }result
}
fn exercise(c:&Context<'_>,shapes:[&[usize];3],indices:[&str;3],mapped:[Distribution;3]){
    let mut a=make(c,shapes[0].to_vec());let mut b=make(c,shapes[1].to_vec());let mut output=make(c,shapes[2].to_vec());
    a.transform(|key,v|*v=a_value(key));b.transform(|key,v|*v=b_value(key));output.transform(|_,v|*v=3.);
    let old=output.distribution().clone();
    output.contract_from_mapped(indices[2],&a,indices[0],&b,indices[1],mapped,None,2.,3.);
    assert_eq!(output.distribution(),&old);
    for(key,value)in output.local_pairs(){let expected=2.*reference(shapes,indices,key)+9.;
        assert!(value.is_finite()&&(value-expected).abs()<1e-6,"key {key}: {value} != {expected}");}
}
fn run(c:&Context<'_>){
    let shape:[&[usize];3]=[&[3,3],&[3,2],&[3,2]];let indices=["ik","kj","ij"];
    let topology=Topology::new(if c.size()==4{vec![2,2]}else{vec![c.size()]});
    let problem=Problem::new(shape,indices).unwrap();
    for permutation in 0..6{
        let mapped=problem.map_to_topology(&topology,permutation,[None;3]).unwrap();
        assert!(ctf::mapping_preflight::check(mapped.each_ref(),indices));exercise(c,shape,indices,mapped);
    }
    // Virtualized k phases, uneven i/k, and empty true local fragments at 4 ranks.
    if c.size()==4{
        let shape:[&[usize];3]=[&[1,3],&[3,1],&[1,1]];
        let space=VariantSpace::new(shape,indices,Topology::new(vec![2,2])).unwrap();
        for variant in 3..6{exercise(c,shape,indices,space.decode(variant).unwrap().distributions);}
    }
    // Two source panel levels, with a size-one physical pair retained explicitly.
    let topology=Topology::new(vec![1,1,if c.size()==4{2}else{c.size()},if c.size()==4{2}else{1}]);
    let p=|axis|Mapping::Physical{axis,processes:topology.dimensions[axis],child:Box::new(Mapping::Unmapped)};
    let shapes:[&[usize];3]=[&[2,1,2,3],&[2,3,1,2],&[2,1,1,2]];
    let mut maps=[vec![p(1),p(3),p(0),p(2)],vec![p(1),p(3),p(0),p(2)],vec![p(1),p(3),p(0),p(2)]];
    // At two ranks the second physical pair is rectangular [2,1]: match l
    // phases with the source-required virtual child on B's size-one axis.
    if c.size()==2{maps[1][1].augment_virtual(2);}
    let mapped=std::array::from_fn(|o|Distribution::new(shapes[o].to_vec(),topology.clone(),maps[o].clone()));
    exercise(c,shapes,["imkl","kljn","imjn"],mapped);

    let topology=Topology::new(vec![c.size()]);
    let k=Mapping::Physical{axis:0,processes:c.size(),child:Box::new(Mapping::Unmapped)};
    let batch=Mapping::Virtual{copies:2,child:Box::new(Mapping::Unmapped)};
    let shapes:[&[usize];3]=[&[3,3,3],&[3,2,3],&[3,2,3]];
    let maps=[vec![Mapping::Unmapped,k.clone(),batch.clone()],vec![k,Mapping::Unmapped,batch.clone()],
        vec![Mapping::Unmapped,Mapping::Unmapped,batch]];
    let mapped=std::array::from_fn(|o|Distribution::new(shapes[o].to_vec(),topology.clone(),maps[o].clone()));
    exercise(c,shapes,["ika","kja","ija"],mapped);

    let catalog:Vec<_>=topology_candidates::all_shapes(c.size()).into_iter().map(|topology|
        TopologyFacts{nodes_per_axis:vec![1.;topology.dimensions.len()],topology}).collect();
    let models=Models::upstream(1);
    let mut cache=SearchCache::new(c,&catalog,&models,8,false,false,Options{memory_limit:1_000_000,weight:0.,allow_exhaustive:true,enable_folding:false});
    let mut a=make(c,vec![3,3]);let mut b=make(c,vec![3,2]);let mut output=make(c,vec![3,2]);
    a.transform(|key,v|*v=a_value(key));b.transform(|key,v|*v=b_value(key));output.transform(|_,v|*v=3.);
    for(iteration,alpha,beta)in[(0,2.,3.),(1,3.,0.)]{
        if iteration==1{a.transform(|key,v|*v=2.*a_value(key));}
        let selected=cache.prepare([a.distribution(),b.distribution(),output.distribution()],[&[1.];3],indices).unwrap().unwrap();
        output.contract_from_mapped("ij",&a,"ik",&b,"kj",selected.distributions.clone(),None,alpha,beta);
        for(key,value)in output.local_pairs(){let expected=if iteration==0{2.*reference(shape,indices,key)+9.}else{6.*reference(shape,indices,key)};
            assert!(value.is_finite()&&(value-expected).abs()<1e-6);}
    }
    assert_eq!(cache.stats(),ctf::planning::CacheStats{hits:1,misses:1});assert_eq!(cache.len(),1);
    // Pure alpha-renaming is the same structural contraction signature.
    cache.prepare([a.distribution(),b.distribution(),output.distribution()],[&[1.];3],["ab","bc","ac"]).unwrap().unwrap();
    assert_eq!(cache.stats().hits,2);
    a.redistribute(Distribution::new(vec![3,3],Topology::new(vec![c.size()]),
        vec![Mapping::Unmapped,Mapping::Physical{axis:0,processes:c.size(),child:Box::new(Mapping::Unmapped)}]));
    let selected=cache.prepare([a.distribution(),b.distribution(),output.distribution()],[&[1.];3],indices).unwrap().unwrap();
    output.contract_from_mapped("ij",&a,"ik",&b,"kj",selected.distributions.clone(),None,3.,0.);
    for(key,value)in output.local_pairs(){assert!(value.is_finite()&&(value-6.*reference(shape,indices,key)).abs()<1e-6);}
    assert_eq!(cache.stats().misses,2);assert_eq!(cache.len(),2);
    cache.clear();assert!(cache.is_empty());
    cache.prepare([a.distribution(),b.distribution(),output.distribution()],[&[1.];3],indices).unwrap().unwrap();
    assert_eq!(cache.stats().misses,3);
}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let child=world.split(Some((world.rank()%2)as i32),world.rank()as i32).unwrap();run(&child);child.close();
    if world.rank()==0{println!("DIGIT / PASS dense_execution: raw 2D input/output panels, nested levels, padded fragments, cached search and fresh values; world+parity; abs<1e-6");}
    world.close();runtime.finalize();}
