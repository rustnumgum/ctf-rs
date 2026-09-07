use ctf::{algebra::Arithmetic, context::{Context,Runtime}, cost::Models,
    mapping::{Distribution,Mapping,Topology}, sparse::SparseTensor,
    sparse_search::{Options,SearchCache},tensor::Tensor};

fn run(context:&Context<'_>) {
    let catalog=[Topology::new(vec![context.size()])];
    let models=Models::upstream(1);
    let mut cache=SearchCache::new(context,&catalog,&models,8,16,true,
        Options{memory_limit:1<<30,weight:0.,allow_exhaustive:false});
    let mut da=Distribution::cyclic(vec![5,3],context.size());
    let db=Distribution::cyclic(vec![3,4],context.size());
    let dc=Distribution::cyclic(vec![5,4],context.size());
    let mut b=Tensor::new(context,db,Arithmetic::<i64>::new());
    b.transform(|key,value|*value=key as i64-2);
    for switched in [false,true] {
        if switched {
            let topology=Topology::new(vec![1,context.size()]);
            let mut rows=Mapping::Unmapped;rows.augment_physical(&topology,1);
            da=Distribution::new(vec![5,3],topology,vec![rows,Mapping::Unmapped]);
        }
        let mut a=SparseTensor::new(context,da.clone(),Arithmetic::<i64>::new());
        let mut saved=None;
        for empty in [false,true] {
            if empty { a.sparsify(|_|false); }
            else {
                a.write_add(&(0..15).filter(|&key|key%3!=0&&da.owner(key)==context.rank())
                    .map(|key|(key,key as i64%7-3)).collect::<Vec<_>>());
            }
            let mut c=Tensor::new(context,dc.clone(),Arithmetic::<i64>::new());
            c.transform(|_,value|*value=if empty{5}else{3});
            // Renaming all labels is normalized by the existing signature.
            let labels=if empty { ["xy","yz","xz"] } else { ["ik","kj","ij"] };
            let selected=cache.prepare([a.distribution(),b.distribution(),c.distribution()],
                labels,if empty{0}else{10}).unwrap().unwrap();
            let fingerprint=(selected.source_id,selected.distributions.clone(),selected.seconds.to_bits(),selected.memory_bytes);
            if let Some(ref original)=saved { assert_eq!(&fingerprint,original); }
            else { saved=Some(fingerprint); }
            c.contract_sparse_from_mapped(labels[2],&a,labels[0],&b,labels[1],
                selected.distributions.clone(),if empty{7}else{2},3,true);
            assert_eq!(c.distribution(),&dc);
            let keys:Vec<_>=(0..20).collect();
            let expected:Vec<_>=keys.iter().map(|&key| {
                if empty{return 15;}
                let i=key%5;let j=key/5;
                9+(0..3).filter(|&k|(i+5*k)%3!=0)
                    .map(|k|2*((i+5*k)as i64%7-3)*((k+3*j)as i64-2)).sum::<i64>()
            }).collect();
            assert_eq!(c.read(&keys),expected);
        }
    }
    assert_eq!(cache.stats(),ctf::planning::CacheStats{hits:2,misses:2});
    cache.clear();
    assert!(cache.prepare([&da,b.distribution(),&dc],["ik","kj","ij"],0).unwrap().is_some());
    assert_eq!(cache.stats(),ctf::planning::CacheStats{hits:2,misses:3});
}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let child=world.split(Some((world.rank()%2)as i32),world.rank()as i32).unwrap();run(&child);child.close();
    if world.rank()==0{println!("DIGIT / PASS distributed_sparse_search_cache: raw search reuse, changed nnz/values/alpha, renamed labels, original distribution switch, clear; exact i64 and cache stats; world+parity");}
    world.close();runtime.finalize();}
