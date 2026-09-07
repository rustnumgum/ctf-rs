use ctf::{algebra::Arithmetic,context::{Context,Runtime},mapping::{Distribution,Mapping,Topology},
    sparse::SparseTensor,tensor::Tensor};
type Dense<'c,'r>=Tensor<'c,'r,Arithmetic<f64>>;
type Sparse<'c,'r>=SparseTensor<'c,'r,Arithmetic<f64>>;
fn source<'c,'r>(c:&'c Context<'r>)->Sparse<'c,'r>{
    let topology=Topology::new(vec![c.size()]);let mut mode=Mapping::Unmapped;
    mode.augment_physical(&topology,0);mode.augment_virtual(2*c.size());
    let d=Distribution::new(vec![3,2,4],topology,vec![mode,Mapping::Unmapped,Mapping::Unmapped]);
    let entries=vec![(0,2.),(11,-3.),(17,5.),(23,0.)];
    let owned:Vec<_>=entries.into_iter().filter(|(key,_)|d.owner(*key)==c.rank()).collect();
    let mut t=Sparse::new(c,d,Arithmetic::new());t.write_add(&owned);t
}
fn vector(mode:usize,i:usize)->f64{(mode+i+1) as f64}
fn matrix(mode:usize,i:usize,r:usize)->f64{(1+(mode+i+r)%4) as f64}
fn factor<'c,'r>(c:&'c Context<'r>,mode:usize,extent:usize,width:Option<usize>,first:bool)->Dense<'c,'r>{
    let shape=match width{None=>vec![extent],Some(k)=>if first{vec![k,extent]}else{vec![extent,k]}};
    let mut t=Dense::new(c,Distribution::cyclic(shape,c.size()),Arithmetic::new());
    t.transform(|key,v|*v=match width{None=>vector(mode,key),Some(k)=>{
        let(i,r)=if first{(key/k,key%k)}else{(key%extent,key/extent)};matrix(mode,i,r)}});t
}
fn close(a:Vec<f64>,b:Vec<f64>){assert_eq!(a.len(),b.len());for(x,y)in a.into_iter().zip(b){
    assert!(x.is_finite()&&(x-y).abs()<=1e-6,"actual={x} expected={y}");}}
fn run(c:&Context<'_>){
    let entries=[(0,2.),(11,-3.),(17,5.),(23,0.)];let keys=vec![0,11,17,23,7];
    let f0=factor(c,0,3,None,true);let f2=factor(c,2,4,None,true);
    let mut t=source(c);let old_keys:Vec<_>=t.local_pairs().into_iter().map(|p|p.0).collect();
    t.tttp_vectors(&[(0,&f0),(2,&f2)]);
    let mut expected:Vec<_>=entries.iter().map(|&(key,v)|v*vector(0,key%3)*vector(2,key/6)).collect();expected.push(0.);
    close(t.read(&keys),expected);assert_eq!(t.local_pairs().into_iter().map(|p|p.0).collect::<Vec<_>>(),old_keys);
    for first in [true,false]{
        let f0=factor(c,0,3,Some(5),first);let f2=factor(c,2,4,Some(5),first);
        let mut t=source(c);t.tttp_matrices(&[(0,&f0),(2,&f2)],first,2);
        let mut expected:Vec<_>=entries.iter().map(|&(key,v)|v*(0..5).map(|r|matrix(0,key%3,r)*matrix(2,key/6,r)).sum::<f64>()).collect();expected.push(0.);
        close(t.read(&keys),expected);
    }
    let t=source(c);
    let coordinate=|key:usize,mode:usize|[key%3,key/3%2,key/6][mode];
    for output_mode in 0..3 {
        let modes:Vec<_>=(0..3).filter(|&mode|mode!=output_mode).collect();
        for width in [None,Some(5)] {
            let factors:Vec<_>=modes.iter().map(|&mode|factor(c,mode,[3,2,4][mode],width,true)).collect();
            let references:Vec<_>=factors.iter().collect();
            let shape=match width{None=>vec![[3,2,4][output_mode]],Some(k)=>vec![k,[3,2,4][output_mode]]};
            let count=shape.iter().product::<usize>();
            let result=t.mttkrp(output_mode,&references,Distribution::cyclic(shape,c.size()));
            let expected:Vec<_>=(0..count).map(|key|{
                let(row,r)=match width{None=>(key,0),Some(k)=>(key/k,key%k)};
                entries.iter().filter(|(key,_)|coordinate(*key,output_mode)==row).map(|&(key,v)|{
                    v*modes.iter().map(|&mode|match width{None=>vector(mode,coordinate(key,mode)),
                        Some(_)=>matrix(mode,coordinate(key,mode),r)}).product::<f64>()
                }).sum()
            }).collect();
            close(result.read(&(0..count).collect::<Vec<_>>()),expected);
        }
    }
    let d=Distribution::cyclic(vec![1_000_000_000,2],c.size());
    let pairs:Vec<_>=[(0,2.),(1_999_999_999,3.)].into_iter().filter(|(key,_)|d.owner(*key)==c.rank()).collect();
    let mut huge=Sparse::new(c,d,Arithmetic::new());huge.write_add(&pairs);
    let f=factor(c,1,2,None,true);huge.tttp_vectors(&[(1,&f)]);
    close(huge.read(&[0,1_999_999_999]),vec![4.,9.]);
}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let parity=world.split(Some((world.rank()%2) as i32),world.rank() as i32).unwrap();run(&parity);parity.close();
    if world.rank()==0{println!("DIGIT / PASS distributed_sparse_multilinear: stored TTTP, both factor orientations, blocked auxiliary, MTTKRP, huge sparse domain; atol=1e-6; world+parity");}
    world.close();runtime.finalize();}
