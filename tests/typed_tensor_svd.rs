// Existing distributed_tensor_svd normalized reconstruction and factor criteria.
use ctf::{algebra::{Arithmetic,Complex,Group,Monoid,Semiring},context::{Context,Runtime},
    mapping::{Distribution,Topology},multilinear::tensor_svd::TensorSvd,tensor::Tensor};
macro_rules! scalar_case {
    ($name:ident,$t:ty,$value:expr,$conjugate:expr,$norm2:expr) => {
        fn $name(context:&Context<'_>) {
            let value=$value;let conjugate=$conjugate;let norm2=$norm2;let algebra=Arithmetic::<$t>::new();
            let grid=if context.size()==4{[2,2]}else{[context.size(),1]};let topology=Topology::new(grid.to_vec());
            let make=|shape|Tensor::new(context,Distribution::cyclic(shape,context.size()),algebra.clone());
            for (randomized,sparse) in [(false,false),(true,false),(false,true),(true,true)] {
                let mut a=make(vec![3,2,2]);
                let input=|key:usize|{let(i,j,k)=(key%3,key/3%2,key/6);
                    if sparse && key%3==0{algebra.zero()}
                    else if randomized{value(((i+1)*(j+2)*(k+1))as f64/10.,0.)}
                    else{value(((key*7+3)%11)as f64/10.,(key%4)as f64/9.)}};
                a.transform(|key,x|*x=input(key));
                let method=if randomized{TensorSvd::Randomized{rank:1,iterations:1,oversampling:1,seed:19}}
                    else{TensorSvd::Truncated{rank:None,threshold:0.}};
                let(mut left,s,right)=if sparse {
                    let sparse_a=a.clone().into_sparse(|value|*value!=algebra.zero());
                    let original=sparse_a.local_pairs();
                    let factors=if randomized {
                        sparse_a.tensor_svd_randomized("ijk","rki",'r',"jr",grid,1,1,1,19).unwrap()
                    }else{sparse_a.tensor_svd_truncated("ijk","rki",'r',"jr",grid,None,0.).unwrap()};
                    assert_eq!(sparse_a.local_pairs(),original);
                    factors
                } else {a.tensor_svd("ijk","rki",'r',"jr",grid,method).unwrap()};
                let rank=if randomized{1}else{2};
                assert_eq!(left.distribution().shape,vec![rank,2,3]);
                assert_eq!(s.distribution().shape,vec![rank]);assert_eq!(right.distribution().shape,vec![2,rank]);
                for(key,x)in a.local_pairs(){assert!(x==input(key));}
                for(factor,labels,renamed)in[(&left,"rki","ski"),(&right,"jr","js")] {
                    let mut conjugated=factor.clone();conjugated.transform(|_,x|*x=conjugate(*x));
                    let mut gram=make(vec![rank,rank]);
                    gram.contract_from_on_grid("rs",&conjugated,labels,factor,renamed,topology.clone(),algebra.one(),algebra.zero()).unwrap();
                    let mut error=[0.];for(key,x)in gram.local_pairs(){
                        assert!(norm2(x).is_finite());let identity=if key%rank==key/rank{algebra.one()}else{algebra.zero()};
                        error[0]+=norm2(algebra.add(&x,&algebra.negate(&identity))).sqrt();
                    }context.sum_f64(&mut error);
                    assert!(error[0]<=1e-3||error[0]/rank as f64<=1e-3,"tensor SVD orthogonality {}",error[0]);
                }
                let singular=s.read(&(0..rank).collect::<Vec<_>>());
                left.transform(|key,x|*x=algebra.multiply(x,&singular[key%rank]));
                let mut reconstructed=make(vec![3,2,2]);
                reconstructed.contract_from_on_grid("ijk",&left,"rki",&right,"jr",topology.clone(),algebra.one(),algebra.zero()).unwrap();
                let mut error=[0.];for(key,x)in reconstructed.local_pairs(){
                    assert!(norm2(x).is_finite());error[0]+=norm2(algebra.add(&x,&algebra.negate(&input(key))));
                }context.sum_f64(&mut error);
                assert!(error[0].sqrt()/12.<1e-6,"tensor SVD normalized residual {}",error[0].sqrt()/12.);
            }
        }
    }
}
scalar_case!(real32,f32,|r:f64,_:f64|r as f32,|x:f32|x,|x:f32|(x as f64).powi(2));
scalar_case!(real64,f64,|r:f64,_:f64|r,|x:f64|x,|x:f64|x*x);
scalar_case!(complex32,Complex<f32>,|r:f64,i:f64|Complex::new(r as f32,i as f32),|x:Complex<f32>|x.conjugate(),|x:Complex<f32>|(x.re as f64).powi(2)+(x.im as f64).powi(2));
scalar_case!(complex64,Complex<f64>,|r:f64,i:f64|Complex::new(r,i),|x:Complex<f64>|x.conjugate(),|x:Complex<f64>|x.norm_squared());
fn run(c:&Context<'_>){real32(c);real64(c);complex32(c);complex64(c);}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let child=world.split(Some((world.rank()%2)as i32),world.rank()as i32).unwrap();run(&child);child.close();
    if world.rank()==0{println!("DIGIT / PASS typed_tensor_svd: four scalar types, sparse/dense permuted factor indices, truncated/randomized paths, normalized reconstruction<1e-6 and original orthogonality bounds; world+parity");}
    world.close();runtime.finalize();}
