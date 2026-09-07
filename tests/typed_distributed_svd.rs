// Existing scalapack_tests/svd.cxx Frobenius bounds; singular-vector phases are free.
use ctf::{algebra::{Arithmetic,Complex,Group,Monoid,Semiring},context::{Context,Runtime},
    mapping::{Distribution,Topology},tensor::Tensor};
macro_rules! scalar_case{
    ($name:ident,$t:ty,$value:expr,$conjugate:expr,$norm2:expr,$real:expr)=>{
        fn $name(context:&Context<'_>){
            let value=$value;let conjugate=$conjugate;let norm2=$norm2;let real=$real;
            let algebra=Arithmetic::<$t>::new();let np=context.size();
            let grid=if np==4{[2,2]}else{[np,1]};let topology=Topology::new(grid.to_vec());
            let make=|m,n|Tensor::new(context,Distribution::cyclic(vec![m,n],np),algebra.clone());
            for(m,n)in[(13,7),(5,8),(1,1),(1,3)]{
                let mut a=make(m,n);
                let input=|key:usize|value(((key*17+11)%31)as f64/31.+if key%m==key/m{1.}else{0.},(key%3)as f64/7.);
                a.transform(|key,x|*x=input(key));let (mut u,s,vt)=a.svd(grid).unwrap();let k=m.min(n);
                assert_eq!(u.distribution().shape,vec![m,k]);assert_eq!(s.distribution().shape,vec![k]);
                assert_eq!(vt.distribution().shape,vec![k,n]);
                assert_eq!(u.distribution().topology,topology);assert_eq!(vt.distribution().topology,topology);
                for(key,x)in a.local_pairs(){assert!(x==input(key),"input preserved");}
                for (factor,columns)in[(&u,true),(&vt,false)]{
                    let mut adjoint=factor.permute_axes(&[1,0]);adjoint.transform(|_,x|*x=conjugate(*x));
                    let mut gram=make(k,k);
                    let(left,right)=if columns{(&adjoint,factor)}else{(factor,&adjoint)};
                    gram.contract_from_on_grid("ij",left,"ik",right,"kj",topology.clone(),algebra.one(),algebra.zero()).unwrap();
                    let mut error=[0.];for(key,x)in gram.local_pairs(){
                        assert!(norm2(x).is_finite());let one=if key%k==key/k{algebra.one()}else{algebra.zero()};
                        error[0]+=norm2(algebra.add(&x,&algebra.negate(&one)));
                    }context.sum_f64(&mut error);
                    assert!(error[0].sqrt()<=(m*n)as f64*1e-6,"SVD orthogonality {}",error[0].sqrt());
                }
                // Reading only the O(min(m,n)) singular-value vector, never matrix factors.
                let values=s.read(&(0..k).collect::<Vec<_>>());
                assert!(values.iter().all(|&x|norm2(x).is_finite()&&real(x)>=0.&&x==value(real(x),0.)));
                assert!(values.windows(2).all(|pair|real(pair[0])>=real(pair[1])));
                u.transform(|key,x|*x=algebra.multiply(x,&values[key/m]));
                let mut reconstructed=make(m,n);
                reconstructed.contract_from_on_grid("ij",&u,"ik",&vt,"kj",topology.clone(),algebra.one(),algebra.zero()).unwrap();
                let mut error=[0.];for((key,expected),(other,actual))in a.local_pairs().into_iter().zip(reconstructed.local_pairs()){
                    assert_eq!(key,other);assert!(norm2(actual).is_finite());
                    error[0]+=norm2(algebra.add(&expected,&algebra.negate(&actual)));
                }context.sum_f64(&mut error);
                assert!(error[0].sqrt()<=(m*n*n)as f64*1e-6,"SVD residual {}",error[0].sqrt());
            }
        }
    }
}
scalar_case!(real32,f32,|r:f64,_:f64|r as f32,|x:f32|x,|x:f32|(x as f64).powi(2),|x:f32|x as f64);
scalar_case!(real64,f64,|r:f64,_:f64|r,|x:f64|x,|x:f64|x*x,|x:f64|x);
scalar_case!(complex32,Complex<f32>,|r:f64,i:f64|Complex::new(r as f32,i as f32),|x:Complex<f32>|x.conjugate(),|x:Complex<f32>|(x.re as f64).powi(2)+(x.im as f64).powi(2),|x:Complex<f32>|x.re as f64);
scalar_case!(complex64,Complex<f64>,|r:f64,i:f64|Complex::new(r,i),|x:Complex<f64>|x.conjugate(),|x:Complex<f64>|x.norm_squared(),|x:Complex<f64>|x.re);
fn run(c:&Context<'_>){real32(c);real64(c);complex32(c);complex64(c);}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let child=world.split(Some((world.rank()%2)as i32),world.rank()as i32).unwrap();run(&child);child.close();
    if world.rank()==0{println!("DIGIT / PASS typed_distributed_svd: four native scalar GESVD families, rectangular/one-row/empty local shards and subcontexts; ordered real singular values, source Frobenius bounds");}
    world.close();runtime.finalize();}
