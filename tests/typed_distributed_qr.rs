// Frobenius reconstruction/orthogonality bounds from scalapack_tests/qr.cxx.
use ctf::{algebra::{Arithmetic,Complex,Group,Monoid,Semiring},context::{Context,Runtime},
    mapping::{Distribution,Topology},tensor::Tensor};
macro_rules! scalar_case{
    ($name:ident,$t:ty,$value:expr,$conjugate:expr,$norm2:expr)=>{
        fn $name(context:&Context<'_>){
            let value=$value;let conjugate=$conjugate;let norm2=$norm2;
            let algebra=Arithmetic::<$t>::new();let np=context.size();
            let grid=if np==4{[2,2]}else{[np,1]};let topology=Topology::new(grid.to_vec());
            let make=|m,n|Tensor::new(context,Distribution::cyclic(vec![m,n],np),algebra.clone());
            for(m,n)in[(13,7),(5,8),(1,1)]{
                let mut a=make(m,n);
                let input=|key:usize|value(((key*17+11)%31)as f64/31.+if key%m==key/m{1.}else{0.},(key%3)as f64/7.);
                a.transform(|key,x|*x=input(key));
                let(q,r)=a.qr(grid).unwrap();let k=m.min(n);
                assert_eq!(q.distribution().shape,vec![m,k]);assert_eq!(r.distribution().shape,vec![k,n]);
                for(key,x)in a.local_pairs(){assert!(x==input(key),"input preserved");}
                let mut reconstructed=make(m,n);
                reconstructed.contract_from_on_grid("ij",&q,"ik",&r,"kj",topology.clone(),algebra.one(),algebra.zero()).unwrap();
                let mut error=[0.];for((key,expected),(other,actual))in a.local_pairs().into_iter().zip(reconstructed.local_pairs()){
                    assert_eq!(key,other);assert!(norm2(actual).is_finite());
                    error[0]+=norm2(algebra.add(&expected,&algebra.negate(&actual)));
                }context.sum_f64(&mut error);
                assert!(error[0].sqrt()<=(m*n*n)as f64*1e-6,"QR residual {}",error[0].sqrt());
                let mut adjoint=q.permute_axes(&[1,0]);adjoint.transform(|_,x|*x=conjugate(*x));
                let mut gram=make(k,k);
                gram.contract_from_on_grid("ij",&adjoint,"ik",&q,"kj",topology.clone(),algebra.one(),algebra.zero()).unwrap();
                let mut error=[0.];for(key,x)in gram.local_pairs(){
                    assert!(norm2(x).is_finite());let identity=if key%k==key/k{algebra.one()}else{algebra.zero()};
                    error[0]+=norm2(algebra.add(&x,&algebra.negate(&identity)));
                }context.sum_f64(&mut error);
                assert!(error[0].sqrt()<=(m*n)as f64*1e-6,"QR orthogonality {}",error[0].sqrt());
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
    if world.rank()==0{println!("DIGIT / PASS typed_distributed_qr: four native scalar GEQRF/ORGQR/UNGQR families, rectangular/empty local shards and subcontexts; source Frobenius residual/orthogonality bounds");}
    world.close();runtime.finalize();}
