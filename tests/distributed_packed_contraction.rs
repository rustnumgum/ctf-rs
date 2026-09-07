use ctf::{algebra::Arithmetic,context::{Context,Runtime},symmetry::{Layout,Symmetry::*},
    symmetric_contraction::{sequential,sequential_function},symmetric_contraction_comm::{replicated,virtualized}};
fn run(context:&Context<'_>) {
    let algebra=Arithmetic::<i64>::new();let scalar=Layout::new(vec![],vec![]);
    let sy=Layout::new(vec![3,3],vec![SY,NS]);let ns=Layout::new(vec![3,3],vec![NS,NS]);
    let a=vec![1,2,3,4,5,6];let b=vec![2,3,4,5,6,7];
    let function=|a:&i64,b:&i64|a+b+1;
    let mut custom=vec![10;6];
    sequential_function(&algebra,&sy,"ij",&a,&sy,"ij",&b,&sy,"ij",&mut custom,&2,&3,&function);
    assert_eq!(custom,vec![38,42,46,50,54,58]);
    let mut custom_diagonal=vec![10;6];
    sequential_function(&algebra,&sy,"ii",&a,&sy,"ii",&b,&sy,"ii",&mut custom_diagonal,&2,&3,&function);
    assert_eq!(custom_diagonal,vec![38,30,46,30,30,58]);
    let mut custom_scalar=vec![10];
    sequential_function(&algebra,&scalar,"",&[2],&scalar,"",&[3],&scalar,"",&mut custom_scalar,&2,&3,&function);
    assert_eq!(custom_scalar,vec![42]);
    let mut c=vec![10;9];
    sequential(&algebra,&sy,"ik",&a,&sy,"kj",&b,&ns,"ij",&mut c,&2,&3);
    // Raw canonical constraints i<=k<=j, not a full symmetric matrix product.
    assert_eq!(c,vec![34,30,30,52,54,30,120,136,114]);
    let mut diagonal=vec![10;6];
    sequential(&algebra,&sy,"ii",&a,&sy,"ii",&b,&sy,"ii",&mut diagonal,&2,&3);
    // Source raw kernel prescales all C; upper-layer diagonal extraction is
    // required to preserve off-diagonal tensor entries in a public operation.
    assert_eq!(diagonal,vec![34,30,54,30,30,114]);
    let mut dot=vec![10];
    sequential(&algebra,&sy,"ij",&a,&sy,"ij",&b,&scalar,"",&mut dot,&2,&3);
    assert_eq!(dot,vec![254]);
    let mut as_slots=vec![0;6];let strict=Layout::new(vec![3,3],vec![AS,NS]);
    sequential(&algebra,&strict,"ij",&a,&scalar,"",&[2],&strict,"ij",&mut as_slots,&1,&0);
    assert_eq!(as_slots,vec![2,4,6,8,10,12]);
    let mut rank_a=a.iter().map(|v|v+context.rank() as i64).collect::<Vec<_>>();
    let mut rank_b=b.clone();let mut total=vec![10];
    replicated(&algebra,&[],&[],&[context],&sy,&[1,1],"ij",&mut rank_a,
        &sy,&[1,1],"ij",&mut rank_b,&scalar,&[],"",&mut total,&2,&3,true);
    if context.rank()==0 {let np=context.size() as i64;assert_eq!(total,vec![30+224*np+27*np*(np-1)]);}
    let mut broadcast_a=if context.rank()==0{a.clone()}else{vec![99;6]};
    let mut broadcast_b=if context.rank()==0{b.clone()}else{vec![99;6]};let mut product=vec![0;6];
    replicated(&algebra,&[context],&[context],&[],&sy,&[1,1],"ij",&mut broadcast_a,
        &sy,&[1,1],"ij",&mut broadcast_b,&sy,&[1,1],"ij",&mut product,&1,&0,true);
    assert_eq!(product,vec![2,6,12,20,30,42]);
    if context.rank()!=0 {assert_eq!(broadcast_a,vec![0;6]);assert_eq!(broadcast_b,vec![0;6]);}
    let va:Vec<_>=(1..=24).collect();let vb=vec![2;24];let mut sum=vec![10];
    virtualized(&algebra,&sy,&[2,2],"ij",&va,&sy,&[2,2],"ij",&vb,&scalar,&[],"",&mut sum,&1,&3);
    assert_eq!(sum,vec![630]);
    let empty=Layout::new(vec![0,0],vec![SY,NS]);let mut zero=vec![10];
    sequential(&algebra,&empty,"ij",&[],&empty,"ij",&[],&scalar,"",&mut zero,&2,&3);
    assert_eq!(zero,vec![30]);
}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let parity=world.split(Some((world.rank()%2) as i32),world.rank() as i32).unwrap();run(&parity);parity.close();
    if world.rank()==0{println!("DIGIT / PASS distributed_packed_contraction: canonical kernel, virtual reduction, input broadcasts and root Reduce; exact i64; world+parity");}
    world.close();runtime.finalize();}
