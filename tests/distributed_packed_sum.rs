use ctf::{algebra::Arithmetic,context::{Context,Runtime},symmetry::{Layout,Symmetry::*},
    symmetric_sum::sequential,symmetric_sum_comm::replicated};
fn run(context:&Context<'_>) {
    let algebra=Arithmetic::<i64>::new();
    let sy=Layout::new(vec![3,3],vec![SY,NS]);
    let ns=Layout::new(vec![3,3],vec![NS,NS]);
    let strict=Layout::new(vec![3,3],vec![AS,NS]);
    let scalar=Layout::new(vec![],vec![]);
    let a=vec![1,2,3,4,5,6];
    let mut out=vec![10;9];
    sequential(&algebra,&sy,"ij",&a,&ns,"ij",&mut out,&2,&3);
    assert_eq!(out,vec![32,30,30,34,36,30,38,40,42]);
    let mut diagonal=vec![10;6];
    sequential(&algebra,&sy,"ii",&a,&sy,"ii",&mut diagonal,&2,&3);
    assert_eq!(diagonal,vec![32,10,36,10,10,42]);
    // AS packed physical slots still include equal local quotients. Global
    // structural/padding holes are cleared outside this sequential kernel.
    let mut physical=vec![0;6];
    sequential(&algebra,&strict,"ij",&a,&strict,"ij",&mut physical,&1,&0);
    assert_eq!(physical,a);
    let mut trace=vec![10];
    sequential(&algebra,&sy,"ii",&a,&scalar,"",&mut trace,&2,&3);
    assert_eq!(trace,vec![50]);
    let mut input=a.iter().map(|v|v+context.rank() as i64).collect::<Vec<_>>();
    let mut reduced=vec![10;6];
    replicated(&algebra,&[],&[context],&sy,&[1,1],"ij",&mut input,
        &sy,&[1,1],"ij",&mut reduced,&2,&3,true);
    let np=context.size() as i64;
    assert_eq!(reduced,a.iter().map(|v|30+2*np*v+np*(np-1)).collect::<Vec<_>>());
    let mut broadcast=if context.rank()==0 {a.clone()} else {vec![999;6]};
    let mut copied=vec![0;6];
    replicated(&algebra,&[context],&[],&sy,&[1,1],"ij",&mut broadcast,
        &sy,&[1,1],"ij",&mut copied,&1,&0,true);
    assert_eq!(copied,a);
    let mut virtual_input:Vec<_>=(1..=24).collect();let mut total=vec![10];
    replicated(&algebra,&[],&[context],&sy,&[2,2],"ij",&mut virtual_input,
        &scalar,&[],"",&mut total,&1,&2,true);
    assert_eq!(total,vec![20+300*np]);
    let empty=Layout::new(vec![0,0],vec![SY,NS]);let mut zero=vec![5];
    sequential(&algebra,&empty,"ij",&[],&scalar,"",&mut zero,&2,&3);
    assert_eq!(zero,vec![15]);
}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let parity=world.split(Some((world.rank()%2) as i32),world.rank() as i32).unwrap();run(&parity);parity.close();
    if world.rank()==0 {println!("DIGIT / PASS distributed_packed_sum: packed sequential, virtual, broadcast/reduce; exact i64; world+parity");}
    world.close();runtime.finalize();}
