use ctf::{algebra::Arithmetic,context::{Context,Runtime},sparse_formats::Coo,
    sparse_2d::{execute_coo_dense,Panel,Layers}};
fn whole()->Panel<'static,'static>{Panel{comm:None,outer:1,inner:0}}
fn matrix(step:usize)->Coo<i64>{Coo::new(2,2,if step%3==0{vec![]}else{
    vec![(2,2,step as i64+1),(1,1,2),(2,2,-1),(1,2,0)]})}
fn leaf(a:&[Coo<i64>],b:&[Vec<i64>],mut c:Vec<Vec<i64>>,beta:i64,l:Layers)->Vec<Vec<i64>>{
    assert_eq!(l,Layers{count:1,index:0});
    a[0].default_coomm(2,&b[0],&1,&beta,&mut c[0]);c
}
fn contribution(step:usize,b:&[i64])->[i64;4]{if step%3==0{[0;4]}else{
    [2*b[0],step as i64*b[1],2*b[2],step as i64*b[3]]}}
fn run(context:&Context<'_>){
    let p=context.size();let rank=context.rank();let edge=2*p;
    let a:Vec<_>=[rank,rank+p].into_iter().map(matrix).collect();
    let b=vec![1,2,3,4];
    let c=execute_coo_dense(&Arithmetic::<i64>::new(),edge,Layers{count:1,index:0},
        Panel{comm:Some(context),outer:1,inner:1},whole(),whole(),&a,&[b.clone()],vec![vec![10;4]],3,leaf);
    let mut expected=vec![30;4];
    for step in 0..edge{for (old,value) in expected.iter_mut().zip(contribution(step,&b)){*old+=value;}}
    assert_eq!(c,vec![expected]);
    let moving_b:Vec<_>=[rank,rank+p].into_iter().map(|s|vec![s as i64,2,3,4]).collect();
    let c=execute_coo_dense(&Arithmetic::<i64>::new(),edge,Layers{count:1,index:0},
        whole(),Panel{comm:Some(context),outer:1,inner:1},whole(),&[matrix(1)],&moving_b,vec![vec![10;4]],3,leaf);
    assert_eq!(c,vec![vec![30+(edge*(edge-1)) as i64,30+2*edge as i64,30+6*edge as i64,30+4*edge as i64]]);
    let a:Vec<_>=(0..edge).map(matrix).collect();
    for moving in [false,true]{for outer in [1,2]{
        let local_steps=if moving{2}else{edge};
        let c=execute_coo_dense(&Arithmetic::<i64>::new(),edge,Layers{count:1,index:0},
            Panel{comm:None,outer:1,inner:1},whole(),
            Panel{comm:if moving{Some(context)}else{None},outer,inner:1},&a,&[b.clone()],vec![vec![10;4];outer*local_steps],2,
            |a,b,mut c,beta,_|{for (strip,block) in c.iter_mut().enumerate(){a[0].default_coomm(2,&b[0],&(strip as i64+1),&beta,block);}c});
        let mut expected=Vec::new();
        for strip in 0..outer{for local in 0..local_steps{
            let step=if moving{rank+local*p}else{local};let factor=(strip+1)*if moving{p}else{1};
            expected.push(contribution(step,&b).into_iter().map(|v|20+factor as i64*v).collect::<Vec<_>>());
        }}
        assert_eq!(c,expected);
    }}
}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    let child=world.split(Some((world.rank()%2) as i32),world.rank() as i32).unwrap();run(&child);child.close();
    if world.rank()==0{println!("DIGIT / PASS distributed_coo_2d: native COO leaf, variable duplicate/zero panels, A/B broadcast, dense C cyclic reduction and strided scatter; exact i64; world+parity");}
    world.close();runtime.finalize();}
