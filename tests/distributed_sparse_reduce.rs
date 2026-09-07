//! Source cyclic-row sparse reduction: partition exchange, rank-tree merge,
//! recursive communicator reduction, and root-only assembly.
use std::collections::BTreeMap;
use ctf::{algebra::{Arithmetic,Monoid,Wire},context::{Context,Runtime},sparse_formats::Coo};

fn entries(rank:usize,rows:usize)->Vec<(usize,usize,i64)> {
    if rows==0 {vec![]} else {
        vec![(1,1,rank as i64+1),(rows,3,2*rank as i64),(2+rank%3,2,0)]
    }
}

fn structural(context:&Context<'_>,root:usize,rows:usize,csr:bool) {
    let matrix=Coo::new(rows,3,entries(context.rank(),rows));
    let result=if csr {
        matrix.to_csr().reduce(context,root,&Arithmetic::<i64>::new()).map(|m|m.to_coo())
    } else {
        matrix.to_ccsr().reduce(context,root,&Arithmetic::<i64>::new()).map(|m|m.to_coo())
    };
    assert_eq!(result.is_some(),context.rank()==root);
    if let Some(result)=result {
        let mut expected=BTreeMap::new();
        for rank in 0..context.size() {for (row,col,value) in entries(rank,rows) {
            *expected.entry((row,col)).or_insert(0)+=value;
        }}
        let expected:Vec<_>=expected.into_iter().map(|((r,c),v)|(r,c,v)).collect();
        assert_eq!(result.shape(),(rows,3));
        assert_eq!(result.entries(),expected.as_slice());
    }
}

#[derive(Clone,Debug,PartialEq)]struct Matrix([i64;4]);
impl Wire for Matrix {const WIDTH:usize=32;
    fn encode(&self,out:&mut Vec<u8>){for value in self.0{value.encode(out);}}
    fn decode(bytes:&[u8])->Self{Self(std::array::from_fn(|i|i64::decode(&bytes[i*8..(i+1)*8])))}}
#[derive(Clone)]struct OrderedProduct;
impl Monoid for OrderedProduct {type Element=Matrix;
    fn zero(&self)->Matrix{Matrix([1,0,0,1])}
    fn add(&self,a:&Matrix,b:&Matrix)->Matrix{Matrix(std::array::from_fn(|i|{
        let row=i/2;let col=i%2;a.0[2*row]*b.0[col]+a.0[2*row+1]*b.0[2+col]}))}}
fn value(rank:usize)->Matrix{if rank%2==0{Matrix([1,rank as i64+1,0,1])}else{Matrix([1,0,rank as i64+1,1])}}
fn ordered(context:&Context<'_>,root:usize) {
    let matrix=Coo::new(1,1,vec![(1,1,value(context.rank()))]);
    let results=[matrix.to_csr().reduce(context,root,&OrderedProduct).map(|m|m.to_coo()),
        matrix.to_ccsr().reduce(context,root,&OrderedProduct).map(|m|m.to_coo())];
    for result in results {
        assert_eq!(result.is_some(),context.rank()==root);
        if let Some(result)=result {
            let expected=(0..context.size()).fold(OrderedProduct.zero(),|old,rank|OrderedProduct.add(&old,&value(rank)));
            assert_eq!(result.entries(),&[(1,1,expected)]);
        }
    }
}
fn run(context:&Context<'_>) {
    let mut roots=vec![0,context.size()-1];roots.dedup();
    for root in roots {
        for rows in [0,5] {structural(context,root,rows,true);structural(context,root,rows,false);}
        // CCSR keeps storage proportional to represented rows, not row count.
        structural(context,root,1_000_003,false);
        ordered(context,root);
    }
}
fn main(){let runtime=Runtime::initialize();let world=runtime.world();run(&world);
    if world.size()==4 {
        let prime=world.split((world.rank()<3).then_some(0),world.rank()as i32);
        if let Some(prime)=prime {run(&prime);prime.close();}
    }
    let child=world.split(Some((world.rank()%2)as i32),world.rank()as i32).unwrap();run(&child);child.close();
    if world.rank()==0{println!("DIGIT / PASS distributed_sparse_reduce: CSR/CCSR row partition exchange and recursive reduction, root-only assembly, exact keys/zeros and noncommutative rank order, empty/tall matrices; world+parity");}
    world.close();runtime.finalize();}
