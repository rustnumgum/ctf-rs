use ctf::algebra::{Arithmetic, CustomMonoid, CustomSemiring};
use ctf::sparse_formats::{Coo,Csr,Ccsr};

#[test]
fn conversions_and_cyclic_partitions() {
    let coo = Coo::new(7,5,vec![(7,3,9i64),(1,5,2),(4,2,6),(1,1,3),(4,5,0)]);
    let csr = coo.to_csr();
    assert_eq!(csr.row_offsets(), &[1,3,3,3,5,5,5,6]);
    assert_eq!(csr.columns(), &[1,5,2,5,3]);
    assert_eq!(csr.values(), &[3,2,6,0,9]);
    let ccsr = coo.to_ccsr();
    assert_eq!(ccsr.row_encoding(), &[1,4,7]);
    assert_eq!(ccsr.row_offsets(), &[1,3,5,6]);
    assert_eq!(ccsr.columns(), csr.columns());
    assert_eq!(ccsr.values(), csr.values());
    assert_eq!(ccsr.to_coo(),csr.to_coo());
    for count in [1,2,4,9] {
        assert_eq!(Csr::assemble(&csr.partition(count)),csr);
        assert_eq!(Ccsr::assemble(&ccsr.partition(count)),ccsr);
    }
    let parts = ccsr.partition(4);
    assert_eq!(parts[0].row_encoding(), &[1]);
    assert_eq!(parts[2].row_encoding(), &[2]);
    assert_eq!(parts[3].row_encoding(), &[1]);
    assert_eq!(parts[1].row_offsets(), &[1]);
}

#[test]
fn empty_and_large_logical_rows() {
    for rows in [0,3] {
        let empty = Coo::<i64>::new(rows,0,vec![]);
        assert_eq!(empty.to_csr().row_offsets(),vec![1;rows+1]);
        assert_eq!(empty.to_ccsr().row_offsets(), &[1]);
        assert_eq!(Ccsr::assemble(&empty.to_ccsr().partition(4)),empty.to_ccsr());
        assert_eq!(Csr::assemble(&empty.to_csr().partition(4)),empty.to_csr());
    }
    // CCSR conversion/partition must not allocate IA proportional to nrow.
    let large = Coo::new(1usize<<40,3,vec![(1usize<<39,2,7i64)]).to_ccsr();
    assert_eq!(large.row_offsets(), &[1,2]);
    assert_eq!(Ccsr::assemble(&large.partition(4)),large);
    let duplicated = Coo::new(1,1,vec![(1,1,2i64),(1,1,3)]).to_csr();
    assert_eq!(duplicated.row_offsets(), &[1,3]);
    let mut c = [0];
    duplicated.multiply_dense(1,&[4],&1,&0,&mut c,&Arithmetic::<i64>::new());
    assert_eq!(c,[20]);
}

fn matrices() -> (Csr<i64>,Csr<i64>) {
    // A=[1 0 2; 0 0 0; 0 3 0; 4 0 5], B=[6 0; 0 7; 8 9].
    let a = Coo::new(4,3,vec![(1,1,1),(1,3,2),(3,2,3),(4,1,4),(4,3,5)]).to_csr();
    let b = Coo::new(3,2,vec![(1,1,6),(2,2,7),(3,1,8),(3,2,9)]).to_csr();
    (a,b)
}

#[test]
fn mixed_and_sparse_products_affine() {
    let algebra = Arithmetic::<i64>::new();
    let (a,b) = matrices();
    let mut dense = vec![1;8];
    a.multiply_dense(2,&[6,0,8,0,7,9],&2,&3,&mut dense,&algebra);
    assert_eq!(dense,vec![47,3,3,131,39,3,45,93]);
    let old = Coo::new(4,2,vec![(1,1,1),(2,2,1),(4,1,1)]).to_csr();
    let c = a.multiply_sparse(&b,&2,&3,Some(&old),&algebra);
    assert_eq!(c.to_coo().entries(), &[(1,1,47),(1,2,36),(2,2,3),(3,2,42),(4,1,131),(4,2,90)]);
    let c = a.multiply_sparse(&b,&1,&0,Some(&old),&algebra);
    assert_eq!(c.to_coo().entries(), &[(1,1,22),(1,2,18),(3,2,21),(4,1,64),(4,2,45)]);
    let cancel_a = Coo::new(1,2,vec![(1,1,1),(1,2,-1)]).to_csr();
    let cancel_b = Coo::new(2,1,vec![(1,1,1),(2,1,1)]).to_csr();
    assert_eq!(cancel_a.multiply_sparse(&cancel_b,&1,&0,None,&algebra).values(), &[0]);
    let empty = Coo::new(0,3,vec![]).to_csr();
    assert_eq!(empty.multiply_sparse(&b,&1,&0,None,&algebra).row_offsets(), &[1]);
}

#[test]
fn sparse_add_and_compressed_product() {
    let algebra = Arithmetic::<i64>::new();
    let (a,_) = matrices();
    let b = Coo::new(4,4,vec![(1,3,-2),(2,4,5),(4,2,8)]).to_csr();
    let expected = &[(1,1,1),(1,3,0),(2,4,5),(3,2,3),(4,1,4),(4,2,8),(4,3,5)];
    assert_eq!(a.add(&b,&algebra).to_coo().entries(),expected);
    assert_eq!(a.to_coo().to_ccsr().add(&b.to_coo().to_ccsr(),&algebra).to_coo().entries(),expected);
    let a = a.to_coo().to_ccsr();
    let old = Coo::new(4,3,vec![(2,2,4),(4,3,5)]).to_ccsr();
    let c = a.multiply_dense(3,&[6,0,8,0,7,9,0,0,0],&2,&3,Some(&old),&algebra);
    assert_eq!(c.shape(),(4,3));
    assert_eq!(c.to_coo().entries(), &[(1,1,44),(1,2,36),(2,2,12),(3,1,0),(3,2,42),(4,1,128),(4,2,90),(4,3,15)]);
    // Only the final all-zero column is omitted; earlier structural zeros stay.
    let c = a.multiply_dense(3,&[0;9],&1,&0,None,&algebra);
    assert_eq!(c.values(), &[0;6]);
    assert_eq!(c.columns(), &[1,2,1,2,1,2]);
    assert!(a.multiply_dense(1,&[0;3],&1,&0,None,&algebra).values().is_empty());
    assert!(a.multiply_dense(0,&[],&1,&0,None,&algebra).values().is_empty());
}

#[test]
fn custom_boolean_semiring() {
    let algebra = CustomSemiring {
        monoid: CustomMonoid { identity:false, addition:|a:&bool,b:&bool| *a || *b },
        identity:true, multiplication:|a:&bool,b:&bool| *a && *b,
    };
    let a = Coo::new(3,3,vec![(1,2,true),(2,3,true)]).to_csr();
    let product = a.multiply_sparse(&a,&true,&false,None,&algebra);
    assert_eq!(product.to_coo().entries(), &[(1,3,true)]);
    let mut dense = [false;9];
    a.multiply_dense(3,&[false,false,false,true,false,false,false,true,false],
        &true,&false,&mut dense,&algebra);
    assert_eq!(dense,[false,false,false,false,false,false,true,false,false]);
}
