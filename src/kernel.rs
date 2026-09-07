// Adapted from cc4s CTF interface/kernel.h::{Monoid_Kernel,Bivar_Kernel}
// at f69cbb46e23bc2f39cda5722ce096f56301dab4f.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Mixed-type CPU function kernels. Accumulation is explicitly g(f(a,b),c);
//! there is no assumption that g is ordinary addition or f distributes over g.
use crate::{linalg::Transpose,sparse_formats::{Coo,Csr}};

pub fn gemm<A,B,C>(
    transpose_a:Transpose,transpose_b:Transpose,m:usize,n:usize,k:usize,
    a:&[A],b:&[B],c:&mut[C],function:impl Fn(&A,&B)->C,accumulate:impl Fn(C,&mut C),
) {
    assert_eq!(a.len(),m*k);assert_eq!(b.len(),k*n);assert_eq!(c.len(),m*n);
    // The source tests only N versus non-N: transpose never conjugates.
    let (am,ak)=if matches!(transpose_a,Transpose::No){(1,m)}else{(k,1)};
    let (bk,bn)=if matches!(transpose_b,Transpose::No){(1,k)}else{(n,1)};
    for row in 0..m {for column in 0..n {for inner in 0..k {
        accumulate(function(&a[row*am+inner*ak],&b[inner*bk+column*bn]),&mut c[row+column*m]);
    }}}
}

pub fn csr_dense<A:Clone,B,C>(a:&Csr<A>,n:usize,b:&[B],c:&mut[C],
    function:impl Fn(&A,&B)->C,accumulate:impl Fn(C,&mut C)) {
    let (m,k)=a.shape();assert_eq!(b.len(),k*n);assert_eq!(c.len(),m*n);
    for row in 0..m {for column in 0..n {
        for entry in a.row_offsets()[row]-1..a.row_offsets()[row+1]-1 {
            let inner=a.columns()[entry]-1;
            accumulate(function(&a.values()[entry],&b[column*k+inner]),&mut c[column*m+row]);
        }
    }}
}

pub fn csr_sparse_dense<A:Clone,B:Clone,C>(a:&Csr<A>,b:&Csr<B>,c:&mut[C],
    function:impl Fn(&A,&B)->C,accumulate:impl Fn(C,&mut C)) {
    let (m,k)=a.shape();let (rows_b,n)=b.shape();assert_eq!(k,rows_b);assert_eq!(c.len(),m*n);
    for row in 0..m {
        for entry_a in a.row_offsets()[row]-1..a.row_offsets()[row+1]-1 {
            let inner=a.columns()[entry_a]-1;
            for entry_b in b.row_offsets()[inner]-1..b.row_offsets()[inner+1]-1 {
                let column=b.columns()[entry_b]-1;
                accumulate(function(&a.values()[entry_a],&b.values()[entry_b]),&mut c[column*m+row]);
            }
        }
    }
}

/// Monoid_Kernel::xpy, with explicit positive element strides and existing
/// output objects. Object construction/destruction uses Rust rather than memcpy.
pub fn accumulate_strided<C:Clone>(n:usize,x:&[C],stride_x:usize,y:&mut[C],stride_y:usize,
    accumulate:impl Fn(C,&mut C)) {
    for index in 0..n {accumulate(x[index*stride_x].clone(),&mut y[index*stride_y]);}
}

/// Bivar_Kernel::csrmultcsr: symbolic row structure, first-product assignment,
/// subsequent g updates, then CSR_Matrix::csr_add(old, product). No implicit
/// additive identity or coefficient is required from the output type.
pub fn csr_sparse<A:Clone,B:Clone,C:Clone>(a:&Csr<A>,b:&Csr<B>,old:Option<&Csr<C>>,
    function:impl Fn(&A,&B)->C,accumulate:impl Fn(C,&mut C))->Csr<C> {
    let (m,k)=a.shape();let (rows_b,n)=b.shape();assert_eq!(k,rows_b);
    let mut symbolic=vec![0usize;m];let mut present=vec![false;n];
    for row in 0..m {
        present.fill(false);
        for ia in a.row_offsets()[row]-1..a.row_offsets()[row+1]-1 {
            let inner=a.columns()[ia]-1;
            for ib in b.row_offsets()[inner]-1..b.row_offsets()[inner+1]-1 {
                let column=b.columns()[ib]-1;
                if !present[column] {present[column]=true;symbolic[row]+=1;}
            }
        }
    }
    let mut entries=Vec::with_capacity(symbolic.iter().sum());
    let mut values:Vec<Option<C>>=vec![None;n];
    for row in 0..m {
        for ia in a.row_offsets()[row]-1..a.row_offsets()[row+1]-1 {
            let inner=a.columns()[ia]-1;
            for ib in b.row_offsets()[inner]-1..b.row_offsets()[inner+1]-1 {
                let value=function(&a.values()[ia],&b.values()[ib]);
                let slot=&mut values[b.columns()[ib]-1];
                if let Some(previous)=slot {accumulate(value,previous);}else{*slot=Some(value);}
            }
        }
        for (column,value) in values.iter_mut().enumerate() {
            if let Some(value)=value.take(){entries.push((row+1,column+1,value));}
        }
    }
    let product=Coo::new(m,n,entries).to_csr();
    let Some(old)=old.filter(|old|!old.values().is_empty())else{return product;};
    assert_eq!(old.shape().0,m);
    let columns=n.max(old.shape().1);
    let mut values:Vec<Option<C>>=vec![None;columns];
    let mut entries=Vec::new();
    for row in 0..m {
        // Source copies A entries before accumulating B; duplicate old
        // coordinates therefore leave their last stored value.
        for i in old.row_offsets()[row]-1..old.row_offsets()[row+1]-1 {
            values[old.columns()[i]-1]=Some(old.values()[i].clone());
        }
        for i in product.row_offsets()[row]-1..product.row_offsets()[row+1]-1 {
            let slot=&mut values[product.columns()[i]-1];
            if let Some(previous)=slot {accumulate(product.values()[i].clone(),previous);}
            else {*slot=Some(product.values()[i].clone());}
        }
        for (column,value) in values.iter_mut().enumerate() {
            if let Some(value)=value.take(){entries.push((row+1,column+1,value));}
        }
    }
    Coo::new(m,columns,entries).to_csr()
}
