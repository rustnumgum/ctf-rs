// Adapted from cc4s CTF interface/matrix.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Distributed native matrix operations, following interface/matrix.cxx's
//! read_mat -> ScaLAPACK -> tensor/get_tri sequence. No global tensor gather.
use crate::{algebra::Arithmetic,mapping::{Distribution,Mapping,Topology},tensor::Tensor};

fn distribution(shape:&[usize],grid:[usize;2])->Distribution {
    let topology=Topology::new(grid.to_vec());
    let mut row=Mapping::Unmapped;row.augment_physical(&topology,0);
    let mut column=Mapping::Unmapped;column.augment_physical(&topology,1);
    Distribution::new(shape.to_vec(),topology,vec![row,column])
}
impl<'c,'r> Tensor<'c,'r,Arithmetic<f64>> {
    /// Collective PD POTRF on a caller-selected grid with cyclic block size 1.
    /// Returns only the requested triangle in the input's original distribution.
    /// The BLACS grid is explicitly closed before returning, including info errors.
    pub fn cholesky(&self,grid:[usize;2],lower:bool)->Result<Self,i32> {
        assert_eq!(self.distribution().shape.len(),2);let n=self.distribution().shape[0];
        assert_eq!(self.distribution().shape[1],n);
        let mut result=self.clone();result.redistribute(distribution(&[n,n],grid));
        let blacs=self.context().inner.scalapack_grid(grid[0],grid[1]);
        let operation:Result<(),i32>=(|| {
            let desc=blacs.descriptor(n,n,1,1,n.div_ceil(grid[0]).max(1))?;
            let mut values=result.local_storage().to_vec();values.resize(values.len().max(1),0.);
            blacs.cholesky(n,&mut values,&desc,lower)?;
            let dist=result.distribution().clone();let rank=self.context().rank();
            result.transform(|key,value| {
                let row=key%n;let col=key/n;
                *value=if (lower&&row>=col)||(!lower&&row<=col) {values[dist.local_offset(rank,key)]} else {0.};
            });
            Ok(())
        })();
        blacs.close();operation?;
        result.redistribute(self.distribution().clone());Ok(result)
    }
    /// Solve op(T)*X=B or X*op(T)=B using PDTRSM, keeping input tensors unchanged.
    /// `self` is B; diagonal entries of T are non-unit, as in upstream solve_tri.
    pub fn solve_tri(&self,factor:&Self,grid:[usize;2],lower:bool,from_left:bool,transpose:bool)->Result<Self,i32> {
        assert!(std::ptr::eq(self.context(),factor.context()));
        assert_eq!(self.distribution().shape.len(),2);assert_eq!(factor.distribution().shape.len(),2);
        let (m,n)=(self.distribution().shape[0],self.distribution().shape[1]);
        let order=if from_left {m} else {n};assert_eq!(factor.distribution().shape,vec![order,order]);
        let mut a=factor.clone();a.redistribute(distribution(&[order,order],grid));
        let mut result=self.clone();result.redistribute(distribution(&[m,n],grid));
        let blacs=self.context().inner.scalapack_grid(grid[0],grid[1]);
        let operation:Result<(),i32>=(|| {
            let da=blacs.descriptor(order,order,1,1,order.div_ceil(grid[0]).max(1))?;
            let db=blacs.descriptor(m,n,1,1,m.div_ceil(grid[0]).max(1))?;
            let mut av=a.local_storage().to_vec();av.resize(av.len().max(1),0.);
            let mut values=result.local_storage().to_vec();
            values.resize((m.div_ceil(grid[0]).max(1)*n.div_ceil(grid[1])).max(1),0.);
            blacs.solve_tri(m,n,&av,&da,&mut values,&db,lower,from_left,transpose);
            let dist=result.distribution().clone();let rank=self.context().rank();
            result.transform(|key,value|*value=values[dist.local_offset(rank,key)]);Ok(())
        })();
        blacs.close();operation?;
        result.redistribute(self.distribution().clone());Ok(result)
    }
}
