//! Root-only variable-byte gather for final disjoint sparse-row assembly.
use super::{check,Comm};
use mpi_sys as sys;

impl Comm {
    pub(crate) fn gather_bytes(&self,root:usize,bytes:&[u8])->Option<Vec<Vec<u8>>> {
        let is_root=self.rank()==root;
        let count:i32=bytes.len().try_into().unwrap();
        let mut counts=vec![0i32;if is_root{self.size()}else{0}];
        unsafe {check(sys::MPI_Gather(
            (&count as *const i32).cast(),1,sys::RSMPI_INT32_T,
            counts.as_mut_ptr().cast(),1,sys::RSMPI_INT32_T,
            root.try_into().unwrap(),self.raw));}
        let mut total=0i32;
        let displacements:Vec<_>=counts.iter().map(|&count|{
            let offset=total;total=total.checked_add(count).unwrap();offset
        }).collect();
        let length:usize=total.try_into().unwrap();
        let mut received=vec![0u8;length.max(1)];
        let empty=[0u8];
        let send=if bytes.is_empty(){&empty[..]}else{bytes};
        unsafe {check(sys::MPI_Gatherv(
            send.as_ptr().cast(),count,sys::RSMPI_UINT8_T,
            received.as_mut_ptr().cast(),counts.as_ptr(),displacements.as_ptr(),sys::RSMPI_UINT8_T,
            root.try_into().unwrap(),self.raw));}
        if is_root {
            Some(counts.into_iter().zip(displacements).map(|(count,offset)|{
                received[offset as usize..(offset+count)as usize].to_vec()
            }).collect())
        } else {None}
    }
}
