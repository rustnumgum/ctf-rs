use ctf::{sym_indices::align_triple,symmetry::Symmetry::*};
#[test]
fn common_to_three_operands() {
    for (pb,sb) in [(b"ijk".as_slice(),1),(b"jik",-1),(b"kij",1),(b"kji",-1)] {
        for (pc,sc) in [(b"ijk".as_slice(),1),(b"jik",-1),(b"kij",1),(b"kji",-1)] {
            let mut b=pb.to_vec();let mut c=pc.to_vec();
            assert_eq!(align_triple(b"ijk",&[AS,AS,NS],&mut b,&[AS,AS,NS],&mut c,&[AS,AS,NS]),sb*sc);
            assert_eq!(b,b"ijk");assert_eq!(c,b"ijk");
        }
    }
}
#[test]
fn incidence_groups_and_partial_symmetry() {
    let mut b=b"klji".to_vec();let mut c=b"lk".to_vec();
    assert_eq!(align_triple(b"ij",&[AS,NS],&mut b,&[AS,NS,AS,NS],&mut c,&[AS,NS]),1);
    assert_eq!(b,b"klij");assert_eq!(c,b"kl");
    let mut b=b"x".to_vec();let mut c=b"ji".to_vec();
    assert_eq!(align_triple(b"ij",&[SY,NS],&mut b,&[NS],&mut c,&[AS,NS]),-1);assert_eq!(c,b"ij");
    let mut b=b"ji".to_vec();let mut c=b"ij".to_vec();
    assert_eq!(align_triple(b"ij",&[SY,NS],&mut b,&[SY,NS],&mut c,&[NS,NS]),1);
    assert_eq!(b,b"ji"); // C's NS boundary prevents treating i/j as one group.
    let mut b=vec![];let mut c=vec![];assert_eq!(align_triple(b"",&[],&mut b,&[],&mut c,&[]),1);
}
