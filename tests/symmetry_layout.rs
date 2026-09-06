use ctf::{algebra::Arithmetic,symmetry::{Layout,Packed,Symmetry::*}};
#[test]
fn packed_sizes_and_column_major_offsets() {
    for (kind,size) in [(SY,20),(AS,4),(SH,4)] {
        let layout=Layout::new(vec![4;3],vec![kind,kind,NS]);assert_eq!(layout.len(),size);assert_eq!(layout.symmetric_len(),20);
        let mut expected=0;
        for k in 0..4 {for j in 0..=k {for i in 0..=j {
            if kind!=SY && (i==j || j==k) {assert_eq!(layout.locate(&[i,j,k]),None);}
            else {assert_eq!(layout.locate(&[i,j,k]),Some((expected,1)));expected+=1;}
        }}}
        assert_eq!(expected,size);
    }
    let mixed=Layout::new(vec![3,3,2,4,4],vec![SY,NS,NS,AS,NS]);
    // SY rank 4 + standalone axis stride 6 + AS rank 4 * stride 12.
    assert_eq!(mixed.len(),72);assert_eq!(mixed.locate(&[1,2,1,1,3]),Some((58,1)));
    assert_eq!(Layout::new(vec![],vec![]).locate(&[]),Some((0,1)));
    assert_eq!(Layout::new(vec![0,0],vec![SY,NS]).len(),0);
    assert_eq!(Layout::new(vec![2;3],vec![AS,AS,NS]).len(),0);
}
#[test]
fn signed_storage_and_hollow_symmetry() {
    let mut anti=Packed::new(Layout::new(vec![3;3],vec![AS,AS,NS]),Arithmetic::<i64>::new());
    anti.write_add(&[0,1,2],&7);assert_eq!(anti.values(),&[7]);
    for (indices,value) in [([0,1,2],7),([1,0,2],-7),([1,2,0],7),([2,1,0],-7),([0,0,2],0)] {
        assert_eq!(anti.read(&indices),value);
    }
    anti.write_add(&[2,1,0],&3);assert_eq!(anti.values(),&[4]);
    anti.write_add(&[1,1,2],&99);assert_eq!(anti.values(),&[4]);
    let mut hollow=Packed::new(Layout::new(vec![3,3],vec![SH,NS]),Arithmetic::<i64>::new());
    hollow.write_add(&[2,0],&5);assert_eq!(hollow.read(&[0,2]),5);assert_eq!(hollow.read(&[2,2]),0);
}
