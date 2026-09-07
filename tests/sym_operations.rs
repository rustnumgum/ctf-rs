use ctf::{
    algebra::Arithmetic,
    sym_permutations::{circular_generator, enumerate},
    symmetry::{Layout, Packed, Symmetry::*},
};
#[test]
fn broken_symmetry_permutation_discovery() {
    let permutations = enumerate(
        [b"ij".as_slice(), b"ij".as_slice()],
        [&[AS, NS][..], &[NS, NS][..]],
    );
    assert_eq!(permutations.len(), 2);
    assert_eq!(permutations[0].indices, [b"ij".to_vec(), b"ij".to_vec()]);
    assert_eq!(permutations[0].sign, 1);
    assert_eq!(permutations[1].indices, [b"ji".to_vec(), b"ij".to_vec()]);
    assert_eq!(permutations[1].sign, -1);
    let preserved = enumerate(
        [b"ij".as_slice(), b"ji".as_slice()],
        [&[AS, NS][..], &[AS, NS][..]],
    );
    assert_eq!(preserved.len(), 1);
    assert_eq!(preserved[0].sign, -1);
    let permutations = enumerate(
        [b"ijk".as_slice(), b"ijk".as_slice()],
        [&[SY, SY, NS][..], &[NS, NS, NS][..]],
    );
    assert_eq!(permutations.len(), 6);
    assert!(permutations.iter().all(|p| p.sign == 1));
    let permutations = enumerate(
        [b"ij".as_slice(), b"jk".as_slice(), b"ik".as_slice()],
        [&[SY, NS][..], &[NS, NS][..], &[NS, NS][..]],
    );
    assert_eq!(permutations.len(), 2);
    assert_eq!(circular_generator(&[AS, NS, NS]), (vec![1, 0, 2], 2, -1));
    assert_eq!(circular_generator(&[AS, AS, NS]), (vec![1, 2, 0], 3, 1));
}
#[test]
fn packed_iteration_and_diagonal_operations() {
    for kind in [SY, AS, SH] {
        let layout = Layout::new(vec![3, 3, 2, 3, 3], vec![kind, NS, NS, kind, NS]);
        let coordinates: Vec<_> = layout.coordinates().collect();
        assert_eq!(coordinates.len(), layout.len());
        for (offset, coords) in coordinates.iter().enumerate() {
            assert_eq!(layout.locate(coords), Some((offset, 1)));
        }
    }
    assert_eq!(
        Layout::new(vec![], vec![])
            .coordinates()
            .collect::<Vec<_>>(),
        vec![Vec::<usize>::new()]
    );
    assert_eq!(Layout::new(vec![0], vec![NS]).coordinates().count(), 0);
    let mut p = Packed::new(
        Layout::new(vec![3, 3], vec![SY, NS]),
        Arithmetic::<i64>::new(),
    );
    p.values_mut().fill(2);
    p.scale_indexed("ii", &3);
    assert_eq!(p.values(), &[6, 2, 6, 2, 2, 6]);
    p.transform_indexed("ii", |v| *v += 1);
    assert_eq!(p.values(), &[7, 2, 7, 2, 2, 7]);
    p.scale_indexed("ij", &2);
    assert_eq!(p.values(), &[14, 4, 14, 4, 4, 14]);
    let mut p = Packed::new(
        Layout::new(vec![3, 3], vec![AS, NS]),
        Arithmetic::<i64>::new(),
    );
    p.values_mut().fill(5);
    p.transform_indexed("ii", |_| panic!("AS diagonal is not stored"));
    assert_eq!(p.values(), &[5; 3]);
}
