use ctf::{sym_indices::*, symmetry::Symmetry::*};
#[test]
fn signs_and_pair_alignment() {
    assert_eq!(relative_sign(b"abc", b"abc"), 1);
    assert_eq!(relative_sign(b"abc", b"bac"), -1);
    assert_eq!(relative_sign(b"abc", b"bca"), 1);
    assert_eq!(relative_sign(b"", b""), 1);
    for (input, expected_sign) in [
        (b"ijk".as_slice(), 1),
        (b"jik", -1),
        (b"kij", 1),
        (b"kji", -1),
    ] {
        let mut b = input.to_vec();
        assert_eq!(
            align_pair(b"ijk", &[AS, AS, NS], &mut b, &[AS, AS, NS]),
            expected_sign
        );
        assert_eq!(b, b"ijk");
    }
    let mut b = b"kji".to_vec();
    assert_eq!(align_pair(b"ijk", &[SY, SY, NS], &mut b, &[SY, SY, NS]), 1);
    assert_eq!(b, b"ijk");
    let mut b = b"jix".to_vec();
    assert_eq!(align_pair(b"ijk", &[AS, NS, NS], &mut b, &[AS, NS, NS]), -1);
    assert_eq!(b, b"ijx");
}
#[test]
fn source_multiplicity_rules() {
    for kind in [SY, AS, SH] {
        assert_eq!(
            contraction_factor(b"ijk", &[kind, kind, NS], b"ijk", &[kind, kind, NS], b""),
            6
        );
        assert_eq!(
            contraction_factor(b"ijk", &[kind, kind, NS], b"ijk", &[kind, kind, NS], b"ijk"),
            1
        );
        assert_eq!(
            contraction_factor(b"ijx", &[kind, kind, NS], b"ijy", &[kind, kind, NS], b"xy"),
            2
        );
    }
    assert_eq!(summation_factor(b"ijk", &[AS, AS, NS], b""), 0);
    assert_eq!(summation_factor(b"ijk", &[SH, SH, NS], b""), 6);
    assert_eq!(summation_factor(b"ijk", &[SY, SY, NS], b""), 1);
    assert_eq!(summation_factor(b"ijk", &[AS, NS, NS], b"ij"), 1);
    assert_eq!(contraction_factor(b"", &[], b"", &[], b""), 1);
}
