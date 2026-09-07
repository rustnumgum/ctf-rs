// Adapted from cc4s CTF scaling/scale_tsr.cxx.
// Copyright (c) 2011, Edgar Solomonik. See LICENSE.
//! Virtual-block traversal for dense scaling.

/// Offsets of tensor virtual blocks selected by the contraction-wide index
/// tuple. Dimension zero changes fastest, exactly as source `scl_virt::run`.
/// Repeated tensor labels therefore visit only equal virtual coordinates.
pub fn virtual_offsets(virtual_dimensions: &[usize], indices: &[usize]) -> Vec<usize> {
    assert!(virtual_dimensions.iter().all(|&dimension| dimension > 0));
    assert!(indices.iter().all(|&index| index < virtual_dimensions.len()));

    let mut tensor_strides = vec![0usize; indices.len()];
    let mut tensor_blocks = 1usize;
    for (stride, &index) in tensor_strides.iter_mut().zip(indices) {
        *stride = tensor_blocks;
        tensor_blocks *= virtual_dimensions[index];
    }
    let mut index_strides = vec![0usize; virtual_dimensions.len()];
    for (&index, &stride) in indices.iter().zip(&tensor_strides) {
        index_strides[index] += stride;
    }

    let tuple_count: usize = virtual_dimensions.iter().product();
    let mut offsets = Vec::with_capacity(tuple_count);
    let mut tuple = vec![0usize; virtual_dimensions.len()];
    let mut offset = 0usize;
    loop {
        offsets.push(offset);
        let mut index = 0;
        while index < virtual_dimensions.len() {
            offset -= index_strides[index] * tuple[index];
            tuple[index] += 1;
            if tuple[index] == virtual_dimensions[index] {
                tuple[index] = 0;
            }
            offset += index_strides[index] * tuple[index];
            if tuple[index] != 0 {
                break;
            }
            index += 1;
        }
        if index == virtual_dimensions.len() {
            break;
        }
    }
    offsets
}

/// Run a sequential scaling child on each selected virtual block.
pub fn for_each_block_mut<T>(
    data: &mut [T],
    block_size: usize,
    virtual_dimensions: &[usize],
    indices: &[usize],
    mut child: impl FnMut(&mut [T]),
) {
    let tensor_blocks: usize = indices
        .iter()
        .map(|&index| virtual_dimensions[index])
        .product();
    assert_eq!(data.len(), tensor_blocks * block_size);
    for offset in virtual_offsets(virtual_dimensions, indices) {
        child(&mut data[offset * block_size..(offset + 1) * block_size]);
    }
}
