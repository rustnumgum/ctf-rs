use ctf::{
    algebra::Arithmetic,
    context::Context,
    cost::Models,
    kernel,
    mapping::{Distribution, Topology},
    random::Generator,
    sparse::SparseTensor,
    sparse_formats::{Coo, Csr},
    sparse_search::{Options, Pattern, SearchCache, StorageSize},
    tensor::Tensor,
    topology_candidates,
};

const BLOCK_SIZE: usize = 7;
const BLOCK_COUNT: usize = 10;
const NORM_BOUND: f64 = 1.0e-4;

type BlockAlgebra = Arithmetic<f64>;
type Block<'c, 'r> = Tensor<'c, 'r, BlockAlgebra>;
type BlockCsr<'c, 'r> = Csr<Block<'c, 'r>>;

/// The source uses the host C rand stream for the common outer sparsity
/// structure.  The pinned WSL source is glibc, whose rand stream is the
/// degree-31 additive generator below.
struct GlibcRand {
    state: [u32; 31],
    front: usize,
    rear: usize,
}

impl GlibcRand {
    fn new(seed: u32) -> Self {
        let mut state = [0; 31];
        state[0] = if seed == 0 { 1 } else { seed };
        for index in 1..state.len() {
            state[index] = ((16807_i64 * i64::from(state[index - 1])) % 2147483647) as u32;
        }
        let mut result = Self {
            state,
            front: 3,
            rear: 0,
        };
        for _ in 0..(31 * 10) {
            result.next();
        }
        result
    }

    fn next(&mut self) -> u32 {
        let value = self.state[self.front].wrapping_add(self.state[self.rear]);
        self.state[self.front] = value;
        self.front = (self.front + 1) % self.state.len();
        self.rear = (self.rear + 1) % self.state.len();
        value >> 1
    }

    fn index(&mut self, bound: usize) -> usize {
        (self.next() as usize) % bound
    }
}

fn new_block<'c, 'r>(context: &'c Context<'r>, rows: usize, cols: usize) -> Block<'c, 'r> {
    Tensor::new(
        context,
        Distribution::cyclic(vec![rows, cols], context.size()),
        BlockAlgebra::new(),
    )
}

// This is the source's Matrix<Tensor<>> on MPI_COMM_SELF represented by a
// rank-local CSR.  The Tensor values retain the MPI world context for their
// inner collective dense operations; no outer semiring or gather is used.
fn make_a<'c, 'r>(
    context: &'c Context<'r>,
    ranges: &[usize],
    structure: &mut GlibcRand,
    random: &mut Generator,
) -> BlockCsr<'c, 'r> {
    let block_count = ranges.len();
    let mut entries = Vec::with_capacity(block_count);
    for column in 0..block_count {
        let row = structure.index(block_count);
        let mut block = new_block(context, ranges[row], ranges[column]);
        block.fill_random(0., 1., random);
        entries.push((row + 1, column + 1, block));
    }
    Coo::new(block_count, block_count, entries).to_csr()
}

fn make_b<'c, 'r>(
    context: &'c Context<'r>,
    ranges: &[usize],
    structure: &mut GlibcRand,
    random: &mut Generator,
) -> BlockCsr<'c, 'r> {
    let block_count = ranges.len();
    let mut entries = Vec::with_capacity(block_count);
    for row in 0..block_count {
        let column = structure.index(block_count);
        let mut block = new_block(context, ranges[row], ranges[column]);
        block.fill_random(1., 1., random);
        entries.push((row + 1, column + 1, block));
    }
    Coo::new(block_count, block_count, entries).to_csr()
}

fn multiply_blocks<'c, 'r>(
    left: &Block<'c, 'r>,
    right: &Block<'c, 'r>,
    topology: &Topology,
) -> Block<'c, 'r> {
    let left_shape = &left.distribution().shape;
    let right_shape = &right.distribution().shape;
    assert_eq!(left_shape[1], right_shape[0]);
    let mut result = new_block(left.context(), left_shape[0], right_shape[1]);
    result
        .contract_from(
            "ij",
            left,
            "ik",
            right,
            "kj",
            topology.clone(),
            1.,
            0.,
        )
        .unwrap();
    result
}

fn add_blocks<'c, 'r>(
    output: &mut Block<'c, 'r>,
    input: &Block<'c, 'r>,
    topology: &Topology,
) {
    output
        .sum_from("ij", input, "ij", topology.clone(), 1., 1.)
        .unwrap();
}

fn prefix_ranges(ranges: &[usize]) -> Vec<usize> {
    let mut prefix = Vec::with_capacity(ranges.len() + 1);
    prefix.push(0);
    for &range in ranges {
        prefix.push(prefix.last().unwrap() + range);
    }
    prefix
}

fn flatten<'c, 'r>(
    matrix: &BlockCsr<'c, 'r>,
    ranges: &[usize],
    context: &'c Context<'r>,
) -> SparseTensor<'c, 'r, BlockAlgebra> {
    let prefix = prefix_ranges(ranges);
    let mut flat = SparseTensor::new(
        context,
        Distribution::cyclic(
            vec![*prefix.last().unwrap(), *prefix.last().unwrap()],
            context.size(),
        ),
        BlockAlgebra::new(),
    );
    let outer_dimension = matrix.shape().0;
    let block_count = matrix.values().len();
    let mut pairs = Vec::new();
    for row in 0..outer_dimension {
        let start = matrix.row_offsets()[row] - 1;
        let end = matrix.row_offsets()[row + 1] - 1;
        for position in start..end {
            let column = matrix.columns()[position] - 1;
            let key = row + outer_dimension * column;

            // Keep the pinned source quirk: flatten decodes with this matrix's
            // actual local nnz rather than the outer block dimension.
            let block_row = key % block_count;
            let block_column = key / block_count;
            let block = &matrix.values()[position];
            assert_eq!(block.distribution().shape.len(), 2);
            let flat_distribution = flat.distribution();
            for (block_key, value) in block.local_pairs() {
                let coordinates = block.distribution().decode_key(block_key);
                let flat_key = flat_distribution.encode_key(&[
                    prefix[block_row] + coordinates[0],
                    prefix[block_column] + coordinates[1],
                ]);
                pairs.push((flat_key, value));
            }
        }
    }
    flat.write_add(&pairs);
    flat
}

fn global_nnz(matrix: &SparseTensor<'_, '_, BlockAlgebra>) -> u64 {
    let rank = matrix.context().rank();
    let local = matrix
        .local_pairs()
        .into_iter()
        .filter(|(key, _)| matrix.distribution().owner(*key) == rank)
        .count() as u64;
    matrix
        .context()
        .all_reduce(&Arithmetic::<u64>::new(), &local)
}

fn run(context: &Context<'_>) -> f64 {
    let ranges = vec![BLOCK_SIZE; BLOCK_COUNT];
    let mut structure = GlibcRand::new(1000);
    let mut random = Generator::new(context.rank() as u64);
    let topology = Topology::new(vec![context.size()]);

    let a = make_a(context, &ranges, &mut structure, &mut random);
    let b = make_b(context, &ranges, &mut structure, &mut random);
    let c = kernel::csr_sparse(
        &a,
        &b,
        None,
        |left, right| multiply_blocks(left, right, &topology),
        |value, output| add_blocks(output, &value, &topology),
    );

    let flat_a = flatten(&a, &ranges, context);
    let flat_b = flatten(&b, &ranges, context);
    let flat_c = flatten(&c, &ranges, context);
    let mut reference = SparseTensor::new(
        context,
        flat_c.distribution().clone(),
        BlockAlgebra::new(),
    );
    let catalog = topology_candidates::all_shapes(context.size());
    let models = Models::upstream(1);
    let mut cache = SearchCache::new(
        context,
        &catalog,
        &models,
        [StorageSize {
            element_bytes: 8,
            pair_bytes: 16,
        }; 3],
        false,
        Pattern::SparseSparseSparse,
        Options {
            memory_limit: u64::MAX,
            weight: 0.,
            allow_exhaustive: true,
        },
    );
    let selected = cache
        .prepare(
            [
                flat_a.distribution(),
                flat_b.distribution(),
                flat_c.distribution(),
            ],
            ["ik", "kj", "ij"],
            [
                Some(global_nnz(&flat_a)),
                Some(global_nnz(&flat_b)),
                Some(global_nnz(&flat_c)),
            ],
            None,
        )
        .unwrap()
        .expect("automatic sparse reference selection rejected the fixture");
    reference.contract_sparse_from_selected(
        "ij",
        &flat_a,
        "ik",
        &flat_b,
        "kj",
        selected,
        1.,
        0.,
        true,
    );
    reference.sum_from("ij", &flat_c, "ij", -1., 1.);
    reference.norm2()
}

fn main() {
    let (universe, provided) = mpi::initialize_with_threading(mpi::Threading::Funneled)
        .expect("MPI initialization failed");
    assert!(provided >= mpi::Threading::Funneled);
    let world = Context::world(&universe);
    let q = run(&world);
    assert!(q <= NORM_BOUND, "block-sparse residual {q} exceeds {NORM_BOUND}");

    let parity = world
        .split(Some((world.rank() % 2) as i32), world.rank() as i32)
        .unwrap();
    let parity_q = run(&parity);
    assert!(
        parity_q <= NORM_BOUND,
        "block-sparse parity residual {parity_q} exceeds {NORM_BOUND}"
    );
    parity.close();

    if world.rank() == 0 {
        println!(
            "DIGIT / PASS block_sparse: source matrix-of-distributed-Tensor local CSR contraction and flattened reference; n=7; r=10; Q={q}; ref=source flattened product; bound=1e-4; world+parity"
        );
    }
    world.close();
    drop(universe);
}
