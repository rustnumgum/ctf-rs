//! Repeated-index key projection used for tensor diagonal extraction/insertion.
//! Index deletion/insertion follows tensor::extract_diag in the pinned source.
pub(crate) struct Projection {
    pub labels: String,
    pub shape: Vec<usize>,
    axes: Vec<usize>,
    first: Vec<usize>,
}
impl Projection {
    pub fn new(shape: &[usize], labels: &str) -> Self {
        assert!(labels.is_ascii());
        assert_eq!(shape.len(), labels.len());
        let mut unique = Vec::new();
        let mut dimensions = Vec::new();
        let mut axes = Vec::new();
        let mut first = Vec::new();
        for (i, label) in labels.bytes().enumerate() {
            let axis = if let Some(axis) = unique.iter().position(|&l| l == label) {
                assert_eq!(dimensions[axis], shape[i]);
                axis
            } else {
                unique.push(label);
                dimensions.push(shape[i]);
                first.push(i);
                unique.len() - 1
            };
            axes.push(axis);
        }
        Self {
            labels: String::from_utf8(unique).unwrap(),
            shape: dimensions,
            axes,
            first,
        }
    }
    pub fn repeated(&self) -> bool {
        self.axes.len() != self.first.len()
    }
    pub fn project(&self, coordinates: &[usize]) -> Option<Vec<usize>> {
        if self
            .axes
            .iter()
            .enumerate()
            .all(|(i, &axis)| coordinates[i] == coordinates[self.first[axis]])
        {
            Some(self.first.iter().map(|&i| coordinates[i]).collect())
        } else {
            None
        }
    }
    pub fn expand(&self, coordinates: &[usize]) -> Vec<usize> {
        self.axes.iter().map(|&axis| coordinates[axis]).collect()
    }
}
