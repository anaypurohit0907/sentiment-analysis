use rand::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use std::collections::HashMap;

/// Split data into train and test sets
pub fn train_test_split<T: Clone, U: Clone>(
    data: Vec<T>,
    labels: Vec<U>,
    test_size: f32,
    random_seed: u64,
) -> (Vec<T>, Vec<T>, Vec<U>, Vec<U>) {
    assert_eq!(data.len(), labels.len(), "Data and labels must have same length");
    assert!(test_size > 0.0 && test_size < 1.0, "test_size must be between 0 and 1");

    let n = data.len();
    let n_test = (n as f32 * test_size).round() as usize;

    // Create indices and shuffle
    let mut indices: Vec<usize> = (0..n).collect();
    let mut rng = StdRng::seed_from_u64(random_seed);
    indices.shuffle(&mut rng);

    // Split indices
    let test_indices = &indices[..n_test];
    let train_indices = &indices[n_test..];

    // Create splits
    let mut train_data = Vec::new();
    let mut train_labels = Vec::new();
    let mut test_data = Vec::new();
    let mut test_labels = Vec::new();

    for &idx in train_indices {
        train_data.push(data[idx].clone());
        train_labels.push(labels[idx].clone());
    }

    for &idx in test_indices {
        test_data.push(data[idx].clone());
        test_labels.push(labels[idx].clone());
    }

    (train_data, test_data, train_labels, test_labels)
}

/// Stratified train-test split (maintains class distribution)
pub fn stratified_train_test_split<T: Clone>(
    data: Vec<T>,
    labels: Vec<i32>,
    test_size: f32,
    random_seed: u64,
) -> (Vec<T>, Vec<T>, Vec<i32>, Vec<i32>) {
    assert_eq!(data.len(), labels.len(), "Data and labels must have same length");
    assert!(test_size > 0.0 && test_size < 1.0, "test_size must be between 0 and 1");

    // Group indices by label
    let mut label_to_indices: HashMap<i32, Vec<usize>> = HashMap::new();
    for (idx, &label) in labels.iter().enumerate() {
        label_to_indices.entry(label).or_insert_with(Vec::new).push(idx);
    }

    let mut rng = StdRng::seed_from_u64(random_seed);
    let mut train_indices = Vec::new();
    let mut test_indices = Vec::new();

    // Split each class proportionally
    for (_, mut indices) in label_to_indices {
        indices.shuffle(&mut rng);
        let n_test = (indices.len() as f32 * test_size).round() as usize;
        test_indices.extend_from_slice(&indices[..n_test]);
        train_indices.extend_from_slice(&indices[n_test..]);
    }

    // Create splits
    let mut train_data = Vec::new();
    let mut train_labels = Vec::new();
    let mut test_data = Vec::new();
    let mut test_labels = Vec::new();

    for &idx in &train_indices {
        train_data.push(data[idx].clone());
        train_labels.push(labels[idx]);
    }

    for &idx in &test_indices {
        test_data.push(data[idx].clone());
        test_labels.push(labels[idx]);
    }

    (train_data, test_data, train_labels, test_labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_train_test_split() {
        let data: Vec<i32> = (0..100).collect();
        let labels: Vec<i32> = (0..100).collect();
        let (train_data, test_data, train_labels, test_labels) = 
            train_test_split(data, labels, 0.2, 42);
        
        assert_eq!(train_data.len(), 80);
        assert_eq!(test_data.len(), 20);
        assert_eq!(train_labels.len(), 80);
        assert_eq!(test_labels.len(), 20);
    }
}
