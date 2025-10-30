use std::fs::{self, File};
use std::io::{BufReader, BufWriter};
use std::path::Path;
use anyhow::{Result, Context};
use serde::{Deserialize, Serialize};
use chrono::Utc;

use crate::tfidf::vectorizer::TfIdfVectorizer;


/// Metadata about the processed dataset
#[derive(Serialize, Deserialize, Debug)]
pub struct DatasetMetadata {
    pub feature_count: usize,
    pub vocab_size: usize,
    pub train_size: usize,
    pub test_size: usize,
    pub max_features: Option<usize>,
    pub min_df: usize,
    pub max_df: f32,
    pub ngram_range: (usize, usize),
    pub created_at: String,
}

/// Structure to hold TF-IDF matrices
#[derive(Serialize, Deserialize)]
pub struct TfidfMatrices {
    pub train_vectors: Vec<Vec<f32>>,
    pub test_vectors: Vec<Vec<f32>>,
}

/// Save TF-IDF vectorizer to file
pub fn save_vectorizer(vectorizer: &TfIdfVectorizer, path: &str) -> Result<()> {
    let file = File::create(path)
        .context(format!("Failed to create file: {}", path))?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, vectorizer)
        .context("Failed to serialize vectorizer")?;
    println!("✓ Saved vectorizer to {}", path);
    Ok(())
}

/// Load TF-IDF vectorizer from file
pub fn load_vectorizer(path: &str) -> Result<TfIdfVectorizer> {
    let file = File::open(path)
        .context(format!("Failed to open file: {}", path))?;
    let reader = BufReader::new(file);
    let vectorizer = bincode::deserialize_from(reader)
        .context("Failed to deserialize vectorizer")?;
    println!("✓ Loaded vectorizer from {}", path);
    Ok(vectorizer)
}

/// Save TF-IDF matrices to file
pub fn save_tfidf_matrices(
    train_vectors: &[Vec<f32>],
    test_vectors: &[Vec<f32>],
    path: &str,
) -> Result<()> {
    let matrices = TfidfMatrices {
        train_vectors: train_vectors.to_vec(),
        test_vectors: test_vectors.to_vec(),
    };
    
    let file = File::create(path)
        .context(format!("Failed to create file: {}", path))?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, &matrices)
        .context("Failed to serialize matrices")?;
    println!("✓ Saved TF-IDF matrices to {}", path);
    Ok(())
}

/// Load TF-IDF matrices from file
pub fn load_tfidf_matrices(path: &str) -> Result<TfidfMatrices> {
    let file = File::open(path)
        .context(format!("Failed to open file: {}", path))?;
    let reader = BufReader::new(file);
    let matrices = bincode::deserialize_from(reader)
        .context("Failed to deserialize matrices")?;
    println!("✓ Loaded TF-IDF matrices from {}", path);
    Ok(matrices)
}

/// Save labels to file
pub fn save_labels(labels: &[f32], path: &str) -> Result<()> {
    let file = File::create(path)
        .context(format!("Failed to create file: {}", path))?;
    let writer = BufWriter::new(file);
    bincode::serialize_into(writer, labels)
        .context("Failed to serialize labels")?;
    println!("✓ Saved labels to {}", path);
    Ok(())
}

/// Load labels from file
pub fn load_labels(path: &str) -> Result<Vec<f32>> {
    let file = File::open(path)
        .context(format!("Failed to open file: {}", path))?;
    let reader = BufReader::new(file);
    let labels = bincode::deserialize_from(reader)
        .context("Failed to deserialize labels")?;
    println!("✓ Loaded labels from {}", path);
    Ok(labels)
}

/// Save metadata to JSON file
pub fn save_metadata(metadata: &DatasetMetadata, path: &str) -> Result<()> {
    let json = serde_json::to_string_pretty(metadata)
        .context("Failed to serialize metadata")?;
    fs::write(path, json)
        .context(format!("Failed to write metadata to {}", path))?;
    println!("✓ Saved metadata to {}", path);
    Ok(())
}

/// Load metadata from JSON file
pub fn load_metadata(path: &str) -> Result<DatasetMetadata> {
    let json = fs::read_to_string(path)
        .context(format!("Failed to read metadata from {}", path))?;
    let metadata = serde_json::from_str(&json)
        .context("Failed to deserialize metadata")?;
    println!("✓ Loaded metadata from {}", path);
    Ok(metadata)
}

/// Create metadata from vectorizer and data
pub fn create_metadata(
    vectorizer: &TfIdfVectorizer,
    train_size: usize,
    test_size: usize,
) -> DatasetMetadata {
    DatasetMetadata {
        feature_count: vectorizer.vocab_size(),
        vocab_size: vectorizer.vocab_size(),
        train_size,
        test_size,
        max_features: vectorizer.max_features,
        min_df: vectorizer.min_df,
        max_df: vectorizer.max_df,
        ngram_range: vectorizer.ngram_range,
        created_at: Utc::now().to_rfc3339(),
    }
}

/// Save all processed data to a directory
pub fn save_all(
    data_dir: &str,
    vectorizer: &TfIdfVectorizer,
    train_vectors: &[Vec<f32>],
    test_vectors: &[Vec<f32>],
    train_labels: &[f32],
    test_labels: &[f32],
) -> Result<()> {
    // Create directory if it doesn't exist
    fs::create_dir_all(data_dir)
        .context(format!("Failed to create directory: {}", data_dir))?;

    // Save vectorizer
    let vectorizer_path = format!("{}/tfidf_vectorizer.bin", data_dir);
    save_vectorizer(vectorizer, &vectorizer_path)?;

    // Save matrices
    let matrices_path = format!("{}/tfidf_matrices.bin", data_dir);
    save_tfidf_matrices(train_vectors, test_vectors, &matrices_path)?;

    // Save labels
    let train_labels_path = format!("{}/y_train.bin", data_dir);
    let test_labels_path = format!("{}/y_test.bin", data_dir);
    save_labels(train_labels, &train_labels_path)?;
    save_labels(test_labels, &test_labels_path)?;

    // Save metadata
    let metadata = create_metadata(vectorizer, train_vectors.len(), test_vectors.len());
    let metadata_path = format!("{}/metadata.json", data_dir);
    save_metadata(&metadata, &metadata_path)?;

    println!("\n✓ All data saved to {}", data_dir);
    Ok(())
}

/// Load all processed data from a directory
pub struct LoadedData {
    pub vectorizer: TfIdfVectorizer,
    pub train_vectors: Vec<Vec<f32>>,
    pub test_vectors: Vec<Vec<f32>>,
    pub train_labels: Vec<f32>,
    pub test_labels: Vec<f32>,
    pub metadata: DatasetMetadata,
}

pub fn load_all(data_dir: &str) -> Result<LoadedData> {
    if !Path::new(data_dir).exists() {
        anyhow::bail!("Data directory does not exist: {}", data_dir);
    }

    // Load vectorizer
    let vectorizer_path = format!("{}/tfidf_vectorizer.bin", data_dir);
    let vectorizer = load_vectorizer(&vectorizer_path)?;

    // Load matrices
    let matrices_path = format!("{}/tfidf_matrices.bin", data_dir);
    let matrices = load_tfidf_matrices(&matrices_path)?;

    // Load labels
    let train_labels_path = format!("{}/y_train.bin", data_dir);
    let test_labels_path = format!("{}/y_test.bin", data_dir);
    let train_labels = load_labels(&train_labels_path)?;
    let test_labels = load_labels(&test_labels_path)?;

    // Load metadata
    let metadata_path = format!("{}/metadata.json", data_dir);
    let metadata = load_metadata(&metadata_path)?;

    println!("\n✓ All data loaded from {}", data_dir);

    Ok(LoadedData {
        vectorizer,
        train_vectors: matrices.train_vectors,
        test_vectors: matrices.test_vectors,
        train_labels,
        test_labels,
        metadata,
    })
}
