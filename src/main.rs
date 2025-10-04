use anyhow::Result;

// Folder modules
mod preprocessing;
mod tfidf;

// Public API from modules
use crate::preprocessing::tokenizer::tokenize;
use crate::tfidf::TfIdfVectorizer;

// --- Assume these exist elsewhere in the project ---
#[allow(dead_code)]
#[derive(Clone, Copy)]
enum Init { HeUniform }
#[allow(dead_code)]
fn initialize_network(_sizes: &[usize], _init: Init, _seed: u64)
    -> (Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>) {
    // stub for demo
    (vec![], vec![])
}
#[allow(dead_code)]
fn check_shapes(
    _sizes: &[usize],
    _weights: &Vec<Vec<Vec<f32>>>,
    _biases: &Vec<Vec<f32>>,
) -> std::result::Result<(), &'static str> {
    Ok(())
}
#[allow(dead_code)]
fn preview_params(_weights: &Vec<Vec<Vec<f32>>>, _biases: &Vec<Vec<f32>>, _n: usize) {}
#[allow(dead_code)]
fn write_params_to_txt(
    _w_path: &str, _b_path: &str,
    _weights: &Vec<Vec<Vec<f32>>>, _biases: &Vec<Vec<f32>>
) -> Result<()> { Ok(()) }

// Optional helper: convert sparse TF-IDF map to dense vector for NN input.
fn sparse_to_dense(
    sparse: &std::collections::HashMap<usize, f32>,
    dim: usize,
) -> Vec<f32> {
    let mut dense = vec![0.0f32; dim];
    for (i, v) in sparse.iter() {
        if *i < dim { dense[*i] = *v; }
    }
    dense
}

fn main() -> Result<()> {
    // 1) Preprocess flow (old)
    let text = "This is a test sentence, with punctuation!";
    let tokens = tokenize(text);
    println!("Tokens: {:?}", tokens);

    // 2) Vectorization flow (new): fit on a tiny corpus, then transform and L2-normalize
    let corpus = vec![tokens.clone()]; // demo corpus; fit on more docs in practice
    let vectorizer = TfIdfVectorizer::fit(&corpus);

    let mut tfidf_sparse = vectorizer.transform(&tokens);
    TfIdfVectorizer::l2_normalize(&mut tfidf_sparse);
    println!("TF-IDF (sparse index -> weight): {:?}", tfidf_sparse);

    // Optional: make a dense 1 x V input for later NN usage
    let tfidf_dense = sparse_to_dense(&tfidf_sparse, vectorizer.vocab.len());
    println!("TF-IDF (dense, first 16 dims): {:?}", &tfidf_dense.iter().take(16).collect::<Vec<_>>());

    // 3) NN initialization flow (existing)
    let sizes = vec![300, 256, 256, 128, 32]; // example architecture
    let (weights, biases) = initialize_network(&sizes, Init::HeUniform, 42);

    // Shape validation
    check_shapes(&sizes, &weights, &biases).map_err(|e| anyhow::anyhow!(e))?;

    // Preview a small slice for quick inspection
    preview_params(&weights, &biases, 3);

    // Write full parameter tensors to files
    write_params_to_txt("weights.txt", "biases.txt", &weights, &biases)?;
    println!("Parameters written to weights.txt and biases.txt");

    Ok(())
}
