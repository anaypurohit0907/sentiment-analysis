use anyhow::Result;
use sentiment_analysis::preprocessing::{serialization, tokenizer::Tokenizer};

fn main() -> Result<()> {
    println!("=== Loading Preprocessed TF-IDF Data ===\n");

    // Load all preprocessed data
    let data = serialization::load_all("data")?;

    println!("\n=== Dataset Summary ===");
    println!("Training samples: {}", data.train_vectors.len());
    println!("Test samples: {}", data.test_vectors.len());
    println!("Feature dimensions: {}", data.vectorizer.vocab_size());
    println!("Class distribution in training:");
    let mut label_counts = std::collections::HashMap::new();
    for &label in &data.train_labels {
        *label_counts.entry(label as i32).or_insert(0) += 1;
    }
    println!("  {:?}", label_counts);

    println!("\n=== Sample TF-IDF Vector ===");
    if let Some(first_vec) = data.train_vectors.first() {
        let non_zero: Vec<(usize, f32)> = first_vec
            .iter()
            .enumerate()
            .filter(|&(_, v)| *v != 0.0)
            .take(10)
            .map(|(i, v)| (i, *v))
            .collect();
        println!("Non-zero features (first 10): {:?}", non_zero);
    }

    println!("\n=== Transform New Text Example ===");
    let tokenizer = Tokenizer::new()
        .with_stopwords(true)
        .with_min_length(2);
    
    let new_text = "I love this movie! It's absolutely amazing!";
    let tokens = tokenizer.tokenize(new_text);
    println!("Original text: {}", new_text);
    println!("Tokenized: {:?}", tokens);
    
    let sparse_vec = data.vectorizer.transform(&tokens);
    let dense_vec = data.vectorizer.to_dense(&sparse_vec);
    
    let non_zero: Vec<(usize, f32)> = dense_vec
        .iter()
        .enumerate()
        .filter(|&(_, v)| *v != 0.0)
        .map(|(i, v)| (i, *v))
        .collect();
    println!("TF-IDF vector (non-zero features): {} features", non_zero.len());
    println!("Sample values: {:?}", &non_zero[..non_zero.len().min(5)]);

    println!("\n=== Ready for Neural Network Training ===");
    println!("You can now use:");
    println!("  - data.train_vectors: Training features [{}, {}]", 
             data.train_vectors.len(), 
             data.train_vectors.first().map(|v| v.len()).unwrap_or(0));
    println!("  - data.train_labels: Training labels [{}]", data.train_labels.len());
    println!("  - data.test_vectors: Test features [{}, {}]", 
             data.test_vectors.len(), 
             data.test_vectors.first().map(|v| v.len()).unwrap_or(0));
    println!("  - data.test_labels: Test labels [{}]", data.test_labels.len());
    println!("  - data.vectorizer: Transform new text at inference time");

    Ok(())
}
