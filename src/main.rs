use anyhow::Result;
use bincode::{DefaultOptions, Options};
use std::fs::File;
use std::io::BufReader;

// Library modules
use sentiment_analysis::preprocessing::serialization;
use sentiment_analysis::NN::ops::forward_pass;

fn load_model_bin(path: &str) -> Result<(Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>)> {
    let f = BufReader::new(File::open(path)?);
    let (weights, biases): (Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>) =
        DefaultOptions::new().deserialize_from(f)?;
    Ok((weights, biases))
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&z| (z - max_logit).exp()).collect();
    let sum: f32 = exps.iter().sum();
    if sum == 0.0 {
        vec![1.0 / logits.len() as f32; logits.len()]
    } else {
        exps.into_iter().map(|e| e / sum).collect()
    }
}

fn main() -> Result<()> {
    // 1) Load saved model
    let (weights, biases) = load_model_bin("model.bin")?;

    // 2) Load the vectorized dataset (or your own vectorized inputs)
    let data = serialization::load_all("data")?;

    // 3) Exactly one scalar per row
    let idx_to_label = [-1.0, 0.0, 1.0]; // fixed order [-1, 0, +1]
    for i in 0..3 {
        let x = &data.test_vectors[i];
        let logits = forward_pass(x, &weights, &biases);
        let probs = softmax(&logits);
        let (argmax, _) = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap();
        let y_hat = idx_to_label[argmax]; // single output in {-1, 0, +1}

        // Print only the scalar (uncomment one of the lines below depending on what you want)
        println!("{}", y_hat);                                   // strict single value
        // println!("row={} pred={} p={:.3}", i, y_hat, probs[argmax]); // scalar + confidence
    }

    Ok(())
}
