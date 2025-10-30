// src/NN/train.rs
use anyhow::Result;
use crate::NN::initialize_network::{initialize_network, Init};
use crate::NN::ops::{check_shapes, forward_with_cache, update_params};
use crate::NN::backprop::backward_pass;
use crate::NN::io::write_params_to_txt;

// Use the shared loader exposed by your preprocessing module.
use crate::preprocessing::serialization;

pub fn train_model(data_dir: &str, mut sizes: Vec<usize>, epochs: usize, lr: f32) -> Result<()> {
    // 1) Load vectorized dataset bundle (train/test matrices, labels, vectorizer)
    let data = serialization::load_all(data_dir)?; // must return {train_vectors, test_vectors, train_labels, test_labels, vectorizer}
    let input_size = data.vectorizer.vocab_size();
    if sizes.first().copied() != Some(input_size) {
        sizes[0] = input_size; // Align NN input width with TF-IDF feature dimension
    }

    // 2) Initialize and sanity-check shapes
    let (mut weights, mut biases) = initialize_network(&sizes, Init::HeUniform, 42);
    check_shapes(&sizes, &weights, &biases).map_err(|e| anyhow::anyhow!(e))?;

    // 3) One-hot mapping for {-1, 0, +1} → 3 classes
    fn to_one_hot(y: f32) -> [f32; 3] {
        if y < 0.0 { [1.0, 0.0, 0.0] } else if y > 0.0 { [0.0, 0.0, 1.0] } else { [0.0, 1.0, 0.0] }
    }

    // 4) Epoch loop (SGD per row; add batching later if needed)
    for epoch in 0..epochs {
        let mut epoch_loss = 0.0f32;

        for (x, &y) in data.train_vectors.iter().zip(data.train_labels.iter()) {
            let target: Vec<f32> = to_one_hot(y).to_vec();

            let (acts, preacts) = forward_with_cache(x, &weights, &biases);
            let pred = acts.last().unwrap();

            // MSE per sample
            let loss: f32 = pred.iter()
                                .zip(target.iter())
                                .map(|(p, t)| (p - t) * (p - t))
                                .sum::<f32>() / (pred.len() as f32);
            epoch_loss += loss;

            let (dW, dB) = backward_pass(&acts, &preacts, &weights, &biases, &target);
            update_params(&mut weights, &mut biases, &dW, &dB, lr);
        }

        epoch_loss /= data.train_vectors.len() as f32;
        println!("epoch {} loss {:.6}", epoch, epoch_loss);

        // Quick validation accuracy
        let mut correct = 0usize;
        for (x, &y) in data.test_vectors.iter().zip(data.test_labels.iter()) {
            let (acts, _) = forward_with_cache(x, &weights, &biases);
            let pred = acts.last().unwrap();
            let argmax = pred.iter()
                             .enumerate()
                             .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                             .unwrap()
                             .0;
            let y_idx = if y < 0.0 { 0 } else if y > 0.0 { 2 } else { 1 };
            if argmax == y_idx { correct += 1; }
        }
        let acc = correct as f32 / data.test_vectors.len() as f32;
        println!("val acc {:.4}", acc);
    }

    // 5) Save model
    write_params_to_txt("weights_final.txt", "biases_final.txt", &weights, &biases)?;
    save_model_bin("model.bin", &weights, &biases)?;
    Ok(())
}

fn save_model_bin(path: &str, weights: &Vec<Vec<Vec<f32>>>, biases: &Vec<Vec<f32>>) -> Result<()> {
    use std::fs::File;
    use std::io::BufWriter;
    use bincode::{DefaultOptions, Options};
    let f = BufWriter::new(File::create(path)?);
    DefaultOptions::new().serialize_into(f, &(weights, biases))?;
    Ok(())
}
