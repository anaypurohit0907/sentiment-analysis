use std::fs::File;
use std::io::{BufWriter, Write};
use anyhow::Result;

mod NN;

use NN::initialize_network::{initialize_network, Init};
use NN::ops::{check_shapes, preview_params};
use NN::backprop::{forward_with_cache, backward_pass, update_params};
use NN::io::write_params_to_txt;

fn save_weights_biases_to_files(
    w_path: &str, b_path: &str,
    weights: &Vec<Vec<Vec<f32>>>, biases: &Vec<Vec<f32>>
) -> Result<()> {
    write_params_to_txt(w_path, b_path, weights, biases)?;
    Ok(())
}

/// Save output vector to a text file
fn save_output_to_txt(path: &str, output: &[f32]) -> Result<()> {
    let mut file = BufWriter::new(File::create(path)?);
    for v in output {
        writeln!(file, "{:.6}", v)?;
    }
    Ok(())
}

fn main() -> Result<()> {
    // 1. Define a small network architecture
    let sizes = vec![10, 6, 3];

    // 2. Initialize network weights and biases
    let (mut weights, mut biases) = initialize_network(&sizes, Init::HeUniform, 1234);

    // 3. Check shape validity and preview
    check_shapes(&sizes, &weights, &biases).map_err(|e| anyhow::anyhow!(e))?;
    preview_params(&weights, &biases, 3);

    // 4. Save initial weights/biases
    save_weights_biases_to_files("init_weights.txt", "init_biases.txt", &weights, &biases)?;

    // 5. Define a simple input vector (e.g., 10 features)
    let input: Vec<f32> = (0..sizes[0]).map(|i| i as f32 / 10.0).collect();

    // 6. Forward pass before training
    let output_before = forward_with_cache(&input, &weights, &biases);
    println!("Output before training: {:?}", &output_before.0.last().unwrap());

    // 7. Train for 1 epoch on dummy data (simulate a small dataset)
    let targets = vec![
        vec![0.0, 1.0, 0.0], // example target
        vec![1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let train_data: Vec<(Vec<f32>, Vec<f32>)> = targets.iter().map(|t| (input.clone(), t.clone())).collect();

    // Call training
    let epochs = 1;
    let learning_rate = 0.01;
    for _ in 0..epochs {
        for (x, y) in &train_data {
            // Forward with cache
            let (activations, preactivations) = forward_with_cache(&x, &weights, &biases);
            // Backprop to get gradients
            let (dW, dB) = backward_pass(&activations, &preactivations, &weights, &biases, y);
            // Update weights/biases
            for l in 0..weights.len() {
                for i in 0..weights[l].len() {
                    for j in 0..weights[l][i].len() {
                        weights[l][i][j] -= learning_rate * dW[l][i][j];
                    }
                    biases[l][i] -= learning_rate * dB[l][i];
                }
            }
        }
    }

    // 8. Forward pass after training
    let output_after = forward_with_cache(&input, &weights, &biases);
    println!("Output after training: {:?}", &output_after.0.last().unwrap());

    // 9. Save final weights/biases
    save_weights_biases_to_files("final_weights.txt", "final_biases.txt", &weights, &biases)?;

    // 10. Save output to file
    save_output_to_txt("output_before.txt", &output_before.0.last().unwrap())?;
    save_output_to_txt("output_after.txt", &output_after.0.last().unwrap())?;

    Ok(())
}
