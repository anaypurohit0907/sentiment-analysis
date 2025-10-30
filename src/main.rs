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

fn main() -> Result<()> {
    // 1) Load saved model
    let (weights, biases) = load_model_bin("model.bin")?;

    // 2) Load the same vectorized dataset bundle to get feature width and a few samples
    let data = serialization::load_all("data")?; // provides train/test vectors and labels + input_size

    // 3) Pick a few test samples and run forward
    for i in 0..3 {
        let x = &data.test_vectors[i]; // already TF-IDF vector of length input_size
        let logits = forward_pass(x, &weights, &biases);
        // argmax
        let (argmax, _) = logits.iter()
                                .enumerate()
                                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                                .unwrap();
        let true_y = data.test_labels[i];
        let pred_label = match argmax {
            0 => -1.0,
            1 => 0.0,
            _ => 1.0,
        };
        println!("sample {} true={} pred_class={} logits={:?}", i, true_y, pred_label, logits);
    }

    Ok(())
}
