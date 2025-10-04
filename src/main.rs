mod preprocessing;
mod NN;

use NN::initialize_network::{initialize_network, Init};
use NN::ops::{check_shapes, preview_params};
use NN::io::write_params_to_txt;

fn main() -> anyhow::Result<()> {
    // 1) Preprocess flow (old)
    let text = "This is a test sentence, with punctuation!";
    let tokens = preprocessing::tokenizer::tokenize(text);
    println!("Tokens: {:?}", tokens);

    // 2) NN initialization flow (new)
    let sizes = vec![300, 256, 256, 128, 32];
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
