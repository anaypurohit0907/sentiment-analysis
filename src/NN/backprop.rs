use crate::NN::ops::{mat_vec_mul, vec_add, relu};


pub fn train_nn(
    weights: &mut Vec<Vec<Vec<f32>>>,
    biases: &mut Vec<Vec<f32>>,
    training_data: &[(Vec<f32>, Vec<f32>)],
    epochs: usize,
    learning_rate: f32,
) {
    for epoch in 0..epochs {
        for (input, target) in training_data {
            // Forward pass with caches
            let (activations, pre_activations) = forward_with_cache(input, weights, biases);

            // Backward pass yielding gradients dW, dB
            let (d_weights, d_biases) = backward_pass(
                &activations,
                &pre_activations,
                weights,
                biases,
                target,
            );

            // Update weights/biases with gradients
            update_params(weights, biases, &d_weights, &d_biases, learning_rate);
        }
    }
}

/// Update weights and biases in place using computed gradients and learning rate.
///
/// weights and biases must have the same shapes as d_weights and d_biases respectively.
pub fn update_params(
    weights: &mut Vec<Vec<Vec<f32>>>,
    biases: &mut Vec<Vec<f32>>,
    d_weights: &Vec<Vec<Vec<f32>>>,
    d_biases: &Vec<Vec<f32>>,
    learning_rate: f32,
) {
    for l in 0..weights.len() {
        for i in 0..weights[l].len() {
            for j in 0..weights[l][i].len() {
                weights[l][i][j] -= learning_rate * d_weights[l][i][j];
            }
            biases[l][i] -= learning_rate * d_biases[l][i];
        }
    }
}
fn relu_prime(z: &[f32]) -> Vec<f32> {
    z.iter().map(|&x| if x > 0.0 { 1.0 } else { 0.0 }).collect()
}

fn backward_layer(
    delta_next: &[f32],
    weights_next: &[Vec<f32>],
    z: &[f32],
) -> Vec<f32> {
    let relu_grad = relu_prime(z);
    let mut delta = vec![0.0; z.len()];
    for i in 0..z.len() {
        let mut sum = 0.0;
        for j in 0..delta_next.len() {
            sum += weights_next[j][i] * delta_next[j];
        }
        delta[i] = sum * relu_grad[i];
    }
    delta
}

pub fn forward_with_cache(
    input: &[f32],
    weights: &Vec<Vec<Vec<f32>>>,
    biases: &Vec<Vec<f32>>,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut activations: Vec<Vec<f32>> = Vec::with_capacity(weights.len() + 1);
    let mut pre_activations: Vec<Vec<f32>> = Vec::with_capacity(weights.len());

    // Input layer activation
    activations.push(input.to_vec());

    let mut current_activation = input.to_vec();

    for (w, b) in weights.iter().zip(biases.iter()) {
        let z = mat_vec_mul(w, &current_activation); // z = W * a + b
        let z_biased = vec_add(&z, b);
        pre_activations.push(z_biased.clone());

        // ReLU activation
        let a = relu(&z_biased);
        activations.push(a.clone());
        current_activation = a;
    }

    (activations, pre_activations)
}


pub fn backward_pass(
    activations: &Vec<Vec<f32>>,      // length L+1 (includes input)
    pre_activations: &Vec<Vec<f32>>,  // length L (no input)
    weights: &Vec<Vec<Vec<f32>>>,
    biases: &Vec<Vec<f32>>,
    target: &Vec<f32>,
) -> (Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>) {
    let num_layers = weights.len();
    let mut d_weights: Vec<Vec<Vec<f32>>> = vec![vec![vec![0.0; weights[0][0].len()]; weights[0].len()]; num_layers];
    let mut d_biases: Vec<Vec<f32>> = vec![vec![0.0; biases[0].len()]; num_layers];

    // Start with output layer delta (MSE loss derivative * ReLU derivative if applicable)
    // Assuming MSE loss: delta = (activation - target) * relu'(z)
    let mut delta: Vec<f32> = activations[num_layers]
        .iter()
        .zip(target.iter())
        .map(|(a, t)| a - t)
        .collect();

    // Multiply by ReLU prime on output layer pre-activation
    {
        let relu_deriv = pre_activations[num_layers-1]
            .iter()
            .map(|&z| if z > 0.0 { 1.0 } else { 0.0 });
        for (d, rd) in delta.iter_mut().zip(relu_deriv) {
            *d *= rd;
        }
    }

    // Backpropagate through layers, from last to first
    for l in (0..num_layers).rev() {
        // Compute gradient for biases = delta
        d_biases[l] = delta.clone();

        // Compute gradient for weights = delta outer product with activation of previous layer
        // weights[l] shape: (neurons in layer l) x (neurons in layer l-1)
        let a_prev = &activations[l];
        let rows = delta.len();
        let cols = a_prev.len();

        d_weights[l] = vec![vec![0.0; cols]; rows];
        for i in 0..rows {
            for j in 0..cols {
                d_weights[l][i][j] = delta[i] * a_prev[j];
            }
        }

        // Compute delta for next layer (previous in index/given order)
        if l > 0 {
            let mut new_delta = vec![0.0; weights[l][0].len()];
            // delta_new_j = sum_i w_ij * delta_i * relu'(z_j)
            for j in 0..new_delta.len() {
                let mut sum = 0.0;
                for i in 0..delta.len() {
                    sum += weights[l][i][j] * delta[i];
                }
                new_delta[j] = sum;
            }

            // Multiply by relu' of pre_activation in layer l-1
            for j in 0..new_delta.len() {
                if pre_activations[l-1][j] <= 0.0 {
                    new_delta[j] = 0.0;
                }
            }
            delta = new_delta;
        }
    }

    (d_weights, d_biases)
}