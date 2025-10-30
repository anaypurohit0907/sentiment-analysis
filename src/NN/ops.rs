// src/nn/ops.rs

pub fn forward_pass(
    input: &[f32],
    weights: &Vec<Vec<Vec<f32>>>,
    biases: &Vec<Vec<f32>>,
) -> Vec<f32> {
    let mut activation = input.to_vec();

    for (i, (w, b)) in weights.iter().zip(biases.iter()).enumerate() {
        let z = mat_vec_mul(w, &activation);
        let z_bias = vec_add(&z, b);

        activation = if i < weights.len() - 1 {
            relu(&z_bias)
        } else {
            z_bias
        };
    }
    activation
}

pub fn mat_vec_mul(matrix: &[Vec<f32>], vector: &[f32]) -> Vec<f32> {
    matrix.iter().map(|row| dot_product(row, vector)).collect()
}

pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

pub fn vec_add(a: &[f32], b: &[f32]) -> Vec<f32> {
    a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect()
}

pub fn relu(v: &[f32]) -> Vec<f32> {
    v.iter().map(|&x| x.max(0.0)).collect()
}

pub fn check_shapes(
    sizes: &[usize],
    weights: &Vec<Vec<Vec<f32>>>,
    biases: &Vec<Vec<f32>>,
) -> Result<(), String> {
    if weights.len() != sizes.len() - 1 {
        return Err(format!(
            "weights layers mismatch: {} vs {}",
            weights.len(),
            sizes.len() - 1
        ));
    }
    if biases.len() != sizes.len() - 1 {
        return Err(format!(
            "biases layers mismatch: {} vs {}",
            biases.len(),
            sizes.len() - 1
        ));
    }
    for l in 0..weights.len() {
        let rows = weights[l].len();
        let cols = if rows > 0 { weights[l][0].len() } else { 0 };
        if rows != sizes[l + 1] || cols != sizes[l] {
            return Err(format!(
                "Layer {l} shape mismatch: got {rows}x{cols}, expected {}x{}",
                sizes[l + 1], sizes[l]
            ));
        }
        if biases[l].len() != sizes[l + 1] {
            return Err(format!(
                "Bias length mismatch at layer {l}: got {}, expected {}",
                biases[l].len(),
                sizes[l + 1]
            ));
        }
    }
    Ok(())
}

pub fn preview_params(weights: &Vec<Vec<Vec<f32>>>, biases: &Vec<Vec<f32>>, k: usize) {
    for l in 0..weights.len() {
        let rows = weights[l].len();
        let cols = if rows > 0 { weights[l][0].len() } else { 0 };
        println!("Layer {l}: W[{rows}x{cols}], b[{}]", biases[l].len());

        for r in 0..rows.min(k) {
            let row = &weights[l][r];
            let head = &row[..row.len().min(k)];
            println!("  W[{r}][0..{k}]: {:?}", head);
        }
        let bhead = &biases[l][..biases[l].len().min(k)];
        println!("  b[0..{k}]: {:?}", bhead);
    }
}
pub fn forward_with_cache(
    input: &[f32],
    weights: &Vec<Vec<Vec<f32>>>,
    biases: &Vec<Vec<f32>>,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut activations: Vec<Vec<f32>> = Vec::with_capacity(weights.len() + 1);
    let mut pre_activations: Vec<Vec<f32>> = Vec::with_capacity(weights.len());

    // a^(0) = input
    activations.push(input.to_vec());
    let mut a = input.to_vec();

    for (l, (w, b)) in weights.iter().zip(biases.iter()).enumerate() {
        // z = W a + b
        let z = mat_vec_mul(w, &a);
        let z_bias = vec_add(&z, b);
        pre_activations.push(z_bias.clone());

        // activation: ReLU for hidden, identity for last layer
        a = if l < weights.len() - 1 {
            relu(&z_bias)
        } else {
            z_bias
        };
        activations.push(a.clone());
    }

    (activations, pre_activations)
}
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
