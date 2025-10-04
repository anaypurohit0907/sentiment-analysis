use rand::{rngs::StdRng, Rng, SeedableRng};

/// Initialize a feedforward network from a list of layer sizes.
/// sizes = [in, h1, h2, ..., out]
/// Returns (weights, biases) where:
/// - weights[l] is of shape [sizes[l+1]][sizes[l]]
/// - biases[l] is of length sizes[l+1]
pub fn initialize_network(
    sizes: &[usize],
    init: Init,
    seed: u64,
) -> (Vec<Vec<Vec<f32>>>, Vec<Vec<f32>>) {
    assert!(sizes.len() >= 2, "Need at least input and output layer");

    let mut rng = StdRng::seed_from_u64(seed);

    let mut weights = Vec::with_capacity(sizes.len() - 1);
    let mut biases = Vec::with_capacity(sizes.len() - 1);

    for l in 0..sizes.len() - 1 {
        let fan_in = sizes[l] as f32;
        let fan_out = sizes[l + 1] as f32;

        let (low, high) = match init {
            Init::XavierUniform => {
                // ±sqrt(6/(fan_in + fan_out))
                let a = (6.0f32 / (fan_in + fan_out)).sqrt();
                (-a, a)
            }
            Init::HeUniform => {
                // ±sqrt(6/fan_in)
                let a = (6.0f32 / fan_in.max(1.0)).sqrt();
                (-a, a)
            }
            Init::Normal { std } => (-std, std),
        };

        // Weights: [fan_out][fan_in]
        let mut w_l = Vec::with_capacity(sizes[l + 1]);
        for _ in 0..sizes[l + 1] {
            let mut row = Vec::with_capacity(sizes[l]);
            for _ in 0..sizes[l] {
                row.push(rng.random_range(low..high));
            }
            w_l.push(row);
        }
        weights.push(w_l);

        // Biases: [fan_out]
        let mut b_l = Vec::with_capacity(sizes[l + 1]);
        for _ in 0..sizes[l + 1] {
            b_l.push(rng.random_range(low..high));
        }
        biases.push(b_l);
    }

    (weights, biases)
}

#[derive(Clone, Copy)]
pub enum Init {
    XavierUniform,
    HeUniform,
    Normal { std: f32 },
}
