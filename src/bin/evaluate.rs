use anyhow::{Context, Result};
use sentiment_analysis::preprocessing::serialization;
use sentiment_analysis::fuzzy::neuro_sugeno::{InferLogits, NeuroSugeno, softmax_temp};
use sentiment_analysis::fuzzy::params;

// Local adapter to satisfy orphan rules
#[derive(Clone)]
struct ModelAdapter {
	inner: sentiment_analysis::NN::io::TrainedModel,
}

impl InferLogits for ModelAdapter {
	fn forward_logits(&self, x: &[f32]) -> Vec<f32> {
		sentiment_analysis::NN::ops::forward_pass(x, &self.inner.weights, &self.inner.biases)
	}
}

fn load_trained_model(path: &str) -> Result<ModelAdapter> {
	let model = sentiment_analysis::NN::io::load_trained_model(path)
		.with_context(|| format!("Failed to load trained model from {}. Make sure you've trained and saved it.", path))?;
	Ok(ModelAdapter { inner: model })
}

fn build_hybrid<M: InferLogits>(model: M) -> NeuroSugeno<M> {
	let rules = params::default_rules();
	let provider = params::tuned_provider(model);
	NeuroSugeno { provider, rules }
}

fn argmax(v: &[f32]) -> usize { v.iter().enumerate().max_by(|a,b| a.1.partial_cmp(b.1).unwrap()).map(|(i,_)| i).unwrap_or(0) }
fn map_index_to_label(ix: usize) -> i32 { if ix==0 {-1} else if ix==1 {0} else {1} }

fn main() -> Result<()> {
	let data_dir = "data";
	let model_path = "model.bin";

	println!("=== Evaluate on test set ===");
	let loaded = serialization::load_all(data_dir).context("Failed to load preprocessed artifacts")?;
	println!("✓ Loaded data: test_size {} features {}", loaded.test_vectors.len(), loaded.vectorizer.vocab_size());

	let model = load_trained_model(model_path)?;
	println!("✓ Loaded trained model: {}", model_path);

	// Build NF using the same model
	let nf = build_hybrid(model.clone());

	let mut nn_correct = 0usize;
	let mut nf_correct = 0usize;
	let total = loaded.test_vectors.len();

	for (x, &y_true) in loaded.test_vectors.iter().zip(loaded.test_labels.iter()) {
		// NN-only
		let logits = model.forward_logits(x);
		let probs = softmax_temp(&logits, 1.0);
		let nn_ix = argmax(&probs);
		let nn_pred = map_index_to_label(nn_ix);
		if (y_true as i32) == nn_pred { nn_correct += 1; }

		// Neuro-fuzzy
	let nf_pred = nf.infer_label_with_threshold(x, params::NF_LABEL_THRESHOLD);
		if (y_true as i32) == nf_pred { nf_correct += 1; }
	}

	let nn_acc = nn_correct as f32 / total as f32;
	let nf_acc = nf_correct as f32 / total as f32;

	println!("\nResults on test set (N = {}):", total);
	println!("  NN-only accuracy:       {:.2}% ({} / {})", nn_acc*100.0, nn_correct, total);
	println!("  Neuro-Fuzzy accuracy:   {:.2}% ({} / {})", nf_acc*100.0, nf_correct, total);

	Ok(())
}

