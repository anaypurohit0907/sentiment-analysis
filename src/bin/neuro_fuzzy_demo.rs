use anyhow::{Context, Result};
use sentiment_analysis::fuzzy::neuro_sugeno::{softmax_temp, InferLogits, NeuroSugeno};
use sentiment_analysis::fuzzy::params;
use sentiment_analysis::preprocessing::{serialization, tokenizer::Tokenizer};
use sentiment_analysis::tfidf::vectorizer::TfIdfVectorizer;
use std::io::{self, Write};

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

fn load_vectorizer_only() -> Result<TfIdfVectorizer> {
	let data_dir = "data";
	println!("✓ Loading artifacts from {}", data_dir);
	let loaded = serialization::load_all(data_dir).context("Failed to load preprocessed artifacts")?;
	println!("✓ Loaded vectorizer, features: {}", loaded.vectorizer.vocab_size());
	Ok(loaded.vectorizer)
}

fn load_trained_model() -> Result<ModelAdapter> {
	let path = "model.bin";
	let model = sentiment_analysis::NN::io::load_trained_model(path)
		.with_context(|| format!("Failed to load trained model from {}. Make sure you've trained and saved it.", path))?;
	println!("✓ Loaded trained model from {}", path);
	Ok(ModelAdapter { inner: model })
}

fn build_hybrid<M: InferLogits>(model: M) -> NeuroSugeno<M> {
	let rules = params::default_rules();
	let provider = params::tuned_provider(model);
	NeuroSugeno { provider, rules }
}

fn argmax(v: &[f32]) -> usize {
	let mut best = 0usize;
	let mut best_val = f32::NEG_INFINITY;
	for (i, &x) in v.iter().enumerate() {
		if x > best_val { best = i; best_val = x; }
	}
	best
}

fn map_index_to_label(ix: usize) -> i32 { match ix { 0 => -1, 1 => 0, _ => 1 } }
fn label_text(label: i32) -> &'static str { match label { -1 => "negative", 0 => "neutral", _ => "positive" } }

fn main() -> Result<()> {
	println!("=== Neuro-Fuzzy Sentiment Demo ===");
	let tokenizer = Tokenizer::new().with_stopwords(true).with_min_length(2);
	let vec = load_vectorizer_only()?;
	let model = load_trained_model()?;
	let nf = build_hybrid(model.clone());

	let membership_names = [
		"mu_neg", "mu_neu", "mu_pos",
		"conf_high(margin)", "ambig_high(H)", "pos_over_neg_high(z_gap)"
	];

	println!("Type a sentence and press Enter (empty line to quit).");
	let mut input = String::new();
	loop {
		print!("\n> ");
		io::stdout().flush().ok();
		input.clear();
		if io::stdin().read_line(&mut input)? == 0 { break; }
		let text = input.trim();
		if text.is_empty() { break; }

		// Step 1: Tokenize
		let tokens = tokenizer.tokenize(text);
		println!("Tokens: {:?}", tokens);

		// Step 2: TF-IDF
		let sparse = vec.transform(&tokens);
		let nnz = sparse.iter().filter(|(_, v)| **v != 0.0).count();
		println!("TF-IDF: non-zeros = {}, vocab = {}", nnz, vec.vocab_size());

		// Step 3: Dense vector
		let dense = vec.to_dense(&sparse);
		println!("Dense length: {}", dense.len());

		// Step 4: NN logits and probabilities
		let logits = model.forward_logits(&dense);
		let probs = softmax_temp(&logits, nf.provider.temperature);
		println!("Logits (raw): {:?}", logits);
		println!("Probs (softmax T={}): {:?}", nf.provider.temperature, probs);

		// Diagnostics
		let mut sorted = logits.clone();
		sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
		let margin = if sorted.len() >= 2 { sorted[0] - sorted[1] } else { 0.0 };
		let entropy: f32 = -probs
			.iter()
			.map(|p| if *p > 0.0 { p * p.ln() } else { 0.0 })
			.sum::<f32>();
		let pos_minus_neg = logits.get(2).unwrap_or(&0.0) - logits.get(0).unwrap_or(&0.0);
		println!("Margin(top1-top2): {:.4}, Entropy: {:.4}, z_pos - z_neg: {:.4}", margin, entropy, pos_minus_neg);

		// Step 5: Neuro-fuzzy memberships (6-dim)
		let mu = nf.provider.memberships(&dense);
		for (i, m) in mu.iter().enumerate() {
			println!("  {} = {:.4}", membership_names.get(i).unwrap_or(&"m"), m);
		}

		// Step 6: Rule firing strengths and contributions
		let strengths = nf.rule_strengths(&dense);
		println!("Rule strengths (w) and contributions (w*c):");
		for (ix, w, contrib) in strengths.iter() {
			println!("  R{:02}: w={:.4}, c={:+.2}, w*c={:+.4}", ix, w, nf.rules[*ix].consequent, contrib);
		}

		// Step 7: Neuro-fuzzy score, label text and confidence
		let score = nf.infer_score(&dense);
	let nf_label = nf.infer_label_with_threshold(&dense, params::NF_LABEL_THRESHOLD);
		let nf_conf = match nf_label {
			1 => score.max(0.0),           // confidence grows with positive score
			-1 => (-score).max(0.0),       // confidence grows with negative score
			0 => 1.0 - score.abs(),        // neutrality highest near 0
			_ => 0.0,
		};
		println!(
			"Neuro-Fuzzy: {} (confidence {:.1}%) [score {:+.4}]",
			label_text(nf_label),
			nf_conf.clamp(0.0, 1.0) * 100.0,
			score
		);

		// Step 8: NN-only label (argmax prob) with confidence
		let nn_ix = argmax(&probs);
		let nn_label = map_index_to_label(nn_ix);
		let nn_conf = probs.get(nn_ix).copied().unwrap_or(0.0);
		println!(
			"NN-only: {} (confidence {:.1}%)",
			label_text(nn_label),
			(nn_conf * 100.0)
		);
	}

	println!("\nBye.");
	Ok(())
}


