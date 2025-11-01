# Sentiment Analysis (Rust)

A modular Rust project for preparing a text sentiment dataset with robust tokenization and TF‑IDF feature extraction, saving portable artifacts for training and inference, and demonstrating how to load and use those artifacts. It also includes a hybrid Neuro‑Fuzzy layer on top of a neural network for interpretable post‑processing and a demo/evaluator to showcase it. The code emphasizes reproducibility, interpretability, and simple integration points for modeling.

This README documents the full pipeline, how to run it, the modules, data shapes, and practical tips/troubleshooting.


## Key features

- Clean, configurable tokenizer (stopwords removal, min token length).
- TF‑IDF vectorizer with n‑gram support, min_df/max_df filtering, and optional max_features cap.
- Deterministic vocabulary ordering for reproducible features.
- Binary serialization of vectorizer and matrices (bincode) and human‑readable metadata (JSON).
- A ready‑to‑run data preparation CLI that loads CSV → tokenizes → TF‑IDF → train/test split → saves `data/`.
- A loader example CLI that shows how to consume the saved artifacts and transform new text.


## Project structure

- `src/bin/prepare_data.rs` — End‑to‑end preprocessing pipeline into TF‑IDF and serialized artifacts.
- `src/bin/load_data_example.rs` — Loads the saved artifacts and demonstrates transforming new text.
- `src/bin/neuro_fuzzy_demo.rs` — Interactive REPL demo that runs the NN and Neuro‑Fuzzy hybrid, prints memberships, rule strengths, labels, and confidence.
- `src/bin/evaluate.rs` — Reports NN‑only vs Neuro‑Fuzzy accuracy on the test split from `data/`.
- `src/preprocessing/tokenizer.rs` — Text cleaning and tokenization utilities.
- `src/tfidf/vectorizer.rs` — Configurable TF‑IDF with n‑grams, term filtering, dense/sparse conversions.
- `src/preprocessing/serialization.rs` — Save/load vectorizer, matrices, labels, and metadata.
- `src/NN/…` — Minimal neural network utilities and a `train_model` function (example), for those who want to experiment further.
- `src/utils.rs` — Train/test split helpers, including a stratified variant.
- `src/fuzzy/neuro_sugeno.rs` — Neuro‑Fuzzy hybrid (NN‑driven memberships + Sugeno rules).
- `src/fuzzy/params.rs` — Centralized tuned parameters and default rule base shared by demo/evaluator.
- `data/` — Output folder for serialized artifacts created by the pipeline.


## Requirements

- Rust toolchain (stable). Install via https://rustup.rs if needed.


## Dataset format

- CSV with headers and two columns: `text,label`
- Labels are expected to be numeric in { -1, 0, +1 } (negative, neutral, positive).
- Default path expected by the pipeline: `src/result.csv`

Example (first few lines):

```csv
text,label
"I love this movie",1
"This is okay",0
"Absolutely terrible",-1
```


## How the pipeline works

Inputs → Preprocess → Split → Vectorize → Serialize → Validate → Done

1) Load CSV
- `NN::io::load_sentiment_csv(path)` reads a headered CSV and skips malformed/empty rows.
- Prints class distribution for quick sanity checks.

2) Tokenize
- `Tokenizer::new().with_stopwords(true).with_min_length(2)`
- `tokenize(&str)` uses: clean punctuation → lowercase → filter by length → optional stopwords removal.

3) Split
- `utils::train_test_split` shuffles deterministically (seed) and splits by ratio.
- Optionally use `utils::stratified_train_test_split` to preserve class proportions.

4) Fit TF‑IDF (train only)
- `TfIdfVectorizer::new()` builder options:
  - `.max_features(usize)` — keep top terms by document frequency cap (optional)
  - `.min_df(usize)` — ignore terms appearing in fewer docs than this
  - `.max_df(f32)` — ignore overly frequent terms (as fraction of docs)
  - `.ngram_range(min, max)` — extract n‑grams; e.g. `(1,2)` includes unigrams + bigrams
- Fit computes smoothed IDF: idf(t) = ln((1+N) / (1+df(t))) + 1.
- Vocabulary is stored in deterministic index order (sorted lexicographically after filtering) for reproducibility.

5) Transform train and test
- For each document, `transform(tokens)` builds a sparse map `HashMap<usize, f32>` of TF*IDF weights.
- `to_dense(&sparse)` converts to `Vec<f32>` using the vocabulary size.

6) Serialize
- `preprocessing::serialization::save_all("data", ...)` writes:
  - `data/tfidf_vectorizer.bin` — the vectorizer (bincode)
  - `data/tfidf_matrices.bin` — train/test matrices (bincode)
  - `data/y_train.bin`, `data/y_test.bin` — labels (bincode)
  - `data/metadata.json` — human‑readable metadata (JSON)

7) Validate
- Immediately reads everything back via `load_all("data")` and asserts shapes match.


## Run it

1) Prepare data

```bash
cargo run --bin prepare_data
```

What it does:
- Reads `src/result.csv`.
- Tokenizes, splits, fits TF‑IDF, transforms train/test.
- Saves artifacts in `data/`.
- Prints metadata and a small feature preview.

2) Load artifacts and transform new text (example)

```bash
cargo run --bin load_data_example
```

What it does:
- Loads all artifacts from `data/`.
- Prints dataset summary and shows how to transform a custom sentence into TF‑IDF.

3) Run the Neuro‑Fuzzy interactive demo

```bash
cargo run --bin neuro_fuzzy_demo
```

What it does:
- Loads `data/` artifacts and a trained model from `model.bin`.
- For each input line, computes logits → softmax (with temperature) → Neuro‑Fuzzy memberships and rule strengths → hybrid score and label.
- Prints both the Neuro‑Fuzzy label (with a confidence heuristic) and the NN‑only argmax baseline.

4) Evaluate accuracy on the test set

```bash
cargo run --bin evaluate
```

What it does:
- Loads `data/` test vectors and labels and `model.bin`.
- Reports NN‑only accuracy and Neuro‑Fuzzy accuracy (using the shared tuned parameters and rules).


## Using the artifacts in your own code

Load everything at once:

```rust
use sentiment_analysis::preprocessing::serialization;
let data = serialization::load_all("data")?;

// Feature dimensions and matrices
let d = data.vectorizer.vocab_size();
let x_train: Vec<Vec<f32>> = data.train_vectors;
let y_train: Vec<f32>       = data.train_labels;

// Transform new text at inference time
use sentiment_analysis::preprocessing::tokenizer::Tokenizer;
let tok = Tokenizer::new().with_stopwords(true).with_min_length(2);
let tokens = tok.tokenize("An unexpectedly enjoyable film");
let sparse = data.vectorizer.transform(&tokens);
let dense  = data.vectorizer.to_dense(&sparse);
assert_eq!(dense.len(), d);
```

### Using the Neuro‑Fuzzy hybrid

```rust
use sentiment_analysis::fuzzy::params;
use sentiment_analysis::fuzzy::neuro_sugeno::{InferLogits, NeuroSugeno};

// Your model type must implement InferLogits (see binaries for a simple adapter)
let model = /* load NN model, wrap in adapter */;
let nf: NeuroSugeno<_> = NeuroSugeno { provider: params::tuned_provider(model), rules: params::default_rules() };

let score = nf.infer_score(&dense);
let label = nf.infer_label_with_threshold(&dense, params::NF_LABEL_THRESHOLD); // -1, 0, +1
```


## Module reference (high level)

- `preprocessing/tokenizer.rs`
  - `clean_text(&str) -> String` — strips punctuation to spaces, normalizes whitespace.
  - `tokenize_with_preprocessing(text, remove_stopwords)` — one‑shot convenience.
  - `Tokenizer { remove_stopwords, min_token_length }` with builder methods and `tokenize(&str)`.

- `tfidf/vectorizer.rs`
  - `TfIdfVectorizer::new() -> Builder`
  - `fit(&mut self, corpus: &[Vec<String>])` — build vocab and IDF from tokenized docs.
  - `transform(&self, tokens: &[String]) -> HashMap<usize, f32>` — sparse TF*IDF.
  - `to_dense(&self, &HashMap<usize, f32>) -> Vec<f32>` — dense vector.
  - `get_feature_names(&self) -> Vec<String>` — index → term mapping.
  - `vocab_size(&self) -> usize` — feature dimension.

- `preprocessing/serialization.rs`
  - `save_all`, `load_all` orchestrate binary+JSON artifacts.
  - `DatasetMetadata` documents key preprocessing settings and sizes.

- `utils.rs`
  - `train_test_split` — simple shuffled split.
  - `stratified_train_test_split` — per‑class proportional split.

- `NN/` (optional experimentation)
  - Contains a basic MLP scaffold and an example `train_model` function showing how to consume the TF‑IDF features. This is intentionally minimal and not wired to a CLI by default, so you can integrate with your own training loop or framework.


## Data shapes and contracts

- Labels: stored as `f32` with values in { -1.0, 0.0, +1.0 }.
- Features: `Vec<f32>` length = `vectorizer.vocab_size()`.
- Train/Test sizes are saved in `metadata.json` and should match matrix row counts.
- Vocabulary indices are stable across save/load — crucial for consistent inference.


## Customization recipes

- Use bigrams only:
  - `.ngram_range(2, 2)`
- Larger vocab:
  - `.max_features(20000)`
- Filter rarer terms:
  - `.min_df(5)`
- Filter extremely frequent terms (e.g., near stopwords):
  - `.max_df(0.85)`
- Skip stopword removal:
  - `Tokenizer::new().with_stopwords(false)`
- Keep short tokens:
  - `.with_min_length(1)`
- Stratified split:
  - Replace `train_test_split` with `stratified_train_test_split` (requires integer labels).


## Troubleshooting

- “Failed to open CSV file”
  - Ensure the path matches your file (default is `src/result.csv`) and that it has headers.
- “Failed to parse label”
  - Check labels are numeric in { -1, 0, +1 } with no stray spaces.
- Empty or tiny vocab
  - Try reducing `min_df`, increasing `max_df`, and/or adjusting `ngram_range`.
- Out‑of‑vocabulary at inference
  - New tokens simply map to zeros. This is expected with TF‑IDF; consider retraining the vectorizer when domain shifts.
- Memory/Speed
  - Reduce `max_features`, use unigrams only, and/or enable release mode (`--release`).


## Example: training a simple MLP (optional)

There is a minimal example in `src/NN/train.rs` that demonstrates integrating with the saved artifacts. You can call it from your own binary like:

```rust
// src/bin/train_example.rs (you can create this file)
use anyhow::Result;
use sentiment_analysis::NN::train::train_model;

fn main() -> Result<()> {
    // Sizes: [input_dim (auto‑aligned), hidden1, hidden2, output=3]
    let sizes = vec![0, 128, 64, 3];
    train_model("data", sizes, /*epochs=*/10, /*lr=*/0.01)?;
    Ok(())
}
```

Run it:

```bash
cargo run --bin train_example
```

Notes:
- This is intentionally minimal (MSE, simple update). For production use, prefer a mature framework or extend this to include batching, better losses, and optimizers.


## Design choices and rationale

- Deterministic vocabulary: sorting post‑filtering ensures consistent feature indices across runs and machines.
- Smoothed IDF: avoids division by zero and stabilizes weights on small corpora.
- Separate serialization for vectorizer, matrices, and labels: allows flexible loading and updates.
- Builder pattern for TF‑IDF: encourages explicit choice of preprocessing settings.


## Extending the project

- Add model training CLI and metrics.
- Add cross‑validation or a validation split and perform TF‑IDF hyperparameter sweeps.
- Integrate probability calibration and downstream logic.
- Add confusion matrix and per‑class metrics in an evaluation binary.
- Add a config file for Neuro‑Fuzzy parameters if you want to switch settings without recompiling.


## Neuro‑Fuzzy details

The hybrid combines the NN output with interpretable cues and Sugeno‑style rules:

- Memberships (6‑dim):
  - μ_neg, μ_neu, μ_pos — NN softmax probabilities (with temperature) for each class.
  - conf_high — a margin‑based confidence cue (top‑1 minus top‑2).
  - ambig_high — an entropy‑based ambiguity cue.
  - pos_over_neg_high — a direct logit gap (z_pos − z_neg) cue.
- Rules: product t‑norm over selected memberships; consequents are constants; output is weighted average.
- Label mapping: score in (−∞, ∞) mapped to {−1, 0, +1} using a threshold τ.

Centralized configuration lives in `src/fuzzy/params.rs`:

```rust
pub const NF_TEMPERATURE: f32 = 0.8;
pub const NF_MARGIN_CENTER: f32 = 0.3;
pub const NF_MARGIN_SLOPE: f32 = 5.0;
pub const NF_ENTROPY_CENTER: f32 = 1.0;
pub const NF_ENTROPY_SLOPE: f32 = 6.0;
pub const NF_GAP_CENTER: f32 = 0.2;
pub const NF_GAP_SLOPE: f32 = 4.0;
pub const NF_LABEL_THRESHOLD: f32 = 0.33;
pub const NF_CLASS_ORDER: [usize; 3] = [0, 1, 2];
```

Edit these constants (and/or `default_rules()`) to change behavior globally for both demo and evaluator.

Example current results on the included dataset (for reference; your results may vary):

```
Results on test set (N = 707):
  NN‑only accuracy:     60.11% (425 / 707)
  Neuro‑Fuzzy accuracy: 58.84% (416 / 707)
```


## FAQ

- Q: Can I use a different tokenizer?
  - A: Yes. Swap in your own tokenizer as long as you pass `Vec<String>` tokens into the vectorizer.
- Q: Can I keep the sparse representation?
  - A: The vectorizer returns sparse maps; you can process sparse directly if your model supports it.
- Q: How do I add more labels/classes?
  - A: Adjust your dataset and model accordingly. The TF‑IDF pipeline is label‑agnostic.


## License

This repository is provided as‑is for educational and experimental purposes. Add your preferred license if you plan to distribute.
