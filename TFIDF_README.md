# TF-IDF Data Preparation - README

## Overview
This implementation provides a complete pipeline for TF-IDF vectorization and dataset preparation for neural network training in Rust. All data preprocessing tasks have been successfully completed.

## ✅ Completed Implementation

### 1. Enhanced Tokenizer (`src/preprocessing/tokenizer.rs`)
- **Features**:
  - Text cleaning (punctuation removal, normalization)
  - Stopword filtering (configurable)
  - Minimum token length filtering
  - Configurable tokenizer with builder pattern

```rust
let tokenizer = Tokenizer::new()
    .with_stopwords(true)
    .with_min_length(2);
let tokens = tokenizer.tokenize("Your text here");
```

### 2. Advanced TF-IDF Vectorizer (`src/tfidf/vectorizer.rs`)
- **Features**:
  - Configurable parameters: `max_features`, `min_df`, `max_df`
  - N-gram support (unigrams, bigrams, etc.)
  - Sparse and dense vector representations
  - Serialization support with `serde`
  - Vocabulary management

```rust
let mut vectorizer = TfIdfVectorizer::new()
    .max_features(5000)
    .min_df(2)
    .max_df(0.9)
    .ngram_range(1, 2)
    .build();

vectorizer.fit(&train_texts);
let sparse_vec = vectorizer.transform(&text_tokens);
let dense_vec = vectorizer.to_dense(&sparse_vec);
```

### 3. Data Loading (`src/NN/io.rs`)
- **Features**:
  - CSV loading with error handling
  - Class distribution analysis
  - Automatic handling of malformed rows

```rust
let data = load_sentiment_csv("src/result.csv")?;
println!("Total samples: {}", data.len());
```

### 4. Train-Test Split (`src/utils.rs`)
- **Features**:
  - Random train-test splitting
  - Stratified split (maintains class distribution)
  - Configurable test size and random seed

```rust
let (train_data, test_data, train_labels, test_labels) = 
    train_test_split(data, labels, 0.2, 42);
```

### 5. Serialization Module (`src/preprocessing/serialization.rs`)
- **Features**:
  - Save/load TF-IDF vectorizer
  - Save/load TF-IDF matrices (train and test)
  - Save/load labels
  - JSON metadata with dataset information
  - Batch save/load functions

```rust
// Save all data
serialization::save_all(
    "data",
    &vectorizer,
    &train_vectors,
    &test_vectors,
    &train_labels,
    &test_labels,
)?;

// Load all data
let loaded = serialization::load_all("data")?;
```

### 6. Data Preparation Pipeline (`src/bin/prepare_data.rs`)
Complete end-to-end pipeline that:
1. Loads dataset from CSV
2. Preprocesses text (tokenization, stopword removal)
3. Splits into train/test sets (80-20)
4. Fits TF-IDF vectorizer on training data only
5. Transforms both train and test sets
6. Serializes everything to disk
7. Validates by loading back

## 📊 Results

### Dataset Statistics
- **Total samples**: 3,534
- **Training samples**: 2,827 (80%)
- **Test samples**: 707 (20%)
- **Feature dimensions**: 3,422
- **Class distribution**:
  - Positive (1.0): 1,103 samples
  - Neutral (0.0): 1,430 samples
  - Negative (-1.0): 1,001 samples

### Output Files (in `data/` directory)
```
data/
├── tfidf_vectorizer.bin      (90 KB)  - Fitted TF-IDF vectorizer
├── tfidf_matrices.bin         (47 MB) - Train and test vectors
├── y_train.bin                (12 KB) - Training labels
├── y_test.bin                 (2.8 KB) - Test labels
└── metadata.json              (239 B)  - Dataset metadata
```

## 🚀 Usage

### Prepare Data
```bash
cargo run --bin prepare_data
```

This will:
- Process `src/result.csv`
- Create TF-IDF vectors
- Save all data to `data/` directory

### Load Preprocessed Data
```bash
cargo run --bin load_data_example
```

Or in your code:
```rust
use sentiment_analysis::preprocessing::serialization;

let data = serialization::load_all("data")?;

// Use the data
let train_features = data.train_vectors;  // [2827, 3422]
let train_labels = data.train_labels;      // [2827]
let test_features = data.test_vectors;     // [707, 3422]
let test_labels = data.test_labels;        // [707]
let vectorizer = data.vectorizer;          // For inference
```

### Transform New Text
```rust
use sentiment_analysis::preprocessing::tokenizer::Tokenizer;

let tokenizer = Tokenizer::new();
let tokens = tokenizer.tokenize("New text to classify");

let sparse_vec = data.vectorizer.transform(&tokens);
let dense_vec = data.vectorizer.to_dense(&sparse_vec);
// Now feed dense_vec to your neural network
```

### Next: Neuro‑Fuzzy demo and evaluation

Once you have `data/` prepared and a trained model saved to `model.bin`, you can:

```bash
cargo run --bin neuro_fuzzy_demo   # interactive REPL with memberships and rules
cargo run --bin evaluate           # report NN‑only vs Neuro‑Fuzzy accuracy
```

The tuned parameters and default rules are centralized in `src/fuzzy/params.rs`. Changing them affects both binaries.

## 🔧 Configuration

### TF-IDF Parameters (in `prepare_data.rs`)
```rust
let max_features = 5000;  // Maximum number of features
let min_df = 2;           // Minimum document frequency
let max_df = 0.9;         // Maximum document frequency (90%)
let ngram_range = (1, 2); // Unigrams and bigrams
```

### Train-Test Split
```rust
let test_size = 0.2;      // 20% test split
let random_seed = 42;     // For reproducibility
```

## 📦 Dependencies Added
```toml
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
bincode = "1.3"
chrono = "0.4"
```

## 🧪 Validation
The pipeline includes automatic validation:
- ✅ Data shapes match expectations
- ✅ Vocabulary consistency
- ✅ Serialization/deserialization integrity
- ✅ No data leakage (vectorizer only sees training data)

## 🎯 Next Steps

### Integration with Neural Network
```rust
use sentiment_analysis::preprocessing::serialization;
use sentiment_analysis::NN::initialize_network;

// Load preprocessed data
let data = serialization::load_all("data")?;

// Initialize network with correct input size
let input_size = data.vectorizer.vocab_size();  // 3422
let hidden_size = 128;
let output_size = 3;  // For 3 classes: positive, neutral, negative
let sizes = vec![input_size, hidden_size, output_size];

let (weights, biases) = initialize_network(&sizes, Init::HeUniform, 42);

// Train using data.train_vectors and data.train_labels
// Test using data.test_vectors and data.test_labels
```

### Inference Pipeline
```rust
// 1. Load vectorizer
let data = serialization::load_all("data")?;

// 2. Process new text
let tokenizer = Tokenizer::new();
let tokens = tokenizer.tokenize(user_input);
let sparse_vec = data.vectorizer.transform(&tokens);
let features = data.vectorizer.to_dense(&sparse_vec);

// 3. Feed to neural network
// let prediction = forward_pass(&features, &weights, &biases);
```

## 📝 Code Quality
- ✅ Error handling with `anyhow::Result`
- ✅ Modular, reusable functions
- ✅ Comprehensive documentation
- ✅ Builder patterns for configuration
- ✅ Type-safe serialization

## 🎉 Success Metrics
- All 9 tasks completed successfully
- Pipeline runs end-to-end without errors
- Data validated and ready for training
- Memory-efficient sparse matrix representation
- Fast serialization with bincode
- Production-ready code structure
