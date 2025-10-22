use anyhow::Result;
use sentiment_analysis::NN::io::{load_sentiment_csv, SentimentData};
use sentiment_analysis::preprocessing::tokenizer::Tokenizer;
use sentiment_analysis::preprocessing::serialization;
use sentiment_analysis::tfidf::vectorizer::TfIdfVectorizer;
use sentiment_analysis::utils::train_test_split;

fn main() -> Result<()> {
    println!("=== TF-IDF Dataset Preparation Pipeline ===\n");

    // Configuration
    let csv_path = "src/result.csv";
    let data_dir = "data";
    let test_size = 0.2;
    let random_seed = 42;

    // TF-IDF parameters
    let max_features = 5000;
    let min_df = 2;
    let max_df = 0.9;
    let ngram_range = (1, 2);

    // Step 1: Load dataset
    println!("Step 1: Loading dataset from {}...", csv_path);
    let data = load_sentiment_csv(csv_path)?;
    println!("Total samples: {}\n", data.len());

    // Step 2: Preprocess text
    println!("Step 2: Preprocessing text...");
    let tokenizer = Tokenizer::new()
        .with_stopwords(true)
        .with_min_length(2);
    
    let tokenized_texts: Vec<Vec<String>> = data.texts
        .iter()
        .map(|text| tokenizer.tokenize(text))
        .collect();
    
    println!("Sample tokenized text: {:?}", &tokenized_texts[0]);
    println!("Preprocessing complete.\n");

    // Step 3: Train-test split
    println!("Step 3: Splitting data ({}% test)...", (test_size * 100.0) as i32);
    let (train_texts, test_texts, train_labels, test_labels) = 
        train_test_split(tokenized_texts, data.labels, test_size, random_seed);
    
    println!("Train samples: {}", train_texts.len());
    println!("Test samples: {}\n", test_texts.len());

    // Step 4: Fit TF-IDF vectorizer on training data only
    println!("Step 4: Fitting TF-IDF vectorizer...");
    println!("  max_features: {}", max_features);
    println!("  min_df: {}", min_df);
    println!("  max_df: {}", max_df);
    println!("  ngram_range: {:?}", ngram_range);
    
    let mut vectorizer = TfIdfVectorizer::new()
        .max_features(max_features)
        .min_df(min_df)
        .max_df(max_df)
        .ngram_range(ngram_range.0, ngram_range.1)
        .build();
    
    vectorizer.fit(&train_texts);
    println!("Vocabulary size: {}", vectorizer.vocab_size());
    
    // Display sample features
    let feature_names = vectorizer.get_feature_names();
    println!("Sample features: {:?}", &feature_names[..10.min(feature_names.len())]);
    println!();

    // Step 5: Transform both train and test sets
    println!("Step 5: Transforming texts to TF-IDF vectors...");
    
    let train_vectors: Vec<Vec<f32>> = train_texts
        .iter()
        .map(|text| {
            let sparse = vectorizer.transform(text);
            vectorizer.to_dense(&sparse)
        })
        .collect();
    
    let test_vectors: Vec<Vec<f32>> = test_texts
        .iter()
        .map(|text| {
            let sparse = vectorizer.transform(text);
            vectorizer.to_dense(&sparse)
        })
        .collect();
    
    println!("Train vectors shape: [{}, {}]", train_vectors.len(), 
             train_vectors.first().map(|v| v.len()).unwrap_or(0));
    println!("Test vectors shape: [{}, {}]", test_vectors.len(), 
             test_vectors.first().map(|v| v.len()).unwrap_or(0));
    println!();

    // Step 6: Serialize everything
    println!("Step 6: Saving processed data to {}...", data_dir);
    serialization::save_all(
        data_dir,
        &vectorizer,
        &train_vectors,
        &test_vectors,
        &train_labels,
        &test_labels,
    )?;

    // Step 7: Validate by loading back
    println!("\nStep 7: Validating saved data...");
    let loaded = serialization::load_all(data_dir)?;
    
    assert_eq!(loaded.train_vectors.len(), train_vectors.len(), "Train vectors mismatch!");
    assert_eq!(loaded.test_vectors.len(), test_vectors.len(), "Test vectors mismatch!");
    assert_eq!(loaded.train_labels.len(), train_labels.len(), "Train labels mismatch!");
    assert_eq!(loaded.test_labels.len(), test_labels.len(), "Test labels mismatch!");
    assert_eq!(loaded.vectorizer.vocab_size(), vectorizer.vocab_size(), "Vocabulary size mismatch!");
    
    println!("\n✓ Validation successful!");
    
    // Display metadata
    println!("\n=== Dataset Metadata ===");
    println!("Feature count: {}", loaded.metadata.feature_count);
    println!("Vocabulary size: {}", loaded.metadata.vocab_size);
    println!("Train samples: {}", loaded.metadata.train_size);
    println!("Test samples: {}", loaded.metadata.test_size);
    println!("Max features: {:?}", loaded.metadata.max_features);
    println!("Min DF: {}", loaded.metadata.min_df);
    println!("Max DF: {}", loaded.metadata.max_df);
    println!("N-gram range: {:?}", loaded.metadata.ngram_range);
    println!("Created at: {}", loaded.metadata.created_at);
    
    println!("\n=== Pipeline Complete ===");
    println!("All data has been processed and saved successfully!");
    println!("You can now use the saved data for training your neural network.\n");

    Ok(())
}
