
//! TF-IDF feature extraction module.
//!
//! Exposes a vectorizer that builds a vocabulary and computes smoothed IDF,
//! and transforms tokenized texts into sparse L2-normalized TF-IDF vectors.

pub mod vectorizer;

pub use vectorizer::TfIdfVectorizer;

// Optional convenience alias for sparse vectors produced by transform()
