use std::collections::{HashMap, HashSet};

pub struct TfIdfVectorizer {
    pub vocab: HashMap<String, usize>, // word -> index
    pub idf: Vec<f32>,                // idf for each vocab word
}

impl TfIdfVectorizer {
    // Build vocab and compute IDF from training data
    pub fn fit(corpus: &[Vec<String>]) -> Self { /* ... */ }

    // Transform a tokenized sentence into a TF-IDF vector
    pub fn transform(&self, sentence: &[String]) -> HashMap<usize, f32> { /* ... */ }

    //  L2 normalize a vector
    pub fn l2_normalize(vec: &mut HashMap<usize, f32>) { /* ... */ }
}
