
use std::collections::{HashMap, HashSet};
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Clone)]
pub struct TfIdfVectorizer {
    pub vocab: HashMap<String, usize>, // term -> index
    pub idf: Vec<f32>,                 // idf per index
    pub max_features: Option<usize>,
    pub min_df: usize,
    pub max_df: f32,
    pub ngram_range: (usize, usize),
}

pub struct TfIdfVectorizerBuilder {
    max_features: Option<usize>,
    min_df: usize,
    max_df: f32,
    ngram_range: (usize, usize),
}

impl TfIdfVectorizerBuilder {
    pub fn new() -> Self {
        Self {
            max_features: None,
            min_df: 1,
            max_df: 1.0,
            ngram_range: (1, 1),
        }
    }

    pub fn max_features(mut self, n: usize) -> Self {
        self.max_features = Some(n);
        self
    }

    pub fn min_df(mut self, n: usize) -> Self {
        self.min_df = n;
        self
    }

    pub fn max_df(mut self, f: f32) -> Self {
        self.max_df = f;
        self
    }

    pub fn ngram_range(mut self, min: usize, max: usize) -> Self {
        self.ngram_range = (min, max);
        self
    }

    pub fn build(self) -> TfIdfVectorizer {
        TfIdfVectorizer {
            vocab: HashMap::new(),
            idf: Vec::new(),
            max_features: self.max_features,
            min_df: self.min_df,
            max_df: self.max_df,
            ngram_range: self.ngram_range,
        }
    }
}

impl TfIdfVectorizer {
    pub fn new() -> TfIdfVectorizerBuilder {
        TfIdfVectorizerBuilder::new()
    }

    /// Generate n-grams from tokens
    fn generate_ngrams(tokens: &[String], n: usize) -> Vec<String> {
        if n == 1 {
            return tokens.to_vec();
        }
        
        // Return empty if we don't have enough tokens
        if tokens.len() < n {
            return Vec::new();
        }
        
        let mut ngrams = Vec::new();
        for i in 0..=tokens.len() - n {
            let ngram = tokens[i..i + n].join(" ");
            ngrams.push(ngram);
        }
        ngrams
    }

    /// Extract all n-grams based on ngram_range
    fn extract_features(&self, tokens: &[String]) -> Vec<String> {
        let mut all_features = Vec::new();
        for n in self.ngram_range.0..=self.ngram_range.1 {
            all_features.extend(Self::generate_ngrams(tokens, n));
        }
        all_features
    }

    /// Fit on a corpus of tokenized documents.
    /// Smoothed IDF: idf(t) = ln((1 + n_docs) / (1 + df(t))) + 1
    pub fn fit(&mut self, corpus: &[Vec<String>]) {
        let n_docs = corpus.len() as f32;

        // 1) Extract features (n-grams) and calculate document frequency
        let mut df: HashMap<String, usize> = HashMap::new();
        for doc in corpus {
            let features = self.extract_features(doc);
            let mut seen: HashSet<String> = HashSet::new();
            for feature in features {
                if seen.insert(feature.clone()) {
                    *df.entry(feature).or_insert(0) += 1;
                }
            }
        }

        // 2) Filter by min_df and max_df
        let max_df_count = (n_docs * self.max_df) as usize;
        let filtered_terms: Vec<(String, usize)> = df
            .into_iter()
            .filter(|(_, count)| *count >= self.min_df && *count <= max_df_count)
            .collect();

        // 3) Sort by document frequency (descending) and limit by max_features
        let mut terms = filtered_terms;
        terms.sort_by(|a, b| b.1.cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
        
        if let Some(max_feat) = self.max_features {
            terms.truncate(max_feat);
        }

        // 4) Build vocabulary with deterministic ordering
        terms.sort_by(|a, b| a.0.cmp(&b.0));
        let mut vocab: HashMap<String, usize> = HashMap::new();
        for (i, (term, _)) in terms.iter().enumerate() {
            vocab.insert(term.clone(), i);
        }

        // 5) Compute smoothed IDF
        let mut idf = vec![0.0f32; vocab.len()];
        for (term, df_val) in terms {
            let idx = *vocab.get(&term).unwrap();
            let df_f = df_val as f32;
            let val = ((1.0 + n_docs) / (1.0 + df_f)).ln() + 1.0;
            idf[idx] = val;
        }

        self.vocab = vocab;
        self.idf = idf;
    }

    /// Transform a tokenized sentence into a sparse TF-IDF vector (index -> weight).
    /// Uses natural TF (raw counts) times smoothed IDF; apply L2 normalization separately if desired.
    pub fn transform(&self, sentence: &[String]) -> HashMap<usize, f32> {
        // Extract features (n-grams)
        let features = self.extract_features(sentence);
        
        // TF counts for in-vocab features
        let mut tf_counts: HashMap<usize, usize> = HashMap::new();
        for feature in features {
            if let Some(&idx) = self.vocab.get(&feature) {
                *tf_counts.entry(idx).or_insert(0) += 1;
            }
        }

        // TF * IDF sparse map
        let mut vec_map: HashMap<usize, f32> = HashMap::with_capacity(tf_counts.len());
        for (idx, tf) in tf_counts {
            let w = (tf as f32) * self.idf[idx];
            if w != 0.0 {
                vec_map.insert(idx, w);
            }
        }
        vec_map
    }

    /// Convert sparse vector to dense vector
    pub fn to_dense(&self, sparse: &HashMap<usize, f32>) -> Vec<f32> {
        let mut dense = vec![0.0; self.vocab.len()];
        for (&idx, &val) in sparse {
            if idx < dense.len() {
                dense[idx] = val;
            }
        }
        dense
    }

    /// Get vocabulary size
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get feature names (sorted by index)
    pub fn get_feature_names(&self) -> Vec<String> {
        let mut features: Vec<(String, usize)> = self.vocab.iter()
            .map(|(k, v)| (k.clone(), *v))
            .collect();
        features.sort_by_key(|x| x.1);
        features.into_iter().map(|(name, _)| name).collect()
    }

    /// In-place L2 normalization on a sparse vector (index -> weight).
    pub fn l2_normalize(vec: &mut HashMap<usize, f32>) {
        let mut sum_sq = 0.0f32;
        for (_, v) in vec.iter() {
            sum_sq += (*v) * (*v);
        }
        let norm = sum_sq.sqrt();
        if norm > 0.0 {
            for v in vec.values_mut() {
                *v /= norm;
            }
        }
    }
}
