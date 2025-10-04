
use std::collections::{HashMap, HashSet};

pub struct TfIdfVectorizer {
    pub vocab: HashMap<String, usize>, // term -> index
    pub idf: Vec<f32>,                 // idf per index
}

impl TfIdfVectorizer {
    /// Fit on a corpus of tokenized documents.
    /// Smoothed IDF: idf(t) = ln((1 + n_docs) / (1 + df(t))) + 1
    pub fn fit(corpus: &[Vec<String>]) -> Self {
        let n_docs = corpus.len() as f32;

        // 1) Document frequency per term (unique per document)
        let mut df: HashMap<String, usize> = HashMap::new();
        for doc in corpus {
            let mut seen: HashSet<&str> = HashSet::new();
            for tok in doc {
                if seen.insert(tok.as_str()) {
                    *df.entry(tok.clone()).or_insert(0) += 1;
                }
            }
        }

        // 2) Build vocabulary with deterministic ordering
        let mut vocab: HashMap<String, usize> = HashMap::new();
        let mut terms: Vec<(String, usize)> = df.into_iter().collect();
        terms.sort_by(|a, b| a.0.cmp(&b.0));
        for (i, (term, _)) in terms.iter().enumerate() {
            vocab.insert(term.clone(), i);
        }

        // 3) Compute smoothed IDF
        let mut idf = vec![0.0f32; vocab.len()];
        for (term, df_val) in terms {
            let idx = *vocab.get(&term).unwrap();
            let df_f = df_val as f32;
            let val = ((1.0 + n_docs) / (1.0 + df_f)).ln() + 1.0;
            idf[idx] = val as f32;
        }

        Self { vocab, idf }
    }

    /// Transform a tokenized sentence into a sparse TF-IDF vector (index -> weight).
    /// Uses natural TF (raw counts) times smoothed IDF; apply L2 normalization separately if desired.
    pub fn transform(&self, sentence: &[String]) -> HashMap<usize, f32> {
        // TF counts for in-vocab tokens
        let mut tf_counts: HashMap<usize, usize> = HashMap::new();
        for tok in sentence {
            if let Some(&idx) = self.vocab.get(tok) {
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
