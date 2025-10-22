use std::collections::HashSet;

/// Common English stopwords
fn get_stopwords() -> HashSet<String> {
    let words = vec![
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
        "to", "was", "will", "with", "i", "me", "my", "you", "your",
        "this", "but", "or", "not", "have", "had", "do", "does", "did",
    ];
    words.iter().map(|s| s.to_string()).collect()
}

/// Basic tokenization with lowercase conversion
pub fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|s| s.to_lowercase())
        .collect()
}

/// Clean text by removing/normalizing punctuation
pub fn clean_text(text: &str) -> String {
    text.chars()
        .map(|c| {
            if c.is_alphanumeric() || c.is_whitespace() {
                c
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<&str>>()
        .join(" ")
}

/// Tokenize with preprocessing: clean, lowercase, remove stopwords
pub fn tokenize_with_preprocessing(text: &str, remove_stopwords: bool) -> Vec<String> {
    let cleaned = clean_text(text);
    let tokens: Vec<String> = cleaned
        .split_whitespace()
        .map(|s| s.to_lowercase())
        .filter(|s| !s.is_empty())
        .collect();
    
    if remove_stopwords {
        let stopwords = get_stopwords();
        tokens
            .into_iter()
            .filter(|t| !stopwords.contains(t))
            .collect()
    } else {
        tokens
    }
}

/// Advanced tokenizer with configurable options
pub struct Tokenizer {
    pub remove_stopwords: bool,
    pub min_token_length: usize,
}

impl Tokenizer {
    pub fn new() -> Self {
        Self {
            remove_stopwords: true,
            min_token_length: 2,
        }
    }

    pub fn with_stopwords(mut self, remove: bool) -> Self {
        self.remove_stopwords = remove;
        self
    }

    pub fn with_min_length(mut self, len: usize) -> Self {
        self.min_token_length = len;
        self
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let cleaned = clean_text(text);
        let tokens: Vec<String> = cleaned
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .filter(|s| s.len() >= self.min_token_length)
            .collect();
        
        if self.remove_stopwords {
            let stopwords = get_stopwords();
            tokens
                .into_iter()
                .filter(|t| !stopwords.contains(t))
                .collect()
        } else {
            tokens
        }
    }
}

impl Default for Tokenizer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello, world!");
        assert_eq!(tokens, vec!["hello,", "world!"]);
    }
    
    #[test]
    fn test_clean_text() {
        let cleaned = clean_text("Hello, world! This is a test.");
        assert_eq!(cleaned, "Hello world This is a test");
    }
    
    #[test]
    fn test_tokenize_with_preprocessing() {
        let tokens = tokenize_with_preprocessing("The quick brown fox!", false);
        assert_eq!(tokens, vec!["the", "quick", "brown", "fox"]);
        
        let tokens_no_stop = tokenize_with_preprocessing("The quick brown fox!", true);
        assert_eq!(tokens_no_stop, vec!["quick", "brown", "fox"]);
    }
}
