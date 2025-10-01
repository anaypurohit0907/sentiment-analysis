pub fn tokenize(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|s| s.to_lowercase())
        .collect()
}



#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_tokenize() {
        let tokens = tokenize("Hello, world!");
        assert_eq!(tokens, vec!["hello,", "world!"]);
    }
}
