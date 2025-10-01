mod preprocessing;

fn main() {
    let text = "This is a test sentence, with punctuation!";
    let tokens = preprocessing::tokenizer::tokenize(text);
    println!("Tokens: {:?}", tokens);
}
