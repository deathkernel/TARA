use std::io::{self, Read};
use tara_tokenizer::BpeTokenizer;

fn main() {
    let mut text = String::new();
    io::stdin()
        .read_to_string(&mut text)
        .expect("failed to read stdin");

    if text.is_empty() {
        eprintln!("Provide text on stdin.");
        return;
    }

    let tokenizer = BpeTokenizer::train(&text, 512);
    let ids = tokenizer.encode(&text);

    println!("TARA Rust BPE tokenizer");
    println!("bytes: {}", text.as_bytes().len());
    println!("tokens: {}", ids.len());
    println!(
        "token/byte: {:.4}",
        ids.len() as f64 / text.as_bytes().len() as f64
    );
    println!("vocab: {}", tokenizer.vocab.len());
}
