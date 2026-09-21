use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    pub vocab: Vec<Vec<u8>>,
    pub merges: Vec<(u32, u32)>,
}

impl BpeTokenizer {
    pub fn train(text: &str, target_vocab_size: usize) -> Self {
        assert!(
            target_vocab_size >= 256,
            "vocab must include all byte values"
        );

        let mut vocab: Vec<Vec<u8>> =
            (0u16..=255).map(|b| vec![b as u8]).collect();
        let mut symbols: Vec<u32> =
            text.as_bytes().iter().map(|b| *b as u32).collect();
        let mut merges = Vec::new();

        while vocab.len() < target_vocab_size {
            let mut counts: HashMap<(u32, u32), usize> = HashMap::new();

            for pair in symbols.windows(2) {
                *counts.entry((pair[0], pair[1])).or_default() += 1;
            }

            let best = counts
                .into_iter()
                .filter(|(_, count)| *count >= 2)
                .max_by(|(a, ac), (b, bc)| {
                    ac.cmp(bc).then_with(|| b.cmp(a))
                });

            let Some(((left, right), _)) = best else {
                break;
            };

            let merged_bytes = [
                vocab[left as usize].clone(),
                vocab[right as usize].clone(),
            ]
            .concat();

            let new_id = vocab.len() as u32;
            vocab.push(merged_bytes);
            merges.push((left, right));

            let mut next = Vec::with_capacity(symbols.len());
            let mut index = 0;

            while index < symbols.len() {
                if index + 1 < symbols.len()
                    && symbols[index] == left
                    && symbols[index + 1] == right
                {
                    next.push(new_id);
                    index += 2;
                } else {
                    next.push(symbols[index]);
                    index += 1;
                }
            }

            symbols = next;
        }

        Self { vocab, merges }
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut symbols: Vec<u32> =
            text.as_bytes().iter().map(|b| *b as u32).collect();

        for (offset, &(left, right)) in self.merges.iter().enumerate() {
            let new_id = (256 + offset) as u32;
            let mut next = Vec::with_capacity(symbols.len());
            let mut index = 0;

            while index < symbols.len() {
                if index + 1 < symbols.len()
                    && symbols[index] == left
                    && symbols[index + 1] == right
                {
                    next.push(new_id);
                    index += 2;
                } else {
                    next.push(symbols[index]);
                    index += 1;
                }
            }

            symbols = next;
        }

        symbols
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();

        for &id in ids {
            let piece = self
                .vocab
                .get(id as usize)
                .expect("token id outside vocabulary");
            bytes.extend_from_slice(piece);
        }

        String::from_utf8(bytes)
            .expect("tokenizer vocabulary should reconstruct valid UTF-8")
    }
}

#[cfg(test)]
mod tests {
    use super::BpeTokenizer;
    use std::time::Instant;

    #[test]
    fn round_trip_unicode() {
        let text = "hello hello नमस्ते";
        let tokenizer = BpeTokenizer::train(text, 300);
        let encoded = tokenizer.encode(text);
        assert_eq!(tokenizer.decode(&encoded), text);
    }

    #[test]
    fn compression_improves_with_repetition() {
        let text = "abababababababababab";
        let tokenizer = BpeTokenizer::train(text, 260);
        assert!(tokenizer.encode(text).len() < text.as_bytes().len());
    }

    #[test]
    fn benchmark_small_corpus() {
        let text = "TARA learns language. ".repeat(2_000);
        let started = Instant::now();
        let tokenizer = BpeTokenizer::train(&text, 512);
        let encoded = tokenizer.encode(&text);
        let elapsed = started.elapsed();

        println!("benchmark bytes: {}", text.len());
        println!("benchmark tokens: {}", encoded.len());
        println!(
            "benchmark token/byte: {:.4}",
            encoded.len() as f64 / text.len() as f64
        );
        println!("benchmark vocab: {}", tokenizer.vocab.len());
        println!("benchmark elapsed_ms: {:.3}", elapsed.as_secs_f64() * 1000.0);

        assert!(!encoded.is_empty());
        assert!(tokenizer.decode(&encoded) == text);
    }
}
