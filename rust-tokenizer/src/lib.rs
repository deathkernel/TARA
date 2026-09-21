use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct BpeTokenizer {
    pub vocab: Vec<Vec<u8>>,
    pub merges: Vec<(u32, u32)>,
}

impl BpeTokenizer {
    pub fn train(text: &str, target_vocab_size: usize) -> Self {
        assert!(target_vocab_size >= 256, "vocab must include all byte values");

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
                .max_by(|(a, ac), (b, bc)| ac.cmp(bc).then_with(|| b.cmp(a)));

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

    pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
        let mut bytes = Vec::new();

        for &id in ids {
            let piece = self
                .vocab
                .get(id as usize)
                .ok_or_else(|| format!("token id {id} is outside the vocabulary"))?;
            bytes.extend_from_slice(piece);
        }

        String::from_utf8(bytes)
            .map_err(|_| "tokenizer vocabulary contains invalid UTF-8".to_string())
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
        assert_eq!(tokenizer.decode(&encoded).unwrap(), text);
    }

    #[test]
    fn compression_improves_with_repetition() {
        let text = "abababababababababab";
        let tokenizer = BpeTokenizer::train(text, 260);
        assert!(tokenizer.encode(text).len() < text.as_bytes().len());
    }

    #[test]
    fn decode_rejects_invalid_token_id() {
        let tokenizer = BpeTokenizer::train("hello", 256);
        let error = tokenizer.decode(&[999]).unwrap_err();
        assert!(error.contains("outside the vocabulary"));
    }

    #[test]
    fn decode_rejects_invalid_utf8() {
        let tokenizer = BpeTokenizer {
            vocab: vec![vec![0xff]],
            merges: Vec::new(),
        };
        assert!(tokenizer.decode(&[0]).is_err());
    }

    #[test]
    fn benchmark_holdout_corpus() {
        let training_text = [
            "A small model learns patterns from many examples. ",
            "The tokenizer breaks text into reusable pieces. ",
            "Attention helps each position inspect earlier context. ",
            "A dataset should contain varied language rather than duplicates. ",
            "Training loss measures prediction error on observed tokens. ",
            "Validation loss measures performance on unseen text. ",
            "Rust can provide fast text processing for a research pipeline. ",
            "Python remains useful for experiments and analysis. ",
            "A good benchmark keeps the comparison controlled and reproducible. ",
            "Longer context can expose relationships across more tokens. ",
            "A model should be tested on data it did not train on. ",
            "Compression alone does not prove that a tokenizer is better. ",
            "A vocabulary contains atomic symbols and learned merges. ",
            "Repeated byte pairs can become larger subword units. ",
            "Language data contains punctuation, numbers, and mixed casing. ",
            "Reliable experiments record time, loss, and token counts. ",
        ]
        .concat();

        let validation_text = [
            "The model reads a fresh paragraph after training is complete. ",
            "Unseen examples are useful for checking whether patterns generalize. ",
            "Tokenizer speed matters when large corpora must be prepared. ",
            "Natural text is much less repetitive than an artificial benchmark. ",
            "Different vocabularies can change both sequence length and compute cost. ",
            "A smaller sequence is useful only when the model can learn from it. ",
            "Research results should be based on matched train and validation conditions. ",
            "TARA compares components with the same data budget and random seed. ",
        ]
        .concat();

        let started = Instant::now();
        let tokenizer = BpeTokenizer::train(&training_text, 512);
        let train_ids = tokenizer.encode(&training_text);
        let validation_ids = tokenizer.encode(&validation_text);
        let elapsed = started.elapsed();

        println!("benchmark train_bytes: {}", training_text.len());
        println!("benchmark validation_bytes: {}", validation_text.len());
        println!("benchmark train_tokens: {}", train_ids.len());
        println!("benchmark validation_tokens: {}", validation_ids.len());
        println!(
            "benchmark train_token/byte: {:.4}",
            train_ids.len() as f64 / training_text.len() as f64
        );
        println!(
            "benchmark validation_token/byte: {:.4}",
            validation_ids.len() as f64 / validation_text.len() as f64
        );
        println!("benchmark vocab: {}", tokenizer.vocab.len());
        println!("benchmark elapsed_ms: {:.3}", elapsed.as_secs_f64() * 1000.0);

        assert!(!train_ids.is_empty());
        assert!(!validation_ids.is_empty());
        assert_eq!(tokenizer.decode(&validation_ids).unwrap(), validation_text);
    }
}
