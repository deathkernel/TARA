"""Small tokenizers used by TARA.

Research basis:
- Kudo & Richardson (2018), SentencePiece, motivates subword tokenization.
- OpenAI's tiktoken uses byte-pair encoding (BPE); TARA studies the idea with a
  deliberately small, dependency-free implementation before using an external tokenizer.
"""

from collections import Counter
import heapq


class CharTokenizer:
    """Map characters to integer token IDs and back."""

    def __init__(self, text):
        if not text:
            raise ValueError("training text must not be empty")
        vocabulary = sorted(set(text))
        self.itos = ["<UNK>"] + vocabulary
        self.stoi = {token: index for index, token in enumerate(self.itos)}

    @property
    def vocab_size(self):
        return len(self.itos)

    def encode(self, text):
        return [self.stoi.get(character, self.stoi["<UNK>"]) for character in text]

    def decode(self, ids):
        return "".join(self.itos[index] if 0 <= index < self.vocab_size else "<UNK>" for index in ids)


class BPETokenizer:
    """Dependency-free BPE with bounded training and efficient encoding."""

    UNK = "<UNK>"

    def __init__(self, text, vocab_size=128):
        if not text:
            raise ValueError("training text must not be empty")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")

        base_tokens = sorted(set(text))
        if vocab_size < len(base_tokens) + 1:
            raise ValueError("vocab_size must fit UNK plus all training characters")

        self.itos = [self.UNK] + base_tokens
        self.stoi = {token: index for index, token in enumerate(self.itos)}
        self.merges = []
        self._train(text, vocab_size)
        self._merge_ranks = {pair: index for index, pair in enumerate(self.merges)}

    @property
    def vocab_size(self):
        return len(self.itos)

    def _train(self, text, target_vocab_size):
        # Bounded deterministic merge learning prevents multi-million-character
        # datasets from repeatedly rescanning the entire corpus in Python.
        training_text = text if len(text) <= 50000 else text[:50000]
        symbols = list(training_text)
        while len(self.itos) < target_vocab_size:
            pair_counts = Counter(zip(symbols, symbols[1:]))
            if not pair_counts:
                break
            best_pair, best_count = min(
                pair_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
            if best_count < 2:
                break
            merged = best_pair[0] + best_pair[1]
            if merged in self.stoi:
                break
            self.merges.append(best_pair)
            self.itos.append(merged)
            self.stoi[merged] = len(self.itos) - 1
            symbols = self._merge_pair(symbols, best_pair)

    @staticmethod
    def _merge_pair(symbols, pair):
        merged = []
        i = 0
        while i < len(symbols):
            if i + 1 < len(symbols) and (symbols[i], symbols[i + 1]) == pair:
                merged.append(symbols[i] + symbols[i + 1])
                i += 2
            else:
                merged.append(symbols[i])
                i += 1
        return merged

    def _apply_merges(self, symbols):
        if len(symbols) < 2 or not self.merges:
            return symbols

        nodes = [
            {"value": value, "prev": i - 1,
             "next": i + 1 if i + 1 < len(symbols) else -1,
             "alive": True}
            for i, value in enumerate(symbols)
        ]
        heap = []
        for i in range(len(nodes) - 1):
            rank = self._merge_ranks.get((nodes[i]["value"], nodes[i + 1]["value"]))
            if rank is not None:
                heapq.heappush(heap, (rank, i))

        while heap:
            rank, left = heapq.heappop(heap)
            if not nodes[left]["alive"]:
                continue
            right = nodes[left]["next"]
            if right < 0 or not nodes[right]["alive"]:
                continue
            pair = (nodes[left]["value"], nodes[right]["value"])
            if self._merge_ranks.get(pair) != rank:
                continue

            prev = nodes[left]["prev"]
            nxt = nodes[right]["next"]
            nodes[left]["value"] += nodes[right]["value"]
            nodes[left]["next"] = nxt
            nodes[right]["alive"] = False
            if nxt >= 0:
                nodes[nxt]["prev"] = left

            if prev >= 0:
                new_rank = self._merge_ranks.get((nodes[prev]["value"], nodes[left]["value"]))
                if new_rank is not None:
                    heapq.heappush(heap, (new_rank, prev))
            if nxt >= 0:
                new_rank = self._merge_ranks.get((nodes[left]["value"], nodes[nxt]["value"]))
                if new_rank is not None:
                    heapq.heappush(heap, (new_rank, left))

        result = []
        index = 0
        while index >= 0:
            if nodes[index]["alive"]:
                result.append(nodes[index]["value"])
            index = nodes[index]["next"]
        return result

    def encode_tokens(self, text):
        if not text:
            return []
        symbols = [symbol if symbol in self.stoi else self.UNK for symbol in text]
        return self._apply_merges(symbols)

    def encode(self, text):
        return [self.stoi.get(token, self.stoi[self.UNK]) for token in self.encode_tokens(text)]

    def decode(self, ids):
        pieces = []
        for index in ids:
            if index < 0 or index >= self.vocab_size:
                pieces.append(self.UNK)
            else:
                pieces.append(self.itos[index])
        return "".join(pieces)
