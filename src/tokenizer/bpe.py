"""Byte-level Byte Pair Encoding tokenizer, implemented from scratch."""
import json
from collections import Counter, defaultdict
from pathlib import Path

import regex as re

# Pre-tokenization: merges never cross chunk boundaries. Corporate terms and whole financial
# figures ("$1,234.5M", "-$3.2 M", "12,3%", "7.9x") are kept as single chunks so BPE
# learns them as units instead of mixing digits with neighbouring words.
SPLIT_PATTERN = (
    r"'(?:[sdmt]|ll|ve|re)"
    r"| ?(?:EBITDA|CAPEX|OPEX|ROI|YoY|Q[1-4])(?![\p{L}\p{N}])"
    r"| ?-?\$?\p{N}+(?:[.,]\p{N}+)*(?:%|[MBKx](?!\p{L}))?"
    r"| ?\p{L}+"
    r"| ?[^\s\p{L}\p{N}]+"
    r"|\s+(?!\S)"
    r"|\s+"
)
EOT = "<|endoftext|>"


def merge(ids, pair, idx):
    """Replace every occurrence of `pair` in `ids` with `idx`."""
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(idx)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out


class BPETokenizer:
    def __init__(self, pattern=SPLIT_PATTERN, special_tokens=(EOT,)):
        self.pattern = pattern
        self.compiled = re.compile(pattern)
        self.special_names = list(special_tokens)
        self._set_merges({})

    def _set_merges(self, merges):
        """Rebuild vocab and special-token ids from an ordered {pair: id} merge table."""
        self.merges = merges
        self.vocab = {i: bytes([i]) for i in range(256)}
        for (a, b), idx in merges.items():
            self.vocab[idx] = self.vocab[a] + self.vocab[b]
        base = 256 + len(merges)
        self.special_tokens = {s: base + i for i, s in enumerate(self.special_names)}
        for s, idx in self.special_tokens.items():
            self.vocab[idx] = s.encode("utf-8")
        self.special_re = (re.compile("(" + "|".join(re.escape(s) for s in self.special_tokens) + ")")
                           if self.special_tokens else None)
        self._cache = {}

    @property
    def vocab_size(self):
        return len(self.vocab)

    def train(self, text, vocab_size, verbose=False):
        n_merges = vocab_size - 256 - len(self.special_names)
        assert n_merges >= 0, "vocab_size too small"
        if self.special_re:
            text = "".join(self.special_re.split(text)[::2])  # specials never take part in merges
        # Unique chunks weighted by frequency: much less work than iterating the raw text.
        chunk_freqs = Counter(self.compiled.findall(text))
        words = [list(c.encode("utf-8")) for c in chunk_freqs]
        freqs = list(chunk_freqs.values())

        stats, where = Counter(), defaultdict(set)
        for i, (w, f) in enumerate(zip(words, freqs)):
            for p in zip(w, w[1:]):
                stats[p] += f
                where[p].add(i)

        merges, vocab = {}, {i: bytes([i]) for i in range(256)}
        for k in range(n_merges):
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + k
            merges[pair] = idx
            # Incremental update: only chunks containing `pair` change their pair counts.
            for i in where.pop(pair, ()):
                w, f = words[i], freqs[i]
                old = list(zip(w, w[1:]))
                for p in old:
                    stats[p] -= f
                w = words[i] = merge(w, pair, idx)
                for p in zip(w, w[1:]):
                    stats[p] += f
                    where[p].add(i)
                for p in old:
                    if stats[p] <= 0:
                        del stats[p]
            stats.pop(pair, None)
            vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
            if verbose and (k + 1) % 250 == 0:
                print(f"merge {k + 1}/{n_merges}: {pair} -> {idx} {vocab[idx]!r}")
        self._set_merges(merges)

    def _encode_chunk(self, chunk):
        if chunk in self._cache:
            return self._cache[chunk]
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Apply the earliest-learned merge available, exactly as during training.
            pair = min(zip(ids, ids[1:]), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        self._cache[chunk] = ids
        return ids

    def encode_ordinary(self, text):
        return [i for chunk in self.compiled.findall(text) for i in self._encode_chunk(chunk)]

    def encode(self, text, allowed_special=True):
        if not (allowed_special and self.special_re):
            return self.encode_ordinary(text)
        ids = []
        for k, part in enumerate(self.special_re.split(text)):
            if k % 2:
                ids.append(self.special_tokens[part])
            elif part:
                ids.extend(self.encode_ordinary(part))
        return ids

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "pattern": self.pattern,
            "special_tokens": self.special_names,
            "merges": [[a, b] for (a, b) in self.merges],  # dict order == merge rank
        }
        path.write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        tok = cls(pattern=data["pattern"], special_tokens=data["special_tokens"])
        tok._set_merges({(a, b): 256 + i for i, (a, b) in enumerate(data["merges"])})
        return tok
