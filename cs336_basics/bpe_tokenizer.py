
from __future__ import annotations


from enum import unique
import json
import base64
import os
import pickle
import tempfile
from typing import Iterable, Iterator
from unicodedata import name
import regex as re
import tqdm
from collections import defaultdict
from .pretokenization_example import find_chunk_boundaries
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


class BPETokenizer:
    def __init__(self,vocab:dict[int, bytes],
                 merges: list[tuple[bytes, bytes]],
                 special_tokens: list[str] = None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens
        self.encoder: dict[bytes,int] = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}

        #Adding the special tokens
        if special_tokens is not None:
            for st in special_tokens:
                cur_st_encoding = st.encode("utf-8")

                if cur_st_encoding not in self.encoder:
                    idx = len(self.vocab)
                    self.vocab[idx] = cur_st_encoding
                    self.encoder[cur_st_encoding] = idx




    @classmethod
    def from_files(
        cls,
        vocab_filepath: str | os.PathLike,
        merges_filepath: str | os.PathLike,
        special_tokens: list[str] | None = None,
    ) -> "BPETokenizer":
        """Load a tokenizer saved by save_bpe_files().

        The base64 format preserves arbitrary byte tokens, including tokens
        that are not valid UTF-8. A legacy UTF-8 JSON/text format is also
        accepted for compatibility with older files.
        """
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)

        if isinstance(raw_vocab, dict) and raw_vocab.get("format") == "byte-bpe-base64-v1":
            vocab = {
                int(token_id): base64.b64decode(token_b64, validate=True)
                for token_id, token_b64 in raw_vocab["vocab"].items()
            }
        else:
            # Legacy format: {token_as_utf8_string: token_id}.
            vocab = {
                int(token_id): token.encode("utf-8")
                for token, token_id in raw_vocab.items()
            }

        with open(merges_filepath, "r", encoding="utf-8") as f:
            merge_lines = f.read().splitlines()

        # Text format: a header followed by two base64-encoded byte tokens
        # per line. Base64 keeps arbitrary byte sequences reversible while
        # retaining the conventional one-merge-per-line .txt layout.
        if merge_lines and merge_lines[0].strip() == "# byte-bpe-base64-v1":
            merges = []
            for line_number, line in enumerate(merge_lines[1:], start=2):
                if not line.strip() or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(
                        f"Invalid merge at {merges_filepath}:{line_number}; expected two base64 tokens"
                    )
                merges.append((
                    base64.b64decode(parts[0], validate=True),
                    base64.b64decode(parts[1], validate=True),
                ))
        else:
            # Legacy format: one pair of UTF-8 strings per line.
            merges = []
            for line in merge_lines:
                parts = line.split(" ")
                if len(parts) == 2:
                    merges.append((parts[0].encode("utf-8"), parts[1].encode("utf-8")))

        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)

    
    def encode(self, text:str) -> list[int]:

        if len(text) == 0:
            return []
        
        #pretokenize
        freq_table:dict[tuple[bytes,...],int] = {}
        get_frequencies(chunk=text,freq_table=freq_table,special_tokens=self.special_tokens)
        unique_words = list(freq_table.keys())
        # word_counts = [freq_table[w] for w in unique_words]
        del freq_table
        #apply the merges
        og_word_to_new_word:dict[tuple[bytes,...],tuple[bytes,...]] = {}
        for w in unique_words:
            temp_w = list(w)
            for m in self.merges:
                i = 0
                new_token = m[0] + m[1]
                cur_w = []
                while i < len(temp_w):
                    if (
                        i < len(temp_w) - 1
                        and (temp_w[i], temp_w[i + 1]) == m
                    ):
                        cur_w.append(new_token)
                        i += 2
        
                    else:
                        cur_w.append(temp_w[i])
                        i += 1
                temp_w = cur_w
        
            og_word_to_new_word[w] = tuple(temp_w)

        #turn the text to idx
        idx_encoding:list[int] = []
        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            special_pattern = "(" + "|".join(re.escape(token) for token in sorted_specials) + ")"
            mini_chunks = re.split(special_pattern, text)
        else:
            mini_chunks = [text]
    
        for mc in mini_chunks:
            if not mc:
                continue
                
            if self.special_tokens is not None and mc in self.special_tokens:
                # Handle special token directly as a single ID (do NOT split with PAT)
                idx_encoding.append(self.encoder[mc.encode('utf-8')])
            else:
                # Normal text: split with PAT and run BPE
                for cur in re.finditer(PAT, mc):
                    word = tuple(bytes([b]) for b in cur.group().encode('utf-8'))
                    new_word = og_word_to_new_word[word]
                    for byte in new_word:
                        idx_encoding.append(self.encoder[byte])  

        return idx_encoding      
                

                
        

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text_chunk in iterable:
            # Yields each token ID one by one from encode()
            yield from self.encode(text_chunk)

    def decode(self, ids: list[int]) -> str:
        raw_bytes = b"".join(self.vocab[token_id] for token_id in ids)
    
        return raw_bytes.decode("utf-8", errors="replace")
        


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    min_vocab = 256 + len(special_tokens)

    if vocab_size < min_vocab:
        raise ValueError(f"vocab_size must be at least {min_vocab}")

    # 1. Initialize vocabulary
    vocab: dict[int, bytes] = {
        i: bytes([i]) for i in range(256)
    }

    current_vocab_sz = 256

    for st in special_tokens:
        vocab[current_vocab_sz] = st.encode("utf-8")
        current_vocab_sz += 1

    merges: list[tuple[bytes, bytes]] = []

    # 2. Pre-tokenize corpus
    
    unique_words , word_counts = pretokenize(input_path=input_path,special_tokens=special_tokens,**kwargs)

    # 4. Initialize pair counts and inverted index
    pair_counts: dict[tuple[bytes, bytes], int] = defaultdict(int)

    inverted_index: dict[tuple[bytes, bytes], set[int]] = defaultdict(set)

    for w_idx, token in enumerate(unique_words):

        count = word_counts[w_idx]

        for i in range(len(token) - 1):

            pair = (token[i], token[i + 1])

            pair_counts[pair] += count
            inverted_index[pair].add(w_idx)

    # 5. BPE merging iterations

    pbar = tqdm.tqdm(total=(vocab_size-current_vocab_sz),desc="Training BPE Tokenizer")
    while current_vocab_sz < vocab_size:

        if not pair_counts:
            break

        # Select highest-frequency pair.
        # Ties: lexicographically greatest byte pair.
        best_pair = max(
            pair_counts,
            key=lambda p: (pair_counts[p], p)
        )

        if pair_counts[best_pair] <= 0:
            break

        left, right = best_pair

        # Save vocabulary entry and merge
        vocab[current_vocab_sz] = left + right
        merges.append((left, right))

        merged_index = current_vocab_sz
        current_vocab_sz += 1
        pbar.update(n=1)

        # Only update words containing the selected pair
        words_to_update = list(inverted_index[best_pair])

        update_table(
            unique_words=unique_words,
            word_counts=word_counts,
            merged_pair=best_pair,
            new_token=left + right,
            words_to_update=words_to_update,
            pair_counts=pair_counts,
            inverted_index=inverted_index
        )

    pbar.close()
    return vocab, merges



def process_chunk(args):
    input_path, special_tokens, start, end, output_path = args

    local_freq = {}

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8")

    get_frequencies(
        chunk=chunk,
        freq_table=local_freq,
        special_tokens=special_tokens
    )

    with open(output_path, "wb") as f:
        pickle.dump(local_freq, f, protocol=pickle.HIGHEST_PROTOCOL)

    del local_freq
    return output_path

def pretokenize(
    input_path,
    special_tokens,
    **kwargs    
)-> tuple[list[tuple[bytes, ...]],list[int]]:
    
    num_processes = min(
        kwargs.get("num_processes", 2),
        os.cpu_count() or 1,
        2
    )

    # Small chunks reduce peak worker memory.
    num_chunks = 64

    input_path = os.fspath(input_path)
    freq = {}

    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            num_chunks,
            b"<|endoftext|>"
        )

    with tempfile.TemporaryDirectory(prefix="bpe_") as tmpdir:

        tasks = [
            (
                input_path,
                special_tokens,
                start,
                end,
                os.path.join(tmpdir, f"chunk_{i}.pkl")
            )
            for i, (start, end) in enumerate(
                zip(boundaries[:-1], boundaries[1:])
            )
        ]

        with ProcessPoolExecutor(
            max_workers=num_processes,
            mp_context=mp.get_context("spawn")
        ) as executor:

            futures = [
                executor.submit(process_chunk, task)
                for task in tasks
            ]

            with tqdm.tqdm(total=len(futures), desc="Pretokenizer") as pbar:

                for future in as_completed(futures):

                    output_path = future.result()

                    with open(output_path, "rb") as f:
                        local_freq = pickle.load(f)

                    for word, count in local_freq.items():
                        freq[word] = freq.get(word, 0) + count

                    del local_freq
                    os.remove(output_path)

                    pbar.update(1)

    # 3. Convert unique pre-tokens into indexed lists
    unique_words = list(freq.keys())
    word_counts = [freq[w] for w in unique_words]

    del freq

    return unique_words , word_counts

def update_table(
    unique_words: list[tuple[bytes, ...]],
    word_counts: list[int],
    merged_pair: tuple[bytes, bytes],
    new_token: bytes,
    words_to_update: list[int],
    pair_counts: dict[tuple[bytes, bytes], int],
    inverted_index: dict[tuple[bytes, bytes], set[int]]
) -> None:

    # Process only affected words
    for w_idx in words_to_update:

        token = unique_words[w_idx]
        count = word_counts[w_idx]

        # 1. Remove old pair contributions
        for i in range(len(token) - 1):

            old_pair = (token[i], token[i + 1])

            pair_counts[old_pair] -= count

            if pair_counts[old_pair] <= 0:
                pair_counts.pop(old_pair, None)

            members = inverted_index.get(old_pair)

            if members is not None:
                members.discard(w_idx)

                if not members:
                    inverted_index.pop(old_pair, None)

        # 2. Merge occurrences from left to right
        new_word = []
        i = 0

        while i < len(token):

            if (
                i < len(token) - 1
                and (token[i], token[i + 1]) == merged_pair
            ):
                new_word.append(new_token)
                i += 2

            else:
                new_word.append(token[i])
                i += 1

        new_word = tuple(new_word)

        # 3. Update the word
        unique_words[w_idx] = new_word

        # 4. Add new pair contributions
        for i in range(len(new_word) - 1):

            new_pair = (new_word[i], new_word[i + 1])

            pair_counts[new_pair] += count
            inverted_index[new_pair].add(w_idx)


def get_frequencies(
    chunk: str,
    freq_table: dict[tuple[bytes, ...], int],
    special_tokens: list[str]
):
    if special_tokens:
        sorted_specials = sorted(special_tokens, key=len, reverse=True)
        special_pattern = "|".join(re.escape(token) for token in sorted_specials)
        mini_chunks = re.split(special_pattern, chunk)
    else:
        mini_chunks = [chunk]

    for mc in mini_chunks:
        for cur in re.finditer(PAT, mc):
            word = tuple(bytes([b]) for b in cur.group().encode('utf-8'))        
            if word not in freq_table:
                freq_table[word] = 1
            else:
                freq_table[word] += 1


def save_bpe_files(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    vocab_filepath: str | os.PathLike,
    merges_filepath: str | os.PathLike,
) -> None:
    """Save a trained byte-level BPE vocabulary and merges for from_files().

    The vocabulary is stored as JSON. Merges are stored as a readable .txt
    file with one pair per line; each token is base64-encoded because BPE
    tokens are arbitrary byte strings and are not necessarily valid UTF-8.
    Both files can be loaded with BPETokenizer.from_files().
    """
    vocab_path = os.fspath(vocab_filepath)
    merges_path = os.fspath(merges_filepath)

    vocab_payload = {
        "format": "byte-bpe-base64-v1",
        "vocab": {
            str(token_id): base64.b64encode(token_bytes).decode("ascii")
            for token_id, token_bytes in sorted(vocab.items())
        },
    }
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_payload, f, ensure_ascii=True, indent=2)

    with open(merges_path, "w", encoding="utf-8") as f:
        f.write("# byte-bpe-base64-v1\n")
        for left, right in merges:
            left_b64 = base64.b64encode(left).decode("ascii")
            right_b64 = base64.b64encode(right).decode("ascii")
            f.write(f"{left_b64} {right_b64}\n")