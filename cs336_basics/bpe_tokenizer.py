
from __future__ import annotations

import os
import regex as re
from collections import defaultdict
from .pretokenization_example import find_chunk_boundaries
from concurrent.futures import ProcessPoolExecutor

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def process_chunk(args):
    input_path, special_tokens, start, end = args

    local_freq = {}

    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8")

    get_frequencies(
        chunk=chunk,
        freq_table=local_freq,
        special_tokens=special_tokens
    )

    return local_freq


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
    
    num_processes = kwargs.get(
        "num_processes",
        os.cpu_count() or 1
    )

    input_path = os.fspath(input_path)

    with open(input_path, "rb") as f:

        boundaries = find_chunk_boundaries(
            f,
            num_processes,
            b"<|endoftext|>"
        )

    tasks = [
        (input_path, special_tokens, start, end)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]

    freq = {}

    with ProcessPoolExecutor(
        max_workers=num_processes
    ) as executor:

        for local_freq in executor.map(process_chunk, tasks):

            for word, count in local_freq.items():

                freq[word] = freq.get(word, 0) + count


    # 3. Convert unique pre-tokens into indexed lists
    unique_words = list(freq.keys())
    word_counts = [freq[w] for w in unique_words]

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

    return vocab, merges


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
        special_pattern = "|".join(re.escape(token) for token in special_tokens)
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
