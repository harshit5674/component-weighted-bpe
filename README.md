# component-weighted-bpe

Exact incremental byte-pair encoding (BPE) in O(log t) worst-case time per appended byte, where t is the length of the longest token.

This repository contains the code for

> Harshit Verma and Rex Ying. **Component-Weighted Centroid Search for Exact Incremental BPE.** AXIOM: Foundations of Efficient Deep Learning, NeurIPS 2026 Workshop. [arXiv:2609.40016](https://arxiv.org/abs/2609.40016)

## What it does

An incremental tokenizer receives text one byte at a time. After each byte, `CentroidStream` returns the last token of the standard BPE tokenization of the text so far and stores it with the position where that token starts. Following these positions back gives the tokenization of any prefix, so a growing prompt never has to be tokenized again from the start.

The algorithm is the one of [Jiang and Gong (ICML 2026)](https://arxiv.org/abs/2605.30813). Their search for the new last token passes through O(log t) nested subproblems. At each one it chooses among the child intervals of a node by binary search, which can cost another O(log t). The crate implements that search and ours side by side:

| `CentroidSearch` | How it searches a node's child intervals | Worst case per byte |
|---|---|---|
| `CountBalanced` | binary search, as in Jiang and Gong | O(log² t) |
| `ComponentWeighted` | a search tree balanced by the size of the subproblem each interval leads to (this paper) | O(log t) |

Both return exactly the same tokens. Matching, validity intervals, the centroid decomposition and the stored history are shared.

## Quick start

You need a stable Rust toolchain. The crate has no dependencies.

```rust
use component_weighted_bpe::*;
use std::sync::Arc;

// Alphabet "ab". Rule 0 merges a+b into token 2; rule 1 merges ab+b into token 3.
let dictionary = Arc::new(Dictionary::new(b"ab", &[(0, 1), (2, 1)], 100)?);
let options = CentroidOptions {
    search: CentroidSearch::ComponentWeighted,
    ..Default::default()
};
let compiled = Arc::new(CentroidCompiled::compile(dictionary, &options)?);
let mut stream = CentroidStream::with_capacity(compiled, 3)?;
stream.extend(b"abb")?;
assert_eq!(stream.reconstruct(3)?, vec![3]); // "abb" is a single token
```

```bash
cargo test --release
cargo run --example quickstart
```

Token IDs `0..alphabet.len()` are the bytes of the alphabet, in order, and rule `i` produces token `alphabet.len() + i`. `append_profiled` returns the same token as `append` together with operation counts (`SearchStats`), including the number of interval probes.

## Repository layout

| Path | Contents |
|---|---|
| `src/dictionary.rs` | validated BPE dictionaries, and literal and heap reference tokenizers |
| `src/centroid/` | the compiler, both branch searches and the stream |
| `src/witness.rs` | the constructed dictionaries of Section 4 and Appendix C |
| `tests/` | exhaustive and randomized checks against literal BPE, the probe formulas, allocation-free appends |
| `examples/witness_probes.rs` | probe counts on the constructed family (Figure 2) |
| `examples/vocab_stats.rs` | probe statistics and update times on real vocabularies (Section 5, Appendix E) |
| `scripts/prepare_vocab.py` | converts a Hugging Face `tokenizer.json` into the format read by `vocab_stats` |
| `scripts/fetch_data.sh` | downloads the vocabularies and texts used in Section 5 |
| `scripts/check_construction.py` | an independent Python check of the construction |

## Reproducing the paper

### Correctness

`cargo test --release` checks both searches against literal BPE on:

* every normalized dictionary over two letters with at most three rules, on every 7-byte input: 399 dictionaries and 715,008 prefix comparisons;
* 48 random dictionaries over four letters with up to 20 rules, and regression cases for self-pair rules and long successor chains;
* the constructed family for k ≤ 9, where it also checks the exact probe counts below.

A counting allocator also checks that appends to a stream with reserved capacity allocate nothing.

### Figure 2: the constructed family

```bash
cargo run --release --example witness_probes -- 12
```

builds the dictionary with t = 2^k for k = 4 to 12, runs both searches over its input, and prints the interval probes on the final update:

| k | t | count-balanced | component-weighted |
|---:|---:|---:|---:|
| 4 | 16 | 11 | 4 |
| 5 | 32 | 16 | 5 |
| 6 | 64 | 22 | 6 |
| 7 | 128 | 29 | 7 |
| 8 | 256 | 37 | 8 |
| 9 | 512 | 46 | 9 |
| 10 | 1024 | 56 | 10 |
| 11 | 2048 | 67 | 11 |
| 12 | 4096 | 79 | 12 |

These are exactly k(k+1)/2 + 1 and k. The vocabulary of this family has a total spelling length of about t²/2, and this crate's dense matching automaton grows with it. On an Apple M4, k = 11 took 4 seconds and 1.5 GB of memory, and k = 12 took 27 seconds and 4 GB. The paper's largest point, t = 8192, needs roughly 16 GB with this implementation. The example also prints the time of the final update, but a single append takes well under a microsecond here, close to the resolution of the system clock, so those times are coarse.

### Real vocabularies (Section 5)

```bash
sh scripts/fetch_data.sh
cargo run --release --example vocab_stats -- data/gpt2.ibpe data/english.txt data/code.txt
cargo run --release --example vocab_stats -- data/roberta-base.ibpe data/english.txt data/code.txt
cargo run --release --example vocab_stats -- data/gpt-neox-20b.ibpe data/english.txt data/code.txt
```

`fetch_data.sh` needs `curl`, `git` and Python 3. It downloads the GPT-2, RoBERTa-base and GPT-NeoX-20B tokenizers from Hugging Face, *War and Peace* from Project Gutenberg and the TheAlgorithms/Rust repository, cuts one million bytes of English and of Rust code, and converts the tokenizers. Texts are tokenized as continuous byte strings, without pre-tokenization, normalization or special tokens.

For each text and search, `vocab_stats` prints one JSON line: the share of updates that searched at least one interval, the longest list searched, the 50th/95th/99th/99.9th percentiles and maximum of probes per update, the median time per update, and the number of prefixes checked against offline BPE. It stops with an error if the two searches ever return different tokens.

The paper's text snapshots are not included, so results can differ slightly from the paper; the script records hashes of the files it downloads. Times depend on the machine. Run on an idle machine and raise `--repeats` for stable values.

### Independent check of the construction

```bash
python3 scripts/check_construction.py
```

This standard-library Python script implements literal BPE, the validity intervals, the centroid decomposition and both searches separately from the Rust code. It checks every claim of Appendices C and D for k ≤ 6 with distinct letters, for repeated letters, and for the fixed-alphabet variant with k ≤ 4. It runs in under a second.

## Notes on this implementation

* This is an independent implementation of Jiang and Gong's framework, not their released code ([ModelTC/mtc-inc-bpe](https://github.com/ModelTC/mtc-inc-bpe)). It uses a dense Aho–Corasick automaton with completed transitions, so the matcher needs (automaton states) × (alphabet size) words; Jiang and Gong use a more compact tiled table.
* The count-balanced search finds the last interval whose lower endpoint is at most the query, then checks that interval's right endpoint as a separate probe. This convention fixes the exact count k(k+1)/2 + 1 on the constructed family. Other binary-search conventions change only the additive constant.
* `Dictionary::new` accepts only normalized dictionaries: every token is produced by exactly one rule, and tokenizing its spelling returns that token. It rejects other merge lists rather than normalizing them. `scripts/prepare_vocab.py` drops the merges of a Hugging Face vocabulary that would break this and lists them in a report.
* Compilation charges its tables against `CentroidOptions::max_table_bytes` (256 MiB by default) and fails rather than exceed it.
* Appends to a stream with reserved capacity allocate nothing. Compilation, `reserve` and `reconstruct` may allocate. The library contains no `unsafe` code.

## Citation

```bibtex
@inproceedings{verma2026componentweighted,
  title         = {Component-Weighted Centroid Search for Exact Incremental {BPE}},
  author        = {Verma, Harshit and Ying, Rex},
  booktitle     = {AXIOM: Foundations of Efficient Deep Learning, NeurIPS 2026 Workshop},
  year          = {2026},
  eprint        = {2609.40016},
  archivePrefix = {arXiv}
}
```

Please also cite the work this builds on:

```bibtex
@inproceedings{jiang2026incremental,
  title         = {Incremental {BPE} Tokenization},
  author        = {Jiang, Shenghu and Gong, Ruihao},
  booktitle     = {Proceedings of the 43rd International Conference on Machine Learning},
  series        = {Proceedings of Machine Learning Research},
  volume        = {306},
  year          = {2026},
  eprint        = {2605.30813},
  archivePrefix = {arXiv}
}
```

## License

Licensed under either the [Apache License, Version 2.0](LICENSE-APACHE) or the [MIT license](LICENSE-MIT), at your option.
