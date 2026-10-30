#![forbid(unsafe_code)]
//! Exact incremental byte-pair encoding (BPE) with centroid search.
//!
//! After every appended byte, a [`CentroidStream`] returns the last token of the
//! standard BPE tokenization of the text so far and stores it with a
//! back-pointer, so the tokenization of any prefix can be reconstructed. The
//! algorithm is the one of Jiang and Gong (ICML 2026), with two interchangeable
//! branch searches:
//!
//! * [`CentroidSearch::CountBalanced`] searches a node's child intervals by
//!   ordinary binary search, as Jiang and Gong do: O(log² t) worst-case time
//!   per byte, where t is the maximum token length.
//! * [`CentroidSearch::ComponentWeighted`] searches them with a tree balanced by
//!   the size of the component each child leads to: O(log t) worst-case time
//!   per byte, with exactly the same output.
//!
//! The dictionary must be normalized: every token is produced by exactly one
//! rule and tokenizing its spelling returns it. [`Dictionary::new`] checks this
//! and rejects other merge lists. Compilation charges its tables against
//! [`CentroidOptions::max_table_bytes`]. Appends to a stream with reserved
//! capacity allocate nothing.
//!
//! ```
//! use component_weighted_bpe::*;
//! use std::sync::Arc;
//!
//! // Alphabet "ab"; rule 0 merges a+b into token 2, rule 1 merges ab+b into token 3.
//! let dictionary = Arc::new(Dictionary::new(b"ab", &[(0, 1), (2, 1)], 100)?);
//! for search in [CentroidSearch::CountBalanced, CentroidSearch::ComponentWeighted] {
//!     let options = CentroidOptions { search, ..CentroidOptions::default() };
//!     let compiled = Arc::new(CentroidCompiled::compile(dictionary.clone(), &options)?);
//!     let mut stream = CentroidStream::with_capacity(compiled, 3)?;
//!     stream.extend(b"abb")?;
//!     assert_eq!(stream.reconstruct(3)?, vec![3]);
//! }
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

mod budget;
mod centroid;
mod dictionary;
mod error;
mod trie;
pub mod witness;

pub use centroid::{
    CentroidCompiled, CentroidOptions, CentroidReport, CentroidSearch, CentroidStream, Prefix,
    SearchStats,
};
pub use dictionary::{Dictionary, Rule};
pub use error::{Error, StreamError};

/// Token identifiers. IDs `0..alphabet.len()` are the single bytes, in alphabet
/// order, and rule `i` (counting from 0) produces token `alphabet.len() + i`.
pub type TokenId = u32;
