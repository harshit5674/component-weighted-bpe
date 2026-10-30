//! The constructed dictionaries of Section 4 and Appendix C of the paper.
//!
//! On the final update of [`Witness::input`], count-balanced search makes
//! k(k+1)/2 + 1 interval probes and component-weighted search makes k, where
//! the maximum token length is t = 2^k.
//!
//! With N = 2^k, the target string is S = a_0 a_1 … a_{N-2} [`END`] and the
//! input is [`CONTEXT`] followed by S. The letters are bytes 1..=254. The rules,
//! in priority order, are block rules that build each level's block of letters
//! from right to left, the context rule (CONTEXT, a_0), and suffix rules that
//! join the rest of each position's block to the suffix after that block, from
//! the shortest suffix to the longest. The construction needs only that
//! adjacent pairs of letters are distinct.
//!
//! The vocabulary's total spelling length grows as t², and so does the dense
//! matching automaton: compiling both searches takes about 1.5 GB of memory
//! for k = 11 and 4 GB for k = 12, and k = 13 needs roughly four times more.
//! See `examples/witness_probes.rs`.

use crate::{Dictionary, Error, TokenId};
use std::collections::BTreeMap;
use std::sync::Arc;

/// The end marker, the last byte of every suffix token.
pub const END: u8 = 0;
/// The context symbol, which merges with the first letter and so makes the
/// longest candidate invalid.
pub const CONTEXT: u8 = 255;

/// How the letters a_0, …, a_{N-2} are chosen.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Letters {
    /// One distinct byte per position. Needs 2^k − 1 ≤ 254, so k ≤ 7.
    Distinct,
    /// As few distinct bytes as possible such that no adjacent pair repeats,
    /// taken from an order-two de Bruijn walk.
    DistinctPairs,
}

/// One member of the family, with its dictionary and the input whose final
/// byte triggers the expensive update.
pub struct Witness {
    pub k: usize,
    pub dictionary: Arc<Dictionary>,
    /// [`CONTEXT`] followed by the target string S, which ends in [`END`].
    pub input: Vec<u8>,
}

impl Witness {
    pub fn new(k: usize, letters: Letters) -> Result<Self, Error> {
        if !(1..=15).contains(&k) {
            return Err(Error::Limit {
                resource: "witness parameter k",
                requested: k,
                limit: 15,
            });
        }
        let n = 1usize << k;
        let mut target = match letters {
            Letters::Distinct => {
                if n - 1 > 254 {
                    return Err(Error::Limit {
                        resource: "distinct witness letters",
                        requested: n - 1,
                        limit: 254,
                    });
                }
                (1..n).map(|b| b as u8).collect()
            }
            Letters::DistinctPairs => distinct_pair_walk(n - 1),
        };
        target.push(END);

        let mut alphabet = target.clone();
        alphabet.push(CONTEXT);
        alphabet.sort_unstable();
        alphabet.dedup();
        let mut ids: BTreeMap<Vec<u8>, TokenId> = alphabet
            .iter()
            .enumerate()
            .map(|(i, &a)| (vec![a], i as TokenId))
            .collect();
        let mut merges = Vec::new();
        let mut add = |left: &[u8], right: &[u8]| {
            merges.push((ids[left], ids[right]));
            let id = (alphabet.len() + merges.len() - 1) as TokenId;
            ids.insert([left, right].concat(), id);
        };
        let r = |j: usize| (1usize << j) - 1;
        // Block rules: level j owns positions r(j-1)..r(j), built right to left.
        for j in 1..=k {
            let (b, q) = (r(j - 1), r(j));
            for p in (b..q - 1).rev() {
                add(&target[p..p + 1], &target[p + 1..q]);
            }
        }
        // The context rule.
        add(&[CONTEXT], &target[..1]);
        // Suffix rules, from the shortest suffix to the longest.
        for p in (0..n - 1).rev() {
            let q = parent(p);
            add(&target[p..q], &target[q..]);
        }

        let dictionary = Arc::new(Dictionary::new(&alphabet, &merges, usize::MAX)?);
        let mut input = vec![CONTEXT];
        input.extend_from_slice(&target);
        Ok(Self {
            k,
            dictionary,
            input,
        })
    }

    /// The maximum token length t = 2^k, the length of S.
    pub fn max_token_length(&self) -> usize {
        1 << self.k
    }

    /// Exact probes of this crate's count-balanced search on the final update:
    /// k(k+1)/2 + 1. Its binary search ends with a separate right-endpoint check.
    pub fn count_balanced_probes(&self) -> usize {
        self.k * (self.k + 1) / 2 + 1
    }

    /// Exact probes of the component-weighted search on the final update: k.
    pub fn component_weighted_probes(&self) -> usize {
        self.k
    }
}

/// The parent of node p of T_k in postorder: the end of p's block,
/// 2^(⌊log₂(p+1)⌋ + 1) − 1.
fn parent(p: usize) -> usize {
    let level = (usize::BITS - 1 - (p + 1).leading_zeros()) as usize;
    (2usize << level) - 1
}

/// `count` bytes from 1..=m, with m as small as possible, in which no pair of
/// adjacent bytes repeats: a prefix of an Eulerian circuit of the complete
/// directed graph with loops on m vertices (Hierholzer's algorithm).
fn distinct_pair_walk(count: usize) -> Vec<u8> {
    let mut m = 1;
    while m * m + 1 < count {
        m += 1;
    }
    let mut out: Vec<Vec<usize>> = (0..m).map(|_| (0..m).collect()).collect();
    let mut stack = vec![0];
    let mut circuit = Vec::with_capacity(m * m + 1);
    while let Some(&v) = stack.last() {
        match out[v].pop() {
            Some(w) => stack.push(w),
            None => circuit.push(stack.pop().unwrap()),
        }
    }
    circuit.reverse();
    circuit.truncate(count);
    circuit.into_iter().map(|v| v as u8 + 1).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parents_end_each_block() {
        let expected = [1, 3, 3, 7, 7, 7, 7, 15];
        for (p, &q) in expected.iter().enumerate() {
            assert_eq!(parent(p), q, "p={p}");
        }
    }

    #[test]
    fn walks_have_distinct_adjacent_pairs_and_few_letters() {
        for count in 1..=600 {
            let walk = distinct_pair_walk(count);
            assert_eq!(walk.len(), count);
            let pairs: std::collections::BTreeSet<_> = walk.windows(2).collect();
            assert_eq!(pairs.len(), count.saturating_sub(1));
            let fewest = (1..).find(|m| m * m + 1 >= count).unwrap();
            assert!(walk.iter().all(|&b| (1..=fewest).contains(&(b as usize))));
        }
    }
}
