use crate::error::limit;
use crate::{Error, StreamError, TokenId};
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

/// One merge rule: `left` followed by `right` becomes `output`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rule {
    pub left: TokenId,
    pub right: TokenId,
    pub output: TokenId,
}

/// A validated, normalized BPE dictionary. Token IDs `0..alphabet.len()` follow
/// alphabet order; rule `i` produces ID `alphabet.len() + i`. Rules apply in
/// rank order, each in one left-to-right pass over non-overlapping pairs.
/// No normalization is implicit: [`Dictionary::new`] rejects merge lists that
/// are not already normalized.
#[derive(Debug)]
pub struct Dictionary {
    alphabet: Vec<u8>,
    byte_ids: [Option<TokenId>; 256],
    rules: Vec<Rule>,
    /// (left, right) -> (rank, output), for the offline heap tokenizer.
    pairs: BTreeMap<(TokenId, TokenId), (usize, TokenId)>,
    spellings: Vec<Vec<u8>>,
    spelling_size: usize,
}

impl Dictionary {
    /// Validates the merge list: operands must exist, spellings must be unique,
    /// and each rule's output must tokenize to exactly its two operands under
    /// the earlier rules. The spelling-size limit is checked before each new
    /// spelling is built.
    pub fn new(
        alphabet: &[u8],
        merges: &[(TokenId, TokenId)],
        max_spelling_bytes: usize,
    ) -> Result<Self, Error> {
        if alphabet.is_empty() || alphabet.len() > 256 {
            return Err(Error::InvalidAlphabet);
        }
        let mut byte_ids = [None; 256];
        for (i, &a) in alphabet.iter().enumerate() {
            if byte_ids[a as usize].replace(i as TokenId).is_some() {
                return Err(Error::InvalidAlphabet);
            }
        }
        let count = alphabet
            .len()
            .checked_add(merges.len())
            .ok_or(Error::Overflow)?;
        limit("token IDs", count, u32::MAX as usize)?;
        limit(
            "explicit spelling bytes",
            alphabet.len(),
            max_spelling_bytes,
        )?;
        let mut spellings: Vec<Vec<u8>> = alphabet.iter().map(|&a| vec![a]).collect();
        let mut seen: BTreeSet<Vec<u8>> = spellings.iter().cloned().collect();
        let mut rules = Vec::new();
        let mut pairs = BTreeMap::new();
        let mut spelling_size = alphabet.len();
        for (i, &(left, right)) in merges.iter().enumerate() {
            for token in [left, right] {
                if token as usize >= spellings.len() {
                    return Err(Error::InvalidOperand { rule: i + 1, token });
                }
            }
            let len = spellings[left as usize]
                .len()
                .checked_add(spellings[right as usize].len())
                .ok_or(Error::Overflow)?;
            spelling_size = spelling_size.checked_add(len).ok_or(Error::Overflow)?;
            limit("explicit spelling bytes", spelling_size, max_spelling_bytes)?;
            let mut word = Vec::new();
            word.try_reserve_exact(len).map_err(|_| Error::Allocation)?;
            word.extend_from_slice(&spellings[left as usize]);
            word.extend_from_slice(&spellings[right as usize]);
            if seen.contains(&word) {
                return Err(Error::DuplicateSpelling { rule: i + 1 });
            }
            let atoms: Vec<_> = word
                .iter()
                .map(|&a| byte_ids[a as usize].unwrap())
                .collect();
            let actual = heap_tokenize(&atoms, &pairs);
            if actual != [left, right] {
                return Err(Error::NonCanonical {
                    rule: i + 1,
                    actual,
                });
            }
            let output = spellings.len() as TokenId;
            seen.insert(word.clone());
            spellings.push(word);
            rules.push(Rule {
                left,
                right,
                output,
            });
            pairs.insert((left, right), (i, output));
        }
        Ok(Self {
            alphabet: alphabet.to_vec(),
            byte_ids,
            rules,
            pairs,
            spellings,
            spelling_size,
        })
    }

    /// A dictionary over all 256 byte values, so token ID `b` is byte `b`.
    pub fn byte_dictionary(
        merges: &[(TokenId, TokenId)],
        max_spelling_bytes: usize,
    ) -> Result<Self, Error> {
        Self::new(&(0..=255).collect::<Vec<_>>(), merges, max_spelling_bytes)
    }
    pub fn alphabet(&self) -> &[u8] {
        &self.alphabet
    }
    pub fn rules(&self) -> &[Rule] {
        &self.rules
    }
    pub fn token_count(&self) -> usize {
        self.spellings.len()
    }
    /// Total length of all token spellings in bytes.
    pub fn spelling_size(&self) -> usize {
        self.spelling_size
    }
    pub fn spelling(&self, token: TokenId) -> Option<&[u8]> {
        self.spellings.get(token as usize).map(Vec::as_slice)
    }
    pub fn atom(&self, byte: u8) -> Option<TokenId> {
        self.byte_ids[byte as usize]
    }
    pub(crate) fn raw_len(&self, token: TokenId) -> usize {
        self.spellings[token as usize].len()
    }

    /// Literal BPE: every rule in rank order, one left-to-right pass each.
    /// Independent of the incremental algorithm; O(input length × rules).
    pub fn tokenize_reference(&self, input: &[u8]) -> Result<Vec<TokenId>, StreamError> {
        let mut tokens = self.atoms(input)?;
        for rule in &self.rules {
            let mut next = Vec::with_capacity(tokens.len());
            let mut i = 0;
            while i < tokens.len() {
                if i + 1 < tokens.len() && tokens[i] == rule.left && tokens[i + 1] == rule.right {
                    next.push(rule.output);
                    i += 2;
                } else {
                    next.push(tokens[i]);
                    i += 1;
                }
            }
            tokens = next;
        }
        Ok(tokens)
    }

    /// Offline BPE that repeatedly merges the lowest-ranked adjacent pair,
    /// leftmost first. For a normalized dictionary it equals
    /// [`Dictionary::tokenize_reference`], in O(n log n) time for n bytes.
    pub fn tokenize_heap(&self, input: &[u8]) -> Result<Vec<TokenId>, StreamError> {
        Ok(heap_tokenize(&self.atoms(input)?, &self.pairs))
    }

    fn atoms(&self, input: &[u8]) -> Result<Vec<TokenId>, StreamError> {
        input
            .iter()
            .map(|&a| self.atom(a).ok_or(StreamError::UnknownByte(a)))
            .collect()
    }
}

// Heap-based BPE. Ordering by rank and then by left position gives left-to-right
// self-pair ties. Entries check both token identities and adjacency before
// merging, so stale heap entries cannot act on merged positions.
fn heap_tokenize(
    input: &[TokenId],
    pairs: &BTreeMap<(TokenId, TokenId), (usize, TokenId)>,
) -> Vec<TokenId> {
    let n = input.len();
    if n == 0 {
        return Vec::new();
    }
    let mut tokens = input.to_vec();
    let mut prev: Vec<_> = (0..n).map(|i| i.checked_sub(1)).collect();
    let mut next: Vec<_> = (0..n).map(|i| (i + 1 < n).then_some(i + 1)).collect();
    let mut alive = vec![true; n];
    let mut heap = BinaryHeap::new();
    let offer = |i: usize, j: usize, tokens: &[TokenId], heap: &mut BinaryHeap<_>| {
        if let Some(&(rank, out)) = pairs.get(&(tokens[i], tokens[j])) {
            heap.push(Reverse((rank, i, j, tokens[i], tokens[j], out)));
        }
    };
    for i in 0..n - 1 {
        offer(i, i + 1, &tokens, &mut heap);
    }
    while let Some(Reverse((_, i, j, x, y, z))) = heap.pop() {
        if !alive[i] || !alive[j] || next[i] != Some(j) || tokens[i] != x || tokens[j] != y {
            continue;
        }
        tokens[i] = z;
        alive[j] = false;
        next[i] = next[j];
        if let Some(k) = next[i] {
            prev[k] = Some(i);
            offer(i, k, &tokens, &mut heap);
        }
        if let Some(k) = prev[i] {
            offer(k, i, &tokens, &mut heap);
        }
    }
    let mut result = Vec::new();
    let mut i = Some(0);
    while let Some(j) = i {
        result.push(tokens[j]);
        i = next[j];
    }
    result
}
