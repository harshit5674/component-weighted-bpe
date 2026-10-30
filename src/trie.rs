use crate::budget::Budget;
use crate::error::checked_product;
use crate::Error;
use std::collections::{BTreeMap, VecDeque};

/// Aho–Corasick automaton over token spellings with completed transitions:
/// `go[v * m + a]` is the next state, so matching never walks failure links.
pub(crate) struct Trie {
    pub go: Vec<usize>,
    pub failure: Vec<usize>,
    /// States in breadth-first order, so a state's failure target comes first.
    pub order: Vec<usize>,
}

impl Trie {
    /// Returns one terminal state per input pattern, including duplicates.
    pub fn build(
        patterns: &[Vec<usize>],
        m: usize,
        budget: &mut Budget,
    ) -> Result<(Self, Vec<usize>), Error> {
        let row = m.checked_add(8).ok_or(Error::Overflow)?;
        let mut children = vec![BTreeMap::new()];
        let mut terminals = Vec::new();
        for pattern in patterns {
            let mut v = 0;
            for &a in pattern {
                if a >= m {
                    return Err(Error::Invariant("trie symbol outside alphabet"));
                }
                if let Some(&u) = children[v].get(&a) {
                    v = u;
                    continue;
                }
                // Charge the node's dense row and metadata before extending the
                // trie. Rows are allocated once below, without quadratic growth.
                budget.charge::<usize>(row)?;
                let u = children.len();
                children[v].insert(a, u);
                children.push(BTreeMap::new());
                v = u;
            }
            terminals.push(v);
        }
        budget.charge::<usize>(row)?;
        let n = children.len();
        let mut go = vec![0; checked_product(n, m)?];
        let mut failure = vec![0; n];
        let mut queue = VecDeque::from([0]);
        let mut order = Vec::with_capacity(n);
        while let Some(v) = queue.pop_front() {
            order.push(v);
            for a in 0..m {
                if let Some(&u) = children[v].get(&a) {
                    go[v * m + a] = u;
                    failure[u] = if v == 0 { 0 } else { go[failure[v] * m + a] };
                    queue.push_back(u);
                } else if v != 0 {
                    go[v * m + a] = go[failure[v] * m + a];
                }
            }
        }
        Ok((Self { go, failure, order }, terminals))
    }
}
