//! Jiang and Gong's centroid search with either branch search.
//!
//! Both modes share the Aho–Corasick matcher, the successor forest and its
//! validity intervals, the centroid decomposition and the history. Only the
//! search among a valid centroid's child intervals differs.

mod compiler;
mod search;
mod stream;

pub use stream::{CentroidStream, Prefix};

use crate::{Dictionary, TokenId};
use std::ops::Range;
use std::sync::Arc;

const NONE: usize = usize::MAX;

/// The search used among the child intervals of a valid centroid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CentroidSearch {
    /// Binary search over the sorted intervals (Jiang and Gong): O(log² t) per byte.
    CountBalanced,
    /// A search tree balanced by the size of the component each interval
    /// leads to: O(log t) per byte.
    ComponentWeighted,
}

#[derive(Debug, Clone)]
pub struct CentroidOptions {
    pub search: CentroidSearch,
    /// Budget for the cumulative compiled table payload. This is not a
    /// peak-memory bound: spellings, scratch maps, vector slack and allocator
    /// metadata are not charged.
    pub max_table_bytes: usize,
}

impl Default for CentroidOptions {
    fn default() -> Self {
        Self {
            search: CentroidSearch::CountBalanced,
            max_table_bytes: 256 * 1024 * 1024,
        }
    }
}

/// Sizes of the compiled tables.
#[derive(Debug, Clone)]
pub struct CentroidReport {
    pub search: CentroidSearch,
    pub max_token_length: usize,
    pub spelling_bytes: usize,
    pub automaton_states: usize,
    pub automaton_transition_cells: usize,
    pub centroid_nodes: usize,
    pub downward_intervals: usize,
    pub largest_suffix_tree: usize,
    pub max_centroid_levels: usize,
    pub max_downward_degree: usize,
    pub charged_table_bytes: usize,
    pub history_bytes_per_prefix: usize,
}

/// Per-append operation counts, available through
/// [`CentroidStream::append_profiled`]. A probe reads one stored interval. The
/// count-balanced search ends with a separate check of the predecessor's right
/// endpoint, which counts as one more probe even though the record was read.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct SearchStats {
    pub validity_checks: usize,
    pub history_queries: usize,
    pub centroid_visits: usize,
    pub downward_lookups: usize,
    pub interval_probes: usize,
    pub endpoint_comparisons: usize,
    pub upward_moves: usize,
    pub downward_moves: usize,
    /// Empty interval lists also count as a terminal miss, with zero probes.
    pub terminal_misses: usize,
    /// Length of the longest nonempty interval list searched in this append.
    pub largest_branch_list: usize,
    pub longest_suffix_fast_path: bool,
}

#[derive(Clone, Copy, Default)]
struct Token {
    dfs: usize,
    len: usize,
    right_len: usize,
    lo: usize,
    hi: usize,
}

struct Centroid {
    token: TokenId,
    up: usize,
    down: Range<usize>,
    weighted_root: usize,
    size: usize,
}

#[derive(Clone, Copy)]
struct Interval {
    lo: usize,
    hi: usize,
    target: usize,
    weight: usize,
}

#[derive(Clone, Copy)]
struct Links {
    left: usize,
    right: usize,
}

/// Immutable compiled tables, shareable between streams through `Arc`. All
/// online matching transitions are completed table reads; no failure-link walk
/// or suffix scan happens during an append.
pub struct CentroidCompiled {
    dictionary: Arc<Dictionary>,
    tokens: Vec<Token>,
    alphabet_size: usize,
    go: Vec<usize>,
    longest: Vec<usize>,
    roots: Vec<usize>,
    centroids: Vec<Centroid>,
    intervals: Vec<Interval>,
    /// Weighted-tree links; empty for the count-balanced search.
    links: Vec<Links>,
    report: CentroidReport,
}

impl CentroidCompiled {
    pub fn dictionary(&self) -> &Dictionary {
        &self.dictionary
    }
    pub fn report(&self) -> &CentroidReport {
        &self.report
    }
}

#[cfg(test)]
mod tests;
