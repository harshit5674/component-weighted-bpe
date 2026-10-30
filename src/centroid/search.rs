use super::stream::Record;
use super::*;

impl CentroidCompiled {
    pub(super) fn last<const WEIGHTED: bool, const TRACK: bool>(
        &self,
        matched: usize,
        history: &[Record],
        stats: &mut SearchStats,
    ) -> TokenId {
        // The longest matched suffix, if valid, must be the endpoint of the
        // monotonic path. Both modes use this upstream fast path identically.
        if self.tokens[matched].len == history.len() || self.valid::<TRACK>(matched, history, stats)
        {
            if TRACK {
                stats.longest_suffix_fast_path = true;
            }
            return matched as TokenId;
        }
        let mut current = self.roots[matched];
        loop {
            if TRACK {
                stats.centroid_visits += 1;
            }
            let node = &self.centroids[current];
            if !self.valid::<TRACK>(node.token as usize, history, stats) {
                // An invalid centroid excludes its whole descendant subtree.
                // The invariant guarantees an upward component exists here.
                assert_ne!(node.up, NONE, "invalid centroid without upward component");
                current = node.up;
                if TRACK {
                    stats.upward_moves += 1;
                }
                continue;
            }
            if TRACK {
                stats.downward_lookups += 1;
                stats.largest_branch_list = stats.largest_branch_list.max(node.down.len());
            }
            let next = if node.down.is_empty() {
                NONE
            } else {
                let x = query::<TRACK>(history, self.tokens[node.token as usize].len, stats);
                if WEIGHTED {
                    weighted::<TRACK>(&self.intervals, &self.links, node.weighted_root, x, stats)
                } else {
                    count_balanced::<TRACK>(&self.intervals, node.down.clone(), x, stats)
                }
            };
            if next == NONE {
                if TRACK {
                    stats.terminal_misses += 1;
                }
                return node.token;
            }
            if TRACK {
                stats.downward_moves += 1;
            }
            current = next;
        }
    }

    fn valid<const TRACK: bool>(
        &self,
        token: usize,
        history: &[Record],
        stats: &mut SearchStats,
    ) -> bool {
        if TRACK {
            stats.validity_checks += 1;
        }
        let t = self.tokens[token];
        if t.len == 1 {
            return true;
        }
        let x = query::<TRACK>(history, t.right_len, stats);
        t.lo <= x && x < t.hi
    }
}

fn query<const TRACK: bool>(history: &[Record], skip: usize, stats: &mut SearchStats) -> usize {
    if TRACK {
        stats.history_queries += 1;
    }
    // history includes the empty prefix but not the byte currently being added:
    // its length is precisely the new raw endpoint.
    history[history.len() - skip].dfs
}

pub(super) fn count_balanced<const TRACK: bool>(
    intervals: &[Interval],
    range: Range<usize>,
    x: usize,
    stats: &mut SearchStats,
) -> usize {
    let (mut lo, mut hi) = (range.start, range.end);
    // Explicit predecessor convention pins probe counts independently of future
    // standard-library binary_search implementation changes.
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        if TRACK {
            stats.interval_probes += 1;
            stats.endpoint_comparisons += 1;
        }
        if x < intervals[mid].lo {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    if lo == range.start {
        return NONE;
    }
    if TRACK {
        stats.interval_probes += 1;
        stats.endpoint_comparisons += 1;
    }
    let candidate = intervals[lo - 1];
    if x < candidate.hi {
        candidate.target
    } else {
        NONE
    }
}

pub(super) fn weighted<const TRACK: bool>(
    intervals: &[Interval],
    links: &[Links],
    mut current: usize,
    x: usize,
    stats: &mut SearchStats,
) -> usize {
    while current != NONE {
        let record = intervals[current];
        if TRACK {
            stats.interval_probes += 1;
            stats.endpoint_comparisons += 1;
        }
        if x < record.lo {
            current = links[current].left;
        } else {
            if TRACK {
                stats.endpoint_comparisons += 1;
            }
            if x >= record.hi {
                current = links[current].right;
            } else {
                return record.target;
            }
        }
    }
    NONE
}
