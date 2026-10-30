use super::*;
use crate::budget::Budget;
use crate::trie::Trie;
use crate::Error;

impl CentroidCompiled {
    pub fn compile(dictionary: Arc<Dictionary>, options: &CentroidOptions) -> Result<Self, Error> {
        let mut budget = Budget::new(options.max_table_bytes);
        let tokens = forest(&dictionary, &mut budget)?;
        let alphabet_size = dictionary.alphabet().len();
        let patterns: Vec<Vec<usize>> = (0..dictionary.token_count())
            .map(|t| {
                dictionary
                    .spelling(t as TokenId)
                    .unwrap()
                    .iter()
                    .map(|&b| dictionary.atom(b).unwrap() as usize)
                    .collect()
            })
            .collect();
        let (trie, terminals) = Trie::build(&patterns, alphabet_size, &mut budget)?;
        drop(patterns);
        let mut longest = budget.filled(trie.failure.len(), NONE)?;
        for (t, &v) in terminals.iter().enumerate() {
            longest[v] = t;
        }
        for &v in &trie.order {
            if v != 0 && longest[v] == NONE {
                longest[v] = longest[trie.failure[v]];
            }
        }
        // Strict longest token suffix links enumerate *all* suffix candidates.
        // They differ from successor links, which are generating right operands.
        let suffix: Vec<_> = terminals
            .iter()
            .map(|&v| longest[trie.failure[v]])
            .collect();
        let mut roots = budget.filled(dictionary.token_count(), NONE)?;
        let mut builder = Builder {
            tokens: &tokens,
            centroids: Vec::new(),
            intervals: Vec::new(),
            links: Vec::new(),
            weighted: options.search == CentroidSearch::ComponentWeighted,
            budget: &mut budget,
            max_levels: 0,
            max_degree: 0,
        };
        // Reuse a token-ID map rather than clearing a vocabulary-sized map for
        // every tree. Each suffix node is set once and cleared once per tree.
        let mut local_id = vec![NONE; dictionary.token_count()];
        let mut largest = 0;
        for (t, tree_root) in roots.iter_mut().enumerate() {
            let mut chain = Vec::new();
            let mut v = t;
            while v != NONE {
                local_id[v] = chain.len();
                chain.push(v);
                v = suffix[v];
            }
            largest = largest.max(chain.len());
            let mut children = vec![Vec::new(); chain.len()];
            for (i, &token) in chain.iter().enumerate() {
                if token >= alphabet_size {
                    let parent = dictionary.rules()[token - alphabet_size].right as usize;
                    let p = local_id[parent];
                    if p == NONE {
                        return Err(Error::Invariant("successor missing from suffix tree"));
                    }
                    children[p].push(i);
                }
            }
            let tree_len = chain.len();
            let mut tree = Tree {
                chain,
                children,
                removed: vec![false; tree_len],
                sizes: vec![0; tree_len],
            };
            *tree_root = builder.build(&mut tree, tree_len - 1, 1)?;
            for &token in &tree.chain {
                local_id[token] = NONE;
            }
        }
        let Builder {
            centroids,
            intervals,
            links,
            max_levels,
            max_degree,
            ..
        } = builder;
        let report = CentroidReport {
            search: options.search,
            max_token_length: tokens.iter().map(|t| t.len).max().unwrap(),
            spelling_bytes: dictionary.spelling_size(),
            automaton_states: longest.len(),
            automaton_transition_cells: trie.go.len(),
            centroid_nodes: centroids.len(),
            downward_intervals: intervals.len(),
            largest_suffix_tree: largest,
            max_centroid_levels: max_levels,
            max_downward_degree: max_degree,
            charged_table_bytes: budget.charged,
            history_bytes_per_prefix: std::mem::size_of::<super::stream::Record>(),
        };
        Ok(Self {
            dictionary,
            tokens,
            alphabet_size,
            go: trie.go,
            longest,
            roots,
            centroids,
            intervals,
            links,
            report,
        })
    }
}

fn forest(dictionary: &Dictionary, budget: &mut Budget) -> Result<Vec<Token>, Error> {
    let n = dictionary.token_count();
    let mut children = vec![Vec::new(); n];
    // Lower priority (larger rank) children are visited first. Token IDs of
    // non-atoms increase with rank, so this is the required decreasing ID order.
    for rule in dictionary.rules().iter().rev() {
        children[rule.right as usize].push(rule.output as usize);
    }
    let mut tokens = budget.filled(n, Token::default())?;
    let mut ends = vec![0; n];
    let mut clock = 1; // coordinate zero is the empty-prefix sentinel
    let mut stack = Vec::new();
    for root in 0..dictionary.alphabet().len() {
        stack.push((root, false));
        while let Some((v, exit)) = stack.pop() {
            if exit {
                ends[v] = clock;
            } else {
                tokens[v].dfs = clock;
                tokens[v].len = dictionary.raw_len(v as TokenId);
                clock = clock.checked_add(1).ok_or(Error::Overflow)?;
                stack.push((v, true));
                for &child in children[v].iter().rev() {
                    stack.push((child, false));
                }
            }
        }
    }
    for rule in dictionary.rules() {
        let v = rule.output as usize;
        let left = rule.left as usize;
        let boundary = children[left].partition_point(|&c| c > v);
        tokens[v].lo = tokens[left].dfs;
        tokens[v].hi = children[left]
            .get(boundary)
            .map_or(ends[left], |&c| tokens[c].dfs);
        tokens[v].right_len = tokens[rule.right as usize].len;
    }
    Ok(tokens)
}

struct Tree {
    chain: Vec<usize>,
    children: Vec<Vec<usize>>,
    removed: Vec<bool>,
    sizes: Vec<usize>,
}

struct Builder<'a, 'b> {
    tokens: &'a [Token],
    centroids: Vec<Centroid>,
    intervals: Vec<Interval>,
    links: Vec<Links>,
    weighted: bool,
    budget: &'b mut Budget,
    max_levels: usize,
    max_degree: usize,
}

impl Builder<'_, '_> {
    // Recursion is only along centroid levels (logarithmic), never token length.
    fn build(&mut self, tree: &mut Tree, root: usize, level: usize) -> Result<usize, Error> {
        let mut order = vec![root];
        let mut scan = 0;
        while scan < order.len() {
            let v = order[scan];
            tree.sizes[v] = 1;
            order.extend(
                tree.children[v]
                    .iter()
                    .copied()
                    .filter(|&c| !tree.removed[c]),
            );
            scan += 1;
        }
        for &v in order.iter().rev() {
            for &c in &tree.children[v] {
                if !tree.removed[c] {
                    tree.sizes[v] += tree.sizes[c];
                }
            }
        }
        let size = order.len();
        let mut center = root;
        // Strict-half convention: when two centroids exist, keep the one nearer
        // the original root. The constructed family (`witness`) relies on it.
        while let Some(c) = tree.children[center]
            .iter()
            .copied()
            .find(|&c| !tree.removed[c] && tree.sizes[c] > size / 2)
        {
            center = c;
        }
        let mut downward: Vec<_> = tree.children[center]
            .iter()
            .copied()
            .filter(|&c| !tree.removed[c])
            .map(|c| (c, tree.sizes[c]))
            .collect();
        downward.sort_by_key(|&(c, _)| self.tokens[tree.chain[c]].lo);
        for pair in downward.windows(2) {
            if self.tokens[tree.chain[pair[0].0]].hi > self.tokens[tree.chain[pair[1].0]].lo {
                return Err(Error::Invariant(
                    "overlapping suffix-sibling validity intervals",
                ));
            }
        }
        let up_size = size - tree.sizes[center];
        if up_size > size / 2 || downward.iter().any(|&(_, s)| s > size / 2) {
            return Err(Error::Invariant(
                "centroid child exceeds half the component",
            ));
        }
        tree.removed[center] = true;
        self.budget.charge::<Centroid>(1)?;
        self.budget.charge::<Interval>(downward.len())?;
        if self.weighted {
            self.budget.charge::<Links>(downward.len())?;
        }
        let id = self.centroids.len();
        let start = self.intervals.len();
        for &(c, weight) in &downward {
            let t = self.tokens[tree.chain[c]];
            self.intervals.push(Interval {
                lo: t.lo,
                hi: t.hi,
                target: NONE,
                weight,
            });
            if self.weighted {
                self.links.push(Links {
                    left: NONE,
                    right: NONE,
                });
            }
        }
        let end = self.intervals.len();
        let weighted_root = if self.weighted {
            weighted_links(
                &self.intervals[start..end],
                &mut self.links[start..end],
                start,
            )?
        } else {
            NONE
        };
        self.centroids.push(Centroid {
            token: tree.chain[center] as TokenId,
            up: NONE,
            down: start..end,
            weighted_root,
            size,
        });
        self.max_levels = self.max_levels.max(level);
        self.max_degree = self.max_degree.max(downward.len());
        drop(order);
        if up_size != 0 {
            let up = self.build(tree, root, level + 1)?;
            if self.centroids[up].size != up_size {
                return Err(Error::Invariant("upward component size"));
            }
            self.centroids[id].up = up;
        }
        for (offset, (c, weight)) in downward.into_iter().enumerate() {
            let target = self.build(tree, c, level + 1)?;
            if self.centroids[target].size != weight {
                return Err(Error::Invariant(
                    "interval weight differs from selected component size",
                ));
            }
            self.intervals[start + offset].target = target;
        }
        Ok(id)
    }
}

/// Ordered weighted medians, with strict sides at most half the total weight.
/// Links are global interval indices. Prefix sums are compilation scratch only.
pub(super) fn weighted_links(
    intervals: &[Interval],
    links: &mut [Links],
    base: usize,
) -> Result<usize, Error> {
    let mut sums = vec![0usize];
    for interval in intervals {
        if interval.weight == 0 {
            return Err(Error::Invariant("zero interval weight"));
        }
        sums.push(
            sums.last()
                .unwrap()
                .checked_add(interval.weight)
                .ok_or(Error::Overflow)?,
        );
    }
    fn build(sums: &[usize], links: &mut [Links], base: usize, lo: usize, hi: usize) -> usize {
        if lo == hi {
            return NONE;
        }
        let weight = sums[hi] - sums[lo];
        let half = sums[lo] + weight / 2 + weight % 2;
        let pivot = lo + sums[lo + 1..=hi].partition_point(|&x| x < half);
        links[pivot] = Links {
            left: build(sums, links, base, lo, pivot),
            right: build(sums, links, base, pivot + 1, hi),
        };
        base + pivot
    }
    Ok(build(&sums, links, base, 0, intervals.len()))
}
