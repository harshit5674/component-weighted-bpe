"""Independent check of the constructed family (Appendices C and D).

This script is separate from the Rust implementation.  It implements literal
BPE, Jiang and Gong's valid intervals, the strict-half centroid decomposition
and both branch searches, and checks every statement the appendix makes about
the family:

* every token is canonical (the dictionary is normalized);
* the candidate tree of S is T_k, with parent map pi;
* at the root of each T_j, the branch intervals come in the order
  U_{q-1}, ..., U_b, with weights 1, ..., 1, 2^(j-1);
* both searches return the literal-BPE last token on every prefix of @S;
* on the final update, count-balanced search makes k(k+1)/2 + 1 probes and
  component-weighted search makes k.

Run:  python3 check_construction.py
"""

import sys

sys.setrecursionlimit(100000)


# ------------------------------------------------------------------ BPE
def bpe(text, rules):
    """Literal BPE: each rule in rank order, one left-to-right pass."""
    seq = list(text)
    for x, y in rules:
        out, i = [], 0
        while i < len(seq):
            if i + 1 < len(seq) and seq[i] == x and seq[i + 1] == y:
                out.append(x + y)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        seq = out
    return seq


def last_token(text, rules):
    return bpe(text, rules)[-1] if text else None


class Dictionary:
    """A ranked rule list; checks normalization and builds valid intervals."""

    def __init__(self, alphabet, rules):
        self.rules = rules
        self.vocab = set(alphabet)
        self.rank, self.pre, self.suc = {}, {}, {}
        for r, (x, y) in enumerate(rules):
            z = x + y
            assert x in self.vocab and y in self.vocab, ("operand not yet created", x, y)
            assert z not in self.vocab, ("duplicate spelling", z)
            self.vocab.add(z)
            self.rank[z], self.pre[z], self.suc[z] = r, x, y
        for z in self.rank:
            assert bpe(z, rules) == [z], ("token is not canonical", z)

        # Successor forest; children visited from lowest to highest priority.
        self.kids = {v: [] for v in self.vocab}
        for z in self.rank:
            self.kids[self.suc[z]].append(z)
        for v in self.kids:
            self.kids[v].sort(key=lambda z: -self.rank[z])
        self.dfs_in, self.dfs_out = {}, {}
        clock = [1]  # 0 is reserved for the empty prefix

        def visit(v):
            self.dfs_in[v] = clock[0]
            clock[0] += 1
            for c in self.kids[v]:
                visit(c)
            self.dfs_out[v] = clock[0]

        for a in sorted(alphabet):
            visit(a)

        # Valid intervals (Jiang and Gong, Sec. 4.3).
        self.interval = {}
        for z in self.rank:
            x = self.pre[z]
            lo, hi = self.dfs_in[x], self.dfs_out[x]
            for c in self.kids[x]:
                if self.rank[c] <= self.rank[z]:
                    hi = self.dfs_in[c]
                    break
            self.interval[z] = (lo, hi)


# ------------------------------------------------- tree and search
def candidate_tree(D, tau):
    nodes = [tau[i:] for i in range(len(tau)) if tau[i:] in D.vocab]
    parent = {v: (D.suc[v] if v in D.rank else None) for v in nodes}
    return nodes, parent


def locate_count_balanced(ivs, x):
    """Predecessor search on lower endpoints, then a separate right-end probe."""
    lo, hi, probes = 0, len(ivs), 0
    while lo < hi:
        mid = lo + (hi - lo) // 2
        probes += 1
        if x < ivs[mid][0][0]:
            hi = mid
        else:
            lo = mid + 1
    if lo == 0:
        return None, probes
    probes += 1
    (_, right), node, _ = ivs[lo - 1]
    return (node if x < right else None), probes


def locate_weighted(ivs, x):
    """Weight-balanced tree: the root is the first interval reaching W/2."""
    probes = 0

    def go(lo, hi):
        nonlocal probes
        if lo >= hi:
            return None
        total = sum(w for _, _, w in ivs[lo:hi])
        acc, p = 0, lo
        for p in range(lo, hi):
            acc += ivs[p][2]
            if 2 * acc >= total:
                break
        probes += 1
        (left, right), node, _ = ivs[p]
        if x < left:
            return go(lo, p)
        if x >= right:
            return go(p + 1, hi)
        return node

    return go(0, len(ivs)), probes


def centroid_search(D, text, tau, weighted):
    """Returns (answer, probes, trace of (centroid, branch list) per step)."""
    nodes, parent = candidate_tree(D, tau)
    children = {v: [] for v in nodes}
    for v in nodes:
        if parent[v] is not None:
            children[parent[v]].append(v)

    def coord(drop):
        h = last_token(text[: len(text) - drop], D.rules)
        return 0 if h is None else D.dfs_in[h]

    def valid(v):
        if v not in D.rank:
            return True
        lo, hi = D.interval[v]
        return lo <= coord(len(D.suc[v])) < hi

    def size_in(v, comp):
        return 1 + sum(size_in(c, comp) for c in children[v] if c in comp)

    def piece(v, rest):
        seen, todo = set(), [v]
        while todo:
            w = todo.pop()
            if w in seen or w not in rest:
                continue
            seen.add(w)
            todo.extend(children[w])
            if parent[w] is not None:
                todo.append(parent[w])
        return seen

    comp, probes, trace = set(nodes), 0, []
    while True:
        u = next(v for v in comp if parent[v] not in comp)  # top node
        while True:  # strict-half descent
            heavy = [c for c in children[u] if c in comp and 2 * size_in(c, comp) > len(comp)]
            if not heavy:
                break
            u = heavy[0]
        rest = comp - {u}
        if not valid(u):
            trace.append((u, None))
            comp = piece(parent[u], rest)
            continue
        down = sorted((c for c in children[u] if c in comp), key=lambda c: D.interval[c][0])
        ivs = [(D.interval[c], c, len(piece(c, rest))) for c in down]
        for (a, _, _), (b, _, _) in zip(ivs, ivs[1:]):
            assert a[1] <= b[0], "sibling intervals overlap"
        trace.append((u, ivs))
        hit, p = (locate_weighted if weighted else locate_count_balanced)(ivs, coord(len(u)))
        probes += p
        if hit is None:
            return u, probes, trace
        comp = piece(hit, rest)


# ------------------------------------------------------ the family
def levels(k):
    N = 2 ** k
    r = [2 ** j - 1 for j in range(k + 1)]
    pi = {N - 1: N - 1}
    for j in range(1, k + 1):
        for p in range(r[j - 1], r[j]):
            pi[p] = r[j]
    return N, r, pi


def build(k, letters, end="$", context="@"):
    """Appendix C over arbitrary letter tokens; returns (S, rules, pi, r, U)."""
    N, r, pi = levels(k)
    a = letters[: N - 1]
    P = lambda p, q: "".join(a[p:q])
    U = lambda p: "".join(a[p:]) + end
    rules = []
    for j in range(1, k + 1):  # block rules, right to left
        b, q = r[j - 1], r[j]
        for p in range(q - 2, b - 1, -1):
            rules.append((a[p], P(p + 1, q)))
    rules.append((context, a[0]))  # context rule
    for p in range(N - 2, -1, -1):  # suffix rules, shortest first
        rules.append((P(p, pi[p]), U(pi[p])))
    return U(0), rules, pi, r, U


def codeword_rules(k):
    """Appendix D: codewords [bin_k(p)] assembled before everything else."""
    N = 2 ** k
    codes = ["[" + format(p, "0%db" % k) + "]" for p in range(N - 1)]
    rules, made = [], set()
    for length in range(2, k + 2):
        for c in codes:
            suffix = c[1:][-length:]
            if suffix not in made:
                made.add(suffix)
                rules.append((suffix[0], suffix[1:]))
    for c in codes:
        rules.append(("[", c[1:]))
    return codes, rules


def distinct_pair_letters(count, alphabet):
    """A walk with distinct adjacent pairs (Hierholzer on the de Bruijn graph)."""
    out = {x: list(alphabet) for x in alphabet}
    stack, path = [alphabet[0]], []
    while stack:
        v = stack[-1]
        if out[v]:
            stack.append(out[v].pop())
        else:
            path.append(stack.pop())
    seq = path[::-1][:count]
    assert len(seq) == count and len(set(zip(seq, seq[1:]))) == count - 1
    return seq


def check(k, S, rules, alphabet, pi, r, U, t_expected):
    D = Dictionary(alphabet, rules)
    N = 2 ** k
    assert len(S) == t_expected

    # The candidate tree of S is T_k with parent map pi.
    nodes, parent = candidate_tree(D, S)
    assert sorted(nodes, key=len) == [U(p) for p in range(N - 1, -1, -1)]
    for p in range(N - 1):
        assert parent[U(p)] == U(pi[p]), ("wrong link", p)

    # Literal BPE agrees with both searches on every prefix of @S.
    text = "@" + S
    for i in range(1, len(text) + 1):
        prefix = text[:i]
        want = last_token(prefix, rules)
        tau = max((prefix[j:] for j in range(i) if prefix[j:] in D.vocab), key=len)
        for weighted in (False, True):
            got = centroid_search(D, prefix, tau, weighted)[0]
            assert got == want, (k, prefix, want, got)

    # The forced update: heavy path, ordered intervals, exact probe counts.
    ans_c, probes_c, trace_c = centroid_search(D, text, S, False)
    ans_w, probes_w, trace_w = centroid_search(D, text, S, True)
    assert ans_c == ans_w == U(1) == last_token(text, rules)
    assert [u for u, _ in trace_c] == [U(r[j]) for j in range(k, 0, -1)]
    assert [u for u, _ in trace_w] == [u for u, _ in trace_c]
    for (u, ivs), j in zip(trace_c, range(k, 0, -1)):
        b, q = r[j - 1], r[j]
        assert [node for _, node, _ in ivs] == [U(p) for p in range(q - 1, b - 1, -1)]
        assert [w for _, _, w in ivs] == [1] * (q - 1 - b) + [2 ** (j - 1)]
    assert probes_c == k * (k + 1) // 2 + 1, (k, probes_c)
    assert probes_w == k, (k, probes_w)
    return probes_c, probes_w


def main():
    for k in range(1, 7):
        letters = [chr(0x4E00 + i) for i in range(2 ** k - 1)]
        S, rules, pi, r, U = build(k, letters)
        pc, pw = check(k, S, rules, set(letters) | {"$", "@"}, pi, r, U, 2 ** k)
        print(f"distinct letters  k={k}: t={len(S):5d}  count-balanced {pc:2d}  weighted {pw}")
    for k, m in [(3, 3), (4, 4), (5, 6), (6, 8)]:
        letters = distinct_pair_letters(2 ** k - 1, [chr(ord("a") + i) for i in range(m)])
        S, rules, pi, r, U = build(k, letters)
        pc, pw = check(k, S, rules, set(letters) | {"$", "@"}, pi, r, U, 2 ** k)
        print(f"repeated letters  k={k}: t={len(S):5d}  count-balanced {pc:2d}  weighted {pw}"
              f"  ({''.join(letters)})")
    for k in range(1, 5):
        codes, code_rules = codeword_rules(k)
        S, rules, pi, r, U = build(k, codes)
        t = (2 ** k - 1) * (k + 2) + 1
        pc, pw = check(k, S, code_rules + rules, set("01[]$@"), pi, r, U, t)
        print(f"fixed alphabet    k={k}: t={len(S):5d}  count-balanced {pc:2d}  weighted {pw}")
    print("all checks passed")


if __name__ == "__main__":
    main()
