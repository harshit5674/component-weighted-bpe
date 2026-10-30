#!/usr/bin/env python3
"""Convert a Hugging Face byte-level BPE tokenizer.json into an IBPE1 file.

    python3 scripts/prepare_vocab.py TOKENIZER_JSON OUTPUT.ibpe [--audit]

The crate only accepts normalized dictionaries, so merges are kept in rank
order and a merge is dropped, and listed in the report, when its output
spelling already exists, when an operand is not a token, or when its spelling
does not tokenize to exactly its two operands under the earlier merges. The
semantics are continuous byte BPE: no regex pre-tokenization, normalization or
special tokens.

Writes OUTPUT.ibpe and a JSON report next to it (OUTPUT.import.json). With
--audit, the `tokenizers` package checks every merge's spelling against the
source model. Without it, only the standard library is needed.

IBPE1 format: b"IBPE1", alphabet length (u32 LE), alphabet bytes, rule count
(u32 LE), then (left, right) token-ID pairs (u32 LE). Token IDs 0..n-1 are
the alphabet bytes in order, and rule i produces ID n + i.
"""
import hashlib
import heapq
import json
import struct
import sys
from pathlib import Path


def byte_alphabet():
    """GPT-2's map from bytes to the printable characters used in vocab files."""
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    chars = values[:]
    other = [b for b in range(256) if b not in values]
    return dict(zip(values + other, map(chr, chars + list(range(256, 256 + len(other))))))


def heap_bpe(word, atoms, pairs):
    """Offline BPE: merge the lowest-ranked adjacent pair, leftmost first."""
    tokens = [atoms[b] for b in word]
    n = len(tokens)
    before, after = list(range(-1, n - 1)), list(range(1, n + 1))
    live = [True] * n
    heap = []

    def offer(i, j):
        if i >= 0 and j < n and (tokens[i], tokens[j]) in pairs:
            rank, out = pairs[tokens[i], tokens[j]]
            heapq.heappush(heap, (rank, i, j, tokens[i], tokens[j], out))

    for i in range(n - 1):
        offer(i, i + 1)
    while heap:
        _, i, j, x, y, out = heapq.heappop(heap)
        if not (live[i] and live[j] and after[i] == j and tokens[i] == x and tokens[j] == y):
            continue
        tokens[i], live[j], after[i] = out, False, after[j]
        if after[i] < n:
            before[after[i]] = i
            offer(i, after[i])
        offer(before[i], i)
    return [t for t, alive in zip(tokens, live) if alive]


def prepare(source, output, audit):
    document = json.loads(source.read_text())
    model = document["model"]
    if model.get("type", "BPE") != "BPE":
        raise SystemExit(f"{source}: not a BPE model")
    assert model.get("dropout") is None
    assert not model.get("continuing_subword_prefix") and not model.get("end_of_word_suffix")
    assert not model.get("byte_fallback", False) and not model.get("ignore_merges", False)
    encode = byte_alphabet()
    decode = {v: k for k, v in encode.items()}
    alphabet = bytes(b for b in range(256) if encode[b] in model["vocab"])
    atoms = {b: i for i, b in enumerate(alphabet)}
    words = [bytes([b]) for b in alphabet]
    ids = {w: i for i, w in enumerate(words)}
    merges, pairs, rejected = [], {}, []
    for rank, merge in enumerate(model["merges"]):
        x, y = merge.split(" ") if isinstance(merge, str) else merge
        x, y = bytes(decode[c] for c in x), bytes(decode[c] for c in y)
        z = x + y
        reason = None
        if z in ids:
            reason = "duplicate_spelling"
        elif x not in ids or y not in ids:
            reason = "noncanonical_operand"
        elif heap_bpe(z, atoms, pairs) != [ids[x], ids[y]]:
            reason = "noncanonical_generating_pair"
        if reason:
            rejected.append({"source_rank": rank + 1, "reason": reason, "spelling_hex": z.hex()})
            continue
        left, right = ids[x], ids[y]
        out = len(words)
        merges.append((left, right))
        pairs[left, right] = (len(merges), out)
        ids[z] = out
        words.append(z)

    payload = b"IBPE1" + struct.pack("<I", len(alphabet)) + alphabet + struct.pack("<I", len(merges))
    payload += b"".join(struct.pack("<II", x, y) for x, y in merges)
    output.write_bytes(payload)

    audited = 0
    if audit:
        import tokenizers

        source_ids = [model["vocab"]["".join(encode[b] for b in w)] for w in words]
        oracle = tokenizers.Tokenizer.from_file(str(source)).model
        # Every merge output, including dropped ones, must tokenize as in the
        # source model; sequences are compared through the original IDs.
        for merge in model["merges"]:
            x, y = merge.split(" ") if isinstance(merge, str) else merge
            raw = bytes(decode[c] for c in x + y)
            expected = [t.id for t in oracle.tokenize(x + y)]
            actual = [source_ids[t] for t in heap_bpe(raw, atoms, pairs)]
            assert actual == expected, ((x, y), actual, expected)
            audited += 1

    report = dict(
        source=source.name,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        prepared_sha256=hashlib.sha256(payload).hexdigest(),
        alphabet_bytes=len(alphabet),
        missing_bytes=[b for b in range(256) if b not in atoms],
        original_rules=len(model["merges"]),
        accepted_rules=len(merges),
        rejected_rules=rejected,
        source_model_audits=audited,
        spelling_bytes=sum(map(len, words)),
        max_token_length=max(map(len, words)),
        semantics="continuous byte BPE; no pre-tokenizer, normalizer or special tokens",
    )
    report_path = output.with_suffix(".import.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{output}: {len(merges)} of {len(model['merges'])} merges kept, "
          f"{len(rejected)} dropped, max token length {report['max_token_length']}, "
          f"{audited} merges audited; report in {report_path}")


def main(argv):
    args = [a for a in argv if a != "--audit"]
    if len(args) != 2:
        raise SystemExit(__doc__)
    prepare(Path(args[0]), Path(args[1]), "--audit" in argv)


if __name__ == "__main__":
    main(sys.argv[1:])
