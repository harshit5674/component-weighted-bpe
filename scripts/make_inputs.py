#!/usr/bin/env python3
"""Make the synthetic inputs of Appendix E.

    python3 scripts/make_inputs.py [DIR]

Writes into DIR (default: data):

* agent-json.txt: 1,024 synthetic tool-call records in JSON, one per line,
  with queries in four languages (254,804 bytes);
* repeat-a.txt, repeat-space.txt and repeat-hyphen.txt: 131,072 bytes of one
  repeated byte.
"""
import json
import sys
from pathlib import Path

QUERIES = [
    'incremental tokenization and cache reuse',
    '如何保留所有前缀的分词结果？',
    'كيف يمكن تقليل زمن الاستجابة؟',
    'Comment mesurer la latence ?',
]


def main(argv):
    out = Path(argv[0] if argv else "data")
    out.mkdir(parents=True, exist_ok=True)
    records = [
        json.dumps(
            {
                "sequence": i,
                "tool": "search",
                "query": QUERIES[i % 4],
                "status": "ok",
                "results": [{"id": i * 3 + j, "score": (j + 1) / 7} for j in range(3)],
            },
            ensure_ascii=False,
        )
        for i in range(1024)
    ]
    (out / "agent-json.txt").write_text("\n".join(records) + "\n", encoding="utf-8")
    for name, byte in [("a", b"a"), ("space", b" "), ("hyphen", b"-")]:
        (out / f"repeat-{name}.txt").write_bytes(byte * 131072)


if __name__ == "__main__":
    main(sys.argv[1:])
