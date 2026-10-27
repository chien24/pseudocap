"""YC2 train caption style -> metadata/yc2_stats.json (prompt uses median as target length, p90 as max).

Run: python -I -X utf8 tools/yc2_style_stats.py
"""
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def pct(xs, q):
    """Nearest-rank percentile on sorted ints."""
    return xs[min(len(xs) - 1, max(0, round(q / 100 * len(xs)) - 1))]


def main():
    with open(ROOT / "metadata" / "yc2_train_text.jsonl", encoding="utf-8") as f:
        caps = [json.loads(l)["sentence"].strip() for l in f if l.strip()]
    n = sorted(len(c.split()) for c in caps)
    first = Counter(c.split()[0].lower() for c in caps if c.split())
    stats = {
        "source": "metadata/yc2_train_text.jsonl (YC2 subset 'training' minus dev, 1229 videos)",
        "n_captions": len(caps),
        "words": {"median": pct(n, 50), "p10": pct(n, 10), "p90": pct(n, 90), "min": n[0], "max": n[-1],
                  "mean": round(sum(n) / len(n), 2)},
        "lowercase_rate": round(sum(c == c.lower() for c in caps) / len(caps), 4),
        "trailing_period_rate": round(sum(c.endswith(".") for c in caps) / len(caps), 4),
        "top30_first_words": first.most_common(30),
    }
    assert pct([1, 2, 3, 4], 50) == 2 and pct(list(range(1, 11)), 90) == 9
    with open(ROOT / "metadata" / "yc2_stats.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(stats, f, ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
