"""Mark which manifest videos exist -> metadata/availability.csv, and has_video in manifest.jsonl.

HT-Step: the hf_path tar must be in the HF repo file list (names only, no download).
YC2: videos are on Drive, which is not visible locally; the local mirror D:\\Học\\KL\\Data\\YouCookII\\videos is used as proxy
(file names only) and notebook cell 1 re-checks every path on Drive.
Run after build_manifest.py: python -I -X utf8 tools/check_availability.py
"""
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "metadata" / "manifest.jsonl"
YC2_LOCAL_VIDEOS = ROOT.parents[1] / "YouCookII" / "videos"


def main():
    with open(MANIFEST, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    repo_files = set(HfApi().list_repo_files("nguyenminh04/htstep-videos", repo_type="dataset"))
    local_yc2 = set(os.listdir(YC2_LOCAL_VIDEOS))

    status = {}
    for r in rows:
        if r["dataset"] == "htstep":
            ok = r["hf_path"] is not None and r["hf_path"] in repo_files
        else:
            ok = Path(r["hf_path"]).name in local_yc2
        status[(r["dataset"], r["video_id"])] = (r["split"], r["hf_path"], ok)
        r["has_video"] = ok

    with open(ROOT / "metadata" / "availability.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "split", "video_id", "hf_path", "status"])
        for (ds, vid), (split, path, ok) in sorted(status.items()):
            w.writerow([ds, split, vid, path or "", "ok" if ok else "missing"])
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    vids, segs = defaultdict(Counter), defaultdict(Counter)
    for (ds, _), (split, _, ok) in status.items():
        vids[f"{ds}/{split}"][ok] += 1
    for r in rows:
        segs[f"{r['dataset']}/{r['split']}"][r["has_video"]] += 1
    for k in sorted(vids):
        v, s = vids[k], segs[k]
        print(f"{k}: videos {v[True]}/{v[True] + v[False]} ({v[True] / (v[True] + v[False]):.1%}), "
              f"segments {s[True]}/{s[True] + s[False]} ({s[True] / (s[True] + s[False]):.1%})")


if __name__ == "__main__":
    main()
