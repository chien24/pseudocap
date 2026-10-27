"""Mark which manifest videos exist -> metadata/availability.csv, and has_video in manifest.jsonl.

HT-Step: the video must be a member of its hf_path tar in metadata/tar_index.csv (tar headers, no download).
YC2: videos are on Drive, which is not visible locally; the local mirror D:\\Học\\KL\\Data\\YouCookII\\videos is used as proxy
(file names only) and notebook cell 1 re-checks every path on Drive.
Run after build_manifest.py: python -I -X utf8 tools/check_availability.py
"""
import csv
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "metadata" / "manifest.jsonl"
YC2_LOCAL_VIDEOS = ROOT.parents[1] / "YouCookII" / "videos"


def main():
    with open(MANIFEST, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    with open(ROOT / "metadata" / "tar_index.csv", encoding="utf-8", newline="") as f:
        member = {(r["tar_path"], r["video_id"]): int(r["size"]) for r in csv.DictReader(f) if r["video_id"]}
    tar_size = json.loads((ROOT / "metadata" / "tar_index_tars.json").read_text(encoding="utf-8"))["tars"]
    local_yc2 = set(os.listdir(YC2_LOCAL_VIDEOS))

    status = {}
    for r in rows:
        if r["dataset"] == "htstep":
            ok = (r["hf_path"], r["video_id"]) in member
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
    need = {(path, vid) for (ds, vid), (_, path, ok) in status.items() if ds == "htstep" and ok}
    tars = sorted({t for t, _ in need})
    gb_tar, gb_vid = sum(tar_size[t]["size"] for t in tars) / 1e9, sum(member[k] for k in need) / 1e9
    print(f"htstep val: cần {len(tars)} tar, tổng {gb_tar:.1f} GB; video val trong đó {gb_vid:.1f} GB "
          f"(tỉ lệ {gb_tar / gb_vid:.1f}x; > 3x thì notebook đọc range từng video)")
    for t in tars:
        print(f"  {t}: {sum(x == t for x, _ in need)} video val / {tar_size[t]['n_members']} member")


if __name__ == "__main__":
    main()
