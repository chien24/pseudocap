"""Mark which manifest videos exist -> metadata/availability.csv, has_video in both manifests, tar_order.txt.

HT-Step: the video must be a member of its hf_path tar in metadata/tar_index.csv (tar headers, no download).
YC2: videos are on Drive, which is not visible locally; the local mirror D:\\Học\\KL\\Data\\YouCookII\\videos is used as proxy
(file names only) and notebook cell 1 re-checks every path on Drive.
tar_order.txt: tars holding HT-Step val videos first (cell 3, mode "extract"), then the rest by number (cell 8b, mode "full").
Run after build_manifest.py: python -I -X utf8 tools/check_availability.py
"""
import csv
import gzip
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "metadata"
MANIFEST, TRAIN = META / "manifest.jsonl", META / "manifest_htstep_train.jsonl.gz"
YC2_LOCAL_VIDEOS = ROOT.parents[1] / "YouCookII" / "videos"


def main():
    with open(MANIFEST, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    with gzip.open(TRAIN, "rt", encoding="utf-8") as f:
        train = [json.loads(l) for l in f if l.strip()]
    with open(META / "tar_index.csv", encoding="utf-8", newline="") as f:
        member = {(r["tar_path"], r["video_id"]): int(r["size"]) for r in csv.DictReader(f) if r["video_id"]}
    tar_size = json.loads((META / "tar_index_tars.json").read_text(encoding="utf-8"))["tars"]
    local_yc2 = set(os.listdir(YC2_LOCAL_VIDEOS))

    status = {}
    for r in rows + train:
        if r["dataset"] == "htstep":
            ok = (r["hf_path"], r["video_id"]) in member
        else:
            ok = Path(r["hf_path"]).name in local_yc2
        status[(r["dataset"], r["video_id"])] = (r["split"], r["hf_path"], ok)
        r["has_video"] = ok

    with open(META / "availability.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "split", "video_id", "hf_path", "status"])
        for (ds, vid), (split, path, ok) in sorted(status.items()):
            w.writerow([ds, split, vid, path or "", "ok" if ok else "missing"])
    with open(MANIFEST, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(TRAIN, "wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as f:
        f.write("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in train).encode("utf-8"))

    vids, segs = defaultdict(Counter), defaultdict(Counter)
    for (ds, _), (split, _, ok) in status.items():
        vids[f"{ds}/{split}"][ok] += 1
    for r in rows + train:
        segs[f"{r['dataset']}/{r['split']}"][r["has_video"]] += 1
    cov = {}
    for k in sorted(vids):
        v, s = vids[k], segs[k]
        cov[k] = {"videos": [v[True], v[True] + v[False]], "segments": [s[True], s[True] + s[False]]}
        print(f"{k}: videos {v[True]}/{v[True] + v[False]} ({v[True] / (v[True] + v[False]):.1%}), "
              f"segments {s[True]}/{s[True] + s[False]} ({s[True] / (s[True] + s[False]):.1%})")
    ht = [cov[k] for k in cov if k.startswith("htstep/")]
    tv, ts = [sum(c[x][0] for c in ht) for x in ("videos", "segments")], [sum(c[x][1] for c in ht) for x in ("videos", "segments")]
    print(f"htstep tổng (train + val): videos {tv[0]}/{ts[0]} ({tv[0] / ts[0]:.1%}), segments {tv[1]}/{ts[1]} ({tv[1] / ts[1]:.1%})")

    need = {(path, vid) for (ds, vid), (split, path, ok) in status.items() if ds == "htstep" and split == "val" and ok}
    tars = sorted({t for t, _ in need})
    gb_tar, gb_vid = sum(tar_size[t]["size"] for t in tars) / 1e9, sum(member[k] for k in need) / 1e9
    print(f"htstep val: cần {len(tars)} tar, tổng {gb_tar:.1f} GB; video val trong đó {gb_vid:.1f} GB "
          f"(tỉ lệ {gb_tar / gb_vid:.1f}x; > 3x thì notebook đọc range từng video)")
    for t in tars:
        n_other = sum(1 for (ds, vid), (split, path, ok) in status.items() if path == t and split != "val" and ok)
        print(f"  {t}: {sum(x == t for x, _ in need)} video val / {tar_size[t]['n_members']} member; "
              f"{n_other} video split khác (không vào lượt full)")
    order = tars + sorted(set(tar_size) - set(tars))
    (META / "tar_order.txt").write_text("".join(t + "\n" for t in order), encoding="utf-8", newline="\n")
    full = Counter(path for (ds, vid), (split, path, ok) in status.items() if ds == "htstep" and ok and path not in tars)
    print(f"tar_order.txt: {len(tars)} tar extract + {len(order) - len(tars)} tar full "
          f"({sum(full.values())} video train có video ngoài tar val)")


if __name__ == "__main__":
    main()
