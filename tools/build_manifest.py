"""Build metadata/manifest.jsonl (YC2 dev+val, HT-Step val), yc2_train_text.jsonl, yc2_dev_val_split.json.

Text only: reads annotation files and the HF repo's shard lists (small json), never a video.
Run: python -I -X utf8 tools/build_manifest.py
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "metadata"
HTSTEP = ROOT.parent / "htstep" / "data"
YC2_META = ROOT.parents[1] / "YouCookII" / "metadata"
YC2_FOOD = ROOT.parents[1] / "data" / "label_foodtype.csv"            # official YC2 recipe_type -> name
HF_VIDEOS, HF_FEATS = "nguyenminh04/htstep-videos", "nguyenminh04/htstep-feats"
YC2_REPO = "drive:Colab Notebooks/code KL"           # YC2 videos live on Drive, not HF (as in Fushion notebooks)
YC2_PATH = "Omni/data/YouCookII/videos/{}.mp4"


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read_json(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def write_jsonl(p, rows):
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def htstep_shard_map():
    """video_id -> tar path. manifest.json is stale (95 of the shards); the rest come from htstep-feats dl/done/<n>.json."""
    api = HfApi()
    tars = sorted(f for f in api.list_repo_files(HF_VIDEOS, repo_type="dataset") if f.endswith(".tar"))
    shards = read_json(hf_hub_download(HF_VIDEOS, "manifest.json", repo_type="dataset"))["shards"]
    where = {v: s for s, d in shards.items() for v in d["videos"]}
    extra = [t for t in tars if t not in shards]
    for t in extra:
        n = t.rsplit("_", 1)[1][:-4]
        for v in read_json(hf_hub_download(HF_FEATS, f"dl/done/{n}.json", repo_type="dataset"))["ok"]:
            assert v not in where, (v, t)
            where[v] = t
    assert not set(shards) - set(tars), "manifest.json names a shard missing from the repo"
    return where, {"tars": len(tars), "in_manifest_json": len(shards), "from_done_json": extra}


def htstep_rows(where):
    anno = read_json(HTSTEP / "annotations.json")
    cand = read_json(HTSTEP / "video_candidate_steps.json")
    with open(HTSTEP / "taxonomy.csv", encoding="utf-8") as f:
        tax = {int(r["global_step_index"]): r for r in csv.DictReader(f)}
    rows = []
    for vid in sorted(v for v, a in anno.items() if a["subset"] == "val_seen"):
        a = anno[vid]
        steps = cand[vid]["candidate_step_ids"]
        assert cand[vid]["activity"] == a["activity"] and cand[vid]["variation"] == a["variation"], vid
        for i, s in enumerate(a["annotations"]):
            t = tax[s["id"]]
            assert t["activity"] == a["activity"] and s["id"] in steps, (vid, s["id"])
            rows.append({
                "key": f"htstep/val/{vid}/{i}", "dataset": "htstep", "split": "val",
                "video_id": vid, "youtube_id": vid, "hf_repo": HF_VIDEOS, "hf_path": where.get(vid),
                "seg_idx": i, "start": s["segment"][0], "end": s["segment"][1], "duration": a["duration"],
                "task": a["activity"], "variation": a["variation"], "step_id": s["id"],
                "headline": t["headline"].strip(), "description": t["paragraph"].strip().rstrip(";").strip(),
                "partial": s["partial"],
                # the step set of this video's task variation (video_candidate_steps.json), parallel lists
                "task_step_ids": steps, "task_headlines": [tax[k]["headline"].strip() for k in steps],
            })
    return rows


def yc2_rows():
    db = read_json(YC2_META / "youcookii_annotations_uemr.json")["database"]
    with open(YC2_FOOD, encoding="utf-8") as f:
        food = {r[0]: r[1].strip() for r in csv.reader(f) if r}
    rows, split = [], {}
    for name, file in [("dev", "train_omni_video.dev.jsonl"), ("val", "val_omni_video.jsonl")]:
        src = read_jsonl(YC2_META / file)
        split[name] = sorted({r["video"][:-4] for r in src})
        for r in src:
            vid, idx = r["id"].rsplit("_", 1)
            v = db[vid]
            gt = v["annotations"][int(idx)]
            assert r["video"] == vid + ".mp4" and gt["sentence"].strip() == r["text"], r["id"]   # 3 raw sentences have a leading space
            assert [float(x) for x in gt["segment"]] == r["timestamps"], r["id"]
            rows.append({
                "key": f"yc2/{name}/{vid}/{idx}", "dataset": "yc2", "split": name,
                "video_id": vid, "youtube_id": vid, "hf_repo": YC2_REPO, "hf_path": YC2_PATH.format(vid),
                "seg_idx": int(idx), "start": r["timestamps"][0], "end": r["timestamps"][1], "duration": v["duration"],
                "task": food[v["recipe_type"]], "recipe_type": v["recipe_type"],
                "gt_caption": r["text"],
            })
    train = [{"video_id": vid, "seg_idx": a["id"], "recipe_type": v["recipe_type"], "task": food[v["recipe_type"]],
              "sentence": a["sentence"]}
             for vid, v in sorted(db.items()) if v["subset"] == "training" for a in v["annotations"]]
    return rows, train, split, db


def checks(ht, yc, train, split, db):
    """A4: printed, and returned for NHAT_KY.md."""
    rep = {}
    anno = read_json(HTSTEP / "annotations.json")
    splits = read_json(HTSTEP / "video_splits.json")
    rep["htstep_val_videos"] = len({r["video_id"] for r in ht})
    rep["htstep_val_videos_video_splits.json"] = len(splits["val_seen"])
    rep["htstep_val_segments"] = len(ht)
    rep["htstep_subsets_in_annotations"] = dict(Counter(a["subset"] for a in anno.values()))
    rep["htstep_test_annotated"] = sum(v in anno for k in ("test_seen", "test_unseen", "val_unseen") for v in splits[k])
    rep["htstep_partial"] = dict(Counter(r["partial"] for r in ht))
    by_vid = defaultdict(list)
    for r in ht:
        by_vid[r["video_id"]].append(r)
    ov = sum(1 for segs in by_vid.values() for i, a in enumerate(segs) for b in segs[i + 1:]
             if min(a["end"], b["end"]) > max(a["start"], b["start"]))
    rep["htstep_overlapping_pairs"] = ov
    rep["htstep_videos_with_overlap"] = sum(
        any(min(a["end"], b["end"]) > max(a["start"], b["start"]) for i, a in enumerate(s) for b in s[i + 1:])
        for s in by_vid.values())
    rep["htstep_repeated_step_in_video"] = sum(
        len(s) - len({r["step_id"] for r in s}) for s in by_vid.values())
    rep["htstep_steps_per_task_median"] = sorted(len(r["task_step_ids"]) for r in ht)[len(ht) // 2]
    rep["start_ge_end"] = [r["key"] for r in ht + yc if not r["start"] < r["end"]]
    rep["htstep_missing_headline"] = [r["key"] for r in ht if not r["headline"]]
    rep["htstep_with_description"] = sum(bool(r["description"]) for r in ht)
    rep["yc2"] = {s: {"videos": len(split[s]), "segments": sum(r["split"] == s for r in yc)} for s in split}
    val_anno = sum(len(db[v]["annotations"]) for v in split["val"])
    rep["yc2_val_segments_in_annotations"] = val_anno       # val_omni_video.jsonl (UEMR) has one fewer
    have = {r["key"].rsplit("/", 2)[1] + "_" + r["key"].rsplit("/", 1)[1] for r in yc if r["split"] == "val"}
    rep["yc2_val_dropped_by_uemr"] = [f"{v}_{a['id']}" for v in split["val"] for a in db[v]["annotations"]
                                      if f"{v}_{a['id']}" not in have]
    train_vids = {t["video_id"] for t in train}
    rep["yc2_train"] = {"videos": len(train_vids), "captions": len(train)}
    rep["yc2_train_overlap_dev_val"] = sorted(train_vids & (set(split["dev"]) | set(split["val"])))
    rep["yc2_dev_overlap_val"] = sorted(set(split["dev"]) & set(split["val"]))
    rep["yc2_start_ge_end_or_zero_len"] = [r["key"] for r in yc if r["end"] - r["start"] <= 0]
    assert not rep["start_ge_end"] and not rep["htstep_missing_headline"]
    assert not rep["yc2_train_overlap_dev_val"] and not rep["yc2_dev_overlap_val"]
    assert rep["yc2"]["dev"] == {"videos": 104, "segments": 773} and rep["yc2"]["val"] == {"videos": 394, "segments": 3030}
    return rep


def main():
    where, shard_info = htstep_shard_map()
    ht = htstep_rows(where)
    yc, train, split, db = yc2_rows()
    rep = checks(ht, yc, train, split, db)
    rep["hf_shards"] = shard_info
    rep["hf_videos_total"] = len(where)

    write_jsonl(OUT / "manifest.jsonl", yc + ht)
    write_jsonl(OUT / "yc2_train_text.jsonl", train)
    split_out = {"dev": split["dev"], "val": split["val"],
                 "source": {"dev": "YouCookII/metadata/train_omni_video.dev.jsonl",
                            "val": "YouCookII/metadata/val_omni_video.jsonl",
                            "defined_in": "Code/Fushion/configs/final_protocol.json (dev 104/773, val 394/3030)"},
                 "sha256": {f: sha256(YC2_META / f) for f in
                            ("train_omni_video.dev.jsonl", "val_omni_video.jsonl", "youcookii_annotations_uemr.json")}}
    with open(OUT / "yc2_dev_val_split.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(split_out, f, ensure_ascii=False, indent=1)
    with open(OUT / "build_report.json", "w", encoding="utf-8", newline="\n") as f:
        json.dump(rep, f, ensure_ascii=False, indent=1)
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
