"""Index every HT-Step tar on HF by reading tar headers over HTTP range requests (no video data is downloaded).

-> metadata/tar_index.csv: tar_path, member_name, video_id, offset_data, size
-> metadata/tar_index_tars.json: {tar_path: {size, n_members}} (resume: a tar with the same size is not re-read)
Cross-checks the result against the repo's own (stale) manifest.json and htstep-feats dl/done/<n>.json.
Run first: python -I -X utf8 tools/build_tar_index.py   (HF_TOKEN from env if the repo is private)
"""
import csv
import json
import os
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_download, hf_hub_url

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "metadata"
REPO, FEATS = "nguyenminh04/htstep-videos", "nguyenminh04/htstep-feats"
TOKEN = os.environ.get("HF_TOKEN")
VIDEO_EXT = {".mp4", ".mkv", ".webm"}


def range_reader(url):
    """Resolve the HF redirect once, then read byte ranges straight from the CDN (one request per tar header)."""
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {TOKEN}"} if TOKEN else {})
    r = s.head(url, allow_redirects=False, timeout=60)
    loc = r.headers.get("Location") if r.is_redirect else url
    s.headers.pop("Authorization", None)                 # do not send the token to the CDN

    def read(off, n):
        for attempt in range(5):
            try:
                g = s.get(loc, headers={"Range": f"bytes={off}-{off + n - 1}"}, timeout=60)
                g.raise_for_status()
                assert g.status_code == 206, g.status_code
                return g.content
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
    return read


def read_headers(read):
    """[(member_name, offset_data, size)] of regular files; handles GNU long names and pax path records."""
    out, off, pending = [], 0, None
    while True:
        buf = read(off, 512)
        if len(buf) < 512 or buf == b"\0" * 512:
            return out
        ti = tarfile.TarInfo.frombuf(buf, "utf-8", "surrogateescape")
        data, nxt = off + 512, off + 512 + -(-ti.size // 512) * 512
        if ti.type == tarfile.GNUTYPE_LONGNAME:
            pending = read(data, ti.size).rstrip(b"\0").decode("utf-8")
        elif ti.type == tarfile.XHDTYPE:
            for rec in read(data, ti.size).decode("utf-8").splitlines():
                k, _, v = rec.partition(" ")[2].partition("=")
                if k == "path":
                    pending = v
        elif ti.type != tarfile.XGLTYPE:
            if ti.isreg():
                out.append((pending or ti.name, data, ti.size))
            pending = None
        off = nxt


def video_id(name):
    """Rule: basename without extension, e.g. 'abc123.mp4' or 'videos/abc123.mp4' -> 'abc123'. None for non-video members."""
    p = Path(name)
    return p.stem if p.suffix.lower() in VIDEO_EXT else None


def main():
    assert video_id("x/abc_D-1.mp4") == "abc_D-1" and video_id("meta.json") is None
    api = HfApi(token=TOKEN)
    rev = api.dataset_info(REPO).sha
    infos = {p.path: p.size for p in api.get_paths_info(REPO, [f.path for f in api.list_repo_tree(
        REPO, "videos", repo_type="dataset", revision=rev) if f.path.endswith(".tar")], repo_type="dataset", revision=rev)}
    side_p, csv_p = OUT / "tar_index_tars.json", OUT / "tar_index.csv"
    side = json.loads(side_p.read_text(encoding="utf-8"))["tars"] if side_p.exists() else {}
    rows = {}
    if csv_p.exists():
        with open(csv_p, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                rows.setdefault(r["tar_path"], []).append(r)
    todo = sorted(t for t, s in infos.items() if side.get(t, {}).get("size") != s)
    print(f"revision {rev[:8]}: {len(infos)} tar, {len(todo)} cần đọc header", flush=True)

    for t in set(rows) - set(infos):                     # tar removed from the repo
        rows.pop(t)
        side.pop(t, None)

    def save():
        """Rewrite both files after every tar (resume after an interrupt)."""
        with open(csv_p, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, ["tar_path", "member_name", "video_id", "offset_data", "size"], lineterminator="\n")
            w.writeheader()
            w.writerows(r for t in sorted(rows) if t in side for r in rows[t])
        side_p.write_text(json.dumps({"revision": rev, "tars": {t: side[t] for t in sorted(side)}}, ensure_ascii=False,
                                     indent=1) + "\n", encoding="utf-8", newline="\n")

    t0 = time.time()
    with ThreadPoolExecutor(24) as ex:
        futs = {ex.submit(lambda t: read_headers(range_reader(hf_hub_url(REPO, t, repo_type="dataset", revision=rev))), t): t
                for t in todo}
        for i, fu in enumerate(as_completed(futs), 1):
            t = futs[fu]
            mem = fu.result()
            end = max((o + s for _, o, s in mem), default=0)
            assert end <= infos[t], (t, end, infos[t])
            rows[t] = [{"tar_path": t, "member_name": n, "video_id": video_id(n), "offset_data": o, "size": s}
                       for n, o, s in mem]
            side[t] = {"size": infos[t], "n_members": len(mem)}
            save()
            print(f"  {i}/{len(todo)} {t}: {len(mem)} member, {time.time() - t0:.0f}s", flush=True)
    save()
    flat = [r for t in sorted(rows) for r in rows[t]]

    # cross-check with the repo's own lists
    odd = [r["member_name"] for r in flat if not r["video_id"]]
    by_vid = {}
    for r in flat:
        if r["video_id"]:
            by_vid.setdefault(r["video_id"], []).append(r["tar_path"])
    dup = {v: ts for v, ts in by_vid.items() if len(ts) > 1}
    listed = {v: s for s, d in json.load(open(hf_hub_download(REPO, "manifest.json", repo_type="dataset", token=TOKEN),
                                              encoding="utf-8"))["shards"].items() for v in d["videos"]}
    for t in sorted(set(infos) - set(listed.values())):
        n = t.rsplit("_", 1)[1][:-4]
        try:
            for v in json.load(open(hf_hub_download(FEATS, f"dl/done/{n}.json", repo_type="dataset", token=TOKEN),
                                    encoding="utf-8"))["ok"]:
                listed[v] = t
        except Exception as e:
            print("  không có done json cho", t, type(e).__name__)
    rep = {"revision": rev, "tars": len(infos), "members": len(flat), "videos": len(by_vid),
           "non_video_members": odd[:20], "n_non_video_members": len(odd), "videos_in_several_tars": dict(list(dup.items())[:20]),
           "n_videos_in_several_tars": len(dup),
           "repo_lists_vs_headers": {"listed": len(listed), "listed_not_in_headers": sum(v not in by_vid for v in listed),
                                     "headers_not_listed": sum(v not in listed for v in by_vid),
                                     "different_tar": sum(v in by_vid and t not in by_vid[v] for v, t in listed.items())}}
    (OUT / "tar_index_report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1) + "\n", encoding="utf-8",
                                               newline="\n")
    print(json.dumps(rep, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
