"""Write metadata/UPLOAD.md: files to upload, Drive target, line counts + sha256 that notebook cell 1 checks.

Run last (after build_manifest, check_availability, yc2_style_stats): python -I -X utf8 tools/write_upload.py
"""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "metadata"
FILES = ["manifest.jsonl", "yc2_train_text.jsonl", "yc2_dev_val_split.json", "availability.csv", "yc2_stats.json"]
DRIVE = "MyDrive/Colab Notebooks/code KL/Fushion/htstep_cap"     # same root as the UEMR notebooks (.../code KL/Fushion/uemr)


def main():
    counts = {}
    for f in FILES:
        b = (META / f).read_bytes()
        counts[f] = [b.decode("utf-8").count("\n") + (not b.endswith(b"\n")), hashlib.sha256(b).hexdigest()]
    lines = [
        "# Upload lên Drive",
        "",
        f"Thư mục đích: `{DRIVE}/metadata/` (cùng gốc `Colab Notebooks/code KL/Fushion/` với notebook UEMR).",
        "",
        "| File | Số dòng | sha256 |",
        "|---|---:|---|",
        *[f"| `{f}` | {n} | `{h[:16]}…` |" for f, (n, h) in counts.items()],
        "| `UPLOAD.md` (file này) | | |",
        "",
        f"Notebook: `notebooks/htstep_pseudocap_AZ.ipynb` → `{DRIVE}/` rồi mở bằng Colab (GPU T4).",
        "",
        "Video không cần upload: YC2 đọc từ `Colab Notebooks/code KL/Omni/data/YouCookII/videos/<video_id>.mp4`",
        "(như các notebook UEMR), HT-Step tải từ HF `nguyenminh04/htstep-videos` (tar shard, cột `hf_path`).",
        "",
        "Ô 1 của notebook đọc dòng dưới đây và kiểm số dòng + sha256 từng file:",
        "",
        f"<!-- COUNTS {json.dumps(counts)} -->",
        "",
    ]
    (META / "UPLOAD.md").write_text("\n".join(lines), encoding="utf-8", newline="\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
