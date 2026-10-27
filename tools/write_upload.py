"""Write metadata/UPLOAD.md: line counts + sha256 that notebook cell 1 checks after cloning the repo.

Run last (after build_manifest, check_availability, yc2_style_stats): python -I -X utf8 tools/write_upload.py
"""
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
META = ROOT / "metadata"
FILES = ["manifest.jsonl", "manifest_htstep_train.jsonl.gz", "yc2_train_text.jsonl", "yc2_dev_val_split.json",
         "availability.csv", "tar_index.csv", "tar_index_tars.json", "tar_order.txt", "yc2_stats.json"]
DRIVE = "MyDrive/Colab Notebooks/code KL/Fushion/htstep_cap"     # same root as the UEMR notebooks (.../code KL/Fushion/uemr)


def main():
    counts = {}
    for f in FILES:
        b = (META / f).read_bytes()
        t = gzip.decompress(b) if f.endswith(".gz") else b              # .gz: số dòng sau giải nén, sha256 của file nén
        counts[f] = [t.decode("utf-8").count("\n") + (not t.endswith(b"\n")), hashlib.sha256(b).hexdigest()]
    lines = [
        "# Metadata cho notebook",
        "",
        "Không cần upload: ô 1 clone `https://github.com/chien24/pseudocap.git` và đọc `metadata/` từ đó.",
        "Sửa metadata thì chạy lại tools rồi commit + push (trước khi khóa protocol).",
        f"Kết quả ghi lên Drive `{DRIVE}/` (cùng gốc `Colab Notebooks/code KL/Fushion/` với notebook UEMR).",
        "",
        "| File | Số dòng | sha256 |",
        "|---|---:|---|",
        *[f"| `{f}` | {n} | `{h[:16]}…` |" for f, (n, h) in counts.items()],
        "",
        "Notebook: upload `notebooks/htstep_pseudocap_AZ.ipynb` (hoặc mở từ GitHub trong Colab), GPU T4.",
        "",
        "Video không cần upload: YC2 đọc từ `Colab Notebooks/code KL/Omni/data/YouCookII/videos/<video_id>.mp4`",
        "(như các notebook UEMR), HT-Step lấy từ HF `nguyenminh04/htstep-videos` theo từng tar (`tar_index.csv`:",
        "tar, member, offset, size; revision ghim trong `tar_index_tars.json`). Tiến trình ở Drive `.../htstep_cap/progress/`.",
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
