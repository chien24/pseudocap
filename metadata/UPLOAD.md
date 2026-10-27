# Metadata cho notebook

Không cần upload: ô 1 clone `https://github.com/chien24/pseudocap.git` và đọc `metadata/` từ đó.
Sửa metadata thì chạy lại tools rồi commit + push (trước khi khóa protocol).
Kết quả ghi lên Drive `MyDrive/Colab Notebooks/code KL/Fushion/htstep_cap/` (cùng gốc `Colab Notebooks/code KL/Fushion/` với notebook UEMR).

| File | Số dòng | sha256 |
|---|---:|---|
| `manifest.jsonl` | 7244 | `bfb28391816e3f34…` |
| `manifest_htstep_train.jsonl.gz` | 103293 | `2fcc56e30aece99e…` |
| `yc2_train_text.jsonl` | 9564 | `ce42f78acbe29fcf…` |
| `yc2_dev_val_split.json` | 514 | `f0d5d2f7d45f0794…` |
| `availability.csv` | 18571 | `f86ddcc62063c2a3…` |
| `tar_index.csv` | 7454 | `6865b0772402ef2a…` |
| `tar_index_tars.json` | 457 | `35934323f1f43572…` |
| `tar_order.txt` | 113 | `c56e569b89f9e46c…` |
| `yc2_stats.json` | 136 | `0e90590cac8c2014…` |

Notebook: upload `notebooks/htstep_pseudocap_AZ.ipynb` (hoặc mở từ GitHub trong Colab), GPU T4.

Video không cần upload: YC2 đọc từ `Colab Notebooks/code KL/Omni/data/YouCookII/videos/<video_id>.mp4`
(như các notebook UEMR), HT-Step lấy từ HF `nguyenminh04/htstep-videos` theo từng tar (`tar_index.csv`:
tar, member, offset, size; revision ghim trong `tar_index_tars.json`). Tiến trình ở Drive `.../htstep_cap/progress/`.

Ô 1 của notebook đọc dòng dưới đây và kiểm số dòng + sha256 từng file:

<!-- COUNTS {"manifest.jsonl": [7244, "bfb28391816e3f3430e98219ebcfa86d1698007b4cfc7d40814823b7b6c99362"], "manifest_htstep_train.jsonl.gz": [103293, "2fcc56e30aece99e58c954b04c9e1a6a9141ab084009ee232fefb220f22ada0e"], "yc2_train_text.jsonl": [9564, "ce42f78acbe29fcf53d8198492f900a62ef6078eb6950a66d2c7eae61614120f"], "yc2_dev_val_split.json": [514, "f0d5d2f7d45f0794cffbb39b8a1f0240c1c9f7d0e6e196cab6af6f07316e522d"], "availability.csv": [18571, "f86ddcc62063c2a3b50a31a3bcdb552aec08605cfbf73d1cf8cbd255301736a4"], "tar_index.csv": [7454, "6865b0772402ef2aba945d8b2b919383272d8d5bfd317eea38fd641c03ff7853"], "tar_index_tars.json": [457, "35934323f1f43572fc73bfec3852917de207857470f20868931f97b4d13f4b3a"], "tar_order.txt": [113, "c56e569b89f9e46c2d4d9b9d977a094f8b0e3cfec90162ece42248dea0286533"], "yc2_stats.json": [136, "0e90590cac8c2014ce8fc48cb54302636540cc2246bdfb5114a5a283251e056b"]} -->
