# Metadata cho notebook

Không cần upload: ô 1 clone `https://github.com/chien24/pseudocap.git` và đọc `metadata/` từ đó.
Sửa metadata thì chạy lại tools rồi commit + push (trước khi khóa protocol).
Kết quả ghi lên Drive `MyDrive/Colab Notebooks/code KL/Fushion/htstep_cap/` (cùng gốc `Colab Notebooks/code KL/Fushion/` với notebook UEMR).

| File | Số dòng | sha256 |
|---|---:|---|
| `manifest.jsonl` | 7244 | `bfb28391816e3f34…` |
| `yc2_train_text.jsonl` | 9564 | `ce42f78acbe29fcf…` |
| `yc2_dev_val_split.json` | 514 | `f0d5d2f7d45f0794…` |
| `availability.csv` | 1099 | `7e107616cd12503f…` |
| `yc2_stats.json` | 136 | `0e90590cac8c2014…` |

Notebook: upload `notebooks/htstep_pseudocap_AZ.ipynb` (hoặc mở từ GitHub trong Colab), GPU T4.

Video không cần upload: YC2 đọc từ `Colab Notebooks/code KL/Omni/data/YouCookII/videos/<video_id>.mp4`
(như các notebook UEMR), HT-Step tải từ HF `nguyenminh04/htstep-videos` (tar shard, cột `hf_path`).

Ô 1 của notebook đọc dòng dưới đây và kiểm số dòng + sha256 từng file:

<!-- COUNTS {"manifest.jsonl": [7244, "bfb28391816e3f3430e98219ebcfa86d1698007b4cfc7d40814823b7b6c99362"], "yc2_train_text.jsonl": [9564, "ce42f78acbe29fcf53d8198492f900a62ef6078eb6950a66d2c7eae61614120f"], "yc2_dev_val_split.json": [514, "f0d5d2f7d45f0794cffbb39b8a1f0240c1c9f7d0e6e196cab6af6f07316e522d"], "availability.csv": [1099, "7e107616cd12503fecddd0aa61ef4e3464f2f8f23d76df58fc672b8f686c4500"], "yc2_stats.json": [136, "0e90590cac8c2014ce8fc48cb54302636540cc2246bdfb5114a5a283251e056b"]} -->
