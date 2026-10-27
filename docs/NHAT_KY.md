# Nhật ký: pseudo-caption HT-Step

Plan: `Data/Htstep/HTSTEP_PSEUDO_CAPTION.md`. Mọi số dưới đây lấy từ `metadata/build_report.json` và output của `tools/*.py`
(chạy local 2026-10-27, chỉ đọc file text và danh sách file trên HF, không đọc video).

## 2026-10-27: Phần A (metadata local) + notebook

### A0. Nguồn

**HT-Step** (`Data/Htstep/htstep/data/`, chỉ đọc):

| File | Cột / khóa thật |
|---|---|
| `annotations.json` | `{video_id: {subset, duration, activity, variation, annotations: [{segment: [s, e], partial, id}]}}`; `id` = `global_step_index` của taxonomy; `partial` ∈ {`Full Match`, `Partial Match`} |
| `taxonomy.csv` | `global_step_index, activity, step_index, variation, variation_index, headline, paragraph` (4.958 bước, 433 activity; `paragraph` = mô tả wikiHow, rỗng ở 1.016 bước) |
| `video_splits.json` | `train` 17.472, `val_seen` 600, `val_unseen` 963, `test_seen` 600, `test_unseen` 1.000 video |
| `video_candidate_steps.json` | `{video_id: {activity, variation, candidate_step_ids}}` |

- `annotations.json` chỉ có `train` và `val_seen`. **Test (seen/unseen) và val_unseen không có annotation công khai** (0 video
  trùng), chấm qua EvalAI. → "HT-Step val" trong plan = `val_seen`: 600 video, 3.441 segment (khớp `video_splits.json`).
  Không đối chiếu với số trong paper (README không ghi số; không tự điền từ trí nhớ).
- `task_headlines` lấy theo `candidate_step_ids` của video (bước của đúng **biến thể** task), không gộp mọi biến thể của
  activity: các biến thể có headline gần trùng nhau, gộp vào sẽ làm 5.2 khó một cách giả tạo. Đã kiểm mọi step GT nằm trong
  tập này. Trung vị 10 bước/task.

**YC2**:
- `YouCookII/metadata/youcookii_annotations_uemr.json`: `database[video_id] = {duration, subset, recipe_type, annotations:
  [{segment, id, sentence}], video_url}`; subset `training` 1.229 / `dev` 104 / `validation` 457.
- `recipe_type` là mã số; tên món lấy từ `Data/data/label_foodtype.csv` (file chính thức của YC2, nằm ngoài danh sách thư mục
  của plan nhưng chỉ đọc).
- Split dev/val UEMR: `Fushion/configs/final_protocol.json` dòng 10–11 định nghĩa dev = `train_omni_video.dev.jsonl`
  (104 video / 773 câu), val = `val_omni_video.jsonl` (394 / 3.030). Đã chép danh sách sang `metadata/yc2_dev_val_split.json`
  (kèm sha256 3 file nguồn). Annotation có 3.031 segment val; UEMR bỏ `hs2h7nb5PHQ_4`, manifest theo UEMR (3.030).
- 3 câu gốc có dấu cách đầu (`NTyhMGmuWik_3`, …); so khớp sau `strip()`.
- YC2 train text (few-shot, thống kê) = subset `training` (1.229 video, 9.564 câu), không giao dev/val (đã assert).

**Video**:
- HT-Step: HF dataset `nguyenminh04/htstep-videos`, public, revision `541f392c`. Video đóng gói trong
  `videos/htstep_videos_NNNN.tar` (≤ 100 `<video_id>.mp4`/tar, ~1,5–2,5 GB). Repo có 113 tar nhưng `manifest.json` của repo
  chỉ ghi 95 (README ghi trạng thái 2026-10-17); 18 tar `4027–4044` lấy danh sách video từ
  `nguyenminh04/htstep-feats/dl/done/<n>.json` (trường `ok`; đã kiểm `done/1000.json` khớp `manifest.json`). Tổng 7.453 video.
  `hf_path` trong manifest = đường dẫn tar; notebook tải tar, chỉ giải nén video cần, xóa tar.
- YC2: không có trên HF; trên Drive `Colab Notebooks/code KL/Omni/data/YouCookII/videos/<video_id>.mp4` (như `DRIVE_DATA`
  trong `Fushion/notebooks/uemr/*.ipynb`). `hf_repo = "drive:Colab Notebooks/code KL"`, `hf_path` = đường dẫn tương đối.

### A2. Tỉ lệ có video

| | video | segment |
|---|---|---|
| HT-Step val | 343/600 (57,2%) | 2.195/3.441 (63,8%) |
| YC2 dev | 104/104 | 773/773 |
| YC2 val | 394/394 | 3.030/3.030 |

Mọi video val HT-Step có trên HF nằm trong 5 tar `1000–1004` (~8,5 GB tải tạm). YC2 kiểm local bằng bản sao
`Data/YouCookII/videos` (1.500 mp4, chỉ tên file); ô 1 kiểm lại trên Drive.

### A3. Văn phong YC2 train (`metadata/yc2_stats.json`)

Số từ: median 8, p10 5, p90 14 (max 43). 100% chữ thường, 0% dấu chấm cuối. Từ đầu câu phổ biến nhất: add 3.147, place 529,
mix 416, cut 378, pour 336, … (đếm từ đầu câu, không gán POS; 30/30 từ đầu đều là động từ).

### A4. Kiểm tra

- Mọi `step_id` tra được headline; mọi segment `start < end` (cả hai dataset).
- HT-Step val: 47 cặp segment chồng lấn trong 37 video; 1.054 segment lặp lại một bước đã có trong cùng video;
  `Partial Match` 2.440 / `Full Match` 1.001.
- Không video nào vừa ở YC2 train vừa ở dev/val; dev ∩ val = ∅.

### Notebook (`notebooks/htstep_pseudocap_AZ.ipynb`, sinh từ `tools/make_notebook.py`)

Thay đổi so với plan và lý do:

1. **Gốc Drive** `MyDrive/Colab Notebooks/code KL/Fushion/htstep_cap/` thay vì `MyDrive/Fushion/htstep_cap/`: thống nhất với
   notebook UEMR (`.../code KL/Fushion/uemr`), theo A5.
2. **Frame lưu Drive dạng 1 tar/video** (`cache/frames/<ds>/<video_id>.tar`, bên trong đúng cấu trúc
   `<video_id>/<seg_idx>/f{0..7}.jpg`), giải nén về `/content/htcap/frames/` mỗi phiên. Lý do: ~48k JPEG nhỏ ghi thẳng lên Drive
   rất chậm.
3. **HT-Step video ở dạng tar shard** nên "tải từng video" = tải shard rồi giải nén từng video cần, xóa ngay.
4. **YC2 không có headline**: caption hỏng sau lần sinh lại được cắt về p90 từ (`cap_source = truncated`), hoặc `empty`.
5. **8 frame** lấy ở tâm 8 khoảng đều trong `[start, end]`: `start + (i + 0.5)(end − start)/8`.
6. **Hòa trong lưới chọn**: các cấu hình có CIDEr > max − 1 (thang ×100); chọn model nhanh hơn (đo bằng P2 ở ô 5), rồi CIDEr cao
   nhất của model đó.
7. **CI**: bootstrap theo video trên trung bình theo segment. CIDEr dùng điểm từng segment (IDF từ tập tham chiếu, chung cho mọi hệ)
   nên trung bình = điểm toàn tập; BLEU-4 chỉ báo toàn tập.
8. **5.2**: dự đoán đúng nếu headline được chọn trùng chữ với headline GT (một task có thể có hai bước cùng chữ). Báo thêm acc@1
   bỏ các caption `headline_fallback`.
9. **5.4b mẫu theo video**: xáo video theo seed, lấy mọi segment của từng video đến đủ n (300 YC2 / 600 HT-Step). Từ vựng nguyên
   liệu để sửa caption = danh từ (spaCy) xuất hiện ≥ 10 lần trong YC2 train, trừ danh sách chung chung/dụng cụ trong code.
   Nếu T1 trượt, T6 = N/A (bỏ 5.4b, đúng plan) và không chặn kết luận.
10. **Export**: chỉ video HT-Step có trên HF; bản headline dùng headline chữ thường bỏ dấu chấm cuối (cột `headline` giữ bản gốc).
11. **Khóa protocol** ngoài hash còn so mã nguồn `build_prompt/clean/valid/caption_one/asr_window`, few-shot, `CRITERIA`, `DECODE`
    với bản đã khóa (ô 7–9 FAIL nếu đổi).

Model thay thế: chưa (chưa chạy). Ô 2b in model nào resolve được trên HF; dự phòng theo bảng mục 3.

Phiên bản ghim (chưa kiểm trên Colab): transformers 4.53.3, accelerate 1.9.0, bitsandbytes 0.46.1, faster-whisper 1.1.1,
sentence-transformers 5.0.0, bert-score 0.3.13, pycocoevalcap 1.2, spacy 3.8.7 + en_core_web_sm 3.8.0; torch giữ bản Colab.

### Chờ chạy trên Colab

Tốc độ, `CHECK` từng ô: chưa có. Ghi vào đây khi user gửi log.

Commit: `f66a17f` (tools + metadata + notebook + nhật ký), sau đó `.gitattributes` (`* -text`, giữ nguyên byte để sha256 ở ô 1 khớp).
