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

## 2026-10-27: plan cập nhật lần 3 ([v3])

**A2: chỉ mục tar (`tools/build_tar_index.py`).** Đọc header 512 byte của từng tar bằng HTTP range (resolve redirect HF một lần
mỗi tar rồi đọc thẳng trên CDN; 24 luồng, khoảng 9 phút cho 113 tar; không tải dữ liệu video). Xử lý cả header GNU longname và
pax. Ghi `metadata/tar_index.csv` (`tar_path, member_name, video_id, offset_data, size`), `tar_index_tars.json` (revision
`541f392c`, dung lượng + số member từng tar; resume theo dung lượng) và `tar_index_report.json`.
- Quy tắc `video_id` = tên file bỏ đuôi (`-25-1nbOki0.mp4` → `-25-1nbOki0`). Không có member lạ (0 member không phải video).
- 113 tar, 7.453 member = 7.453 video, không video nào nằm trong 2 tar.
- Đối chiếu với danh sách của repo (`manifest.json` + `htstep-feats/dl/done/<n>.json`): khớp 7.453/7.453, 0 lệch tar.
  → `hf_path` trong manifest giữ nguyên, `manifest.jsonl` và `availability.csv` không đổi byte (sha256 như cũ).
- HT-Step val: cần 5 tar (`1000–1004`), tổng 8,6 GB, video val trong đó 7,0 GB (1,2×). Tar 1004 chỉ có 3/75 video val.

**Ô 3 viết lại theo 4.0:**
- YC2 đọc từ Drive (copy về `/content/tmp`, không ghi vào thư mục YC2).
- HT-Step đi theo từng tar, thứ tự cố định. Tiến trình nằm trong `progress/{tars.json, videos_done.txt, failed.csv}`, ghi nguyên
  tử bằng file tạm + `os.replace`.
- Khóa `in_progress` theo `session_id`, quá `STALE_HOURS = 3` thì coi như session đó đã chết.
- Kiểm tra đĩa trước khi tải; thiếu đĩa thì CHECK FAIL.
- Sau mỗi tar xóa thư mục tạm, kể cả khi có lỗi, và CHECK đã xóa.
- Tải tar lỗi: tar trở về `pending`, kèm `error`.

**Thay đổi so với plan:**
12. Chọn `tar` hay `range` (mục 4.0, "tối ưu") **theo từng tar** thay vì cho cả tập: range nếu dung lượng tar > 3 × tổng video
    cần trong tar đó. Với số liệu hiện tại: 1000–1003 tải cả tar, 1004 đọc range 3 video.
13. `session_id` lưu ở `/content/htcap/session_id`: còn giữ khi restart kernel, mất khi Colab cấp máy mới. Khi đó tar đang làm dở
    bị khóa 3 giờ; user có thể đặt `STALE_HOURS = 0` nếu chắc không còn session nào khác.
14. `progress/` nằm dưới gốc `.../code KL/Fushion/htstep_cap/`, giống mục 1 của bảng thay đổi.
15. Thông tin từng video (frame lặp, có audio) ghi riêng ở `cache/info/<ds>/<vid>.json`. `videos_done.txt` chỉ chứa
    `dataset/video_id`. Dấu `cache/done/` của bản notebook trước được tự chuyển sang `videos_done.txt`.

**Xem tiến trình:**
- Ô 1b `progress_report()` in trạng thái từng tar, số video đã trích/lỗi/còn lại, ASR, caption, judge, protocol và dung lượng
  đĩa trống. Ô 3 gọi hàm này lúc bắt đầu và lúc kết thúc.
- Thanh tqdm có ở: các vòng tar/video, ASR, đo tốc độ, sinh caption (lưới dev, YC2 val, HT-Step val), SigLIP, 5.1, 5.4a và
  giám khảo.

Mục 0.6 (vẫn giữ frame + ASR; không làm bản chỉ dùng text) trùng với notebook hiện tại, không cần sửa.

## 2026-10-27: v4, pseudo-caption cho toàn bộ HT-Step có video

**Split có annotation công khai:** chỉ `train` và `val_seen`. Test và `val_unseen` không có annotation (xem A0), nên không có
gì để thêm. `has_video` lấy theo `tar_index.csv` đã có, không lập chỉ mục lại.

| Split HT-Step | Video có trên HF | Segment có video |
|---|---|---|
| train | 7.110 / 17.472 (40,7%) | 44.151 / 103.293 (42,7%) |
| val (val_seen) | 343 / 600 (57,2%) | 2.195 / 3.441 (63,8%) |
| **tổng** | **7.453 / 18.072 (41,2%)** | **46.346 / 106.734 (43,4%)** |

Mọi segment train đều có `start < end` và có headline (0 lỗi). Mọi member trong 113 tar đều là video có annotation (train
hoặc val).

**Thay đổi so với yêu cầu v4:**
16. Các dòng train **không** nằm trong `manifest.jsonl`. Nếu ghi dạng jsonl, mỗi dòng kèm `task_headlines` và mô tả, file sẽ
    khoảng 135 MB, quá giới hạn 100 MB/file của GitHub (ô 1 clone repo). Vì vậy chúng nằm ở
    `metadata/manifest_htstep_train.jsonl.gz`: cùng schema, 103.293 dòng, 14,7 MB, gzip với mtime 0 nên byte cố định. Ô 1 đọc
    cả hai file thành một `MAN`. `manifest.jsonl` không đổi byte. `UPLOAD.md` kiểm số dòng sau giải nén và sha256 của file nén.
17. `tar_order.txt`: 5 tar val (1000–1004) trước, rồi 108 tar còn lại theo số hiệu.
18. **Lỗ hổng:** tar 1004 (mode extract) chứa thêm **72 video train / 487 segment**. Theo v4, lượt "full" chỉ chạy trên tar
    không có video val, còn tar val đã done thì không làm lại, nên 72 video này **không có caption**. Lượt full gồm 108 tar, 7.038
    video, 43.664 segment.
19. Cổng của ô 8b:
    - T4 trượt nếu cận **trên** CI của acc@1 < 0,85.
    - T6 là ngưỡng "≤ 20%", nên trượt nếu cận **dưới** CI của tỉ lệ unsupported > 0,20.
    - T1 trượt thì T6 = N/A và không chặn (như mục 6).
    - Thiếu điều kiện (protocol, caption val, file kết quả 5.2/5.4b) thì luôn chặn. `FORCE_FULL_PASS` chỉ bỏ qua trường hợp
      T4/T6 trượt.
    - Ô 8b chặn bằng cách in `CHECK full pass gate STOP` và **không raise exception**, để các ô 9–11 sau đó vẫn chạy được.
20. Thêm cờ `RUN_FULL_PASS`, mặc định False. Nếu không có cờ này, ô 8b sẽ gọi `verify_protocol()` ngay ở phiên 1 và FAIL. Phiên
    3 trở đi chỉ bật cờ này.
21. Lượt full trong một tar đi theo pha: (1) tải/đọc range, trích frame + audio vào `/content`, **xóa tar + video ngay sau pha
    này** (CHECK đã xóa); (2) ASR từng video; (3) sinh caption theo split; (4) xóa frame/audio (CHECK), ghi `videos_done`, tar
    done. Thứ tự này cho phép nạp ASR rồi đến model sinh **lần lượt** trong từng tar khi VRAM không đủ cho cả hai.
    - Cách quyết định giữ cả hai model: đỉnh VRAM của model sinh (ô 5) + VRAM của ASR (ô 4, đo bằng `torch.cuda.mem_get_info`,
      vì ctranslate2 không đi qua bộ cấp phát của torch) + 1 GB < VRAM của T4.
    - VRAM đo được ghi vào `tars.json[...].vram_gb` và `results/full_pass_vram.json`. Chép số này vào đây khi có log.
22. Resume trong một tar:
    - Bỏ qua video đã done, ASR đã có, key caption đã có.
    - Video đã có ASR và đủ caption (session trước bị ngắt trước khi ghi done) thì đánh dấu done luôn, không tải lại.
23. Chọn cách lấy video cho tar full: đọc range nếu tar có < 20% số member cần dùng, ngược lại tải cả tar. Với 108 tar hiện tại,
    mọi member đều là video train nên luôn tải cả tar.
24. Ước lượng thời gian:
    - Ô 5 ghi `results/full_pass_estimate.json`: giờ GPU cho caption theo từng model (P2, chưa có anchor) và giờ ASR (tổng thời
      lượng video / tốc độ realtime đo ở ô 4). Chưa tính thời gian tải tar và trích frame.
    - Sau tar full đầu tiên, ô 8b ghi `full_pass_estimate_measured.json` (gồm cả tải, trích, ASR và caption).
    - Ô 1b in ước lượng theo số đo của mọi tar đã xong.
25. Đánh giá:
    - 5.2 (`eval_5_2_htstep_<split>.json`) và 5.4a chạy theo từng split, với split train chỉ tính phần đã có caption, ghi
      `complete`.
    - T7 báo theo từng split (`T7`, `T7_train`); kết luận vẫn dùng T7 của val.
    - Ô 11 xuất `htstep_<split>_{pseudo,headline}.json` (subset `training`/`validation`, kèm `complete`), chạy được khi split
      chưa xong.

Không sửa file plan (`Data/Htstep/HTSTEP_PSEUDO_CAPTION.md` nằm ngoài `pseudocap/`); các thay đổi v4 ghi ở đây.

2026-10-27 (sau): metadata không upload nữa: ô 1 clone/pull `github.com/chien24/pseudocap` (public) vào `/content/pseudocap`, đọc `metadata/` từ đó; protocol ghi thêm `pseudocap_commit`; `verify_protocol` kiểm sha256 metadata không đổi sau khi khóa. Commit `dcde5bb`.

## 2026-10-28: log ô 3 (phiên 1, lần 2)

- Tải tuần tự đúng: 1000.tar 1,79 GB trong 10 s (176 MB/s), 1001.tar 1,69 GB trong 9 s; mỗi tar `CHECK ... đã xóa OK` trước tar
  sau; khoảng 650 s/tar (gần hết là trích frame/audio). Ô bị dừng khi đang tải 1002 (chưa thấy traceback).
- 8 video HT-Step lỗi `no frame at ...` ngay frame đầu (1000: 7, 1001: 1). **[giả thuyết]** OpenCV bản pip không giải mã được
  codec (có thể AV1). Sửa: frame nào cv2 không đọc được thì lấy bằng ffmpeg (`-ss t -frames:v 1`, PNG qua pipe, rồi resize + JPEG
  q85 bằng cv2 như cũ); cv2 chưa đọc được frame nào thì cả video dùng ffmpeg. `cache/info/<ds>/<vid>.json` ghi `ffmpeg_frames`.
- Cờ `RETRY_FAILED` (ô 0): xóa video YC2/HT-Step val khỏi `failed.csv`, đưa tar của chúng về `pending`; ô 3 tải lại tar đó và chỉ
  xử lý video chưa có trong `videos_done.txt`.

## 2026-10-28: log ô 3 (phiên 1, lần 3, `RETRY_FAILED = True`)

- `RETRY_FAILED: thử lại 8 video`; tar 1000, 1001 về `pending`; 1002 `in_progress` của session cũ 912 phút trước (> `STALE_HOURS`)
  nên được nhận lại. Đầu ô: HT-Step val xong 168/343.
- 5 tar extract xong trong 40:41 (khoảng 503 s/tar; tải 10–38 s/tar, 49–172 MB/s). Mỗi tar `CHECK ... đã xóa OK`. Tar 1004 đọc
  range 3 video.
- Cuối ô: YC2 dev 104/104, YC2 val 394/394, HT-Step val 343/343, **lỗi 0** → 8 video lỗi trước đều qua (dự phòng ffmpeg; số frame
  lấy bằng ffmpeg nằm ở `cache/info/htstep/<vid>.json`, chưa xem). Codec thật vẫn chưa kiểm.
- Giải nén frame về `/content`: 841 video trong 12:03 (1,29 s/video). Segment có frame: yc2/dev 773, yc2/val 3030, htstep/val 2195
  (khớp manifest). `CHECK every video done or logged as failed OK`, `CHECK frames ready OK`.

## 2026-10-28: ô 4 (ASR), phiên 1

- Lỗi 1: `av.open() got an unexpected keyword argument 'metadata_errors'` (PyAV trên Colab mới hơn bản faster-whisper 1.1.1
  dùng). Sửa: `load_audio` giải mã bằng ffmpeg (mono 16 kHz s16 / 32768, như `decode_audio`), đưa mảng vào `transcribe`. `ac1f7e6`.
- Lỗi 2: `libcublas.so.12 is not found`. Colab: **torch CUDA 13.0**; ctranslate2 (bản mới nhất 4.8.4) vẫn build cho CUDA 12, cuDNN
  không bắt buộc từ 4.6.3. Sửa: `cuda12_cublas()` cài `nvidia-cublas-cu12` và nạp sẵn `libcublasLt.so.12`, `libcublas.so.12`
  bằng `RTLD_GLOBAL`; `load_asr` gọi hàm này (ô 8b dùng chung). `804f716`.
- Kết quả: `large-v3-turbo`, 841/841 video trong 2:12:15 (8,13 s/video), audio 290.054 s (80,6 h), **36,6x realtime**, VRAM cả GPU
  khi chạy ASR **4,96 GB**. Ngôn ngữ tự nhận: en 788, es 33, de 4, tl 3, fr 2, nl 2, ja 2, el 1 (53 video ngoài en, chưa kiểm đúng
  sai). 9 video không có từ nào. `CHECK asr for every ready video OK 841/841`, `CHECK one ASR model for all OK`.

## 2026-10-28: ô 5 (đo tốc độ), phiên 1

- **q7b4 không tải được**: bitsandbytes 0.46.1 không có `libbitsandbytes_cuda130.so` (torch CUDA 13.0) → `SPEED['q7b4'] = None`
  (đã tải 16,5 GB trọng số vào cache HF của phiên). Chưa chạy ô 6, để protocol không khóa khi thiếu q7b4.
- **q3b**: `seg_per_h_P2` 2640, `model_ok` 20/20, `peak_vram_gb` 8,44. Ước lượng giờ: dev grid 0,9, YC2 val 1,1, HT-Step val 0,8.
- Lượt full (ô 8b): 7038 video, 43.664 segment, audio 783,6 h, ASR 21,4 giờ GPU, caption q3b 16,5 giờ GPU (chưa tính tải + trích).
- Cảnh báo `generation flags are not valid ... ['temperature']` lặp mỗi segment: vô hại (greedy).
- Sửa: ô 2 ghim `bitsandbytes==0.50.2` (wheel có `libbitsandbytes_cuda130.so`, đã đọc danh sách file trong wheel bằng range
  request); ô 2b thêm `CHECK bitsandbytes has a binary for torch CUDA`. Chưa kiểm trên Colab: bnb 0.50.2 với transformers 4.53.3,
  bản cuda130 chạy trên T4 (sm_75).

## 2026-10-28: ô 2b + ô 5 sau khi sửa (phiên 1b, notebook `a2693f0`)

- Môi trường: torch 2.11.0+cu130, transformers 4.53.3, bitsandbytes 0.50.2, faster-whisper 1.1.1, ctranslate2 4.8.4,
  opencv-python 5.0.0.93, GPU Tesla T4 14,6 GB. `CHECK bitsandbytes has a binary for torch CUDA OK libbitsandbytes_cuda130.so`.
- Revision: q7b4 `cc594898137f460bfe9f0759e9844b3ce807cfb5`, q3b `66285546d2b821cf421d4f5eb2576359d3770cd3`, judge
  InternVL3-8B-hf `259a3b64a14623c0ec91a045cb43f7c5af5fa6af`, SigLIP `9fdffc58afc957d1a03a25b10dba0329ab15c2a3`, mpnet
  `e8c3b32edf5434bc2275fc9bab85f82640a19130`.
- Tốc độ (P2, 20 segment YC2 dev): q7b4 2094 seg/h, 20/20, VRAM đỉnh 7,73 GB; q3b 2895 seg/h, 20/20, 7,38 GB (lần đo trước 2640
  seg/h, 8,44 GB). Ước lượng giờ: q7b4 dev grid 1,1 / YC2 val 1,4 / HT-Step val 1,0; q3b 0,8 / 1,0 / 0,8.
- Lượt full: ASR 21,4 + caption q7b4 20,9 hoặc q3b 15,1 giờ GPU. ASR (4,96) + model sinh (≤ 7,73) + 1 < 14,6 GB → ô 8b sẽ giữ cả
  hai trên GPU.

## 2026-10-28: ô 6 lỗi METEOR

- Lưới dev chạy đủ: 6 file `captions/dev_grid/yc2_dev__{q7b4,q3b}__{P0,P1,P2}.jsonl` đều 773/773 (q7b4 P2 2,12 s/seg; q3b P0/P1/P2
  1,13 / 1,21 / 1,30 s/seg). Protocol **chưa ghi**.
- Lỗi khi chấm q7b4 P0: BLEU xong, `Meteor.compute_score` đọc phải một dòng stats (`'9.0 7.0 4.0 ...'`) chỗ cần một số → `ValueError`.
  pycocoevalcap 1.2 gửi mỗi segment một lệnh `SCORE` (đọc 1 dòng) rồi một lệnh `EVAL` (đọc N+1 dòng); dòng thừa nghĩa là một lệnh
  `SCORE` bị jar tách thành hai dòng. GT YC2 dev toàn ASCII (đã quét 773 câu ở local), nên lỗi nằm ở caption sinh ra.
  **[giả thuyết]** caption có ký tự ngoài ASCII (Java giải mã thành ký tự xuống dòng như U+0085) hoặc `|`.
- Sửa: `ascii_text()` (bỏ ký tự ngoài ASCII, đổi `|` và mọi kiểu xuống dòng thành dấu cách), áp cho cả refs và hyps trước
  PTBTokenizer trong `coco()` (dùng cho ô 6 và 9a, cùng một cách cho mọi hệ). `coco()` in số hyp bị đổi.
- Lần 2 (sau `2d24180`): `coco: 1/773 hyp có ký tự ngoài ASCII` nhưng vẫn lỗi y hệt (cùng dòng stats `'9.0 7.0 4.0 ...'`) →
  giả thuyết ký tự ngoài ASCII **sai** (hoặc không đủ). Không chạy được jar ở local để tìm nguyên nhân (bị chặn chạy code tải về).
  Đọc hằng chuỗi trong `meteor-1.5.jar`: jar tách trường bằng regex `\|\|\|`, có chế độ file in `Segment <i> score:` và
  `Final score:`.
- Sửa: `meteor_files()` gọi cùng jar ở **chế độ file** (`java -Xmx2G -jar meteor-1.5.jar hyp.txt ref.txt -l en -norm`, cwd = thư
  mục jar), đọc điểm từng segment + điểm toàn tập; lỗi (rc ≠ 0, thiếu segment) thì raise kèm stdout/stderr. Cùng jar, cùng tham số,
  cùng chuỗi đã token hóa như `-stdio`; điểm toàn tập vẫn từ stats cộng dồn. Vẫn giữ `ascii_text()`.
- Lần 3: ô 6 treo > 11 phút sau `CHECK yc2_dev__q7b4__P2.jsonl complete OK`, trước khi nạp q3b (chưa rõ vì sao; nghi 2 tiến trình
  Java METEOR `-Xmx2G` của 2 lần lỗi trước còn sống chiếm RAM). Sửa: ô 6 không nạp model khi đủ 3 file dev grid của model đó.
