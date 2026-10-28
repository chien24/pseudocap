"""Write notebooks/htstep_pseudocap_AZ.ipynb from the cell sources below (keeps the .ipynb JSON valid).

Run: python -I -X utf8 tools/make_notebook.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CELLS = []


def md(s):
    CELLS.append(("markdown", s.strip("\n")))


def code(s):
    CELLS.append(("code", s.strip("\n")))


md(r'''
# HT-Step pseudo-caption kiểu YouCook2: sinh + đánh giá tự động (A–Z)

Plan: `Data/Htstep/HTSTEP_PSEUDO_CAPTION.md`. Metadata do `pseudocap/tools/*.py` tạo ở local; ô 1 tự clone repo
`github.com/chien24/pseudocap` để lấy `metadata/`, chỉ cần upload notebook này.

**Cách chạy.** Luôn chạy tuần tự từ ô 0; cờ `RUN_*` ở ô 0 chỉ bật/tắt phần nặng (ô tắt vẫn định nghĩa hàm cho ô sau).
Mọi kết quả trung gian nằm trên Drive và có resume, Colab ngắt thì chạy lại từ ô 0.

**Xem tiến trình:** mỗi vòng lặp dài có thanh tqdm; ô 1b (`progress_report()`) in trạng thái từng tar, số video đã trích/lỗi,
ASR, caption, judge, protocol. Chạy ô 0 → 1 → 1b bất cứ lúc nào (kể cả ở một session khác) để xem, không làm gì nặng.
Ô 3 xử lý HT-Step từng tar (tải → trích → xóa) và ghi `progress/` trên Drive; Colab ngắt khi đang làm dở một tar thì tar đó
bị khóa `STALE_HOURS` giờ (đặt `STALE_HOURS = 0` ở ô 0 nếu chắc không còn session nào khác đang chạy).

1. Phiên 1: `RUN_EXTRACT`, `RUN_ASR`, `RUN_SPEED`, `RUN_DEV_GRID` = True → ô 6 in sha256 của protocol.
2. Tải `htstep_cap_protocol.json` về `pseudocap/configs/` (commit + push), dán sha256 vào `PROTOCOL_SHA256`.
3. Phiên 2: tắt 4 cờ trên, bật `RUN_YC2_VAL`, `RUN_HTSTEP_VAL`, `RUN_EVAL`, `RUN_JUDGE` → ô 7, 8, 9a–9e, 10, 11.
4. Phiên 3 trở đi (lượt full, HT-Step train): chỉ bật `RUN_FULL_PASS`, chạy 0 → 8b. Ô 8b chỉ chạy khi protocol khóa, caption
   HT-Step val đủ và T4 (ô 9b), T6 (ô 9e) đã có trong `results/`; T4/T6 trượt thì dừng trừ khi `FORCE_FULL_PASS = True`.
   Mỗi session làm tiếp các tar chưa done (`MAX_TARS_THIS_SESSION` để giới hạn). Ô 11 xuất được khi split chưa xong.

Mỗi ô in `CHECK ... OK/FAIL`; `FAIL` dừng ô bằng exception.
''')

code(r'''
# ô 0. Cấu hình + hàm dùng chung
import os, re, gc, csv, json, time, uuid, random, shutil, hashlib, tarfile, inspect, subprocess
from pathlib import Path
from collections import Counter, defaultdict
from tqdm.auto import tqdm

DRIVE      = '/content/drive/MyDrive/Colab Notebooks/code KL'
BASE       = f'{DRIVE}/Fushion/htstep_cap'
PROGRESS   = f'{BASE}/progress'                          # tars.json, videos_done.txt, failed.csv
REPO_URL   = 'https://github.com/chien24/pseudocap.git'  # metadata/ lấy từ repo, không upload
REPO_DIR   = '/content/pseudocap'
META       = f'{REPO_DIR}/metadata'
YC2_VIDEOS = f'{DRIVE}/Omni/data/YouCookII/videos'     # chỉ đọc; = DRIVE_DATA/videos của notebook UEMR
HF_REPO    = 'nguyenminh04/htstep-videos'                # video trong tar videos/htstep_videos_XXXX.tar (tar_index.csv)
LOCAL      = '/content/htcap'                            # đĩa Colab: frame giải nén
TMP        = '/content/tmp'                              # tar/video tạm, xóa ngay sau khi trích
FETCH_MODE = 'auto'      # 'tar' tải cả tar | 'range' đọc từng video qua HTTP range | 'auto': theo từng tar, range nếu
                         # dung lượng tar > 3x tổng video cần trong tar đó (vd. tar có 3/75 video val)
STALE_HOURS = 3          # tar 'in_progress' của session khác cũ hơn mức này thì làm lại. Chắc chắn không còn session nào
                         # khác đang chạy (vd. Colab vừa ngắt) thì đặt 0 để làm lại ngay.
DISK_MARGIN_GB = 5
SEED       = 0
N_BOOT     = 1000

RUN_EXTRACT    = True
RUN_ASR        = False    # 841/841 video đã có ASR trên Drive (2026-10-28); False: ô 4 chỉ đọc lại
RUN_SPEED      = True
RUN_DEV_GRID   = True
RUN_YC2_VAL    = False    # chỉ bật sau khi đã điền PROTOCOL_SHA256
RUN_HTSTEP_VAL = False
RUN_EVAL       = False
RUN_JUDGE      = False
RETRY_FAILED   = False    # ô 3: xóa video YC2/HT-Step val khỏi failed.csv, đưa tar của chúng về pending rồi làm lại
RUN_FULL_PASS  = False    # ô 8b: HT-Step ngoài tar val (train). Cổng: protocol khóa + caption HT-Step val xong + T4, T6 đã tính
FORCE_FULL_PASS = False   # True: chạy ô 8b dù T4/T6 trượt
MAX_TARS_THIS_SESSION = None   # số tar 'full' tối đa mỗi session (None = không giới hạn)
PROTOCOL_SHA256 = ''      # dán sha256 mà ô 6 in ra

GEN_MODELS   = {'q7b4': ('Qwen/Qwen2.5-VL-7B-Instruct', True),    # 4-bit NF4, fp16 compute
                'q3b':  ('Qwen/Qwen2.5-VL-3B-Instruct', False)}   # fp16
ASR_MODELS   = ['large-v3-turbo', 'medium.en']                    # [chính, dự phòng], faster-whisper
JUDGE_MODELS = ['OpenGVLab/InternVL3-8B-hf', 'OpenGVLab/InternVL3-2B-hf']
VIS_MODELS   = ['google/siglip-so400m-patch14-384', 'openai/clip-vit-large-patch14']
TXT_MODELS   = ['sentence-transformers/all-mpnet-base-v2', 'BAAI/bge-large-en-v1.5']
DECODE       = dict(do_sample=False, max_new_tokens=40, temperature=None, top_p=None, top_k=None)   # greedy; None: bỏ sampling của generation_config
VARIANTS     = {'P0': ('frames', 'task'), 'P1': ('frames', 'task', 'asr'), 'P2': ('frames', 'task', 'asr', 'fewshot')}
N_FRAMES, ASR_PAD, ASR_MAX_WORDS, N_FEWSHOT = 8, 3.0, 120, 8
N_JUDGE_CAL, N_JUDGE_HT, N_JUDGE_YC2 = 300, 600, 300

# Tiêu chí mục 6. Sửa ngưỡng TRƯỚC ô 6; ô 6 khóa vào protocol, ô 7+ kiểm không đổi.
CRITERIA = {'T1_judge_pass_gt': 0.85, 'T1_judge_catch_corrupt': 0.70,
            'T2_cider_pseudo_minus_asr_lo': 0.0, 'T3_r1_pseudo_over_gt': 0.90, 'T4_step_acc1': 0.85,
            'T5_r1_pseudo_minus_headline_lo': 0.0, 'T6_unsupported_max': 0.20, 'T7_fallback_max': 0.05}


def check(name, ok, info=''):
    print(f"CHECK {name} {'OK' if ok else 'FAIL'} {info}".rstrip(), flush=True)
    if not ok:
        raise RuntimeError(f'CHECK FAIL: {name} {info}')

def read_jsonl(p):
    if not os.path.exists(p):
        return []
    with open(p, encoding='utf-8') as f:
        return [json.loads(l) for l in f if l.strip()]

def append_jsonl(p, rows):
    with open(p, 'a', encoding='utf-8') as f:
        f.write(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))

def write_text(p, s):
    """Ghi nguyên tử: file tạm rồi đổi tên (session bị ngắt giữa chừng không để lại file hỏng)."""
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p + '.tmp', 'w', encoding='utf-8') as f:
        f.write(s)
    os.replace(p + '.tmp', p)

def write_json(p, obj):
    write_text(p, json.dumps(obj, ensure_ascii=False, indent=1))

def gb_free(p='/content'):
    return shutil.disk_usage(p).free / 2**30

# ---- tiến trình trích frame/audio (Drive: PROGRESS/) ----
def videos_done():
    p = f'{PROGRESS}/videos_done.txt'
    return set(open(p, encoding='utf-8').read().split()) if os.path.exists(p) else set()

def mark_video_done(ds, *vids):
    d = videos_done() | {f'{ds}/{v}' for v in vids}
    write_text(f'{PROGRESS}/videos_done.txt', ''.join(x + '\n' for x in sorted(d)))

def failed_rows():
    p = f'{PROGRESS}/failed.csv'
    return list(csv.DictReader(open(p, encoding='utf-8'))) if os.path.exists(p) else []

def failed_set():
    return {f"{x['dataset']}/{x['video_id']}" for x in failed_rows()}   # thử lại: xóa dòng trong failed.csv

def write_failed(rows):
    import io
    buf = io.StringIO()
    w = csv.DictWriter(buf, ['dataset', 'video_id', 'tar_path', 'error'])
    w.writeheader()
    w.writerows(rows)
    write_text(f'{PROGRESS}/failed.csv', buf.getvalue())

def log_failed(ds, vid, src, err):
    write_failed(failed_rows() + [{'dataset': ds, 'video_id': vid, 'tar_path': src, 'error': str(err)[:300]}])
    print('  FAILED', ds, vid, str(err)[:120], flush=True)

def load_tars():
    p = f'{PROGRESS}/tars.json'
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else {}

def set_tar(tar, **kw):
    t = load_tars()
    t[tar] = {**t.get(tar, {}), **kw, 'updated_at': time.time()}
    write_json(f'{PROGRESS}/tars.json', t)

os.makedirs(LOCAL, exist_ok=True)
if not os.path.exists(f'{LOCAL}/session_id'):          # giữ qua restart kernel, mất khi Colab cấp máy mới
    write_text(f'{LOCAL}/session_id', uuid.uuid4().hex[:12])
SESSION_ID = open(f'{LOCAL}/session_id').read().strip()
print('session', SESSION_ID)

def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def free():
    import torch
    gc.collect(); torch.cuda.empty_cache()

def gpu_used_gb():
    """VRAM đang dùng của cả GPU (gồm cả ctranslate2/faster-whisper, torch không thấy)."""
    import torch
    f, t = torch.cuda.mem_get_info()
    return (t - f) / 2**30

def load_caps(name):
    return {r['key']: r for r in read_jsonl(f'{BASE}/captions/{name}.jsonl')}

def load_res(n):
    p = f'{BASE}/results/{n}.json'
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else None

def boot_ci(vals, vids, n=N_BOOT, seed=SEED):
    """Trung bình theo segment, CI 95% bằng bootstrap theo video (seed cố định). So cặp: truyền hiệu từng segment."""
    import numpy as np
    v = np.asarray(vals, float)
    _, inv = np.unique(np.asarray(vids), return_inverse=True)
    s, c = np.bincount(inv, v), np.bincount(inv)
    idx = np.random.default_rng(seed).integers(0, len(s), (n, len(s)))
    m = s[idx].sum(1) / c[idx].sum(1)
    return {'mean': float(v.mean()), 'lo': float(np.percentile(m, 2.5)), 'hi': float(np.percentile(m, 97.5)),
            'n': int(len(v)), 'videos': int(len(s))}

def fmt(ci, k=100):
    return f"{ci['mean'] * k:6.2f} [{ci['lo'] * k:6.2f}, {ci['hi'] * k:6.2f}]"

def frames_dir(r):
    return f"{LOCAL}/frames/{r['dataset']}/{r['video_id']}/{r['seg_idx']}"

def has_frames(r):
    return os.path.exists(f'{frames_dir(r)}/f{N_FRAMES - 1}.jpg')

def load_frames(r):
    from PIL import Image
    return [Image.open(f'{frames_dir(r)}/f{i}.jpg').convert('RGB') for i in range(N_FRAMES)]

def restore_frames(rows):
    """Giải nén tar frame (Drive) của video đã xong về LOCAL, một lần mỗi phiên; trả về các segment có frame."""
    for ds, vid in tqdm(sorted({(r['dataset'], r['video_id']) for r in rows}), desc='giải nén frame về /content'):
        tar = f'{BASE}/cache/frames/{ds}/{vid}.tar'
        if not os.path.isdir(f'{LOCAL}/frames/{ds}/{vid}') and os.path.exists(tar):
            with tarfile.open(tar) as t:
                t.extractall(f'{LOCAL}/frames/{ds}', filter='data')
    return [r for r in rows if has_frames(r)]

_ASR = {}
def asr_words(r):
    k = (r['dataset'], r['video_id'])
    if k not in _ASR:
        p = f'{BASE}/asr/{k[0]}/{k[1]}.json'
        _ASR[k] = json.load(open(p, encoding='utf-8'))['words'] if os.path.exists(p) else []
    return _ASR[k]

def asr_window(r, max_words=ASR_MAX_WORDS):
    """Transcript các từ có tâm trong [start-3s, end+3s], tối đa max_words từ."""
    lo, hi = r['start'] - ASR_PAD, r['end'] + ASR_PAD
    words = ' '.join(w for s, e, w in asr_words(r) if lo <= (s + e) / 2 <= hi).split()
    return ' '.join(words[:max_words])

_b = boot_ci([1, 0, 1, 1], ['a', 'a', 'b', 'b'], n=200)
assert _b['mean'] == 0.75 and _b['videos'] == 2 and 0.5 <= _b['lo'] <= _b['hi'] <= 1
print('ô 0 xong')
''')

code(r'''
# ô 1. Mount Drive, clone/pull repo pseudocap lấy metadata, kiểm khớp UPLOAD.md
import gzip
from google.colab import drive
drive.mount('/content/drive')
if os.path.isdir(f'{REPO_DIR}/.git'):
    subprocess.run(['git', '-C', REPO_DIR, 'pull', '-q', '--ff-only'], check=True)
else:
    subprocess.run(['git', 'clone', '-q', REPO_URL, REPO_DIR], check=True)
GIT_COMMIT = subprocess.run(['git', '-C', REPO_DIR, 'rev-parse', 'HEAD'], capture_output=True, text=True, check=True).stdout.strip()
print('pseudocap commit', GIT_COMMIT)
for d in ['cache/frames/yc2', 'cache/frames/htstep', 'cache/audio/yc2', 'cache/audio/htstep', 'cache/info/yc2',
          'cache/info/htstep', 'progress', 'asr/yc2', 'asr/htstep', 'captions/dev_grid', 'judge', 'results', 'export']:
    os.makedirs(f'{BASE}/{d}', exist_ok=True)

def open_meta(name):
    p = f'{META}/{name}'
    return gzip.open(p, 'rt', encoding='utf-8') if p.endswith('.gz') else open(p, encoding='utf-8')

COUNTS = json.loads(re.search(r'<!-- COUNTS (\{.*?\}) -->', open(f'{META}/UPLOAD.md', encoding='utf-8').read(), re.S).group(1))
for name, (n, sha) in COUNTS.items():
    with open_meta(name) as f:
        lines = sum(1 for _ in f)
    check(f'{name} lines', lines == n, f'{lines} vs UPLOAD.md {n}')
    check(f'{name} sha256', sha256_file(f'{META}/{name}') == sha)

# manifest.jsonl = YC2 dev/val + HT-Step val; HT-Step train cùng schema nằm ở file gzip riêng (giới hạn 100 MB/file của GitHub)
with open_meta('manifest_htstep_train.jsonl.gz') as f:
    MAN = read_jsonl(f'{META}/manifest.jsonl') + [json.loads(l) for l in f if l.strip()]
SPLIT = json.load(open(f'{META}/yc2_dev_val_split.json', encoding='utf-8'))
STATS = json.load(open(f'{META}/yc2_stats.json', encoding='utf-8'))
TRAIN_TEXT = read_jsonl(f'{META}/yc2_train_text.jsonl')
MED, P90 = STATS['words']['median'], STATS['words']['p90']
DEVVAL = set(SPLIT['dev']) | set(SPLIT['val'])
cnt = Counter(f"{r['dataset']}/{r['split']}" for r in MAN)
check('manifest splits', cnt == {'yc2/dev': 773, 'yc2/val': 3030, 'htstep/val': 3441, 'htstep/train': 103293}, str(dict(cnt)))
check('yc2 train text disjoint from dev/val', not ({r['video_id'] for r in TRAIN_TEXT} & DEVVAL))

# YC2: local chỉ kiểm bằng bản sao trên máy, ở đây kiểm lại từng file trên Drive
yc2_vids = sorted({r['video_id'] for r in MAN if r['dataset'] == 'yc2'})
miss = [v for v in yc2_vids if not os.path.exists(f'{YC2_VIDEOS}/{v}.mp4')]
for r in MAN:
    if r['dataset'] == 'yc2':
        r['has_video'] = r['video_id'] not in miss
print(f'YC2 video thiếu trên Drive: {len(miss)}/{len(yc2_vids)}', miss[:10])
check('yc2 video folder on Drive', len(miss) < len(yc2_vids), YC2_VIDEOS)
ROWS = [r for r in MAN if r['has_video']]
for k in sorted(cnt):
    n_ok = sum(f"{r['dataset']}/{r['split']}" == k for r in ROWS)
    nv = len({r['video_id'] for r in MAN if f"{r['dataset']}/{r['split']}" == k})
    nv_ok = len({r['video_id'] for r in ROWS if f"{r['dataset']}/{r['split']}" == k})
    print(f'  {k}: có video {nv_ok}/{nv} video ({nv_ok / nv:.1%}), {n_ok}/{cnt[k]} segment ({n_ok / cnt[k]:.1%})')

# HT-Step: video -> (tar, member, offset, size) từ tar_index.csv (header tar đọc ở local, revision ghim)
TARS_META = json.load(open(f'{META}/tar_index_tars.json', encoding='utf-8'))
HF_REV = TARS_META['revision']
HT_TAR_OF = {r['video_id']: r['hf_path'] for r in ROWS if r['dataset'] == 'htstep'}
VIDX = {r['video_id']: {**r, 'offset_data': int(r['offset_data']), 'size': int(r['size'])}
        for r in csv.DictReader(open(f'{META}/tar_index.csv', encoding='utf-8'))
        if HT_TAR_OF.get(r['video_id']) == r['tar_path']}
check('every HT-Step video with has_video is in tar_index', set(VIDX) == set(HT_TAR_OF), f'{len(VIDX)}/{len(HT_TAR_OF)}')

# Hai chế độ tar: 'extract' (tar có video val, ô 3: frame/audio cache lên Drive) và 'full' (tar còn lại, ô 8b: một lượt
# tải -> ASR -> caption, không lưu frame/audio). Thứ tự theo tar_order.txt: tar val trước.
with open_meta('tar_order.txt') as f:
    TAR_ORDER = f.read().split()
VAL_VIDS = {r['video_id'] for r in ROWS if r['dataset'] == 'htstep' and r['split'] == 'val'}
EXTRACT_TARS = [t for t in TAR_ORDER if t in {VIDX[v]['tar_path'] for v in VAL_VIDS}]
FULL_TARS = [t for t in TAR_ORDER if t not in EXTRACT_TARS and t in {x['tar_path'] for x in VIDX.values()}]
check('tar_order: extract tars first', TAR_ORDER[:len(EXTRACT_TARS)] == EXTRACT_TARS)
FULL_VIDS = {v for v, x in VIDX.items() if x['tar_path'] in set(FULL_TARS)}
EX_ROWS = [r for r in ROWS if r['dataset'] == 'yc2' or r['split'] == 'val']                  # ô 3–9
FULL_ROWS = [r for r in ROWS if r['dataset'] == 'htstep' and r['video_id'] in FULL_VIDS]       # ô 8b
HT_SPLITS = ['val'] + sorted({r['split'] for r in ROWS if r['dataset'] == 'htstep'} - {'val'})
gap = [v for v, x in VIDX.items() if x['tar_path'] in EXTRACT_TARS and v not in VAL_VIDS]
print(f'tar extract: {len(EXTRACT_TARS)} ({len(VAL_VIDS)} video val) | tar full: {len(FULL_TARS)} '
      f'({len(FULL_VIDS)} video, {len(FULL_ROWS)} segment, split {sorted({r["split"] for r in FULL_ROWS})})')
print(f'video không phải val nằm trong tar extract (không vào lượt full): {len(gap)}, '
      f'{sum(r["video_id"] in set(gap) for r in ROWS)} segment')
''')

code(r'''
# ô 1b. XEM TIẾN TRÌNH: chạy bất cứ lúc nào sau ô 1 (kể cả từ một session khác), chỉ đọc file trên Drive
def progress_report():
    vd, fs, tars = videos_done(), failed_set(), load_tars()
    print(f"=== tiến trình {time.strftime('%H:%M:%S')} | session {SESSION_ID} | đĩa /content trống {gb_free():.1f} GB")
    for mode, tl in (('extract', EXTRACT_TARS), ('full', FULL_TARS)):
        st = Counter(tars.get(t, {}).get('status', 'pending') for t in tl)
        print(f"tar {mode:7}: done {st['done']} / in_progress {st['in_progress']} / pending {st['pending']} (tổng {len(tl)})")
        for t in tl:
            x = tars.get(t)
            if x and x.get('status') != 'done':
                print(f"   {t}: {x['status']} session {x.get('session_id')} {(time.time() - x['updated_at']) / 60:.0f} phút trước"
                      f"{' lỗi: ' + x['error'][:100] if x.get('error') else ''}")
        timed = [tars[t] for t in tl if tars.get(t, {}).get('status') == 'done' and tars[t].get('segments')]
        if timed:
            s = sum(x['seconds'] for x in timed) / sum(x['segments'] for x in timed)
            left = sum(1 for r in FULL_ROWS if tars.get(VIDX[r['video_id']]['tar_path'], {}).get('status') != 'done')
            print(f"   đo thật: {s:.1f} s/segment, {len(timed)} tar; còn {left} segment ≈ {left * s / 3600:.1f} giờ GPU; "
                  f"VRAM tối đa {max(x.get('vram_gb', 0) for x in timed):.1f} GB")
    asr_have = {ds: {p[:-5] for p in os.listdir(f'{BASE}/asr/{ds}')} for ds in ('yc2', 'htstep')}
    scopes = [('yc2', 'dev', None), ('yc2', 'val', None), ('htstep', 'val', None)] + \
             [('htstep', s, FULL_VIDS) for s in HT_SPLITS if s != 'val']
    for ds, split, only in scopes:
        vids = {r['video_id'] for r in ROWS if r['dataset'] == ds and r['split'] == split and (only is None or r['video_id'] in only)}
        d, f = sum(f'{ds}/{v}' in vd for v in vids), sum(f'{ds}/{v}' in fs for v in vids)
        print(f'video {ds}/{split:5}: xong {d}/{len(vids)} | lỗi {f} | còn {len(vids) - d - f} | ASR {len(vids & asr_have[ds])}')
    want = {'dev_grid': 773, 'yc2_val': sum(r['dataset'] == 'yc2' and r['split'] == 'val' for r in ROWS),
            **{f'htstep_{s}': sum(r['dataset'] == 'htstep' and r['split'] == s for r in (EX_ROWS if s == 'val' else FULL_ROWS))
               for s in HT_SPLITS}}
    for p in sorted(Path(f'{BASE}/captions').rglob('*.jsonl')):
        n, w = sum(1 for _ in open(p, encoding='utf-8')), want.get('dev_grid' if 'dev_grid' in str(p) else p.stem)
        print(f"caption {p.relative_to(f'{BASE}/captions')}: {n}/{w}" + (f' ({n / w:.1%})' if w else ''))
    for p in sorted(Path(f'{BASE}/judge').glob('*.jsonl')):
        print(f'judge {p.name}: {sum(1 for _ in open(p, encoding="utf-8"))}')
    pp = f'{BASE}/htstep_cap_protocol.json'
    print('protocol:', sha256_file(pp) if os.path.exists(pp) else 'chưa có')
    print('results:', sorted(os.listdir(f'{BASE}/results')))

progress_report()
''')

code(r'''
# ô 2. Cài thư viện (ghim phiên bản), in phiên bản, kiểm model trên HF
%pip -q install transformers==4.53.3 accelerate==1.9.0 bitsandbytes==0.50.2 faster-whisper==1.1.1 sentence-transformers==5.0.0 bert-score==0.3.13 pycocoevalcap==1.2 spacy==3.8.7
!python -m spacy download en_core_web_sm-3.8.0 --direct -q
!apt-get -qq install -y default-jre > /dev/null
''')

code(r'''
# ô 2b. Phiên bản + revision model (nếu pip đòi restart: restart rồi chạy lại 0, 1, 2b)
import torch, transformers, importlib.metadata as im
from huggingface_hub import HfApi
print('torch', torch.__version__, '| cuda', torch.version.cuda, '| transformers', transformers.__version__)
print('GPU', torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1), 'GB')
for p in ['accelerate', 'bitsandbytes', 'faster-whisper', 'ctranslate2', 'sentence-transformers', 'bert-score',
          'pycocoevalcap', 'spacy', 'en_core_web_sm', 'opencv-python', 'opencv-python-headless']:
    try:
        print(f'  {p} {im.version(p)}')
    except im.PackageNotFoundError:
        pass
check('transformers pinned', transformers.__version__ == '4.53.3', transformers.__version__)
# Colab hiện dùng torch CUDA 13.0; bitsandbytes 0.46.1 không có bản build cuda130 nên q7b4 (4-bit) không tải được.
bnb_lib = f"libbitsandbytes_cuda{torch.version.cuda.replace('.', '')}.so"
check('bitsandbytes has a binary for torch CUDA',
      os.path.exists(os.path.join(os.path.dirname(im.distribution('bitsandbytes').locate_file('bitsandbytes/__init__.py')), bnb_lib)), bnb_lib)
check('java + ffmpeg', bool(shutil.which('java')) and bool(shutil.which('ffmpeg')))

try:
    from google.colab import userdata
    HF_TOKEN = userdata.get('HF_TOKEN')
except Exception:
    HF_TOKEN = None
api = HfApi(token=HF_TOKEN)

def resolve(cands):
    """[chính, dự phòng] -> (repo, revision sha) của model đầu tiên còn trên HF."""
    for c in cands:
        try:
            return c, api.model_info(c).sha
        except Exception as e:
            print('  không dùng được', c, type(e).__name__)
    return None, None

REV = {code: resolve([repo])[1] for code, (repo, _) in GEN_MODELS.items()}
JUDGE, JUDGE_REV = resolve(JUDGE_MODELS)
VIS, VIS_REV = resolve(VIS_MODELS)
TXT, TXT_REV = resolve(TXT_MODELS)
print('gen', REV, '\njudge', JUDGE, JUDGE_REV, '\nvis', VIS, VIS_REV, '\ntxt', TXT, TXT_REV)
check('models resolvable', any(REV.values()) and JUDGE and VIS and TXT)
''')

code(r'''
# ô 3. Trích 8 frame/segment + audio cả video, mode 'extract'. YC2: đọc từ Drive. HT-Step val: từng tar từ HF (tải -> trích
#      -> xóa), tiến trình trên Drive (progress/tars.json, videos_done.txt, failed.csv). Chạy lại ở session mới sẽ tiếp tục.
#      Hàm trong ô này (extract_video, claim_tar, fetch_videos) dùng lại ở ô 8b.
import cv2, requests
import numpy as np
from huggingface_hub import hf_hub_download, hf_hub_url

def ffmpeg_frame(path, t):
    """Một frame BGR tại t giây bằng ffmpeg (đọc được codec mà OpenCV pip không đọc, vd. AV1); None nếu không có."""
    r = subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-ss', f'{max(t, 0):.3f}', '-i', path, '-frames:v', '1',
                        '-f', 'image2pipe', '-vcodec', 'png', '-'], capture_output=True)
    return cv2.imdecode(np.frombuffer(r.stdout, np.uint8), cv2.IMREAD_COLOR) if r.stdout else None

def seg_times(s, e):
    return [s + (i + 0.5) * (e - s) / N_FRAMES for i in range(N_FRAMES)]

for p in Path(f'{BASE}/cache/done').glob('*/*.json'):     # dấu done của bản notebook trước
    if f'{p.parent.name}/{p.stem}' not in videos_done():
        mark_video_done(p.parent.name, p.stem)

def extract_video(path, ds, vid, segs, save=True):
    """Frame vào LOCAL/frames, audio vào LOCAL/audio. save=True (mode extract): copy lên Drive + đánh dấu done.
    save=False (mode full): để lại ở /content cho ASR + caption, người gọi xóa và đánh dấu done."""
    cap = cv2.VideoCapture(path)
    fps, nfr = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    dur = nfr / fps if fps > 0 and nfr > 0 else None
    last, n_repeat, n_ffmpeg, use_cv2 = None, 0, 0, True
    for r in segs:
        out = frames_dir(r)
        os.makedirs(out, exist_ok=True)
        for i, t in enumerate(seg_times(r['start'], r['end'])):
            if dur:
                t = min(t, dur - 0.1)
            fr = None
            if use_cv2:
                cap.set(cv2.CAP_PROP_POS_MSEC, max(t, 0) * 1000)
                ok, fr = cap.read()
                fr = fr if ok else None
            if fr is None:                          # cv2 không giải mã được (vd. codec AV1) hoặc seek lỗi: thử ffmpeg
                fr = ffmpeg_frame(path, t)
                if fr is not None:
                    n_ffmpeg += 1
                    use_cv2 = use_cv2 and last is not None   # cv2 chưa đọc được frame nào: cả video dùng ffmpeg
            if fr is None:                          # quá cuối video: lặp frame trước
                fr, n_repeat = last, n_repeat + 1
            if fr is None:
                raise RuntimeError(f'no frame at {t:.1f}s')
            last = fr
            h, w = fr.shape[:2]
            if max(h, w) > 448:
                fr = cv2.resize(fr, (round(w * 448 / max(h, w)), round(h * 448 / max(h, w))), interpolation=cv2.INTER_AREA)
            cv2.imwrite(f'{out}/f{i}.jpg', fr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    cap.release()
    aud = f'{LOCAL}/audio/{ds}/{vid}.opus'
    os.makedirs(os.path.dirname(aud), exist_ok=True)
    res = subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-y', '-i', path, '-vn', '-ac', '1', '-ar', '16000',
                          '-c:a', 'libopus', '-b:a', '32k', aud], capture_output=True, text=True)
    has_audio = res.returncode == 0 and os.path.exists(aud)
    if not save:
        return
    # mọi thứ ghi ở /content trước, xong cả video mới copy lên Drive; Drive chậm với nhiều file nhỏ nên frame = 1 tar/video
    tmp = f'{LOCAL}/{vid}.tar'
    with tarfile.open(tmp, 'w') as t:
        t.add(f'{LOCAL}/frames/{ds}/{vid}', arcname=vid)
    if has_audio:
        shutil.copy(aud, f'{BASE}/cache/audio/{ds}/{vid}.opus')
    shutil.copy(tmp, f'{BASE}/cache/frames/{ds}/{vid}.tar')
    os.remove(tmp)
    write_json(f'{BASE}/cache/info/{ds}/{vid}.json', {'segments': len(segs), 'repeated_frames': n_repeat,
                                                      'duration_cv2': dur, 'audio': has_audio, 'ffmpeg_frames': n_ffmpeg,
                                                      'ffmpeg_err': res.stderr[-300:]})
    mark_video_done(ds, vid)

def process(path, ds, vid, src):
    """Trích một video (mode extract) rồi xóa file video; lỗi thì ghi failed.csv, không dừng notebook."""
    try:
        extract_video(path, ds, vid, BY_VID[(ds, vid)])
    except Exception as e:
        log_failed(ds, vid, src, e)
    finally:
        if os.path.exists(path):
            os.remove(path)

def need_disk(gb, what):
    check(f'disk for {what}', gb_free() >= gb + DISK_MARGIN_GB,
          f'cần {gb:.1f} + {DISK_MARGIN_GB} GB, trống {gb_free():.1f} GB')

def range_get(tar, off, size, out):
    """Đọc byte [off, off+size) của tar trên HF bằng HTTP range, ghi ra out. Resolve URL CDN mỗi video (URL ký có hạn)."""
    url = hf_hub_url(HF_REPO, tar, repo_type='dataset', revision=HF_REV)
    auth = {'Authorization': f'Bearer {HF_TOKEN}'} if HF_TOKEN else {}
    for attempt in range(4):
        try:
            r = requests.head(url, allow_redirects=False, timeout=60, headers=auth)
            loc, hdr = (r.headers['Location'], {}) if r.is_redirect else (url, auth)
            with requests.get(loc, headers={**hdr, 'Range': f'bytes={off}-{off + size - 1}'}, stream=True, timeout=120) as g:
                g.raise_for_status()
                assert g.status_code == 206, f'HTTP {g.status_code}, không phải range'
                with open(out, 'wb') as f:
                    for ch in g.iter_content(1 << 22):
                        f.write(ch)
            if os.path.getsize(out) == size:
                return
            raise IOError(f'nhận {os.path.getsize(out)} byte, cần {size}')
        except Exception:
            if attempt == 3:
                raise
            time.sleep(5 * 2 ** attempt)

def claim_tar(tar, mode):
    """False nếu tar đã done hoặc đang được session khác làm (mới hơn STALE_HOURS); không thì đánh dấu in_progress."""
    st = load_tars().get(tar, {})
    if st.get('status') == 'done':
        return False
    age_h = (time.time() - st.get('updated_at', 0)) / 3600
    if st.get('status') == 'in_progress' and st.get('session_id') != SESSION_ID and age_h < STALE_HOURS:
        print(f"  bỏ qua {tar}: session {st['session_id']} đang làm ({age_h * 60:.0f} phút trước; xem STALE_HOURS ở ô 0)")
        return False
    set_tar(tar, status='in_progress', mode=mode, session_id=SESSION_ID, error=None)
    return True

def fetch_videos(tar, vids, work, how):
    """yield (video_id, đường dẫn video trong work/); how='tar' tải cả tar, 'range' đọc từng video. Người gọi xóa video."""
    if how == 'tar':
        need_disk((TARS_META['tars'][tar]['size'] + max(VIDX[v]['size'] for v in vids)) / 2**30, tar)
        p = hf_hub_download(HF_REPO, tar, repo_type='dataset', revision=HF_REV, local_dir=work, token=HF_TOKEN)
        with tarfile.open(p) as t:
            for v in tqdm(vids, desc=f'  {Path(tar).stem} video', leave=False):
                try:
                    m = t.getmember(VIDX[v]['member_name'])
                except KeyError:
                    log_failed('htstep', v, tar, 'member not in tar')
                    continue
                t.extract(m, f'{work}/v', filter='data')
                yield v, f'{work}/v/{m.name}'
    else:
        for v in tqdm(vids, desc=f'  {Path(tar).stem} video (range)', leave=False):
            need_disk(VIDX[v]['size'] / 2**30, v)
            out = f'{work}/{v}.mp4'
            try:
                range_get(tar, VIDX[v]['offset_data'], VIDX[v]['size'], out)
            except Exception as e:
                log_failed('htstep', v, tar, f'range read: {e}')
                continue
            yield v, out

def tar_mode(tar, vids):
    """'auto': extract -> range nếu tar > 3x dung lượng video cần; full -> range nếu tar có < 20% số video cần dùng."""
    if FETCH_MODE != 'auto':
        return FETCH_MODE
    if tar in EXTRACT_TARS:
        return 'range' if TARS_META['tars'][tar]['size'] > 3 * sum(VIDX[v]['size'] for v in vids) else 'tar'
    return 'range' if len(vids) < 0.2 * TARS_META['tars'][tar]['n_members'] else 'tar'

BY_VID = defaultdict(list)
for r in EX_ROWS:
    BY_VID[(r['dataset'], r['video_id'])].append(r)
VAL_BY_TAR = {t: sorted(v for v in VAL_VIDS if VIDX[v]['tar_path'] == t) for t in EXTRACT_TARS}

if RUN_EXTRACT and RETRY_FAILED:
    fr_rows = failed_rows()
    retry = [x for x in fr_rows if (x['dataset'], x['video_id']) in BY_VID]
    write_failed([x for x in fr_rows if x not in retry])
    for t in {x['tar_path'] for x in retry} & set(EXTRACT_TARS):
        set_tar(t, status='pending', error=None)      # tar done -> làm lại, chỉ các video chưa có trong videos_done
    print(f'RETRY_FAILED: thử lại {len(retry)} video', Counter(x['error'][:40] for x in retry).most_common(5))

if RUN_EXTRACT:
    print('--- ô 3 bắt đầu'); progress_report()
    os.makedirs(TMP, exist_ok=True)

    skip = videos_done() | failed_set()
    for vid in tqdm(sorted(v for ds, v in BY_VID if ds == 'yc2' and f'yc2/{v}' not in skip), desc='YC2 video (Drive)'):
        src, tmp = f'{YC2_VIDEOS}/{vid}.mp4', f'{TMP}/{vid}.mp4'
        try:
            shutil.copy(src, tmp)                    # chỉ đọc thư mục YC2, không ghi gì vào đó
        except Exception as e:
            log_failed('yc2', vid, src, e)
            continue
        process(tmp, 'yc2', vid, src)

    print('HT-Step val, chế độ:', {Path(t).stem: tar_mode(t, VAL_BY_TAR[t]) for t in EXTRACT_TARS})
    bar = tqdm(EXTRACT_TARS, desc='HT-Step tar (extract)')
    for tar in bar:
        bar.set_postfix_str(Path(tar).stem)
        if not claim_tar(tar, 'extract'):
            continue
        skip = videos_done() | failed_set()
        mine = VAL_BY_TAR[tar]
        vids = [v for v in mine if f'htstep/{v}' not in skip]
        work = f'{TMP}/{Path(tar).stem}'
        os.makedirs(work, exist_ok=True)
        try:
            if vids:
                for v, path in fetch_videos(tar, vids, work, tar_mode(tar, mine)):
                    process(path, 'htstep', v, tar)
        except RuntimeError:                          # CHECK FAIL (thiếu đĩa): dừng và báo
            set_tar(tar, status='pending', error='thiếu đĩa')
            raise
        except Exception as e:                        # tải tar lỗi: tar về pending, chạy lại ô 3 sau
            set_tar(tar, status='pending', error=f'{type(e).__name__}: {e}'[:300])
            print('  LỖI', tar, type(e).__name__, str(e)[:200], flush=True)
            continue
        finally:
            shutil.rmtree(work, ignore_errors=True)   # xóa tar + video giải nén, kể cả khi có lỗi
            check(f'{Path(tar).stem}: tar/video tạm đã xóa', not os.path.exists(work))
        vd, fs = videos_done(), failed_set()
        set_tar(tar, status='done', mode='extract', session_id=SESSION_ID, n_ok=sum(f'htstep/{v}' in vd for v in mine),
                n_fail=sum(f'htstep/{v}' in fs for v in mine))
    print('--- ô 3 kết thúc'); progress_report()

finished = videos_done() | failed_set()
left = [k for k in BY_VID if f'{k[0]}/{k[1]}' not in finished]
READY = restore_frames(EX_ROWS)
print('segment có frame:', dict(Counter(f"{r['dataset']}/{r['split']}" for r in READY)))
if RUN_EXTRACT:
    check('every video done or logged as failed', not left, f'{len(left)} video chưa xử lý: chạy lại ô 3')
check('frames ready', len(READY) > 0)
''')

code(r'''
# ô 4. ASR cho mọi video đã trích (cache asr/<dataset>/<video_id>.json, timestamp từng từ). Cùng một ASR cho YC2 và HT-Step;
#      load_asr/transcribe dùng lại ở ô 8b (cùng model, cùng tham số).
import numpy as np
def cuda12_cublas():
    """ctranslate2 (faster-whisper) build cho CUDA 12 và dlopen('libcublas.so.12') lúc chạy (cuDNN không bắt buộc từ 4.6.3);
    torch trên Colab không mang thư viện này -> cài nvidia-cublas-cu12 và nạp sẵn bằng RTLD_GLOBAL để dlopen tìm thấy."""
    import ctypes, sys, glob, site
    find = lambda: sorted(p for d in site.getsitepackages() for p in glob.glob(f'{d}/nvidia/**/libcublas*.so.12', recursive=True))
    libs = find()
    if not libs:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'nvidia-cublas-cu12'], check=True)
        libs = find()
    for p in sorted(libs, key=lambda p: 'Lt' not in p):   # libcublasLt trước, libcublas cần nó
        ctypes.CDLL(p, mode=ctypes.RTLD_GLOBAL)
    import torch
    print('torch CUDA', torch.version.cuda, '| nạp', [os.path.basename(p) for p in libs])

def load_asr(names=ASR_MODELS):
    cuda12_cublas()
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    for name in names:
        try:
            return name, BatchedInferencePipeline(model=WhisperModel(name, device='cuda', compute_type='float16'))
        except Exception as e:
            print('ASR không tải được', name, e)
    return None, None

def load_audio(aud, sr=16000):
    """Giải mã như faster_whisper.decode_audio (mono, 16 kHz, s16 / 32768) nhưng bằng ffmpeg: PyAV mới trên Colab không nhận
    tham số metadata_errors mà faster-whisper 1.1.1 truyền vào av.open."""
    r = subprocess.run(['ffmpeg', '-nostdin', '-loglevel', 'error', '-i', aud, '-f', 's16le', '-ac', '1', '-ar', str(sr), '-'],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.int16).astype(np.float32) / 32768.0

def transcribe(pipe, name, aud):
    """-> (json ghi vào asr/, số giây audio). Không có audio: words rỗng."""
    if not os.path.exists(aud):
        return {'model': name, 'language': None, 'words': [], 'note': 'no audio stream'}, 0.0
    segs, info = pipe.transcribe(load_audio(aud), batch_size=16, word_timestamps=True)
    words = [[round(w.start, 2), round(w.end, 2), w.word.strip()] for s in segs for w in (s.words or [])]
    return {'model': name, 'language': info.language, 'language_probability': round(info.language_probability, 3),
            'words': words}, info.duration

READY_VIDS = sorted({(r['dataset'], r['video_id']) for r in READY})
ASR_SPEED_PATH = f'{BASE}/results/asr_speed.json'
if RUN_ASR:
    ASR_NAME, pipe = load_asr()
    check('asr model loaded', pipe is not None, str(ASR_NAME))
    prev = {json.load(open(f'{BASE}/asr/{ds}/{v}.json'))['model'] for ds, v in READY_VIDS
            if os.path.exists(f'{BASE}/asr/{ds}/{v}.json')}
    check('same ASR model as cached files', prev <= {ASR_NAME}, str(prev))
    t0, n_audio_s, vram = time.time(), 0.0, 0.0
    todo = [k for k in READY_VIDS if not os.path.exists(f'{BASE}/asr/{k[0]}/{k[1]}.json')]
    print(f'ASR: {len(READY_VIDS) - len(todo)} video đã có, còn {len(todo)}')
    bar = tqdm(todo, desc='ASR video')
    for ds, vid in bar:
        res, dur = transcribe(pipe, ASR_NAME, f'{BASE}/cache/audio/{ds}/{vid}.opus')
        write_json(f'{BASE}/asr/{ds}/{vid}.json', res)
        n_audio_s += dur
        vram = max(vram, gpu_used_gb())
        bar.set_postfix_str(f'{n_audio_s / max(time.time() - t0, 1):.0f}x realtime')
    if n_audio_s > 60:                                 # tốc độ + VRAM để ước lượng lượt full (ô 5, ô 8b)
        write_json(ASR_SPEED_PATH, {'model': ASR_NAME, 'audio_s': round(n_audio_s), 'wall_s': round(time.time() - t0),
                                    'realtime_x': round(n_audio_s / (time.time() - t0), 1), 'gpu_used_gb': round(vram, 2)})
        print('ASR:', json.load(open(ASR_SPEED_PATH)))
    pipe = None
    free()

asr = {k: json.load(open(f'{BASE}/asr/{k[0]}/{k[1]}.json', encoding='utf-8')) for k in READY_VIDS
       if os.path.exists(f'{BASE}/asr/{k[0]}/{k[1]}.json')}
ASR_NAME = next(iter({a['model'] for a in asr.values()}), None)
print('ASR model:', ASR_NAME, '| ngôn ngữ:', Counter(a['language'] for a in asr.values()).most_common(8))
print('video không có từ nào:', sum(not a['words'] for a in asr.values()))
check('asr for every ready video', len(asr) == len(READY_VIDS), f'{len(asr)}/{len(READY_VIDS)}')
check('one ASR model for all', len({a['model'] for a in asr.values()}) == 1)
''')

code(r'''
# ô 5. Prompt + sinh caption (một build_prompt cho YC2 và HT-Step) + đo tốc độ 20 segment mỗi model
import torch
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

SYSTEM = (f'You are an annotator for a YouCook2-style dataset of cooking procedures. Write exactly one English sentence '
          f'in the imperative mood, all lowercase, with no final period, about {MED} words and at most {P90} words, '
          f'describing what happens in this clip. Mention only actions, ingredients and tools that are visible in the '
          f'frames or spoken in the transcript. Do not guess and do not add details. Return only that sentence.')

def pick_fewshot(rows, seed=SEED, k=N_FEWSHOT):
    """k caption YC2 train của k món khác nhau, cố định theo seed."""
    rng = random.Random(seed)
    recipes = rng.sample(sorted({r['recipe_type'] for r in rows}), k)
    shots = [rng.choice([r for r in rows if r['recipe_type'] == rt]) for rt in recipes]
    assert not {s['video_id'] for s in shots} & DEVVAL, 'few-shot lấy từ dev/val'
    return shots

FEWSHOT = pick_fewshot(TRAIN_TEXT)
check('few-shot from YC2 train only, distinct dishes',
      len({s['recipe_type'] for s in FEWSHOT}) == N_FEWSHOT and not {s['video_id'] for s in FEWSHOT} & DEVVAL)

def build_prompt(seg, variant, shorter=False):
    parts = VARIANTS[variant] + (('anchor',) if seg['dataset'] == 'htstep' else ())
    text = [f"Dish: {seg['task']}."]
    if 'asr' in parts:
        text.append(f'Transcript around the clip: "{asr_window(seg) or "(no speech)"}"')
    if 'fewshot' in parts:
        text.append('Example captions from other videos (style only):\n' + '\n'.join(f"- {s['sentence']}" for s in FEWSHOT))
    if 'anchor' in parts:
        a = f"The step being performed is: {seg['headline']}"
        if seg.get('description'):
            a += f" Step description: {seg['description']}"
        text.append(a + ' (This step text is for understanding only; do not mention anything from it that is not '
                        'visible or spoken in the clip.)')
    text.append('Write the caption for this clip.' + (f' Keep it short: at most {P90} words.' if shorter else ''))
    content = ([{'type': 'video'}] if 'frames' in parts else []) + [{'type': 'text', 'text': '\n'.join(text)}]
    return [{'role': 'system', 'content': [{'type': 'text', 'text': SYSTEM}]}, {'role': 'user', 'content': content}]

def clean(txt):
    line = next((l for l in txt.strip().splitlines() if l.strip()), '')
    line = re.sub(r'["“”]', '', line).strip().rstrip('.').strip()
    return line.lower()

assert clean(' "Add the Salt."\nsecond line') == 'add the salt' and clean('') == ''

def valid(c):
    return bool(c) and len(c.split()) <= 2 * P90

def caption_one(gen, seg, variant):
    raw = gen(seg, build_prompt(seg, variant))
    c, src = clean(raw), 'model'
    if not valid(c):
        raw = gen(seg, build_prompt(seg, variant, shorter=True))
        c, src = clean(raw), 'model_retry'
    if not valid(c):
        if seg['dataset'] == 'htstep':
            c, src = clean(seg['headline']), 'headline_fallback'
        else:                                       # YC2 không có headline: cắt về p90 từ
            c, src = ' '.join(c.split()[:P90]), ('truncated' if c else 'empty')
    return c, src, raw

def prompt_source():
    return SYSTEM + '\n' + '\n'.join(inspect.getsource(f) for f in (build_prompt, clean, valid, caption_one, asr_window))

def load_gen(code, rev):
    repo, q4 = GEN_MODELS[code]
    kw = dict(torch_dtype=torch.float16, attn_implementation='sdpa', device_map='cuda:0', revision=rev, token=HF_TOKEN)
    if q4:
        kw['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                       bnb_4bit_compute_dtype=torch.float16,
                                                       llm_int8_skip_modules=['visual', 'lm_head'])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(repo, **kw).eval()
    proc = AutoProcessor.from_pretrained(repo, revision=rev, token=HF_TOKEN)

    @torch.inference_mode()
    def gen(seg, msgs):
        text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = proc(text=[text], videos=[load_frames(seg)], return_tensors='pt').to(model.device)
        out = model.generate(**inp, **DECODE)
        return proc.batch_decode(out[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0]
    return model, gen

def caption_rows(rows, path, gen, variant, extra):
    """Sinh cho rows, ghi nối mỗi 50 segment, bỏ qua key đã có trong path."""
    have = {r['key'] for r in read_jsonl(path)}
    todo = [r for r in rows if r['key'] not in have]
    buf = []
    bar = tqdm(todo, desc=os.path.basename(path), initial=len(rows) - len(todo), total=len(rows))
    for i, r in enumerate(bar, 1):
        c, src, raw = caption_one(gen, r, variant)
        buf.append({'key': r['key'], 'caption': c, 'cap_source': src, 'prompt_variant': variant, 'raw': raw, **extra})
        if len(buf) == 50 or i == len(todo):
            append_jsonl(path, buf)
            buf = []
    out = {r['key']: r for r in read_jsonl(path)}
    check(f'{os.path.basename(path)} complete', all(r['key'] in out for r in rows), f'{len(out)}/{len(rows)}')
    return out

DEV = [r for r in READY if r['dataset'] == 'yc2' and r['split'] == 'dev']
YC2_VAL = [r for r in READY if r['dataset'] == 'yc2' and r['split'] == 'val']
HT_VAL = [r for r in READY if r['dataset'] == 'htstep']
SPEED_PATH = f'{BASE}/results/speed.json'

if RUN_SPEED:
    SPEED = {}
    sample = random.Random(SEED).sample(DEV, 20)
    for code in GEN_MODELS:
        if not REV[code]:
            continue
        try:
            torch.cuda.reset_peak_memory_stats()
            model, gen = load_gen(code, REV[code])
        except Exception as e:
            print('không tải được', code, e)
            SPEED[code] = None
            model = gen = None
            free()
            continue
        caption_one(gen, sample[0], 'P2')            # warm-up
        t0 = time.time()
        outs = [caption_one(gen, r, 'P2') for r in tqdm(sample, desc=f'đo tốc độ {code}')]
        dt = time.time() - t0
        SPEED[code] = {'seg_per_h_P2': round(20 / dt * 3600), 'model_ok': sum(o[1] == 'model' for o in outs),
                       'peak_vram_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        print(code, SPEED[code])
        for r, o in list(zip(sample, outs))[:5]:
            print(f"   GT : {r['gt_caption']}\n   GEN: {o[0]}  [{o[1]}]")
        model = gen = None
        free()
    write_json(SPEED_PATH, SPEED)
    for code, s in SPEED.items():
        if s:
            h = {'dev_grid (3 biến thể)': 3 * len(DEV), 'yc2_val': len(YC2_VAL), 'htstep_val': len(HT_VAL)}
            est = {k: round(n / s['seg_per_h_P2'], 1) for k, n in h.items()}
            print(f'{code}: ước lượng giờ', est, '<- VƯỢT 4 GIỜ, báo user' if max(est.values()) > 4 else '')
    check('a generation model gives valid captions on T4 (>= 18/20)', any(s and s['model_ok'] >= 18 for s in SPEED.values()),
          str(SPEED))
else:
    SPEED = json.load(open(SPEED_PATH)) if os.path.exists(SPEED_PATH) else {}
    print('SKIP speed; đọc lại', SPEED)

# Ước lượng giờ GPU cho lượt full (ô 8b) từ số video/segment thật trong manifest (chưa tính tải tar + trích frame;
# đo bằng P2 không anchor). Ô 8b in lại ước lượng bằng số đo thật sau tar 'full' đầu tiên.
ASP = json.load(open(ASR_SPEED_PATH)) if os.path.exists(ASR_SPEED_PATH) else {}
full_audio_h = sum({r['video_id']: r['duration'] for r in FULL_ROWS}.values()) / 3600
FULL_EST = {'videos': len(FULL_VIDS), 'segments': len(FULL_ROWS), 'audio_hours': round(full_audio_h, 1),
            'asr_gpu_hours': round(full_audio_h / ASP['realtime_x'], 1) if ASP.get('realtime_x') else None,
            'caption_gpu_hours': {c: round(len(FULL_ROWS) / s['seg_per_h_P2'], 1) for c, s in SPEED.items() if s}}
print('Lượt full (ô 8b), giờ GPU ước lượng:', FULL_EST)
write_json(f'{BASE}/results/full_pass_estimate.json', FULL_EST)
''')

code(r'''
# ô 6. Lưới chọn trên YC2 dev (P0/P1/P2 x 2 model) -> bảng CIDEr -> ghi protocol + sha256
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

def ascii_text(s):
    """Chỉ giữ ASCII in được, bỏ '|' (jar METEOR tách trường theo '|||'), một dòng; cùng một cách cho refs và mọi hệ."""
    return ' '.join(re.sub(r'[^\x20-\x7e]', '', ' '.join(s.split()).replace('|', ' ')).split())   # split(): mọi kiểu xuống dòng

assert ascii_text('add\u2028salt | pepper\x85 jalape\xf1o\r\n') == 'add salt pepper jalapeo'

def meteor_files(g, h, keys):
    """METEOR bằng đúng jar của pycocoevalcap (meteor-1.5, -l en -norm) nhưng ở chế độ file thay cho -stdio: trên Colab
    (2026-10-28) Meteor.compute_score lệch dòng đọc (đọc phải dòng stats khi chờ điểm) dù input đã là ASCII một dòng."""
    import tempfile
    jar_dir = os.path.dirname(inspect.getfile(Meteor))
    with tempfile.TemporaryDirectory() as d:
        for name, src in (('hyp', h), ('ref', g)):
            with open(f'{d}/{name}.txt', 'w', encoding='utf-8') as f:
                f.write(''.join(src[k][0] + '\n' for k in keys))
        r = subprocess.run(['java', '-Xmx2G', '-jar', 'meteor-1.5.jar', f'{d}/hyp.txt', f'{d}/ref.txt', '-l', 'en', '-norm'],
                           cwd=jar_dir, capture_output=True, text=True)
    seg = sorted((int(i), float(x)) for i, x in re.findall(r'^Segment (\d+) score:\s*(\S+)', r.stdout, re.M))
    final = re.findall(r'^Final score:\s*(\S+)', r.stdout, re.M)
    if r.returncode or not final or [i for i, _ in seg] != list(range(1, len(keys) + 1)):
        raise RuntimeError(f'METEOR file mode: rc={r.returncode}, {len(seg)}/{len(keys)} segment\n'
                           f'{r.stdout[:1500]}\n{r.stderr[-1500:]}')
    return float(final[0]), [x for _, x in seg]

def coco(refs, hyps):
    """refs/hyps {key: str} -> (điểm toàn tập, {metric: {key: điểm}}). BLEU-4 chỉ báo toàn tập."""
    n_non_ascii = sum(ascii_text(h or '') != ' '.join((h or '').split()) for h in hyps.values())
    if n_non_ascii:
        print(f'coco: {n_non_ascii}/{len(hyps)} hyp có ký tự ngoài ASCII hoặc "|" (đã bỏ trước khi chấm)')
    refs = {k: ascii_text(refs[k]) for k in hyps}
    hyps = {k: ascii_text(v or '') for k, v in hyps.items()}
    tok = PTBTokenizer()
    g = tok.tokenize({k: [{'caption': refs[k]}] for k in hyps})
    h = tok.tokenize({k: [{'caption': hyps[k] or 'none'}] for k in hyps})
    h = {k: v if v[0].strip() else ['none'] for k, v in h.items()}
    keys = list(g)
    h = {k: h[k] for k in keys}
    b, _ = Bleu(4).compute_score(g, h)
    out, per = {'BLEU4': b[3]}, {}
    out['METEOR'], arr = meteor_files(g, h, keys)
    per['METEOR'] = dict(zip(keys, arr))
    for name, sc in [('ROUGE_L', Rouge()), ('CIDEr', Cider())]:
        s, arr = sc.compute_score(g, h)
        out[name], per[name] = float(s), dict(zip(keys, map(float, arr)))
    return out, per

PROTO_PATH = f'{BASE}/htstep_cap_protocol.json'
DEV_REFS = {r['key']: r['gt_caption'] for r in DEV}

if RUN_DEV_GRID:
    for code in GEN_MODELS:
        if not (REV[code] and SPEED.get(code)):
            print('bỏ', code, '(không tải được ở ô 5)')
            continue
        paths = {v: f'{BASE}/captions/dev_grid/yc2_dev__{code}__{v}.jsonl' for v in VARIANTS}
        if all({r['key'] for r in read_jsonl(pv)} >= set(DEV_REFS) for pv in paths.values()):
            print(code, ': dev grid đã đủ 3 biến thể, không nạp model')
            continue
        model, gen = load_gen(code, REV[code])
        for v, pv in paths.items():
            caption_rows(DEV, pv, gen, v, {'model': code})
        model = gen = None
        free()

TABLE = []
for code in GEN_MODELS:
    for v in VARIANTS:
        caps = {r['key']: r for r in read_jsonl(f'{BASE}/captions/dev_grid/yc2_dev__{code}__{v}.jsonl')}
        if not caps or set(DEV_REFS) - set(caps):
            continue
        sc, _ = coco(DEV_REFS, {k: caps[k]['caption'] for k in DEV_REFS})
        TABLE.append({'model': code, 'variant': v, **{m: round(x * 100, 2) for m, x in sc.items()},
                      'fallback_or_trunc': sum(c['cap_source'] not in ('model', 'model_retry') for c in caps.values()),
                      'seg_per_h': (SPEED.get(code) or {}).get('seg_per_h_P2')})
print(f"{'model':6} {'var':4} {'BLEU4':>6} {'METEOR':>7} {'ROUGE_L':>8} {'CIDEr':>7} {'fb':>4} {'seg/h':>6}")
for t in TABLE:
    print(f"{t['model']:6} {t['variant']:4} {t['BLEU4']:6.2f} {t['METEOR']:7.2f} {t['ROUGE_L']:8.2f} {t['CIDEr']:7.2f} "
          f"{t['fallback_or_trunc']:4} {t['seg_per_h']}")

if os.path.exists(PROTO_PATH):
    print('protocol đã khóa trước đó, không ghi lại')
elif RUN_DEV_GRID:
    check('dev grid complete', len(TABLE) == 3 * sum(bool(REV[c] and SPEED.get(c)) for c in GEN_MODELS), f'{len(TABLE)} cấu hình')
    best = max(t['CIDEr'] for t in TABLE)
    tied = [t for t in TABLE if t['CIDEr'] > best - 1.0]                 # hòa: chênh < 1 điểm CIDEr
    fastest = max(tied, key=lambda t: t['seg_per_h'] or 0)['model']
    CHOSEN = max((t for t in tied if t['model'] == fastest), key=lambda t: t['CIDEr'])
    proto = {
        'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), 'seed': SEED,
        'gen': {'code': CHOSEN['model'], 'repo': GEN_MODELS[CHOSEN['model']][0], 'revision': REV[CHOSEN['model']],
                'nf4_4bit': GEN_MODELS[CHOSEN['model']][1], 'dtype': 'float16', 'attn_implementation': 'sdpa',
                'variant': CHOSEN['variant'], 'decode': DECODE},
        'prompt': {'system': SYSTEM, 'variants': VARIANTS, 'anchor': 'HT-Step only', 'n_frames': N_FRAMES,
                   'asr_pad_s': ASR_PAD, 'asr_max_words': ASR_MAX_WORDS, 'source': prompt_source()},
        'fewshot': [{'video_id': s['video_id'], 'seg_idx': s['seg_idx'], 'sentence': s['sentence']} for s in FEWSHOT],
        'yc2_stats': STATS, 'criteria': CRITERIA, 'asr_model': ASR_NAME,
        'eval_models': {'judge': [JUDGE, JUDGE_REV], 'vision': [VIS, VIS_REV], 'text': [TXT, TXT_REV],
                        'bertscore': 'roberta-large (bert-score default layer, no baseline rescale)'},
        'selection': {'rule': 'max CIDEr trên YC2 dev; hòa (chênh < 1 điểm) -> model nhanh hơn', 'dev_table': TABLE,
                      'speed': SPEED},
        'n_boot': N_BOOT, 'judge_n': [N_JUDGE_CAL, N_JUDGE_YC2, N_JUDGE_HT],
        'metadata_sha256': {k: v[1] for k, v in COUNTS.items()}, 'pseudocap_commit': GIT_COMMIT,
    }
    write_json(PROTO_PATH, proto)
    print('CHỌN:', CHOSEN)
if os.path.exists(PROTO_PATH):
    print('sha256 protocol =', sha256_file(PROTO_PATH), '-> dán vào PROTOCOL_SHA256 (ô 0), tải file về pseudocap/configs/')
    check('protocol written', True)
else:
    print('SKIP: chưa có protocol (bật RUN_DEV_GRID)')

def verify_protocol():
    """Ô chấm val/HT-Step: hash khớp, prompt/few-shot/tiêu chí không đổi sau khi khóa."""
    check('protocol exists', os.path.exists(PROTO_PATH))
    sha = sha256_file(PROTO_PATH)
    check('protocol sha256 == PROTOCOL_SHA256', bool(PROTOCOL_SHA256) and sha == PROTOCOL_SHA256, sha)
    p = json.load(open(PROTO_PATH, encoding='utf-8'))
    check('prompt code unchanged since lock', p['prompt']['source'] == prompt_source())
    check('few-shot unchanged', [s['sentence'] for s in p['fewshot']] == [s['sentence'] for s in FEWSHOT])
    check('criteria unchanged', p['criteria'] == CRITERIA)
    check('decode unchanged', p['gen']['decode'] == DECODE)
    check('metadata unchanged since lock', p['metadata_sha256'] == {k: v[1] for k, v in COUNTS.items()})
    return p, sha
''')

code(r'''
# ô 7. Kiểm hash protocol, sinh YC2 val (không anchor), một lần
if RUN_YC2_VAL:
    PROTO, PSHA = verify_protocol()
    g = PROTO['gen']
    model, gen = load_gen(g['code'], g['revision'])
    caption_rows(YC2_VAL, f'{BASE}/captions/yc2_val.jsonl', gen, g['variant'],
                 {'model': f"{g['repo']}@{g['revision'][:8]}", 'protocol_sha': PSHA})
    model = gen = None
    free()
else:
    print('SKIP yc2 val:', len(read_jsonl(f'{BASE}/captions/yc2_val.jsonl')), 'caption đã có')
''')

code(r'''
# ô 8. Sinh HT-Step val: cấu hình đã khóa + [anchor]
if RUN_HTSTEP_VAL:
    PROTO, PSHA = verify_protocol()
    g = PROTO['gen']
    model, gen = load_gen(g['code'], g['revision'])
    caps = caption_rows(HT_VAL, f'{BASE}/captions/htstep_val.jsonl', gen, g['variant'],
                        {'model': f"{g['repo']}@{g['revision'][:8]}", 'protocol_sha': PSHA})
    model = gen = None
    free()
    print('cap_source:', Counter(c['cap_source'] for c in caps.values()))
    for r in random.Random(SEED).sample(HT_VAL, 5):
        print(f"  headline: {r['headline']}\n  pseudo  : {caps[r['key']]['caption']}\n  asr     : {asr_window(r)[:150]}")
else:
    print('SKIP htstep val:', len(read_jsonl(f'{BASE}/captions/htstep_val.jsonl')), 'caption đã có')
''')

code(r'''
# ô 8b. Lượt 'full': HT-Step ngoài tar val (train). Mỗi tar một lượt: tải -> frame + audio ở /content -> ASR (asr/htstep/ trên
#       Drive) -> caption mọi segment bằng cấu hình đã khóa + anchor (captions/htstep_<split>.jsonl) -> xóa frame/audio ->
#       videos_done -> tar done. Không lưu frame/audio lên Drive. Cổng: protocol khóa, caption HT-Step val xong, T4 + T6 đã tính.
FULL_BY_VID = defaultdict(list)
for r in FULL_ROWS:
    FULL_BY_VID[r['video_id']].append(r)
FULL_BY_TAR = {t: sorted(v for v in FULL_VIDS if VIDX[v]['tar_path'] == t) for t in FULL_TARS}

def full_gate():
    """-> (thiếu điều kiện, tiêu chí trượt). Trượt = cả đầu CI thuận lợi vẫn không đạt ngưỡng."""
    missing, fails = [], []
    caps = load_caps('htstep_val')
    if not HT_VAL or not all(r['key'] in caps for r in HT_VAL):
        missing.append(f'caption HT-Step val chưa xong ({len(caps)}/{len(HT_VAL)}, ô 8)')
    r52, r54b = load_res('eval_5_2_htstep_val'), load_res('eval_5_4b_judge')
    if not r52:
        missing.append('T4 chưa tính (ô 9b)')
    else:
        t4 = r52['systems']['pseudo']
        print(f"  T4 acc@1 HT-Step val {fmt(t4)} (ngưỡng {CRITERIA['T4_step_acc1']:.2f})")
        if t4['hi'] < CRITERIA['T4_step_acc1']:
            fails.append(f"T4: cận trên CI {t4['hi']:.3f} < {CRITERIA['T4_step_acc1']}")
    if not r54b:
        missing.append('T6 chưa tính (ô 9e)')
    else:
        s = r54b['sets']
        t1 = s['cal_gt']['mean'] >= CRITERIA['T1_judge_pass_gt'] and s['cal_corrupt']['mean'] >= CRITERIA['T1_judge_catch_corrupt']
        t6 = s['htstep_val_pseudo']
        print(f"  T6 unsupported HT-Step val {fmt(t6)} (ngưỡng <= {CRITERIA['T6_unsupported_max']:.2f}), T1 {'đạt' if t1 else 'trượt'}")
        if not t1:
            print('  T1 trượt -> T6 = N/A (giám khảo không tin được), không chặn')
        elif t6['lo'] > CRITERIA['T6_unsupported_max']:
            fails.append(f"T6: cận dưới CI {t6['lo']:.3f} > {CRITERIA['T6_unsupported_max']}")
    return missing, fails

def drop_local(v):
    shutil.rmtree(f'{LOCAL}/frames/htstep/{v}', ignore_errors=True)
    if os.path.exists(f'{LOCAL}/audio/htstep/{v}.opus'):
        os.remove(f'{LOCAL}/audio/htstep/{v}.opus')

def full_estimate(sec_per_seg, note):
    fin = videos_done() | failed_set()
    left = [r for r in FULL_ROWS if f"htstep/{r['video_id']}" not in fin]
    est = {'note': note, 'sec_per_segment': round(sec_per_seg, 2), 'segments_left': len(left),
           'gpu_hours_left': round(len(left) * sec_per_seg / 3600, 1), 'at': time.strftime('%Y-%m-%d %H:%M')}
    print('ƯỚC LƯỢNG lượt full:', est)
    write_json(f'{BASE}/results/full_pass_estimate_measured.json', est)

def run_full_pass(PROTO, PSHA):
    g = PROTO['gen']
    sp, asp = SPEED.get(g['code']) or {}, (json.load(open(ASR_SPEED_PATH)) if os.path.exists(ASR_SPEED_PATH) else {})
    total = torch.cuda.get_device_properties(0).total_memory / 2**30
    keep_both = bool(sp.get('peak_vram_gb') and asp.get('gpu_used_gb')) and sp['peak_vram_gb'] + asp['gpu_used_gb'] + 1.0 < total
    print(f"VRAM: model sinh ~{sp.get('peak_vram_gb')} GB + ASR ~{asp.get('gpu_used_gb')} GB / {total:.1f} GB -> "
          f"{'giữ cả hai' if keep_both else 'nạp lần lượt trong từng tar'}")
    M = {}
    def use(which):
        if which in M:
            return M[which]
        if not keep_both:
            M.clear(); free()
        if which == 'asr':
            name, pipe = load_asr([PROTO['asr_model']])
            check('ASR of the locked protocol', name == PROTO['asr_model'], str(name))
            M['asr'] = pipe
        else:
            M['gen'] = load_gen(g['code'], g['revision'])[1]
        return M[which]

    extra = {'model': f"{g['repo']}@{g['revision'][:8]}", 'protocol_sha': PSHA}
    n_tars, vram_max = 0, 0.0
    bar = tqdm(FULL_TARS, desc='HT-Step tar (full)')
    for tar in bar:
        bar.set_postfix_str(Path(tar).stem)
        if MAX_TARS_THIS_SESSION is not None and n_tars >= MAX_TARS_THIS_SESSION:
            print(f'dừng: đã làm {n_tars} tar trong session này (MAX_TARS_THIS_SESSION)')
            break
        if not claim_tar(tar, 'full'):
            continue
        t0, vram = time.time(), 0.0
        mine = FULL_BY_TAR[tar]
        skip = videos_done() | failed_set()
        have = {s: {r['key'] for r in read_jsonl(f'{BASE}/captions/htstep_{s}.jsonl')}
                for s in {r['split'] for v in mine for r in FULL_BY_VID[v]}}
        todo = [v for v in mine if f'htstep/{v}' not in skip]
        # resume: video đã có ASR và đủ caption (session trước ngắt trước khi ghi done) thì không cần tải lại
        complete = [v for v in todo if os.path.exists(f'{BASE}/asr/htstep/{v}.json')
                    and all(r['key'] in have[r['split']] for r in FULL_BY_VID[v])]
        if complete:
            mark_video_done('htstep', *complete)
        todo = [v for v in todo if v not in complete]
        work, ok_vids = f'{TMP}/{Path(tar).stem}', []
        os.makedirs(work, exist_ok=True)
        try:
            if todo:
                for v, path in fetch_videos(tar, todo, work, tar_mode(tar, todo)):
                    try:
                        extract_video(path, 'htstep', v, FULL_BY_VID[v], save=False)
                        ok_vids.append(v)
                    except Exception as e:
                        log_failed('htstep', v, tar, e)
                        drop_local(v)
                    finally:
                        os.remove(path)
        except RuntimeError:                          # CHECK FAIL (thiếu đĩa)
            set_tar(tar, status='pending', error='thiếu đĩa')
            raise
        except Exception as e:
            set_tar(tar, status='pending', error=f'{type(e).__name__}: {e}'[:300])
            print('  LỖI', tar, type(e).__name__, str(e)[:200], flush=True)
            for v in ok_vids:
                drop_local(v)
            continue
        finally:
            shutil.rmtree(work, ignore_errors=True)   # tar + video xóa ngay sau khi trích frame/audio
            check(f'{Path(tar).stem}: tar/video tạm đã xóa', not os.path.exists(work))

        for v in tqdm([v for v in ok_vids if not os.path.exists(f'{BASE}/asr/htstep/{v}.json')], desc='  ASR', leave=False):
            res, _ = transcribe(use('asr'), PROTO['asr_model'], f'{LOCAL}/audio/htstep/{v}.opus')
            write_json(f'{BASE}/asr/htstep/{v}.json', res)
            vram = max(vram, gpu_used_gb())
        by_split = defaultdict(list)
        for v in ok_vids:
            for r in FULL_BY_VID[v]:
                by_split[r['split']].append(r)
        for s, rows in by_split.items():
            if any(r['key'] not in have[s] for r in rows):
                caption_rows(rows, f'{BASE}/captions/htstep_{s}.jsonl', use('gen'), g['variant'], extra)
                vram = max(vram, gpu_used_gb())
        for v in ok_vids:
            drop_local(v)
        check(f'{Path(tar).stem}: frame/audio tạm đã xóa',
              not any(os.path.exists(f'{LOCAL}/frames/htstep/{v}') or os.path.exists(f'{LOCAL}/audio/htstep/{v}.opus') for v in ok_vids))
        if ok_vids:
            mark_video_done('htstep', *ok_vids)
        vd, fs = videos_done(), failed_set()
        n_seg = sum(len(FULL_BY_VID[v]) for v in ok_vids)
        set_tar(tar, status='done', mode='full', session_id=SESSION_ID, n_ok=sum(f'htstep/{v}' in vd for v in mine),
                n_fail=sum(f'htstep/{v}' in fs for v in mine), seconds=round(time.time() - t0), segments=n_seg,
                vram_gb=round(vram, 2), keep_both=keep_both)
        vram_max = max(vram_max, vram)
        n_tars += 1
        print(f'  {tar}: {len(ok_vids)} video, {n_seg} segment, {time.time() - t0:.0f}s, VRAM {vram:.1f} GB', flush=True)
        if n_tars == 1 and n_seg:
            full_estimate((time.time() - t0) / n_seg, f'đo trên {tar} (tải + trích + ASR + caption)')
    M.clear()
    free()
    write_json(f'{BASE}/results/full_pass_vram.json', {'keep_both': keep_both, 'gpu_total_gb': round(total, 1),
                                                       'gen_peak_gb_cell5': sp.get('peak_vram_gb'),
                                                       'asr_gpu_used_gb_cell4': asp.get('gpu_used_gb'),
                                                       'full_pass_gpu_used_max_gb': round(vram_max, 2)})

if RUN_FULL_PASS:
    PROTO, PSHA = verify_protocol()
    missing, fails = full_gate()
    if missing:
        print('CHECK full pass gate STOP: thiếu', missing)
    elif fails and not FORCE_FULL_PASS:
        print('CHECK full pass gate STOP: tiêu chí trượt', fails, '-> chỉ chạy tiếp khi đặt FORCE_FULL_PASS = True ở ô 0')
    else:
        print('CHECK full pass gate OK' + (f' (FORCE_FULL_PASS, trượt: {fails})' if fails else ''))
        os.makedirs(TMP, exist_ok=True)
        print('--- ô 8b bắt đầu'); progress_report()
        run_full_pass(PROTO, PSHA)
        print('--- ô 8b kết thúc'); progress_report()
else:
    print('SKIP full pass:', {s: len(read_jsonl(f'{BASE}/captions/htstep_{s}.jsonl')) for s in HT_SPLITS if s != 'val'},
          f'caption / {len(FULL_ROWS)} segment')
''')

md(r'''
## 9. Đánh giá tự động (mục 5)

CI: bootstrap 1.000 lần theo video, seed cố định, trên trung bình theo segment. CIDEr/METEOR/ROUGE-L dùng điểm từng segment
(IDF của CIDEr tính trên toàn bộ caption tham chiếu, giống nhau cho mọi hệ so sánh), nên trung bình của chúng bằng điểm
toàn tập; BLEU-4 chỉ báo toàn tập.
''')

code(r'''
# ô 9a. (5.1) Độ giống caption người viết, YC2 val
def by_video(rows):
    d = defaultdict(list)
    for r in rows:
        d[r['video_id']].append(r)
    return d

if RUN_EVAL:
    PROTO, PSHA = verify_protocol()
    CAP_Y = load_caps('yc2_val')
    rows = [r for r in YC2_VAL if r['key'] in CAP_Y]
    check('yc2 val captions complete', len(rows) == len(YC2_VAL), f'{len(rows)}/{len(YC2_VAL)}')
    refs = {r['key']: r['gt_caption'] for r in rows}
    vid = {r['key']: r['video_id'] for r in rows}
    # sàn: caption sinh của segment kế tiếp trong cùng video (xoay vòng); bỏ video chỉ có 1 segment
    floor = {}
    for rs in by_video(rows).values():
        if len(rs) > 1:
            rs = sorted(rs, key=lambda r: r['seg_idx'])
            floor.update({a['key']: CAP_Y[b['key']]['caption'] for a, b in zip(rs, rs[1:] + rs[:1])})
    systems = {'pseudo': {k: CAP_Y[k]['caption'] for k in refs},
               'asr_as_caption': {r['key']: ' '.join(asr_window(r).split()[:MED]) for r in rows},
               'shuffled_in_video': floor}
    from bert_score import score as bert_score
    R51 = {'n_segments': len(rows), 'systems': {}}
    PER = {}
    for name, hyp in tqdm(systems.items(), desc='5.1 hệ'):
        sc, per = coco({k: refs[k] for k in hyp}, hyp)
        _, _, f1 = bert_score([hyp[k] or 'none' for k in hyp], [refs[k] for k in hyp], model_type='roberta-large',
                              lang='en', batch_size=64, verbose=False)
        per['BERTScore_F1'] = dict(zip(hyp, f1.tolist()))
        PER[name] = per
        R51['systems'][name] = {'BLEU4': sc['BLEU4'], **{m: boot_ci([per[m][k] for k in hyp], [vid[k] for k in hyp])
                                                         for m in ('METEOR', 'ROUGE_L', 'CIDEr', 'BERTScore_F1')}}
        s = R51['systems'][name]
        print(f"{name:18} n={len(hyp):5} BLEU4 {s['BLEU4'] * 100:5.2f} | METEOR {fmt(s['METEOR'])} | ROUGE_L {fmt(s['ROUGE_L'])}"
              f" | CIDEr {fmt(s['CIDEr'])} | BERTScore {fmt(s['BERTScore_F1'])}")
    keys = list(systems['pseudo'])
    R51['T2_cider_pseudo_minus_asr'] = boot_ci([PER['pseudo']['CIDEr'][k] - PER['asr_as_caption']['CIDEr'][k] for k in keys],
                                               [vid[k] for k in keys])
    print('CIDEr pseudo - ASR:', fmt(R51['T2_cider_pseudo_minus_asr']))
    R51['per_segment'] = [{'key': k, 'gt': refs[k], 'pseudo': systems['pseudo'][k], 'asr': systems['asr_as_caption'][k],
                           **{f'{m}_{n}': PER[n][m].get(k) for n in PER for m in ('CIDEr', 'BERTScore_F1')}} for k in keys]
    write_json(f'{BASE}/results/eval_5_1_yc2_val.json', R51)
    free()
    check('5.1 done', True)
else:
    print('SKIP 5.1')
''')

code(r'''
# ô 9b. (5.2) Bám bước theo từng split HT-Step: caption -> headline gần nhất trong các bước của task (sentence embedding).
#       val phải đủ caption; split khác (lượt full) tính trên phần đã có caption, ghi % hoàn thành.
if RUN_EVAL:
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(TXT, revision=TXT_REV, device='cuda')
    for split in HT_SPLITS:
        CAP_H = load_caps(f'htstep_{split}')
        scope = HT_VAL if split == 'val' else [r for r in FULL_ROWS if r['split'] == split]
        rows = [r for r in scope if r['key'] in CAP_H]
        if split == 'val':
            check('htstep val captions complete', len(rows) == len(HT_VAL), f'{len(rows)}/{len(HT_VAL)}')
        if not rows:
            print(f'  {split}: chưa có caption, bỏ')
            continue
        texts = {'pseudo': {r['key']: CAP_H[r['key']]['caption'] for r in rows},
                 'asr': {r['key']: asr_window(r) for r in tqdm(rows, desc=f'5.2 ASR {split}')}}
        pool = sorted({h for r in rows for h in r['task_headlines']} | {t for d in texts.values() for t in d.values()})
        E = dict(zip(pool, st.encode(pool, batch_size=128, normalize_embeddings=True, show_progress_bar=True)))

        def pred(r, t):
            return r['task_headlines'][max(range(len(r['task_headlines'])), key=lambda j: float(E[t] @ E[r['task_headlines'][j]]))]

        R52 = {'split': split, 'n_segments': len(rows), 'complete': round(len(rows) / len(scope), 4), 'model': [TXT, TXT_REV],
               'systems': {}}
        vids = [r['video_id'] for r in rows]
        for name, d in texts.items():
            hit = [pred(r, d[r['key']]) == r['headline'] for r in rows]       # headline trùng chữ coi là đúng
            R52['systems'][name] = boot_ci(hit, vids)
            if name == 'pseudo':
                nf = [(h, r['video_id']) for h, r in zip(hit, rows) if CAP_H[r['key']]['cap_source'] != 'headline_fallback']
                if nf:
                    R52['systems']['pseudo_no_fallback'] = boot_ci([h for h, _ in nf], [v for _, v in nf])
                errors = [{'key': r['key'], 'caption': d[r['key']], 'headline': r['headline'], 'chosen': pred(r, d[r['key']])}
                          for h, r in zip(hit, rows) if not h][:20]
        R52['systems']['random'] = boot_ci([1 / len(set(r['task_headlines'])) for r in rows], vids)
        print(f"  {split}: {len(rows)}/{len(scope)} segment ({R52['complete']:.1%})")
        for n, s in R52['systems'].items():
            print(f'    acc@1 {n:20} {fmt(s)}')
        R52['first_20_errors'] = errors
        if split == 'val':
            for e in errors:
                print(f"    x {e['caption']!r}\n        đúng: {e['headline']}\n        chọn: {e['chosen']}")
        write_json(f'{BASE}/results/eval_5_2_htstep_{split}.json', R52)
    st = None
    free()
    check('5.2 done', True)
else:
    print('SKIP 5.2')
''')

code(r'''
# ô 9c. (5.3) Bám hình: R@1 trong cùng video (ảnh = trung bình embedding 8 frame, text của mọi segment cùng video)
if RUN_EVAL:
    import numpy as np, torch.nn.functional as F
    from transformers import AutoModel
    vm = AutoModel.from_pretrained(VIS, revision=VIS_REV, torch_dtype=torch.float16).cuda().eval()
    vp = AutoProcessor.from_pretrained(VIS, revision=VIS_REV)
    TXT_KW = dict(padding='max_length', max_length=64, truncation=True) if 'siglip' in VIS else dict(padding=True, truncation=True, max_length=77)

    @torch.inference_mode()
    def img_emb(r):
        x = vp(images=load_frames(r), return_tensors='pt')['pixel_values'].cuda().half()
        f = F.normalize(vm.get_image_features(pixel_values=x).float(), dim=-1).mean(0)
        return F.normalize(f, dim=0).cpu().numpy()

    @torch.inference_mode()
    def txt_emb(texts):
        out = {}
        texts = sorted(set(texts))
        for i in range(0, len(texts), 64):
            t = vp(text=texts[i:i + 64], return_tensors='pt', **TXT_KW).to('cuda')
            f = F.normalize(vm.get_text_features(**t).float(), dim=-1).cpu().numpy()
            out.update(zip(texts[i:i + 64], f))
        return out

    def r1_in_video(rows, text_of, same, IMG, T):
        hit, vids = [], []
        for v, rs in by_video(rows).items():
            if len(rs) < 2:
                continue
            S = np.stack([IMG[r['key']] for r in rs]) @ np.stack([T[text_of(r)] for r in rs]).T
            for i, r in enumerate(rs):
                hit.append(float(same(r, rs[int(S[i].argmax())])))
                vids.append(v)
        return hit, vids

    R53 = {'model': [VIS, VIS_REV], 'yc2_val': {}, 'htstep_val': {},
           'convention_htstep': 'đúng nếu text đứng đầu có cùng step_id (một video có thể lặp một bước)'}
    for ds, rows, caps, systems, same in [
        ('yc2_val', YC2_VAL, load_caps('yc2_val'), {'pseudo': None, 'gt': 'gt_caption', 'asr': 'asr'},
         lambda a, b: a['key'] == b['key']),
        ('htstep_val', HT_VAL, load_caps('htstep_val'), {'pseudo': None, 'headline': 'headline', 'asr': 'asr'},
         lambda a, b: a['step_id'] == b['step_id'])]:
        rows = [r for r in rows if r['key'] in caps]
        IMG = {r['key']: img_emb(r) for r in tqdm(rows, desc=f'SigLIP frame {ds}')}
        getters = {n: (lambda r: caps[r['key']]['caption']) if f is None else (lambda r: asr_window(r)) if f == 'asr'
                   else (lambda r, f=f: r[f]) for n, f in systems.items()}
        T = txt_emb([g(r) for g in getters.values() for r in rows])
        hits = {}
        for n, g in getters.items():
            h, v = r1_in_video(rows, g, same, IMG, T)
            hits[n] = h
            R53[ds][n] = boot_ci(h, v)
            print(f'  {ds:10} R@1 {n:8} {fmt(R53[ds][n])}')
        other = 'gt' if ds == 'yc2_val' else 'headline'
        R53[ds][f'pseudo_minus_{other}'] = boot_ci(np.subtract(hits['pseudo'], hits[other]), v)
        print(f'  {ds:10} pseudo - {other}: {fmt(R53[ds][f"pseudo_minus_{other}"])}')
    write_json(f'{BASE}/results/eval_5_3_r1_in_video.json', R53)
    vm = None
    free()
    check('5.3 done', True)
else:
    print('SKIP 5.3')
''')

code(r'''
# ô 9d. (5.4a) Danh từ không căn cứ: danh từ của caption (spaCy lemma) không có trong ASR ∪ headline ∪ mô tả ∪ tên task
import spacy
nlp = spacy.load('en_core_web_sm', disable=['ner', 'parser'])
GENERIC = set("""bowl pan pot plate dish tray board knife spoon fork cup glass jar container lid oven stove heat water
mixture ingredient thing piece side top bottom minute second time way hand amount bit rest layer end edge center part
inch half tablespoon teaspoon tsp tbsp spatula whisk towel paper foil wrap surface counter sheet""".split())
_DOC = {}
def docs(texts):
    new = sorted({t for t in texts if t not in _DOC})
    _DOC.update(zip(new, nlp.pipe(new, batch_size=256)))
    return [_DOC[t] for t in texts]

def nouns(doc):
    return {t.lemma_.lower() for t in doc if t.pos_ == 'NOUN' and t.is_alpha} - GENERIC

def words(doc):
    return {t.lemma_.lower() for t in doc} | {t.lower_ for t in doc}

def ground_texts(r):
    return [asr_window(r), r['task']] + ([r['headline'], r['description']] if r['dataset'] == 'htstep' else [])

if RUN_EVAL:
    R54a = {'generic_nouns': sorted(GENERIC), 'systems': {}}
    systems = [('yc2_val_pseudo', YC2_VAL, lambda r, c=load_caps('yc2_val'): c[r['key']]['caption']),
               ('yc2_val_gt', YC2_VAL, lambda r: r['gt_caption']),
               ('htstep_val_pseudo', HT_VAL, lambda r, c=load_caps('htstep_val'): c[r['key']]['caption'])]
    for s in HT_SPLITS[1:]:                              # split của lượt full: phần đã có caption
        c = load_caps(f'htstep_{s}')
        rs = [r for r in FULL_ROWS if r['split'] == s and r['key'] in c]
        if rs:
            systems.append((f'htstep_{s}_pseudo', rs, lambda r, c=c: c[r['key']]['caption']))
    for name, rows, text_of in systems:
        caps = docs([text_of(r) for r in rows])
        bad, top = [], Counter()
        for r, d in tqdm(list(zip(rows, caps)), desc=f'5.4a {name}'):
            g = set().union(*map(words, docs(ground_texts(r))))
            u = nouns(d) - g
            bad.append(bool(u))
            top.update(u)
        R54a['systems'][name] = {**boot_ci(bad, [r['video_id'] for r in rows]), 'top_ungrounded': top.most_common(20)}
        print(f'  {name:18} n={len(rows)} có >= 1 danh từ không căn cứ: {fmt(R54a["systems"][name])}  vd: {top.most_common(6)}')
        _DOC.clear()                                     # Doc spaCy của hàng chục nghìn text tốn RAM
    write_json(f'{BASE}/results/eval_5_4a_nouns.json', R54a)
    check('5.4a done', True)
else:
    print('SKIP 5.4a')
''')

code(r'''
# ô 9e. (5.4b) Giám khảo MLLM (InternVL3, khác họ với Qwen): hiệu chỉnh trên YC2 val rồi chạy trên mẫu
from transformers import AutoModelForImageTextToText

def sample_by_video(rows, n, seed=SEED):
    """Xáo video theo seed, lấy mọi segment của từng video cho đến đủ n."""
    vids = sorted({r['video_id'] for r in rows})
    random.Random(seed).shuffle(vids)
    order = {v: i for i, v in enumerate(vids)}
    return sorted(rows, key=lambda r: (order[r['video_id']], r['seg_idx']))[:n]

JUDGE_Q = ('You check captions of cooking video clips. The images are 8 frames from the clip, in order.\n'
           'Transcript around the clip: "{asr}"\nCaption: "{cap}"\n'
           'Is every action, ingredient and tool mentioned in the caption supported by the frames or the transcript? '
           'Answer only with JSON: {{"supported": true or false, "unsupported": [words in the caption that are not supported]}}')

def parse_judge(txt):
    m = re.search(r'\{.*\}', txt, re.S)
    try:
        d = json.loads(m.group(0))
        return bool(d['supported']), d.get('unsupported', [])
    except Exception:
        m = re.search(r'"?supported"?\s*:\s*(true|false)', txt, re.I)
        return (m.group(1).lower() == 'true', []) if m else (None, [])

def ingredient_vocab(min_count=10):
    """Danh từ nguyên liệu: danh từ trong caption YC2 train, xuất hiện >= min_count lần, trừ danh sách chung chung/dụng cụ."""
    c = Counter(n for d in docs([t['sentence'] for t in TRAIN_TEXT]) for n in nouns(d))
    return sorted(w for w, k in c.items() if k >= min_count)

def corrupt(r, vocab, rng):
    """Thay một danh từ nguyên liệu của gt_caption bằng danh từ nguyên liệu khác không có trong ASR. None nếu không thay được."""
    d = docs([r['gt_caption']])[0]
    cand = [t for t in d if t.pos_ == 'NOUN' and t.lemma_.lower() in vocab]
    if not cand:
        return None
    t = rng.choice(cand)
    banned = words(docs([asr_window(r)])[0]) | words(d)
    new = rng.choice([w for w in vocab if w not in banned])
    return r['gt_caption'][:t.idx] + new + r['gt_caption'][t.idx + len(t.text):]

if RUN_JUDGE:
    PROTO, PSHA = verify_protocol()
    q4 = '8B' in JUDGE
    kw = dict(torch_dtype=torch.float16, device_map='cuda:0', revision=JUDGE_REV)
    if q4:
        kw['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_compute_dtype=torch.float16,
                                                       llm_int8_skip_modules=['vision_tower', 'multi_modal_projector', 'lm_head'])
    jm = AutoModelForImageTextToText.from_pretrained(JUDGE, **kw).eval()
    jp = AutoProcessor.from_pretrained(JUDGE, revision=JUDGE_REV)

    @torch.inference_mode()
    def judge(r, cap):
        msgs = [{'role': 'user', 'content': [{'type': 'image'}] * N_FRAMES +
                 [{'type': 'text', 'text': JUDGE_Q.format(asr=asr_window(r) or '(no speech)', cap=cap)}]}]
        text = jp.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inp = jp(text=text, images=load_frames(r), crop_to_patches=False, return_tensors='pt').to('cuda', torch.float16)
        out = jm.generate(**inp, do_sample=False, max_new_tokens=60, temperature=None, top_p=None, top_k=None)
        return jp.batch_decode(out[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0]

    def run_judge(name, items):
        """items [(row, caption)], cache judge/<name>.jsonl có resume."""
        path = f'{BASE}/judge/{name}.jsonl'
        have = {x['key'] for x in read_jsonl(path)}
        buf = []
        for i, (r, cap) in enumerate(tqdm(items, desc=f'judge {name}'), 1):
            if r['key'] in have:
                continue
            raw = judge(r, cap)
            ok, uns = parse_judge(raw)
            buf.append({'key': r['key'], 'video_id': r['video_id'], 'caption': cap, 'supported': ok, 'unsupported': uns, 'raw': raw})
            if len(buf) == 25 or i == len(items):
                append_jsonl(path, buf)
                buf = []
        append_jsonl(path, buf)
        res = {x['key']: x for x in read_jsonl(path)}
        check(f'judge {name} complete', all(r['key'] in res for r, _ in items), f'{len(res)}/{len(items)}')
        return [res[r['key']] for r, _ in items]

    CAP_Y, CAP_H = load_caps('yc2_val'), load_caps('htstep_val')
    cal = sample_by_video(YC2_VAL, N_JUDGE_CAL)
    vocab, rng = ingredient_vocab(), random.Random(SEED)
    bad_items = [(r, c) for r in cal for c in [corrupt(r, vocab, rng)] if c]
    res = {'cal_gt': run_judge('cal_gt', [(r, r['gt_caption']) for r in cal]),
           'cal_corrupt': run_judge('cal_corrupt', bad_items),
           'yc2_val_pseudo': run_judge('yc2_val_pseudo', [(r, CAP_Y[r['key']]['caption']) for r in sample_by_video(YC2_VAL, N_JUDGE_YC2)]),
           'htstep_val_pseudo': run_judge('htstep_val_pseudo', [(r, CAP_H[r['key']]['caption']) for r in sample_by_video(HT_VAL, N_JUDGE_HT)])}
    jm = None
    free()

    R54b = {'judge': [JUDGE, JUDGE_REV, '4bit' if q4 else 'fp16'], 'ingredient_vocab_size': len(vocab),
            'cal_corruptible': f'{len(bad_items)}/{len(cal)}', 'sets': {}}
    for name, xs in res.items():
        ok = [x for x in xs if x['supported'] is not None]
        val = [x['supported'] for x in ok] if name == 'cal_gt' else [not x['supported'] for x in ok]
        R54b['sets'][name] = {**boot_ci(val, [x['video_id'] for x in ok]), 'parse_fail': len(xs) - len(ok),
                              'metric': 'supported rate' if name == 'cal_gt' else 'unsupported rate'}
        print(f"  {name:18} {R54b['sets'][name]['metric']:16} {fmt(R54b['sets'][name])}  parse_fail {len(xs) - len(ok)}")
    for r, c in bad_items[:5]:
        print(f"  sửa: {r['gt_caption']!r} -> {c!r}")
    write_json(f'{BASE}/results/eval_5_4b_judge.json', R54b)
else:
    print('SKIP 5.4b')
''')

code(r'''
# ô 10. Bảng tiêu chí (mục 6) và kết luận tự động. T7 báo theo từng split; kết luận dùng T7 của val.
r51, r52, r53, r54b = (load_res(n) for n in ('eval_5_1_yc2_val', 'eval_5_2_htstep_val', 'eval_5_3_r1_in_video', 'eval_5_4b_judge'))
C, rows = CRITERIA, []
def crit(t, desc, value, ok):
    rows.append({'T': t, 'desc': desc, 'value': value, 'pass': ok})

if r54b:
    s = r54b['sets']
    t1 = s['cal_gt']['mean'] >= C['T1_judge_pass_gt'] and s['cal_corrupt']['mean'] >= C['T1_judge_catch_corrupt']
    crit('T1', 'giám khảo: qua GT thật / bắt caption sửa', f"{s['cal_gt']['mean']:.3f} / {s['cal_corrupt']['mean']:.3f}", t1)
    v = s['htstep_val_pseudo']
    crit('T6', 'HT-Step: tỉ lệ unsupported (giám khảo)', fmt(v), (v['mean'] <= C['T6_unsupported_max']) if t1 else 'N/A (T1 trượt: bỏ 5.4b)')
if r51:
    d = r51['T2_cider_pseudo_minus_asr']
    crit('T2', 'YC2 val: CIDEr pseudo - ASR, cận dưới CI > 0', fmt(d), d['lo'] > C['T2_cider_pseudo_minus_asr_lo'])
if r53:
    a, b = r53['yc2_val']['pseudo']['mean'], r53['yc2_val']['gt']['mean']
    crit('T3', 'YC2 val: R@1 pseudo / R@1 GT', f'{a:.3f} / {b:.3f} = {a / b:.3f}', a >= C['T3_r1_pseudo_over_gt'] * b)
    d = r53['htstep_val']['pseudo_minus_headline']
    crit('T5', 'HT-Step: R@1 pseudo - headline, cận dưới CI >= 0', fmt(d), d['lo'] >= C['T5_r1_pseudo_minus_headline_lo'])
if r52:
    v = r52['systems']['pseudo']
    crit('T4', 'HT-Step: bám bước acc@1', fmt(v), v['mean'] >= C['T4_step_acc1'])
for s in HT_SPLITS:
    caps_h = load_caps(f'htstep_{s}')
    if caps_h:
        fb = sum(c['cap_source'] == 'headline_fallback' for c in caps_h.values()) / len(caps_h)
        crit('T7' if s == 'val' else f'T7_{s}', f'HT-Step {s}: tỉ lệ headline_fallback ({len(caps_h)} caption)', f'{fb:.4f}',
             fb <= C['T7_fallback_max'])

rows.sort(key=lambda x: x['T'])
for x in rows:
    print(f"  {x['T']}  {str(x['pass']):28} {x['desc']:48} {x['value']}")
need = ['T2', 'T3', 'T4', 'T5', 'T6', 'T7']
got = {x['T']: x['pass'] for x in rows}
missing = [t for t in need if t not in got]
failed = [t for t in need if got.get(t) is False]
if missing:
    verdict = f'CHƯA ĐỦ: thiếu {missing}'
elif failed:
    verdict = f'headline (Mức 1) là cấu hình chính; pseudo-caption chỉ là phụ lục. Trượt: {failed}'
else:
    verdict = 'T2-T7 đạt (N/A không tính): dùng pseudo-caption làm text train cho HT-Step'
print('KẾT LUẬN:', verdict)
write_json(f'{BASE}/results/criteria.json', {'criteria': C, 'rows': rows, 'verdict': verdict})
check('criteria table written', True)
''')

code(r'''
# ô 11. Xuất HT-Step mọi split theo đúng định dạng annotation YC2 (pseudo và headline) + gom kết quả.
#       Chạy được khi split chưa xong: chỉ xuất segment đã có caption, ghi % hoàn thành.
SUBSET = {'val': 'validation', 'train': 'training'}
n_out = 0
for split in HT_SPLITS:
    caps_h = load_caps(f'htstep_{split}')
    scope = HT_VAL if split == 'val' else [r for r in FULL_ROWS if r['split'] == split]
    rows = sorted((r for r in scope if r['key'] in caps_h), key=lambda r: (r['video_id'], r['seg_idx']))
    done = len(rows) / len(scope) if scope else 0.0
    print(f'{split}: {len(rows)}/{len(scope)} segment có caption ({done:.1%})')
    if not rows:
        continue
    for kind in ('pseudo', 'headline'):
        db = {}
        for r in rows:
            v = db.setdefault(r['video_id'], {'duration': r['duration'], 'subset': SUBSET.get(split, split), 'recipe_type': r['task'],
                                              'annotations': [], 'video_url': f"https://www.youtube.com/watch?v={r['youtube_id']}"})
            c = caps_h[r['key']]
            v['annotations'].append({'segment': [r['start'], r['end']], 'id': r['seg_idx'],
                                     'sentence': c['caption'] if kind == 'pseudo' else clean(r['headline']),
                                     'step_id': r['step_id'], 'headline': r['headline'], 'cap_source': c['cap_source']})
        p = f'{BASE}/export/htstep_{split}_{kind}.json'
        write_json(p, {'database': db, 'complete': round(done, 4), 'segments': len(rows), 'segments_in_scope': len(scope),
                       'note': f'HT-Step {split}, chỉ video có trên HF và đã có caption; sentence của bản headline = '
                               'headline chữ thường, bỏ dấu chấm cuối'})
        print(f'  {kind}: {len(db)} video, {len(rows)} segment -> {p}')
    write_json(f'{BASE}/results/per_segment_htstep_{split}.json',
               [{'key': r['key'], 'video_id': r['video_id'], 'start': r['start'], 'end': r['end'], 'step_id': r['step_id'],
                 'headline': r['headline'], 'pseudo': caps_h[r['key']]['caption'], 'cap_source': caps_h[r['key']]['cap_source'],
                 'asr': asr_window(r)} for r in tqdm(rows, desc=f'per-segment {split}')])
    n_out += len(rows)
check('export written', n_out > 0, f'{n_out} segment')
print('Tải về pseudocap/results/: thư mục', f'{BASE}/results', '| pseudocap/export/:', f'{BASE}/export',
      '| pseudocap/configs/: htstep_cap_protocol.json')
''')


def main():
    nb = {"nbformat": 4, "nbformat_minor": 5,
          "metadata": {"accelerator": "GPU", "colab": {"gpuType": "T4", "provenance": []},
                       "kernelspec": {"display_name": "Python 3", "name": "python3"},
                       "language_info": {"name": "python"}},
          "cells": []}
    for i, (kind, src) in enumerate(CELLS):
        cell = {"cell_type": kind, "id": f"c{i:02d}", "metadata": {}, "source": src.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        nb["cells"].append(cell)
    p = ROOT / "notebooks" / "htstep_pseudocap_AZ.ipynb"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(p, len(CELLS), "cells")


if __name__ == "__main__":
    main()
