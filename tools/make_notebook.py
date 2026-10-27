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

Plan: `Data/Htstep/HTSTEP_PSEUDO_CAPTION.md`. Metadata do `pseudocap/tools/*.py` tạo ở local, upload theo `metadata/UPLOAD.md`.

**Cách chạy.** Luôn chạy tuần tự từ ô 0; cờ `RUN_*` ở ô 0 chỉ bật/tắt phần nặng (ô tắt vẫn định nghĩa hàm cho ô sau).
Mọi kết quả trung gian nằm trên Drive và có resume, Colab ngắt thì chạy lại từ ô 0.

1. Phiên 1: `RUN_EXTRACT`, `RUN_ASR`, `RUN_SPEED`, `RUN_DEV_GRID` = True → ô 6 in sha256 của protocol.
2. Tải `htstep_cap_protocol.json` về `pseudocap/configs/`, dán sha256 vào `PROTOCOL_SHA256`.
3. Phiên 2: tắt 4 cờ trên, bật `RUN_YC2_VAL`, `RUN_HTSTEP_VAL`, `RUN_EVAL`, `RUN_JUDGE`.

Mỗi ô in `CHECK ... OK/FAIL`; `FAIL` dừng ô bằng exception.
''')

code(r'''
# ô 0. Cấu hình + hàm dùng chung
import os, re, gc, csv, json, time, random, shutil, hashlib, tarfile, inspect, subprocess
from pathlib import Path
from collections import Counter, defaultdict

DRIVE      = '/content/drive/MyDrive/Colab Notebooks/code KL'
BASE       = f'{DRIVE}/Fushion/htstep_cap'
META       = f'{BASE}/metadata'
YC2_VIDEOS = f'{DRIVE}/Omni/data/YouCookII/videos'     # = DRIVE_DATA/videos của notebook UEMR
HF_REPO    = 'nguyenminh04/htstep-videos'                # dataset public, video trong tar shard (hf_path)
LOCAL      = '/content/htcap'                            # đĩa Colab: video tạm, frame giải nén
SEED       = 0
N_BOOT     = 1000

RUN_EXTRACT    = True
RUN_ASR        = True
RUN_SPEED      = True
RUN_DEV_GRID   = True
RUN_YC2_VAL    = False    # chỉ bật sau khi đã điền PROTOCOL_SHA256
RUN_HTSTEP_VAL = False
RUN_EVAL       = False
RUN_JUDGE      = False
PROTOCOL_SHA256 = ''      # dán sha256 mà ô 6 in ra

GEN_MODELS   = {'q7b4': ('Qwen/Qwen2.5-VL-7B-Instruct', True),    # 4-bit NF4, fp16 compute
                'q3b':  ('Qwen/Qwen2.5-VL-3B-Instruct', False)}   # fp16
ASR_MODELS   = ['large-v3-turbo', 'medium.en']                    # [chính, dự phòng], faster-whisper
JUDGE_MODELS = ['OpenGVLab/InternVL3-8B-hf', 'OpenGVLab/InternVL3-2B-hf']
VIS_MODELS   = ['google/siglip-so400m-patch14-384', 'openai/clip-vit-large-patch14']
TXT_MODELS   = ['sentence-transformers/all-mpnet-base-v2', 'BAAI/bge-large-en-v1.5']
DECODE       = dict(do_sample=False, max_new_tokens=40)
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

def write_json(p, obj):
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)

def sha256_file(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def free():
    import torch
    gc.collect(); torch.cuda.empty_cache()

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
    for ds, vid in sorted({(r['dataset'], r['video_id']) for r in rows}):
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
# ô 1. Mount Drive, đọc metadata, kiểm khớp UPLOAD.md
from google.colab import drive
drive.mount('/content/drive')
for d in ['cache/frames/yc2', 'cache/frames/htstep', 'cache/audio/yc2', 'cache/audio/htstep', 'cache/done/yc2',
          'cache/done/htstep', 'asr/yc2', 'asr/htstep', 'captions/dev_grid', 'judge', 'results', 'export']:
    os.makedirs(f'{BASE}/{d}', exist_ok=True)
os.makedirs(LOCAL, exist_ok=True)

COUNTS = json.loads(re.search(r'<!-- COUNTS (\{.*?\}) -->', open(f'{META}/UPLOAD.md', encoding='utf-8').read(), re.S).group(1))
for name, (n, sha) in COUNTS.items():
    p = f'{META}/{name}'
    lines = sum(1 for _ in open(p, encoding='utf-8'))
    check(f'{name} lines', lines == n, f'{lines} vs UPLOAD.md {n}')
    check(f'{name} sha256', sha256_file(p) == sha)

MAN = read_jsonl(f'{META}/manifest.jsonl')
SPLIT = json.load(open(f'{META}/yc2_dev_val_split.json', encoding='utf-8'))
STATS = json.load(open(f'{META}/yc2_stats.json', encoding='utf-8'))
TRAIN_TEXT = read_jsonl(f'{META}/yc2_train_text.jsonl')
MED, P90 = STATS['words']['median'], STATS['words']['p90']
DEVVAL = set(SPLIT['dev']) | set(SPLIT['val'])
cnt = Counter(f"{r['dataset']}/{r['split']}" for r in MAN)
check('manifest splits', cnt == {'yc2/dev': 773, 'yc2/val': 3030, 'htstep/val': 3441}, str(dict(cnt)))
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
    print(f'  {k}: có video {n_ok}/{cnt[k]} segment ({n_ok / cnt[k]:.1%})')
''')

code(r'''
# ô 2. Cài thư viện (ghim phiên bản), in phiên bản, kiểm model trên HF
%pip -q install transformers==4.53.3 accelerate==1.9.0 bitsandbytes==0.46.1 faster-whisper==1.1.1 sentence-transformers==5.0.0 bert-score==0.3.13 pycocoevalcap==1.2 spacy==3.8.7
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
# ô 3. Tải video (YC2 từ Drive, HT-Step từ tar shard trên HF) -> 8 frame/segment + audio cả video, có resume
import cv2
from huggingface_hub import hf_hub_download
FAILED = f'{BASE}/cache/failed.csv'

def seg_times(s, e):
    return [s + (i + 0.5) * (e - s) / N_FRAMES for i in range(N_FRAMES)]

def done(ds, vid):
    return os.path.exists(f'{BASE}/cache/done/{ds}/{vid}.json')

def failed_set():
    if not os.path.exists(FAILED):
        return set()
    return {(x['dataset'], x['video_id']) for x in csv.DictReader(open(FAILED, encoding='utf-8'))}

def log_failed(ds, vid, err):
    new = not os.path.exists(FAILED)
    with open(FAILED, 'a', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        if new:
            w.writerow(['dataset', 'video_id', 'error'])
        w.writerow([ds, vid, str(err)[:300]])
    print('  FAILED', ds, vid, str(err)[:120])

def extract_video(path, ds, vid, segs):
    cap = cv2.VideoCapture(path)
    fps, nfr = cap.get(cv2.CAP_PROP_FPS), cap.get(cv2.CAP_PROP_FRAME_COUNT)
    dur = nfr / fps if fps > 0 and nfr > 0 else None
    last, n_repeat = None, 0
    for r in segs:
        out = frames_dir(r)
        os.makedirs(out, exist_ok=True)
        for i, t in enumerate(seg_times(r['start'], r['end'])):
            if dur:
                t = min(t, dur - 0.1)
            cap.set(cv2.CAP_PROP_POS_MSEC, max(t, 0) * 1000)
            ok, fr = cap.read()
            if not ok:                              # quá cuối video / lỗi giải mã: lặp frame trước
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
    if has_audio:
        shutil.copy(aud, f'{BASE}/cache/audio/{ds}/{vid}.opus')
    # Drive chậm với hàng chục nghìn file nhỏ: lưu frame thành 1 tar mỗi video, ô sau giải nén về LOCAL
    tmp = f'{LOCAL}/{vid}.tar'
    with tarfile.open(tmp, 'w') as t:
        t.add(f'{LOCAL}/frames/{ds}/{vid}', arcname=vid)
    shutil.copy(tmp, f'{BASE}/cache/frames/{ds}/{vid}.tar')
    os.remove(tmp)
    write_json(f'{BASE}/cache/done/{ds}/{vid}.json', {'segments': len(segs), 'repeated_frames': n_repeat,
                                                      'duration_cv2': dur, 'audio': has_audio, 'ffmpeg_err': res.stderr[-300:]})

BY_VID = defaultdict(list)
for r in ROWS:
    BY_VID[(r['dataset'], r['video_id'])].append(r)

if RUN_EXTRACT:
    todo = {k: v for k, v in BY_VID.items() if not done(*k) and k not in failed_set()}   # thử lại: xóa dòng trong failed.csv
    print('video cần xử lý:', Counter(ds for ds, _ in todo))
    os.makedirs(f'{LOCAL}/video', exist_ok=True)
    t0 = time.time()
    for i, ((ds, vid), segs) in enumerate(sorted((k, v) for k, v in todo.items() if k[0] == 'yc2'), 1):
        tmp = f'{LOCAL}/video/{vid}.mp4'
        try:
            shutil.copy(f'{YC2_VIDEOS}/{vid}.mp4', tmp)
            extract_video(tmp, ds, vid, segs)
        except Exception as e:
            log_failed(ds, vid, e)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
        if i % 50 == 0:
            print(f'  yc2 {i} video, {time.time() - t0:.0f}s', flush=True)

    shards = defaultdict(list)
    for (ds, vid), segs in todo.items():
        if ds == 'htstep':
            shards[segs[0]['hf_path']].append(vid)
    for shard, vids in sorted(shards.items()):
        t0 = time.time()
        try:
            tar_path = hf_hub_download(HF_REPO, shard, repo_type='dataset', local_dir=f'{LOCAL}/dl', token=HF_TOKEN)
        except Exception as e:
            for v in vids:
                log_failed('htstep', v, f'download {shard}: {e}')
            continue
        need, found = set(vids), set()
        with tarfile.open(tar_path) as t:
            for m in t:
                vid = Path(m.name).stem
                if not (m.isfile() and vid in need):
                    continue
                t.extract(m, f'{LOCAL}/video', filter='data')
                path = f'{LOCAL}/video/{m.name}'
                try:
                    extract_video(path, 'htstep', vid, BY_VID[('htstep', vid)])
                    found.add(vid)
                except Exception as e:
                    log_failed('htstep', vid, e)
                    found.add(vid)
                finally:
                    os.remove(path)
        for v in sorted(need - found):
            log_failed('htstep', v, f'not in {shard}')
        os.remove(tar_path)
        print(f'  {shard}: {len(found)}/{len(need)} video, {time.time() - t0:.0f}s', flush=True)

n_done = Counter(ds for ds, vid in BY_VID if done(ds, vid))
failed = failed_set()
left = [k for k in BY_VID if not done(*k) and k not in failed]
print('video xong:', dict(n_done), '| failed.csv:', len(failed), 'video | chưa xử lý:', len(left))
READY = restore_frames(ROWS)
print('segment có frame:', dict(Counter(f"{r['dataset']}/{r['split']}" for r in READY)))
if RUN_EXTRACT:
    check('every video done or logged as failed', not left, f'{len(left)} chưa xử lý')
check('frames ready', len(READY) > 0)
''')

code(r'''
# ô 4. ASR cho mọi video đã trích (cache asr/<dataset>/<video_id>.json, timestamp từng từ). Cùng một ASR cho YC2 và HT-Step.
READY_VIDS = sorted({(r['dataset'], r['video_id']) for r in READY})
if RUN_ASR:
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    wm = None
    for ASR_NAME in ASR_MODELS:
        try:
            wm = WhisperModel(ASR_NAME, device='cuda', compute_type='float16')
            break
        except Exception as e:
            print('ASR không tải được', ASR_NAME, e)
    check('asr model loaded', wm is not None, ASR_NAME)
    prev = {json.load(open(f'{BASE}/asr/{ds}/{v}.json'))['model'] for ds, v in READY_VIDS
            if os.path.exists(f'{BASE}/asr/{ds}/{v}.json')}
    check('same ASR model as cached files', prev <= {ASR_NAME}, str(prev))
    pipe = BatchedInferencePipeline(model=wm)
    t0, n_audio_s = time.time(), 0.0
    for i, (ds, vid) in enumerate(READY_VIDS, 1):
        out = f'{BASE}/asr/{ds}/{vid}.json'
        if os.path.exists(out):
            continue
        aud = f'{BASE}/cache/audio/{ds}/{vid}.opus'
        if not os.path.exists(aud):
            write_json(out, {'model': ASR_NAME, 'language': None, 'words': [], 'note': 'no audio stream'})
            continue
        segs, info = pipe.transcribe(aud, batch_size=16, word_timestamps=True)
        words = [[round(w.start, 2), round(w.end, 2), w.word.strip()] for s in segs for w in (s.words or [])]
        write_json(out, {'model': ASR_NAME, 'language': info.language,
                         'language_probability': round(info.language_probability, 3), 'words': words})
        n_audio_s += info.duration
        if i % 25 == 0:
            print(f'  {i}/{len(READY_VIDS)}  {n_audio_s / max(time.time() - t0, 1):.0f}x realtime', flush=True)
    wm = pipe = None
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
    buf, t0 = [], time.time()
    for i, r in enumerate(todo, 1):
        c, src, raw = caption_one(gen, r, variant)
        buf.append({'key': r['key'], 'caption': c, 'cap_source': src, 'prompt_variant': variant, 'raw': raw, **extra})
        if len(buf) == 50 or i == len(todo):
            append_jsonl(path, buf)
            buf = []
            print(f'  {os.path.basename(path)} {i}/{len(todo)}  {i / (time.time() - t0) * 3600:.0f} seg/h', flush=True)
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
        outs = [caption_one(gen, r, 'P2') for r in sample]
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
''')

code(r'''
# ô 6. Lưới chọn trên YC2 dev (P0/P1/P2 x 2 model) -> bảng CIDEr -> ghi protocol + sha256
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider

def coco(refs, hyps):
    """refs/hyps {key: str} -> (điểm toàn tập, {metric: {key: điểm}}). BLEU-4 chỉ báo toàn tập."""
    tok = PTBTokenizer()
    g = tok.tokenize({k: [{'caption': refs[k]}] for k in hyps})
    h = tok.tokenize({k: [{'caption': hyps[k] or 'none'}] for k in hyps})
    h = {k: v if v[0].strip() else ['none'] for k, v in h.items()}
    keys = list(g)
    h = {k: h[k] for k in keys}
    b, _ = Bleu(4).compute_score(g, h)
    out, per = {'BLEU4': b[3]}, {}
    for name, sc in [('METEOR', Meteor()), ('ROUGE_L', Rouge()), ('CIDEr', Cider())]:
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
        model, gen = load_gen(code, REV[code])
        for v in VARIANTS:
            caption_rows(DEV, f'{BASE}/captions/dev_grid/yc2_dev__{code}__{v}.jsonl', gen, v, {'model': code})
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
        'metadata_sha256': {k: v[1] for k, v in COUNTS.items()},
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

md(r'''
## 9. Đánh giá tự động (mục 5)

CI: bootstrap 1.000 lần theo video, seed cố định, trên trung bình theo segment. CIDEr/METEOR/ROUGE-L dùng điểm từng segment
(IDF của CIDEr tính trên toàn bộ caption tham chiếu, giống nhau cho mọi hệ so sánh), nên trung bình của chúng bằng điểm
toàn tập; BLEU-4 chỉ báo toàn tập.
''')

code(r'''
# ô 9a. (5.1) Độ giống caption người viết, YC2 val
def load_caps(name):
    return {r['key']: r for r in read_jsonl(f'{BASE}/captions/{name}.jsonl')}

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
    for name, hyp in systems.items():
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
# ô 9b. (5.2) Bám bước, HT-Step val: caption -> headline gần nhất trong các bước của task (sentence embedding)
if RUN_EVAL:
    from sentence_transformers import SentenceTransformer
    CAP_H = load_caps('htstep_val')
    rows = [r for r in HT_VAL if r['key'] in CAP_H]
    check('htstep val captions complete', len(rows) == len(HT_VAL), f'{len(rows)}/{len(HT_VAL)}')
    st = SentenceTransformer(TXT, revision=TXT_REV, device='cuda')
    texts = {'pseudo': {r['key']: CAP_H[r['key']]['caption'] for r in rows},
             'asr': {r['key']: asr_window(r) for r in rows}}
    pool = sorted({h for r in rows for h in r['task_headlines']} | {t for d in texts.values() for t in d.values()})
    E = dict(zip(pool, st.encode(pool, batch_size=128, normalize_embeddings=True, show_progress_bar=False)))

    def pred(r, t):
        return r['task_headlines'][max(range(len(r['task_headlines'])), key=lambda j: float(E[t] @ E[r['task_headlines'][j]]))]

    R52 = {'n_segments': len(rows), 'model': [TXT, TXT_REV], 'systems': {}}
    vids = [r['video_id'] for r in rows]
    for name, d in texts.items():
        hit = [pred(r, d[r['key']]) == r['headline'] for r in rows]       # headline trùng chữ coi là đúng
        R52['systems'][name] = boot_ci(hit, vids)
        if name == 'pseudo':
            nf = [(h, r['video_id']) for h, r in zip(hit, rows) if CAP_H[r['key']]['cap_source'] != 'headline_fallback']
            R52['systems']['pseudo_no_fallback'] = boot_ci([h for h, _ in nf], [v for _, v in nf])
            errors = [{'key': r['key'], 'caption': d[r['key']], 'headline': r['headline'], 'chosen': pred(r, d[r['key']])}
                      for h, r in zip(hit, rows) if not h][:20]
    R52['systems']['random'] = boot_ci([1 / len(set(r['task_headlines'])) for r in rows], vids)
    for n, s in R52['systems'].items():
        print(f'  acc@1 {n:20} {fmt(s)}')
    R52['first_20_errors'] = errors
    for e in errors:
        print(f"  x {e['caption']!r}\n      đúng: {e['headline']}\n      chọn: {e['chosen']}")
    write_json(f'{BASE}/results/eval_5_2_htstep_val.json', R52)
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
        IMG = {r['key']: img_emb(r) for r in rows}
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
    for name, rows, text_of in [
        ('yc2_val_pseudo', YC2_VAL, lambda r, c=load_caps('yc2_val'): c[r['key']]['caption']),
        ('yc2_val_gt', YC2_VAL, lambda r: r['gt_caption']),
        ('htstep_val_pseudo', HT_VAL, lambda r, c=load_caps('htstep_val'): c[r['key']]['caption'])]:
        caps = docs([text_of(r) for r in rows])
        bad, top = [], Counter()
        for r, d in zip(rows, caps):
            g = set().union(*map(words, docs(ground_texts(r))))
            u = nouns(d) - g
            bad.append(bool(u))
            top.update(u)
        R54a['systems'][name] = {**boot_ci(bad, [r['video_id'] for r in rows]), 'top_ungrounded': top.most_common(20)}
        print(f'  {name:18} có >= 1 danh từ không căn cứ: {fmt(R54a["systems"][name])}  vd: {top.most_common(6)}')
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
        out = jm.generate(**inp, do_sample=False, max_new_tokens=60)
        return jp.batch_decode(out[:, inp['input_ids'].shape[1]:], skip_special_tokens=True)[0]

    def run_judge(name, items):
        """items [(row, caption)], cache judge/<name>.jsonl có resume."""
        path = f'{BASE}/judge/{name}.jsonl'
        have = {x['key'] for x in read_jsonl(path)}
        buf = []
        for i, (r, cap) in enumerate(items, 1):
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
# ô 10. Bảng tiêu chí (mục 6) và kết luận tự động
def load_res(n):
    p = f'{BASE}/results/{n}.json'
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else None

r51, r52, r53, r54b = (load_res(n) for n in ('eval_5_1_yc2_val', 'eval_5_2_htstep_val', 'eval_5_3_r1_in_video', 'eval_5_4b_judge'))
caps_h = load_caps('htstep_val')
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
if caps_h:
    fb = sum(c['cap_source'] == 'headline_fallback' for c in caps_h.values()) / len(caps_h)
    crit('T7', 'HT-Step: tỉ lệ headline_fallback', f'{fb:.4f}', fb <= C['T7_fallback_max'])

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
# ô 11. Xuất HT-Step val theo đúng định dạng annotation YC2 (pseudo và headline) + gom kết quả
caps_h = load_caps('htstep_val')
rows = sorted((r for r in HT_VAL if r['key'] in caps_h), key=lambda r: (r['video_id'], r['seg_idx']))
out = {}
for kind in ('pseudo', 'headline'):
    db = {}
    for r in rows:
        v = db.setdefault(r['video_id'], {'duration': r['duration'], 'subset': 'validation', 'recipe_type': r['task'],
                                          'annotations': [], 'video_url': f"https://www.youtube.com/watch?v={r['youtube_id']}"})
        c = caps_h[r['key']]
        v['annotations'].append({'segment': [r['start'], r['end']], 'id': r['seg_idx'],
                                 'sentence': c['caption'] if kind == 'pseudo' else clean(r['headline']),
                                 'step_id': r['step_id'], 'headline': r['headline'], 'cap_source': c['cap_source']})
    out[kind] = f'{BASE}/export/htstep_val_{kind}.json'
    write_json(out[kind], {'database': db, 'note': 'HT-Step val_seen, chỉ video có trên HF; sentence của bản headline = '
                                                   'headline chữ thường, bỏ dấu chấm cuối'})
    print(kind, len(db), 'video,', sum(len(v['annotations']) for v in db.values()), 'segment ->', out[kind])
check('export matches captions', len(rows) == len(caps_h) and len(rows) > 0, f'{len(rows)} segment')
write_json(f'{BASE}/results/per_segment_htstep_val.json',
           [{'key': r['key'], 'video_id': r['video_id'], 'start': r['start'], 'end': r['end'], 'step_id': r['step_id'],
             'headline': r['headline'], 'pseudo': caps_h[r['key']]['caption'], 'cap_source': caps_h[r['key']]['cap_source'],
             'asr': asr_window(r)} for r in rows])
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
