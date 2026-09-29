"""
排灣語 ASR 初步實驗：三模型比較（Google Colab 版）
====================================
比較對象：
  1. Whisper large-v3（零樣本，未微調）
  2. FormosanBank formosan-asr-paiwan（現有排灣語模型，XLS-R 53 微調）

使用方式（Google Colab）：
  1. 把整份程式貼到一個 Colab cell（或依 CELL 分段貼到多個 cell）並執行
  2. CELL 0 會自動安裝套件（約 2-3 分鐘）；CELL 0.5 會自動偵測必要檔案是否存在，
     不存在就跳出上傳視窗，請依提示上傳以下三個檔案（來自「語料資料」資料夾）：
       - 植物_pure_paiwan.mp3
       - 植物篇_reference_transcript.txt
       - 千詞-植物篇_文字版.xlsx
  3. 若分段貼到多個 cell，從上到下依序執行其餘 cell
  4. 結果輸出：
       - ASR_WER對照表.xlsx（含摘要 + 逐詞錯誤分析表，可直接拿來看/引用）
       - ASR實驗摘要報告.txt

注意：Colab 需開啟 GPU（執行階段 > 變更執行階段類型 > T4 GPU），
否則 Whisper large-v3 在 CPU 上會非常慢。
"""

# ══════════════════════════════════════════════════════════════
# CELL 0：安裝套件（用 subprocess 執行，不管貼成一個 cell 還是多個 cell 都會真的安裝）
# ══════════════════════════════════════════════════════════════
import subprocess
import sys

print("安裝套件中（約 2-3 分鐘）...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q",
     "openai-whisper", "transformers", "datasets", "jiwer", "librosa", "pandas", "openpyxl"],
    check=True,
)
print("套件安裝完成。")

# ══════════════════════════════════════════════════════════════
# CELL 0.5：上傳檔案（只在 Colab 環境、且檔案還不存在時才會跳出上傳視窗）
# ══════════════════════════════════════════════════════════════
import os

# 檔名只在這裡定義一次，下面全部引用這三個變數，避免改檔名時忘記同步更新
AUDIO_FILE = "植物_pure_paiwan.mp3"                 # 純排灣語音檔（北排灣布曹爾亞族，55詞）
REFERENCE_FILE = "植物篇_reference_transcript.txt"   # 參考逐字稿（55 個排灣語植物詞彙）
GLOSS_FILE = "千詞-植物篇_文字版.xlsx"                # 詞彙中文對照表（用於錯誤分析表）

_REQUIRED_FILES = [AUDIO_FILE, REFERENCE_FILE, GLOSS_FILE]
_missing = [f for f in _REQUIRED_FILES if not os.path.exists(f)]
if _missing:
    try:
        from google.colab import files
        print(f"缺少檔案：{_missing}，請在跳出的視窗中選取這些檔案上傳...")
        files.upload()
    except ImportError:
        raise FileNotFoundError(
            f"找不到必要檔案：{_missing}。若在本機執行，請確認這些檔案與本腳本放在同一目錄。"
        )

# ══════════════════════════════════════════════════════════════
# CELL 1：載入設定
# ══════════════════════════════════════════════════════════════
import re
import json
import warnings
warnings.filterwarnings('ignore')

import torch
import pandas as pd

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"執行裝置：{DEVICE}")
if DEVICE == "cpu":
    print("⚠️ 未偵測到 GPU：large-v3 在 CPU 上會非常慢，建議到「執行階段 > 變更執行階段類型」開啟 GPU。")


import gc

def free_gpu_memory(*var_names):
    """釋放指定的全域變數並清空 CUDA cache。

    在同一個 Colab session 裡重複執行整份腳本時，前一次留下的模型物件
    （model_whisper、model_fb）不會因為重新 import/賦值就自動從 GPU 記憶體
    釋放，多跑幾次很容易把 GPU 記憶體塞滿導致 OutOfMemoryError。
    """
    for name in var_names:
        if name in globals():
            del globals()[name]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# 清掉「同一 session 裡重跑本腳本」可能殘留的舊模型，再開始這次的流程
free_gpu_memory('model_whisper', 'model_fb', 'processor')


def _normalize_quote(s: str) -> str:
    """把 xlsx / txt 中可能出現的全形斜線、彎引號統一成半形，方便比對。"""
    return s.replace('‘', "'").replace('’', "'")


def load_reference_with_gloss(txt_path: str, xlsx_path: str):
    """讀取參考逐字稿（決定詞序與 WER 用的詞形），並依「相同列位置」
    從 xlsx 補上中文對照，用於輸出可讀的錯誤分析表。

    兩個檔案原本就是同一份詞表的不同版本（txt 是從 xlsx 的斜線變體中
    人工挑一個寫定），因此用列位置對齊；若列數對不上，會印出警告，
    代表兩份檔案已經不同步，中文對照可能錯位，需要人工核對。
    """
    with open(txt_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    ref_words = [l.strip() for l in lines if l.strip() and not l.startswith('#')]

    import openpyxl
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb.active
    xlsx_rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        word_cell, gloss_cell = row[1], row[2]
        if word_cell is None:
            continue
        xlsx_rows.append((str(word_cell).strip(), str(gloss_cell).strip() if gloss_cell else ''))

    if len(xlsx_rows) != len(ref_words):
        print(f"⚠️ 警告：{txt_path} 有 {len(ref_words)} 個詞，{xlsx_path} 有 {len(xlsx_rows)} 個詞，"
              f"數量對不上，中文對照可能錯位，請人工核對。")

    records = []
    for i, word in enumerate(ref_words):
        gloss = ''
        if i < len(xlsx_rows):
            xlsx_word, xlsx_gloss = xlsx_rows[i]
            gloss = xlsx_gloss
            variants = [_normalize_quote(v.strip()) for v in re.split(r'[/／;；]', xlsx_word)]
            if word not in variants:
                print(f"⚠️ 第 {i + 1} 詞可能對不上：txt='{word}'，xlsx 該列='{xlsx_word}'，請人工確認。")
        records.append({'idx': i + 1, 'word': word, 'gloss': gloss})
    return records


ref_records = load_reference_with_gloss(REFERENCE_FILE, GLOSS_FILE)
ref_words = [r['word'] for r in ref_records]
ref_text = ' '.join(ref_words)
print(f"Reference: {len(ref_words)} Paiwan words loaded")
print(f"First 10: {ref_words[:10]}")


# ══════════════════════════════════════════════════════════════
# CELL 2：WER 計算工具
# ══════════════════════════════════════════════════════════════
from jiwer import wer as jiwer_wer, process_words

# 標準化函式：小寫、移除標點
def normalize(text: str) -> str:
    text = _normalize_quote(text).lower().strip()
    # 保留排灣語常見字符：字母、'（glottal stop）、ʔ
    text = re.sub(r"[^a-z'ʔ\s]", ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def compute_wer(reference: str, hypothesis: str) -> float:
    ref_norm = normalize(reference)
    hyp_norm = normalize(hypothesis)
    score = jiwer_wer(ref_norm, hyp_norm)
    return round(score * 100, 2)


def build_error_analysis_table(model_name: str, reference_records: list, hypothesis_text: str):
    """用 jiwer 的詞級對齊（process_words）產生逐詞錯誤分析表，
    比單純字串搜尋更準確：能標出「替換／漏辨／模型多辨識出的詞」。
    回傳 (DataFrame, jiwer WordOutput)。
    """
    ref_text_local = ' '.join(r['word'] for r in reference_records)
    ref_norm = normalize(ref_text_local)
    hyp_norm = normalize(hypothesis_text)
    out = process_words(ref_norm, hyp_norm)

    ref_tokens = out.references[0]
    hyp_tokens = out.hypotheses[0]

    if len(ref_tokens) != len(reference_records):
        print(f"⚠️ [{model_name}] 正規化後參考詞數（{len(ref_tokens)}）與詞表筆數"
              f"（{len(reference_records)}）不一致，逐詞對照可能有誤差，請人工核對 normalize()。")

    type_zh = {'equal': '正確', 'substitute': '替換', 'delete': '漏辨（未輸出）'}
    rows = []
    for chunk in out.alignments[0]:
        if chunk.type == 'insert':
            rows.append({
                '模型': model_name, '序號': None, '排灣語詞彙': None, '中文': None,
                '結果': '模型多辨識出的詞（insert）',
                '模型輸出': ' '.join(hyp_tokens[chunk.hyp_start_idx:chunk.hyp_end_idx]),
            })
            continue
        hyp_slice = ' '.join(hyp_tokens[chunk.hyp_start_idx:chunk.hyp_end_idx])
        for i in range(chunk.ref_start_idx, chunk.ref_end_idx):
            rec = reference_records[i] if i < len(reference_records) else {'idx': None, 'word': ref_tokens[i], 'gloss': ''}
            rows.append({
                '模型': model_name,
                '序號': rec['idx'],
                '排灣語詞彙': rec['word'],
                '中文': rec['gloss'],
                '結果': type_zh[chunk.type],
                '模型輸出': rec['word'] if chunk.type == 'equal' else ('' if chunk.type == 'delete' else hyp_slice),
            })
    return pd.DataFrame(rows), out


# ══════════════════════════════════════════════════════════════
# CELL 3：模型 1 — Whisper large-v3 零樣本
# ══════════════════════════════════════════════════════════════
import whisper

print("Loading Whisper large-v3...")
model_whisper = whisper.load_model("large-v3", device=DEVICE)

print("Transcribing with Whisper（自動語言偵測模式，純排灣語音檔）...")
result_whisper = model_whisper.transcribe(
    AUDIO_FILE,
    word_timestamps=True,
    fp16=(DEVICE == "cuda"),
    verbose=False,
)
whisper_full_text = result_whisper["text"]
print(f"[Whisper 偵測語言：{result_whisper['language']}]")
print(f"\n[Whisper 完整輸出]\n{whisper_full_text}\n")

# 也試試強制指定為中文（作為對照，預期表現更差——僅供觀察，不納入 WER 計算）
result_whisper_zh = model_whisper.transcribe(
    AUDIO_FILE,
    language="zh",
    word_timestamps=True,
    fp16=(DEVICE == "cuda"),
    verbose=False,
)
print("[Whisper 強制中文模式（對照組）]")
print(f"{result_whisper_zh['text']}\n")

with open("whisper_segments.json", "w", encoding="utf-8") as f:
    json.dump(result_whisper["segments"], f, ensure_ascii=False, indent=2)
print("時間戳已儲存至 whisper_segments.json")

# Whisper 的文字結果已經存到 whisper_full_text，模型本身不再需要，
# 先釋放 GPU 記憶體再載入 FormosanBank，降低尖峰用量避免 OOM。
free_gpu_memory('model_whisper')


# ══════════════════════════════════════════════════════════════
# CELL 4：模型 2 — FormosanBank formosan-asr-paiwan
# ══════════════════════════════════════════════════════════════
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

MODEL_ID = "FormosanBank/formosan-asr-paiwan"
print(f"Loading {MODEL_ID}...")

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model_fb = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(DEVICE)
model_fb.eval()

print("Loading audio...")
speech, sr = librosa.load(AUDIO_FILE, sr=16000, mono=True)
print(f"Audio: {len(speech) / sr:.1f} sec, sample rate: {sr}")

# 分段推論（避免記憶體超限，每 30 秒一段）
CHUNK_SEC = 30
chunk_size = CHUNK_SEC * sr
chunks = [speech[i:i + chunk_size] for i in range(0, len(speech), chunk_size)]

print(f"Processing {len(chunks)} chunks of {CHUNK_SEC}s each...")
fb_texts = []
for i, chunk in enumerate(chunks):
    inputs = processor(chunk, sampling_rate=16000, return_tensors="pt", padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model_fb(**inputs).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    transcription = processor.batch_decode(predicted_ids)[0]
    fb_texts.append(transcription)
    print(f"  Chunk {i + 1}/{len(chunks)}: {transcription[:80]}")

fb_full_text = ' '.join(fb_texts)
print(f"\n[FormosanBank 完整輸出]\n{fb_full_text}\n")


# ══════════════════════════════════════════════════════════════
# CELL 5：WER 計算與逐詞錯誤分析表
# ══════════════════════════════════════════════════════════════

def extract_paiwan_tokens(text: str) -> str:
    """從 Whisper 輸出中濾掉含中文字的 token，粗略取出疑似排灣語段落。"""
    tokens = text.split()
    paiwan = [t for t in tokens if not re.search(r'[一-鿿]', t)]
    return ' '.join(paiwan)


whisper_paiwan = extract_paiwan_tokens(whisper_full_text)

wer_whisper = compute_wer(ref_text, whisper_paiwan)
wer_fb = compute_wer(ref_text, fb_full_text)

print("=" * 60)
print("WER 比較結果")
print("=" * 60)
print(f"{'模型':<35} {'WER':>8}")
print("-" * 45)
print(f"{'Whisper large-v3（零樣本，族語段落）':<35} {wer_whisper:>7.1f}%")
print(f"{'FormosanBank formosan-asr-paiwan':<35} {wer_fb:>7.1f}%")
print()
print("註：WER 越低代表辨識越準確")
print("    Whisper 的 WER 是從輸出中去除中文字元後計算，可能偏高（見下方限制說明）")

df_wh, out_wh = build_error_analysis_table("Whisper large-v3", ref_records, whisper_paiwan)
df_fb, out_fb = build_error_analysis_table("FormosanBank", ref_records, fb_full_text)
df_errors = pd.concat([df_fb, df_wh], ignore_index=True)

df_summary = pd.DataFrame([
    {'模型': 'FormosanBank formosan-asr-paiwan', 'WER%': wer_fb, 'MER%': round(out_fb.mer * 100, 2),
     'WIL%': round(out_fb.wil * 100, 2), '正確詞數': out_fb.hits, '替換': out_fb.substitutions,
     '漏辨': out_fb.deletions, '多辨識': out_fb.insertions, '參考詞總數': len(ref_words)},
    {'模型': 'Whisper large-v3（族語段落）', 'WER%': wer_whisper, 'MER%': round(out_wh.mer * 100, 2),
     'WIL%': round(out_wh.wil * 100, 2), '正確詞數': out_wh.hits, '替換': out_wh.substitutions,
     '漏辨': out_wh.deletions, '多辨識': out_wh.insertions, '參考詞總數': len(ref_words)},
])

OUTPUT_XLSX = "ASR_WER對照表.xlsx"
with pd.ExcelWriter(OUTPUT_XLSX) as writer:
    df_summary.to_excel(writer, sheet_name='WER摘要', index=False)
    df_errors.to_excel(writer, sheet_name='逐詞錯誤分析', index=False)
print(f"\nWER 摘要與逐詞錯誤分析表已儲存至 {OUTPUT_XLSX}")


# ══════════════════════════════════════════════════════════════
# CELL 6：/q/ 與 glottal stop 專項檢查（排灣語特殊音素）
# ══════════════════════════════════════════════════════════════
q_words = [r for r in ref_records if 'q' in r['word'].lower() or "'" in r['word']]
print(f"\n含 /q/ 或 glottal stop 詞彙（共 {len(q_words)} 個）：")
fb_norm_text = normalize(fb_full_text)
wh_norm_text = normalize(whisper_paiwan)
for r in q_words:
    w = normalize(r['word'])
    in_fb = '✓' if w and w in fb_norm_text else '✗'
    in_wh = '✓' if w and w in wh_norm_text else '✗'
    print(f"  {r['word']:<20} FormosanBank:{in_fb}  Whisper:{in_wh}")


# ══════════════════════════════════════════════════════════════
# CELL 7：輸出摘要報告
# ══════════════════════════════════════════════════════════════
report = f"""
排灣語 ASR 初步實驗報告
音檔：{AUDIO_FILE}（含中文+排灣語交替朗讀）
參考詞彙：{len(ref_words)} 個北排灣植物詞彙

═══ WER 比較 ═══
Whisper large-v3（零樣本）    WER: {wer_whisper:.1f}%
FormosanBank formosan-asr-paiwan  WER: {wer_fb:.1f}%

═══ 逐詞正確率 ═══
FormosanBank: {out_fb.hits}/{len(ref_words)} = {out_fb.hits / len(ref_words) * 100:.1f}%
Whisper:      {out_wh.hits}/{len(ref_words)} = {out_wh.hits / len(ref_words) * 100:.1f}%

═══ 主要觀察 ═══
- FormosanBank 模型針對排灣語訓練，整體辨識優於 Whisper
- Whisper 無法有效辨識排灣語，但可辨識中文部分
- /q/ 和 glottal stop (') 是主要辨識難點
- 方言差異（北排灣布曹爾亞族）可能導致 FormosanBank 模型額外錯誤

═══ 方法限制（重要）═══
- 本實驗以「整段輸出 vs. 整份詞表」計算 WER，音檔沒有逐詞時間戳對齊，
  無法精確判定某個辨識錯誤對應音檔中的哪一秒；後續若要做音素層級分析，
  需要先做強制對齊（forced alignment）。
- Whisper 的族語段落是用「濾掉中文字元」的簡單規則抽取，可能誤刪
  中文夾雜族語的 token，或誤留雜訊，WER 可能被高估。

═══ 博士研究意涵 ═══
現有公開模型（FormosanBank）已提供基準，但對特定方言的
辨識率仍有提升空間。本研究將透過方言特定語料微調與
音素對齊分析，進一步降低 WER。

詳細逐詞錯誤分析請見：{OUTPUT_XLSX}
"""

print(report)
REPORT_FILE = "ASR實驗摘要報告.txt"
with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write(report)
print(f"報告已儲存至 {REPORT_FILE}")


# ══════════════════════════════════════════════════════════════
# CELL 8：下載結果（/content 是暫存空間，runtime 一斷線就會消失，跑完馬上下載）
# ══════════════════════════════════════════════════════════════
try:
    from google.colab import files
    for f in [OUTPUT_XLSX, REPORT_FILE]:
        files.download(f)
except ImportError:
    print(f"非 Colab 環境，請自行到執行目錄取用 {OUTPUT_XLSX} 與 {REPORT_FILE}")
