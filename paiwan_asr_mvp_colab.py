"""
北排灣語 ASR MVP 實驗：切割音檔逐筆評估（Google Colab 版）
=========================================================

比較對象：
  1. FormosanBank/formosan-asr-paiwan（公開 Paiwan ASR baseline）
  2. Whisper large-v3（通用多語模型，零樣本對照）

適用資料（兩位發音人 x 兩種資料集，共 4 個 zip）：
  - F02 植物詞：植物_pure_paiwan_切割音檔.zip
  - F02 短句：voice_17142_姓名與問候_切割音檔.zip
  - F01 植物詞：F01_plants_切割音檔.zip
  - F01 短句：F01_sentences_切割音檔.zip

使用方式（Google Colab）：
  1. 上傳這 4 個 zip（每個 zip 內需包含 *.wav 與對應的切割清單 csv，
     實際檔名見下方 CELL 1 的 DATASETS 設定）。
  2. 把本程式貼到 Colab cell 執行。
  3. 若檔案不存在，程式會跳出上傳視窗。
  4. 輸出：
       - paiwan_asr_mvp_results.xlsx（summary 依 dataset x speaker_id x model_name 分開列，
         不合併成單一分數；per_item_results 逐筆結果；error_counts 錯誤類型統計）
       - paiwan_asr_mvp_summary_report.txt

注意：
  - Colab 請開 GPU（執行階段 > 變更執行階段類型 > T4 GPU）。
  - Whisper large-v3 很大；若 GPU 記憶體不足，可把 WHISPER_MODEL 改成 "medium" 或 "small"。
  - 本程式是 MVP baseline/error analysis，不做微調、不做 forced alignment。
  - 若之後再加新發音人/新資料集，只要在 CELL 1 的 DATASETS list 多加一筆設定即可，
    不需要改其他邏輯。
"""

# ══════════════════════════════════════════════════════════════
# CELL 0：安裝套件
# ══════════════════════════════════════════════════════════════
import subprocess
import sys

print("安裝套件中...")
subprocess.run(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "openai-whisper",
        "transformers",
        "jiwer",
        "librosa",
        "soundfile",
        "pandas",
        "openpyxl",
        "xlsxwriter",
    ],
    check=True,
)
print("套件安裝完成。")


# ══════════════════════════════════════════════════════════════
# CELL 1：設定與檔案上傳 / 解壓縮
# ══════════════════════════════════════════════════════════════
from pathlib import Path
import gc
import json
import os
import re
import shutil
import zipfile
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import torch
from jiwer import cer as jiwer_cer
from jiwer import process_words, wer as jiwer_wer


# 你可以依實際上傳檔名修改這裡。每個 dict 是一個「發音人 x 資料集」組合，
# 之後有新發音人（或新資料集）加入時，只要在這個 list 多加一筆，不用改下面的邏輯。
# 注意：DIR 要填 zip 內部實際的資料夾名稱（用 python -m zipfile -l 檢查），
# 不一定等於 zip 檔案本身的檔名。若你的本機 zip/csv 仍是舊檔名，請先改名
# 為下方 "zip"/"manifest" 欄位指定的中性檔名，或同步修改這裡的路徑字串。
DATASETS = [
    {
        "dataset": "plants",
        "zip": "植物_pure_paiwan_切割音檔.zip",
        "dir": Path("植物_pure_paiwan_切割音檔"),
        "manifest": "植物_pure_paiwan_切割清單.csv",
    },
    {
        "dataset": "sentences",
        "zip": "voice_17142_姓名與問候_切割音檔.zip",
        "dir": Path("voice_17142_姓名與問候_切割音檔"),
        "manifest": "voice_17142_姓名與問候_切割清單.csv",
    },
    {
        "dataset": "plants",
        "zip": "F01_plants_切割音檔.zip",
        "dir": Path("植物_切割音檔"),
        "manifest": "F01_plants_切割清單.csv",
    },
    {
        "dataset": "sentences",
        "zip": "F01_sentences_切割音檔.zip",
        "dir": Path("句子_切割音檔"),
        "manifest": "F01_sentences_切割清單.csv",
    },
]
for _d in DATASETS:
    _d["manifest"] = _d["dir"] / _d["manifest"]

WHISPER_MODEL = "large-v3"
FORMOSANBANK_MODEL = "FormosanBank/formosan-asr-paiwan"

RUN_FORMOSANBANK = True
RUN_WHISPER = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"執行裝置：{DEVICE}")
if DEVICE == "cpu":
    print("提醒：目前未偵測到 GPU，Whisper large-v3 會非常慢。")


def ensure_uploaded(required_files):
    """Colab 上傳輔助：缺檔時跳出上傳視窗。"""
    missing = [f for f in required_files if not Path(f).exists()]
    if not missing:
        return
    try:
        from google.colab import files

        print(f"缺少檔案：{missing}")
        print("請在跳出的視窗中上傳。")
        files.upload()
    except ImportError as exc:
        raise FileNotFoundError(f"找不到必要檔案：{missing}") from exc


def _fix_mojibake_name(name: str) -> str:
    """許多在 macOS 上用 zip/Finder 壓縮的檔案不會設定 ZIP 的 UTF-8 旗標，
    zipfile 預設會用 CP437 解碼檔名，中文檔名/資料夾名因此變成亂碼
    （例如 植物_pure_paiwan_切割音檔 讀出來變成 µñìτë⌐_..._σêçσë▓Θƒ│µ¬ö）。
    這裡嘗試把讀到的亂碼字串「還原」成原本的 UTF-8 檔名。
    """
    try:
        return name.encode("cp437").decode("utf-8")
    except (UnicodeDecodeError, UnicodeEncodeError):
        return name  # 不是這種編碼問題（例如本來就是純英數檔名），維持原樣


def unzip_if_needed(zip_path, target_dir):
    zip_path = Path(zip_path)
    target_dir = Path(target_dir)
    if target_dir.exists() and list(target_dir.glob("*.wav")):
        print(f"已找到資料夾：{target_dir}")
        return
    if not zip_path.exists():
        return
    print(f"解壓縮 {zip_path} -> {target_dir}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            # 若 ZIP 本身已標記 UTF-8（flag_bits 第 11 位），檔名可信，不用修正
            name = info.filename if (info.flag_bits & 0x800) else _fix_mojibake_name(info.filename)
            dest = Path(".") / name
            if info.is_dir() or name.endswith("/"):
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(dest, "wb") as dst:
                shutil.copyfileobj(src, dst)


ensure_uploaded([d["zip"] for d in DATASETS])
for _d in DATASETS:
    unzip_if_needed(_d["zip"], _d["dir"])

_missing_manifests = [str(_d["manifest"]) for _d in DATASETS if not _d["manifest"].exists()]
if _missing_manifests:
    raise FileNotFoundError("找不到切割清單。請確認 zip 內包含：\n" + "\n".join(f"- {m}" for m in _missing_manifests))


# ══════════════════════════════════════════════════════════════
# CELL 2：資料讀取與評估工具
# ══════════════════════════════════════════════════════════════
def normalize_text(text: str) -> str:
    """ASR 評估用正規化：保留排灣語常見字母、apostrophe 與空白。"""
    text = str(text or "")
    text = text.replace("‘", "'").replace("’", "'").replace("ʻ", "'")
    text = text.lower().strip()
    text = re.sub(r"[^a-z'ʔ\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_metrics(reference: str, hypothesis: str):
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if not ref:
        return {"wer": None, "cer": None, "exact_match": False, "hits": 0, "sub": 0, "del": 0, "ins": 0}
    out = process_words(ref, hyp)
    return {
        "wer": round(jiwer_wer(ref, hyp) * 100, 2),
        "cer": round(jiwer_cer(ref, hyp) * 100, 2),
        "exact_match": ref == hyp,
        "hits": out.hits,
        "sub": out.substitutions,
        "del": out.deletions,
        "ins": out.insertions,
    }


def classify_error(reference: str, hypothesis: str) -> str:
    """MVP 用初步錯誤分類，正式論文前仍需人工/專家檢核。"""
    ref = normalize_text(reference)
    hyp = normalize_text(hypothesis)
    if ref == hyp:
        return "exact"
    if not hyp:
        return "no_output"

    labels = []
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()

    if len(ref_tokens) > 1 and len(hyp_tokens) < len(ref_tokens):
        labels.append("word_boundary_or_deletion")
    if len(hyp_tokens) > len(ref_tokens):
        labels.append("insertion")
    if "'" in ref and "'" not in hyp:
        labels.append("glottal_missing")
    if "q" in ref and "q" not in hyp:
        labels.append("q_related")
    if set(ref.replace(" ", "")) & set(hyp.replace(" ", "")):
        labels.append("spelling_shift")
    if not labels:
        labels.append("substitution")
    return "; ".join(labels)


def read_manifest(path: Path, dataset_name: str, audio_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["dataset"] = dataset_name
    df["audio_path"] = df["audio_file"].apply(lambda x: str(audio_dir / str(x)))
    missing = df[~df["audio_path"].apply(lambda p: Path(p).exists())]
    if len(missing) > 0:
        raise FileNotFoundError(f"{dataset_name} 有音檔缺失：\n{missing[['item_id','audio_file']].head(10)}")
    return df


df_items = pd.concat(
    [read_manifest(_d["manifest"], _d["dataset"], _d["dir"]) for _d in DATASETS],
    ignore_index=True,
)
_counts = df_items.groupby(["dataset", "speaker_id"]).size()
print(f"載入資料：total={len(df_items)}")
print(_counts.to_string())
display(df_items[["dataset", "item_id", "speaker_id", "paiwan", "chinese", "audio_file"]].head())


def free_gpu_memory(*names):
    for name in names:
        if name in globals():
            del globals()[name]
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ══════════════════════════════════════════════════════════════
# CELL 3：FormosanBank 逐音檔推論
# ══════════════════════════════════════════════════════════════
def run_formosanbank(df: pd.DataFrame) -> pd.DataFrame:
    import librosa
    from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

    print(f"Loading {FORMOSANBANK_MODEL}...")
    processor = Wav2Vec2Processor.from_pretrained(FORMOSANBANK_MODEL)
    model = Wav2Vec2ForCTC.from_pretrained(FORMOSANBANK_MODEL).to(DEVICE)
    model.eval()

    rows = []
    for idx, row in df.iterrows():
        audio_path = row["audio_path"]
        speech, sr = librosa.load(audio_path, sr=16000, mono=True)
        inputs = processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_ids = torch.argmax(logits, dim=-1)
        hyp = processor.batch_decode(predicted_ids)[0]
        print(f"[FormosanBank] {idx + 1:03d}/{len(df)} {row['item_id']}: {hyp}")
        rows.append(
            {
                **row.to_dict(),
                "model_name": "FormosanBank/formosan-asr-paiwan",
                "model_output": hyp,
            }
        )

    free_gpu_memory("model", "processor")
    return pd.DataFrame(rows)


df_fb = run_formosanbank(df_items) if RUN_FORMOSANBANK else pd.DataFrame()


# ══════════════════════════════════════════════════════════════
# CELL 4：Whisper 逐音檔推論
# ══════════════════════════════════════════════════════════════
def run_whisper(df: pd.DataFrame) -> pd.DataFrame:
    import whisper

    print(f"Loading Whisper {WHISPER_MODEL}...")
    model = whisper.load_model(WHISPER_MODEL, device=DEVICE)

    rows = []
    for idx, row in df.iterrows():
        result = model.transcribe(
            row["audio_path"],
            fp16=(DEVICE == "cuda"),
            verbose=False,
        )
        hyp = result.get("text", "")
        print(f"[Whisper] {idx + 1:03d}/{len(df)} {row['item_id']}: {hyp}")
        rows.append(
            {
                **row.to_dict(),
                "model_name": f"Whisper {WHISPER_MODEL}",
                "model_output": hyp,
                "whisper_detected_language": result.get("language", ""),
            }
        )

    free_gpu_memory("model")
    return pd.DataFrame(rows)


df_whisper = run_whisper(df_items) if RUN_WHISPER else pd.DataFrame()


# ══════════════════════════════════════════════════════════════
# CELL 5：計算 WER/CER/逐筆錯誤類型
# ══════════════════════════════════════════════════════════════
df_results = pd.concat([df_fb, df_whisper], ignore_index=True)
if df_results.empty:
    raise RuntimeError("沒有任何模型結果，請確認 RUN_FORMOSANBANK/RUN_WHISPER。")

metric_rows = []
for _, row in df_results.iterrows():
    m = compute_metrics(row["paiwan"], row["model_output"])
    metric_rows.append(m)

df_metrics = pd.DataFrame(metric_rows)
df_results = pd.concat([df_results.reset_index(drop=True), df_metrics], axis=1)
df_results["error_type"] = df_results.apply(lambda r: classify_error(r["paiwan"], r["model_output"]), axis=1)
df_results["normalized_reference"] = df_results["paiwan"].apply(normalize_text)
df_results["normalized_output"] = df_results["model_output"].apply(normalize_text)


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """依 dataset x speaker_id x model_name 分開統計。

    不同發音人的口音、錄音品質都不同，混在一起算單一分數會掩蓋掉
    發音人之間的差異，所以固定分開報告，不提供跨發音人的合併分數。
    """
    rows = []
    group_cols = ["dataset", "speaker_id", "model_name"]
    for keys, g in df.groupby(group_cols):
        dataset, speaker_id, model_name = keys
        rows.append(
            {
                "dataset": dataset,
                "speaker_id": speaker_id,
                "model_name": model_name,
                "n_items": len(g),
                "avg_WER%": round(g["wer"].mean(), 2),
                "avg_CER%": round(g["cer"].mean(), 2),
                "exact_match_count": int(g["exact_match"].sum()),
                "exact_match_rate%": round(g["exact_match"].mean() * 100, 2),
                "sub_total": int(g["sub"].sum()),
                "del_total": int(g["del"].sum()),
                "ins_total": int(g["ins"].sum()),
            }
        )
    return pd.DataFrame(rows)


df_summary = summarize(df_results)

df_error_counts = (
    df_results.groupby(["dataset", "speaker_id", "model_name", "error_type"])
    .size()
    .reset_index(name="count")
    .sort_values(["dataset", "speaker_id", "model_name", "count"], ascending=[True, True, True, False])
)

print("\n=== Summary ===")
display(df_summary)
print("\n=== Error counts ===")
display(df_error_counts.head(30))


# ══════════════════════════════════════════════════════════════
# CELL 6：輸出 Excel 與文字報告
# ══════════════════════════════════════════════════════════════
OUTPUT_XLSX = "paiwan_asr_mvp_results.xlsx"
REPORT_FILE = "paiwan_asr_mvp_summary_report.txt"

keep_cols = [
    "dataset",
    "item_id",
    "speaker_id",
    "speaker_role",
    "paiwan",
    "chinese",
    "sentence_category",
    "variable_type",
    "recording_priority",
    "audio_file",
    "duration_sec",
    "model_name",
    "model_output",
    "normalized_reference",
    "normalized_output",
    "wer",
    "cer",
    "exact_match",
    "hits",
    "sub",
    "del",
    "ins",
    "error_type",
]
for col in keep_cols:
    if col not in df_results.columns:
        df_results[col] = ""

with pd.ExcelWriter(OUTPUT_XLSX, engine="xlsxwriter") as writer:
    df_summary.to_excel(writer, sheet_name="summary", index=False)
    df_results[keep_cols].to_excel(writer, sheet_name="per_item_results", index=False)
    df_error_counts.to_excel(writer, sheet_name="error_counts", index=False)

    workbook = writer.book
    header_fmt = workbook.add_format({"bold": True, "bg_color": "#0F6D8F", "font_color": "white", "border": 1})
    wrap_fmt = workbook.add_format({"text_wrap": True, "valign": "top"})
    for sheet_name, data in {
        "summary": df_summary,
        "per_item_results": df_results[keep_cols],
        "error_counts": df_error_counts,
    }.items():
        ws = writer.sheets[sheet_name]
        for col_num, value in enumerate(data.columns.values):
            ws.write(0, col_num, value, header_fmt)
        ws.freeze_panes(1, 0)
        ws.autofilter(0, 0, len(data), max(0, len(data.columns) - 1))
        for i, col in enumerate(data.columns):
            width = min(max(len(str(col)) + 2, 12), 36)
            if col in ["paiwan", "chinese", "model_output", "error_type", "audio_file"]:
                width = 28
            ws.set_column(i, i, width, wrap_fmt)

_dataset_counts = df_items.groupby(["dataset", "speaker_id"]).size()
report_lines = [
    "北排灣語 ASR MVP baseline/error analysis",
    "=" * 48,
    f"資料筆數（依 dataset x speaker_id）：",
    *[f"  - {k[0]} / {k[1]}: {v}" for k, v in _dataset_counts.items()],
    f"總筆數：{len(df_items)}",
    f"發音人：{sorted(df_items['speaker_id'].dropna().unique().tolist())}",
    "",
    "模型摘要（依發音人分開統計，不合併成單一分數）：",
]
for _, r in df_summary.iterrows():
    report_lines.append(
        f"- {r['dataset']} / {r['speaker_id']} / {r['model_name']}: "
        f"avg WER={r['avg_WER%']}%, avg CER={r['avg_CER%']}%, "
        f"exact={r['exact_match_count']}/{r['n_items']} ({r['exact_match_rate%']}%)"
    )

report_lines += [
    "",
    "方法限制：",
    "- 本實驗為逐筆切割音檔 baseline 測試，尚未進行 forced alignment 或音素層級標註。",
    "- error_type 為初步規則分類，正式論文前需由族語專家檢核。",
    "- 不同發音人（F01／F02）的分數刻意分開列出，不做跨發音人合併，"
    "避免掩蓋個人口音/錄音品質差異。",
]

with open(REPORT_FILE, "w", encoding="utf-8") as f:
    f.write("\n".join(report_lines))

print(f"Excel 已輸出：{OUTPUT_XLSX}")
print(f"報告已輸出：{REPORT_FILE}")

try:
    from google.colab import files

    files.download(OUTPUT_XLSX)
    files.download(REPORT_FILE)
except ImportError:
    print("非 Colab 環境，請自行取用輸出檔。")

