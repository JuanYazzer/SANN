import os
import io
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from urllib.parse import urlparse, parse_qs
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler

PROJECT_ROOT = Path(__file__).resolve().parent
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from inferenceDAP import predict_and_explain, load_saved_sann_model

MONITORED_CSV_PATH = PROJECT_ROOT / "Monitored_Codes_Dipantau.csv"
SAVED_MODEL_DIR = PROJECT_ROOT / "saved_model"
SAVED_MODEL_DAP_DIR = PROJECT_ROOT / "saved_model_dap"
EVAL_CACHE_JSON = PROJECT_ROOT / "saved_model" / "monitored_test_evaluation.json"
EVAL_RESULTS_CSV = PROJECT_ROOT / "Monitored_Test_Evaluation_Results.csv"


def get_active_model_dir():
    if (SAVED_MODEL_DIR / "sann_dap_weights.h5").exists() and (SAVED_MODEL_DIR / "metadata.json").exists():
        return SAVED_MODEL_DIR
    return SAVED_MODEL_DAP_DIR


def load_training_info():
    model_dir = get_active_model_dir()
    meta_path = model_dir / "metadata.json"
    summary_path = model_dir / "training_summary.json"
    info = {}
    if meta_path.exists():
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            info.update({
                "max_depth": meta.get("max_depth"),
                "max_subtree_length": meta.get("max_subtree_length"),
                "node_vocab_size": meta.get("node_vocab_size"),
                "path_vocab_size": meta.get("path_vocab_size"),
                "embedding_size": meta.get("embedding_size"),
                "train_samples": meta.get("train_samples"),
                "val_samples": meta.get("val_samples"),
                "total_compilable_samples": meta.get("total_compilable_samples"),
                "val_accuracy": meta.get("val_accuracy"),
                "val_precision": meta.get("val_precision"),
                "val_recall": meta.get("val_recall"),
                "val_f1": meta.get("val_f1"),
                "epochs_trained": meta.get("epochs_trained"),
            })
        except Exception:
            pass
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                summary = json.load(f)
            info["history"] = summary.get("history", {})
            info["dataset_source"] = summary.get("dataset_source", "")
        except Exception:
            pass
    info["model_dir"] = str(model_dir)
    info["saved_model_exists"] = (SAVED_MODEL_DIR / "sann_dap_weights.h5").exists()
    info["saved_model_dap_exists"] = (SAVED_MODEL_DAP_DIR / "sann_dap_weights.h5").exists()
    return info


def evaluate_monitored_codes(force_refresh=False):
    model_dir = get_active_model_dir()
    weights_path = model_dir / "sann_dap_weights.h5"
    weights_mtime = weights_path.stat().st_mtime if weights_path.exists() else 0
    csv_mtime = MONITORED_CSV_PATH.stat().st_mtime if MONITORED_CSV_PATH.exists() else 0

    if not force_refresh and EVAL_CACHE_JSON.exists():
        try:
            with open(EVAL_CACHE_JSON, "r", encoding="utf-8") as f:
                cached = json.load(f)
            if (
                cached.get("weights_mtime") == weights_mtime
                and cached.get("csv_mtime") == csv_mtime
                and cached.get("items")
            ):
                return cached
        except Exception:
            pass

    load_saved_sann_model.cache_clear()
    df = pd.read_csv(MONITORED_CSV_PATH)
    items = []
    export_rows = []

    for idx, r in df.iterrows():
        code_str = str(r.get("Code_Yang_Disimpan", ""))
        gt_str = str(r.get("Hasilnya", "")).strip().lower()
        gt_label = 1 if gt_str == "correct" else 0
        orig_state = str(r.get("Status_Asli_Code", ""))
        orig_gt_label = 1 if orig_state.strip().lower().startswith("correct") else 0

        res = predict_and_explain(code_str, model_dir=model_dir, print_report=False)
        pred_cls = int(res["predicted_class"])
        model_pred_cls = int(res.get("model_predicted_class", pred_cls))
        probs = res["probabilities"]
        line_scores = {int(k): float(v) for k, v in res["line_scores"].items()}
        ranked_lines = res["ranked_lines"]
        lines = code_str.splitlines()

        top_line_no = ranked_lines[0] if ranked_lines else None
        top_line_code = lines[top_line_no - 1].strip() if top_line_no and 1 <= top_line_no <= len(lines) else ""
        top_line_weight = line_scores.get(top_line_no, 0.0) if top_line_no else 0.0

        line_breakdown = []
        for ln_no, ln_text in enumerate(lines, start=1):
            w = line_scores.get(ln_no, 0.0)
            rank_pos = ranked_lines.index(ln_no) + 1 if ln_no in ranked_lines else None
            is_syn_err = bool(res.get("syntax_error") and res["syntax_error"].get("line") == ln_no)
            line_breakdown.append({
                "line_no": ln_no,
                "code": ln_text,
                "weight": round(w, 4),
                "weight_pct": round(w * 100, 1),
                "rank": rank_pos,
                "is_syntax_error": is_syn_err,
            })

        is_match = (pred_cls == gt_label)
        is_model_match = (model_pred_cls == gt_label)
        is_orig_match = (model_pred_cls == orig_gt_label)

        item = {
            "index": int(idx + 1),
            "nama": str(r.get("Nama", "")),
            "kelas": str(r.get("Kelas", "")),
            "minggu": str(r.get("Minggu_Ke_Berapa", "")),
            "nama_soal": str(r.get("Nama_Soal", "")),
            "misconception": str(r.get("Jenis_Misconception_Minggu_Itu", "")),
            "sumber_code": str(r.get("Sumber_Code", "")),
            "code": code_str,
            "hasil_gt": gt_str,
            "gt_label": gt_label,
            "kategori_pantauan": str(r.get("Kategori_Pantauan", "")),
            "status_asli_code": orig_state,
            "orig_gt_label": orig_gt_label,
            "waktu_disimpan": str(r.get("Waktu_Disimpan", "")),
            "predicted_class": pred_cls,
            "model_predicted_class": model_pred_cls,
            "predicted_label": res["predicted_label"],
            "prob_correct": round(probs["correct_1"] * 100, 2),
            "prob_incorrect": round(probs["incorrect_0"] * 100, 2),
            "confidence": round(res["confidence"] * 100, 2),
            "syntax_error": res.get("syntax_error"),
            "is_match": is_match,
            "is_model_match": is_model_match,
            "is_orig_match": is_orig_match,
            "top_line_no": top_line_no,
            "top_line_code": top_line_code,
            "top_line_weight_pct": round(top_line_weight * 100, 1),
            "line_breakdown": line_breakdown,
            "top_subtrees": [
                {
                    "subtree_index": st["subtree_index"],
                    "weight_pct": round(st["weight"] * 100, 1),
                    "lines": st["lines"],
                    "tokens": st["tokens"][:12],
                }
                for st in res.get("ranked_subtrees", [])[:5]
            ],
        }
        items.append(item)

        export_rows.append({
            "No": idx + 1,
            "Nama": item["nama"],
            "Kelas": item["kelas"],
            "Minggu_Ke_Berapa": item["minggu"],
            "Nama_Soal": item["nama_soal"],
            "Sumber_Code": item["sumber_code"],
            "Kategori_Pantauan": item["kategori_pantauan"],
            "Status_Asli_Code": item["status_asli_code"],
            "Ground_Truth_Hasil": item["hasil_gt"],
            "Prediksi_SANN_Final": "correct" if pred_cls == 1 else "wrong",
            "Prediksi_SANN_Pure_NN": "correct" if model_pred_cls == 1 else "wrong",
            "Match_Ground_Truth": "MATCH" if is_match else "MISMATCH",
            "Match_Pure_NN_Ground_Truth": "MATCH" if is_model_match else "MISMATCH",
            "Prob_Correct_%": item["prob_correct"],
            "Prob_Incorrect_%": item["prob_incorrect"],
            "Syntax_Error": f"Line {res['syntax_error']['line']}: {res['syntax_error']['message']}" if res.get("syntax_error") else "None",
            "Top_Attention_Line": top_line_no or "-",
            "Top_Attention_Weight_%": item["top_line_weight_pct"],
            "Top_Attention_Code": top_line_code,
            "Code_Yang_Disimpan": code_str,
        })

    # Compute aggregate metrics
    total = len(items)
    tp = sum(1 for it in items if it["gt_label"] == 1 and it["predicted_class"] == 1)
    tn = sum(1 for it in items if it["gt_label"] == 0 and it["predicted_class"] == 0)
    fp = sum(1 for it in items if it["gt_label"] == 0 and it["predicted_class"] == 1)
    fn = sum(1 for it in items if it["gt_label"] == 1 and it["predicted_class"] == 0)

    nn_tp = sum(1 for it in items if it["gt_label"] == 1 and it["model_predicted_class"] == 1)
    nn_tn = sum(1 for it in items if it["gt_label"] == 0 and it["model_predicted_class"] == 0)
    nn_fp = sum(1 for it in items if it["gt_label"] == 0 and it["model_predicted_class"] == 1)
    nn_fn = sum(1 for it in items if it["gt_label"] == 1 and it["model_predicted_class"] == 0)

    wrong_items = [it for it in items if it["gt_label"] == 0]
    corr_items = [it for it in items if it["gt_label"] == 1]

    summary_metrics = {
        "total": total,
        "gt_wrong_count": len(wrong_items),
        "gt_correct_count": len(corr_items),
        "match_count": tp + tn,
        "mismatch_count": fp + fn,
        "accuracy_pct": round((tp + tn) / total * 100, 2) if total else 0.0,
        "wrong_detection_rate_pct": round(tn / len(wrong_items) * 100, 2) if wrong_items else 0.0,
        "correct_detection_rate_pct": round(tp / len(corr_items) * 100, 2) if corr_items else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "nn_match_count": nn_tp + nn_tn,
        "nn_accuracy_pct": round((nn_tp + nn_tn) / total * 100, 2) if total else 0.0,
        "nn_tp": nn_tp,
        "nn_tn": nn_tn,
        "nn_fp": nn_fp,
        "nn_fn": nn_fn,
        "syntax_error_count": sum(1 for it in items if it["syntax_error"] is not None),
        "evaluated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    payload = {
        "weights_mtime": weights_mtime,
        "csv_mtime": csv_mtime,
        "summary": summary_metrics,
        "training_info": load_training_info(),
        "items": items,
    }

    EVAL_CACHE_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(EVAL_CACHE_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    pd.DataFrame(export_rows).to_csv(EVAL_RESULTS_CSV, index=False, encoding="utf-8-sig")
    return payload


HTML_PAGE = r"""<!DOCTYPE html>
<html lang="id">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>SANN-DAP Test & Attention Monitoring Dashboard</title>
  <style>
    :root {
      --bg: #0b1120;
      --surface: #151f32;
      --surface-2: #1e293b;
      --surface-3: #283548;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --accent: #38bdf8;
      --accent-2: #818cf8;
      --success: #10b981;
      --success-bg: rgba(16, 185, 129, 0.14);
      --danger: #f43f5e;
      --danger-bg: rgba(244, 63, 94, 0.14);
      --warn: #f59e0b;
      --warn-bg: rgba(245, 158, 11, 0.15);
      --code-bg: #070b14;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
      background: var(--bg);
      color: var(--text);
      line-height: 1.5;
    }
    header {
      background: rgba(21, 31, 50, 0.95);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid var(--border);
      padding: 16px 24px;
      position: sticky;
      top: 0;
      z-index: 100;
      box-shadow: 0 4px 20px rgba(0,0,0,0.35);
    }
    .container {
      max-width: 1440px;
      margin: 0 auto;
    }
    .header-row {
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 14px;
    }
    .brand h1 {
      margin: 0;
      font-size: 1.3rem;
      font-weight: 800;
      background: linear-gradient(90deg, #38bdf8, #818cf8);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }
    .brand p {
      margin: 3px 0 0;
      font-size: 0.84rem;
      color: var(--muted);
    }
    .header-actions {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
    }
    .btn {
      background: var(--surface-2);
      color: var(--text);
      border: 1px solid var(--border);
      padding: 8px 14px;
      border-radius: 8px;
      font-size: 0.85rem;
      font-weight: 600;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 6px;
      transition: all 0.15s ease;
    }
    .btn:hover {
      border-color: var(--accent);
      background: var(--surface-3);
    }
    .btn-primary {
      background: var(--accent);
      color: #0b1120;
      border-color: var(--accent);
    }
    .btn-primary:hover {
      background: #0284c7;
      color: #fff;
    }
    .tabs {
      display: flex;
      gap: 8px;
      margin-top: 14px;
      border-top: 1px solid var(--border);
      padding-top: 12px;
    }
    .tab-btn {
      background: transparent;
      color: var(--muted);
      border: 1px solid transparent;
      padding: 7px 14px;
      border-radius: 6px;
      font-weight: 600;
      font-size: 0.86rem;
      cursor: pointer;
    }
    .tab-btn.active {
      background: rgba(56, 189, 248, 0.14);
      color: var(--accent);
      border-color: rgba(56, 189, 248, 0.4);
    }
    main {
      max-width: 1440px;
      margin: 22px auto 60px;
      padding: 0 24px;
    }
    .grid-cards {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 14px;
      margin-bottom: 20px;
    }
    .stat-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px;
    }
    .stat-label {
      font-size: 0.76rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
      color: var(--muted);
      font-weight: 700;
    }
    .stat-val {
      font-size: 1.65rem;
      font-weight: 800;
      margin: 6px 0 4px;
    }
    .stat-sub {
      font-size: 0.8rem;
      color: var(--muted);
    }
    .banner-box {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 16px 20px;
      margin-bottom: 20px;
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 16px;
    }
    .filter-bar {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 14px 18px;
      margin-bottom: 20px;
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: flex-end;
    }
    .filter-group {
      display: flex;
      flex-direction: column;
      gap: 4px;
      min-width: 170px;
      flex: 1;
    }
    .filter-group label {
      font-size: 0.74rem;
      text-transform: uppercase;
      color: var(--muted);
      font-weight: 700;
    }
    select, input[type="text"], textarea {
      background: var(--bg);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 8px 12px;
      font-size: 0.88rem;
      outline: none;
    }
    select:focus, input[type="text"]:focus, textarea:focus {
      border-color: var(--accent);
    }
    .case-card {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 12px;
      margin-bottom: 20px;
      overflow: hidden;
      box-shadow: 0 4px 14px rgba(0,0,0,0.2);
    }
    .case-head {
      background: var(--surface-2);
      padding: 14px 20px;
      border-bottom: 1px solid var(--border);
      display: flex;
      flex-wrap: wrap;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }
    .case-title {
      font-weight: 700;
      font-size: 1.02rem;
    }
    .case-meta {
      font-size: 0.82rem;
      color: var(--muted);
      margin-top: 2px;
    }
    .badges {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 4px 10px;
      border-radius: 999px;
      font-size: 0.76rem;
      font-weight: 700;
      border: 1px solid transparent;
    }
    .badge-ok {
      background: var(--success-bg);
      color: #34d399;
      border-color: rgba(16, 185, 129, 0.35);
    }
    .badge-err {
      background: var(--danger-bg);
      color: #fb7185;
      border-color: rgba(244, 63, 94, 0.35);
    }
    .badge-warn {
      background: var(--warn-bg);
      color: #fbbf24;
      border-color: rgba(245, 158, 11, 0.35);
    }
    .badge-info {
      background: rgba(56, 189, 248, 0.14);
      color: var(--accent);
      border-color: rgba(56, 189, 248, 0.35);
    }
    .case-body {
      display: grid;
      grid-template-columns: 1.35fr 1fr;
      gap: 0;
    }
    @media (max-width: 1024px) {
      .case-body { grid-template-columns: 1fr; }
    }
    .code-pane {
      padding: 16px 20px;
      border-right: 1px solid var(--border);
      background: var(--code-bg);
      overflow-x: auto;
    }
    .info-pane {
      padding: 16px 20px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .code-table {
      width: 100%;
      border-collapse: collapse;
      font-family: 'JetBrains Mono', Consolas, monospace;
      font-size: 0.84rem;
    }
    .code-table td {
      padding: 3px 8px;
      vertical-align: middle;
    }
    .ln-col {
      width: 34px;
      text-align: right;
      color: #64748b;
      user-select: none;
      border-right: 1px solid #1e293b;
    }
    .wt-col {
      width: 62px;
      text-align: right;
      font-weight: 700;
      font-size: 0.76rem;
      border-right: 1px solid #1e293b;
    }
    .code-col {
      white-space: pre;
      padding-left: 12px !important;
    }
    .prob-bar-wrap {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 10px 12px;
    }
    .prob-bar-track {
      height: 10px;
      width: 100%;
      background: rgba(244, 63, 94, 0.35);
      border-radius: 999px;
      overflow: hidden;
      display: flex;
      margin-top: 6px;
    }
    .prob-bar-fill-corr {
      background: var(--success);
      height: 100%;
    }
    .prob-bar-fill-inc {
      background: var(--danger);
      height: 100%;
    }
    .subtree-list {
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .subtree-item {
      background: var(--bg);
      border: 1px solid var(--border);
      border-radius: 6px;
      padding: 7px 10px;
      font-size: 0.79rem;
    }
    .subtree-tokens {
      font-family: Consolas, monospace;
      color: #cbd5e1;
      margin-top: 3px;
      word-break: break-all;
    }
    .summary-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 0.85rem;
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 10px;
      overflow: hidden;
    }
    .summary-table th, .summary-table td {
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      text-align: left;
    }
    .summary-table th {
      background: var(--surface-2);
      color: var(--muted);
      font-size: 0.76rem;
      text-transform: uppercase;
    }
    .summary-table tr:hover {
      background: rgba(56, 189, 248, 0.05);
    }
  </style>
</head>
<body>
  <header>
    <div class="container">
      <div class="header-row">
        <div class="brand">
          <h1>SANN-DAP Monitoring Dashboard — Held-Out Monitored Test Set</h1>
          <p>Evaluasi Model SANN (Trained pada Seluruh Data Week 1 Analyze Data Kecuali Monitored_Codes_Dipantau.csv)</p>
        </div>
        <div class="header-actions">
          <button class="btn" onclick="refreshEvaluation(true)">🔄 Re-Run Evaluasi Model</button>
          <a class="btn btn-primary" href="/api/download_results_csv">⬇ Download Hasil Test (CSV)</a>
        </div>
      </div>
      <div class="tabs">
        <button class="tab-btn active" id="tabBtnCards" onclick="switchTab('cards')">🔍 Detail Kartu & Attention per Baris (38 Kasus)</button>
        <button class="tab-btn" id="tabBtnTable" onclick="switchTab('table')">📊 Tabel Ringkasan & Perbandingan</button>
        <button class="tab-btn" id="tabBtnLive" onclick="switchTab('live')">⚡ Live Sandbox Tester (Uji Kode Baru)</button>
      </div>
    </div>
  </header>

  <main>
    <div id="trainingBanner" class="banner-box">
      <div>Memuat informasi model & hasil evaluasi...</div>
    </div>

    <div id="statsGrid" class="grid-cards"></div>

    <!-- TAB 1: CARDS -->
    <div id="viewCards">
      <div class="filter-bar">
        <div class="filter-group">
          <label>Kategori Pantauan (Ground Truth)</label>
          <select id="filterKategori" onchange="renderAll()">
            <option value="ALL">Semua Kategori (38)</option>
            <option value="Wrong with Misconception">Wrong with Misconception (31)</option>
            <option value="Correct with Misconception">Correct with Misconception (7)</option>
          </select>
        </div>
        <div class="filter-group">
          <label>Mode Evaluasi Prediksi</label>
          <select id="filterPredMode" onchange="renderAll()">
            <option value="final">SANN + Syntax Check (Default)</option>
            <option value="pure_nn">Pure SANN Neural Network</option>
          </select>
        </div>
        <div class="filter-group">
          <label>Status Kecocokan (Match / Mismatch)</label>
          <select id="filterMatch" onchange="renderAll()">
            <option value="ALL">Semua Hasil</option>
            <option value="MATCH">✔ Sesuai Ground Truth (Match)</option>
            <option value="MISMATCH">✖ Tidak Sesuai (Mismatch)</option>
          </select>
        </div>
        <div class="filter-group">
          <label>Sumber Kode</label>
          <select id="filterSource" onchange="renderAll()">
            <option value="ALL">Semua Sumber</option>
            <option value="First Submit First Attempt">First Submit First Attempt</option>
            <option value="Final Submit">Final Submit</option>
          </select>
        </div>
        <div class="filter-group">
          <label>Filter Soal</label>
          <select id="filterSoal" onchange="renderAll()">
            <option value="ALL">Semua Soal</option>
          </select>
        </div>
        <div class="filter-group">
          <label>Cari Mahasiswa / NIM / Misconception</label>
          <input type="text" id="searchInput" placeholder="Ketik nama, NIM, atau KC..." oninput="renderAll()" />
        </div>
      </div>

      <div id="cardsContainer"></div>
    </div>

    <!-- TAB 2: TABLE -->
    <div id="viewTable" style="display:none">
      <div id="tableContainer"></div>
    </div>

    <!-- TAB 3: LIVE SANDBOX -->
    <div id="viewLive" style="display:none">
      <div class="case-card" style="padding:20px">
        <h3 style="margin-top:0">⚡ Uji Kode Pseudocode DAP secara Langsung dengan Model SANN</h3>
        <p style="color:var(--muted); font-size:0.88rem">Tempel atau edit kode pseudocode mahasiswa di bawah ini untuk melihat prediksi model SANN dan visualisasi bobot Attention per baris secara real-time.</p>
        <textarea id="liveCodeInput" rows="12" style="width:100%; font-family:Consolas,monospace; margin-bottom:12px">program konversi
kamus
    celcius : integer
    fahrenheit : real
algoritma
    input(celcius)
    fahrenheit <- (9.0 / 5.0) * celcius + 32.0
    output(fahrenheit)
endprogram</textarea>
        <div>
          <button class="btn btn-primary" onclick="runLivePredict()">🚀 Jalankan Prediksi & Attention SANN</button>
        </div>
        <div id="liveResultContainer" style="margin-top:18px"></div>
      </div>
    </div>
  </main>

  <script>
    let evalData = null;

    function escapeHtml(s) {
      return String(s ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
    }

    function switchTab(tab) {
      document.getElementById('viewCards').style.display = tab === 'cards' ? 'block' : 'none';
      document.getElementById('viewTable').style.display = tab === 'table' ? 'block' : 'none';
      document.getElementById('viewLive').style.display = tab === 'live' ? 'block' : 'none';
      document.getElementById('tabBtnCards').classList.toggle('active', tab === 'cards');
      document.getElementById('tabBtnTable').classList.toggle('active', tab === 'table');
      document.getElementById('tabBtnLive').classList.toggle('active', tab === 'live');
    }

    async function refreshEvaluation(force = false) {
      document.getElementById('cardsContainer').innerHTML = '<div class="stat-card">⏳ Sedang mengevaluasi 38 kode pantauan dengan model SANN...</div>';
      const res = await fetch('/api/evaluate' + (force ? '?refresh=1' : ''));
      evalData = await res.json();
      populateSoalFilter();
      renderAll();
    }

    function populateSoalFilter() {
      if (!evalData || !evalData.items) return;
      const sel = document.getElementById('filterSoal');
      const current = sel.value;
      const soals = Array.from(new Set(evalData.items.map(x => x.nama_soal))).sort();
      sel.innerHTML = '<option value="ALL">Semua Soal (' + soals.length + ')</option>' +
        soals.map(s => `<option value="${escapeHtml(s)}">${escapeHtml(s)}</option>`).join('');
      if (soals.includes(current)) sel.value = current;
    }

    function getFilteredItems() {
      if (!evalData || !evalData.items) return [];
      const kat = document.getElementById('filterKategori').value;
      const mode = document.getElementById('filterPredMode').value;
      const matchFilter = document.getElementById('filterMatch').value;
      const srcFilter = document.getElementById('filterSource').value;
      const soalFilter = document.getElementById('filterSoal').value;
      const q = document.getElementById('searchInput').value.trim().toLowerCase();

      return evalData.items.filter(it => {
        if (kat !== 'ALL' && it.kategori_pantauan !== kat) return false;
        if (srcFilter !== 'ALL' && it.sumber_code !== srcFilter) return false;
        if (soalFilter !== 'ALL' && it.nama_soal !== soalFilter) return false;
        const matched = (mode === 'pure_nn') ? it.is_model_match : it.is_match;
        if (matchFilter === 'MATCH' && !matched) return false;
        if (matchFilter === 'MISMATCH' && matched) return false;
        if (q) {
          const hay = `${it.nama} ${it.kelas} ${it.nama_soal} ${it.misconception} ${it.code}`.toLowerCase();
          if (!hay.includes(q)) return false;
        }
        return true;
      });
    }

    function renderAll() {
      if (!evalData) return;
      const t = evalData.training_info || {};
      const s = evalData.summary || {};
      const mode = document.getElementById('filterPredMode').value;

      document.getElementById('trainingBanner').innerHTML = `
        <div>
          <div style="font-weight:700; color:var(--accent); font-size:0.98rem">
            📦 Model SANN Tersimpan di: <code>saved_model/</code> &amp; <code>saved_model_dap/</code>
          </div>
          <div style="font-size:0.83rem; color:var(--muted); margin-top:4px">
            Dataset Training: <strong>${escapeHtml(t.dataset_source || 'Seluruh Kelas Week 1-2 Analyze Data (Kecuali Monitored_Codes_Dipantau.csv)')}</strong>
            &nbsp;|&nbsp; Sampel Compilable: <strong>${t.total_compilable_samples || '-'}</strong> (Train: ${t.train_samples || '-'}, Val: ${t.val_samples || '-'})
            &nbsp;|&nbsp; Max Depth: <strong>${t.max_depth || 3}</strong>, Max Subtree: <strong>${t.max_subtree_length || 64}</strong>
            &nbsp;|&nbsp; Node Vocab: <strong>${t.node_vocab_size || '-'}</strong>, Path Vocab: <strong>${t.path_vocab_size || '-'}</strong>
          </div>
        </div>
        <div class="badges">
          <span class="badge badge-ok">Validation Acc: ${t.val_accuracy ? (t.val_accuracy * 100).toFixed(2) + '%' : '-'}</span>
          <span class="badge badge-info">Val F1: ${t.val_f1 ? (t.val_f1 * 100).toFixed(2) + '%' : '-'}</span>
          <span class="badge badge-warn">Dievaluasi: ${escapeHtml(s.evaluated_at || '-')}</span>
        </div>
      `;

      const accPct = mode === 'pure_nn' ? s.nn_accuracy_pct : s.accuracy_pct;
      const matchCnt = mode === 'pure_nn' ? s.nn_match_count : s.match_count;
      const tp = mode === 'pure_nn' ? s.nn_tp : s.tp;
      const tn = mode === 'pure_nn' ? s.nn_tn : s.tn;
      const fp = mode === 'pure_nn' ? s.nn_fp : s.fp;
      const fn = mode === 'pure_nn' ? s.nn_fn : s.fn;

      document.getElementById('statsGrid').innerHTML = `
        <div class="stat-card">
          <div class="stat-label">Total Data Uji (Dipantau)</div>
          <div class="stat-val">${s.total}</div>
          <div class="stat-sub">31 Wrong w/ Misconception &bull; 7 Correct w/ Misconception</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Akurasi pada Data Uji</div>
          <div class="stat-val" style="color:var(--success)">${accPct}%</div>
          <div class="stat-sub">${matchCnt} dari ${s.total} kasus sesuai Ground Truth</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Deteksi Wrong w/ Misconception</div>
          <div class="stat-val" style="color:var(--accent)">${tn} / ${s.gt_wrong_count}</div>
          <div class="stat-sub">${s.gt_wrong_count ? ((tn / s.gt_wrong_count)*100).toFixed(1) : 0}% kode salah terdeteksi INCORRECT</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Deteksi Correct w/ Misconception</div>
          <div class="stat-val" style="color:#a78bfa">${tp} / ${s.gt_correct_count}</div>
          <div class="stat-sub">${s.gt_correct_count ? ((tp / s.gt_correct_count)*100).toFixed(1) : 0}% kode benar terdeteksi CORRECT</div>
        </div>
        <div class="stat-card">
          <div class="stat-label">Confusion Matrix (Test)</div>
          <div class="stat-val" style="font-size:1.25rem">TN:${tn} | TP:${tp} | FP:${fp} | FN:${fn}</div>
          <div class="stat-sub">Syntax Error Terdeteksi: ${s.syntax_error_count} kode</div>
        </div>
      `;

      const filtered = getFilteredItems();
      renderCards(filtered, mode);
      renderTable(filtered, mode);
    }

    function renderCodeHeatmap(lineBreakdown) {
      const maxW = Math.max(...lineBreakdown.map(l => l.weight_pct), 1);
      const rows = lineBreakdown.map(l => {
        const ratio = l.weight_pct / maxW;
        const alpha = l.weight_pct > 0 ? (0.06 + 0.42 * ratio).toFixed(2) : 0;
        const bg = l.is_syntax_error
          ? 'rgba(244, 63, 94, 0.32)'
          : (l.weight_pct > 0 ? `rgba(244, 63, 94, ${alpha})` : 'transparent');
        let badge = '';
        if (l.is_syntax_error) {
          badge += '<span class="badge badge-err" style="margin-left:8px; padding:1px 6px; font-size:0.68rem">SYNTAX ERROR</span>';
        }
        if (l.rank === 1) {
          badge += '<span class="badge badge-err" style="margin-left:8px; padding:1px 6px; font-size:0.68rem">FOKUS UTAMA #1</span>';
        } else if (l.rank === 2) {
          badge += '<span class="badge badge-warn" style="margin-left:8px; padding:1px 6px; font-size:0.68rem">FOKUS #2</span>';
        } else if (l.rank === 3) {
          badge += '<span class="badge badge-info" style="margin-left:8px; padding:1px 6px; font-size:0.68rem">FOKUS #3</span>';
        }
        return `
          <tr style="background:${bg}">
            <td class="ln-col">${l.line_no}</td>
            <td class="wt-col" style="color:${l.weight_pct > 0 ? '#fda4af' : '#475569'}">${l.weight_pct.toFixed(1)}%</td>
            <td class="code-col">${escapeHtml(l.code)}${badge}</td>
          </tr>
        `;
      }).join('');
      return `<table class="code-table"><tbody>${rows}</tbody></table>`;
    }

    function renderCards(items, mode) {
      const container = document.getElementById('cardsContainer');
      if (items.length === 0) {
        container.innerHTML = '<div class="stat-card">Tidak ada data yang cocok dengan filter saat ini.</div>';
        return;
      }

      container.innerHTML = items.map(it => {
        const predCls = mode === 'pure_nn' ? it.model_predicted_class : it.predicted_class;
        const isMatch = mode === 'pure_nn' ? it.is_model_match : it.is_match;
        const gtBadge = it.gt_label === 1
          ? `<span class="badge badge-ok">GT: Correct (${escapeHtml(it.kategori_pantauan)})</span>`
          : `<span class="badge badge-err">GT: Wrong (${escapeHtml(it.kategori_pantauan)})</span>`;
        const predBadge = predCls === 1
          ? `<span class="badge badge-ok">Prediksi SANN: CORRECT (1)</span>`
          : `<span class="badge badge-err">Prediksi SANN: INCORRECT (0)</span>`;
        const matchBadge = isMatch
          ? `<span class="badge badge-ok">✔ MATCH</span>`
          : `<span class="badge badge-warn">✖ MISMATCH</span>`;

        const subtreesHtml = (it.top_subtrees || []).slice(0, 3).map((st, idx) => `
          <div class="subtree-item">
            <div style="display:flex; justify-content:space-between; font-weight:700; color:var(--accent)">
              <span>Rank #${idx + 1} (Subtree #${st.subtree_index}) — Baris [${st.lines.join(', ')}]</span>
              <span>${st.weight_pct}%</span>
            </div>
            <div class="subtree-tokens">${escapeHtml(st.tokens.join(' → '))}</div>
          </div>
        `).join('');

        return `
          <div class="case-card">
            <div class="case-head">
              <div>
                <div class="case-title">#${it.index} — ${escapeHtml(it.nama)} <span style="color:var(--accent)">(${escapeHtml(it.kelas)})</span></div>
                <div class="case-meta">
                  <strong>${escapeHtml(it.nama_soal)}</strong> &nbsp;|&nbsp; Sumber: <strong>${escapeHtml(it.sumber_code)}</strong> &nbsp;|&nbsp; Status Grader Asli: <strong>${escapeHtml(it.status_asli_code)}</strong>
                </div>
              </div>
              <div class="badges">
                ${gtBadge}
                ${predBadge}
                ${matchBadge}
              </div>
            </div>
            <div class="case-body">
              <div class="code-pane">
                <div style="font-size:0.75rem; text-transform:uppercase; color:var(--muted); margin-bottom:8px; font-weight:700">
                  Visualisasi Bobot Attention SANN Per Baris Kode:
                </div>
                ${renderCodeHeatmap(it.line_breakdown)}
              </div>
              <div class="info-pane">
                <div class="prob-bar-wrap">
                  <div style="display:flex; justify-content:space-between; font-size:0.82rem; font-weight:700">
                    <span style="color:var(--success)">Prob. Correct (1): ${it.prob_correct}%</span>
                    <span style="color:var(--danger)">Prob. Incorrect (0): ${it.prob_incorrect}%</span>
                  </div>
                  <div class="prob-bar-track">
                    <div class="prob-bar-fill-corr" style="width:${it.prob_correct}%"></div>
                    <div class="prob-bar-fill-inc" style="width:${it.prob_incorrect}%"></div>
                  </div>
                  ${it.syntax_error ? `<div style="margin-top:8px; font-size:0.8rem; color:#fda4af">⚠️ <strong>Compiler DAP Syntax Check:</strong> Baris ${it.syntax_error.line} — ${escapeHtml(it.syntax_error.message)}</div>` : ''}
                </div>

                <div style="font-size:0.83rem; background:var(--bg); border:1px solid var(--border); border-radius:8px; padding:10px 12px">
                  <div style="color:var(--muted); font-size:0.74rem; text-transform:uppercase; font-weight:700">Fokus Attention Tertinggi (#1)</div>
                  <div style="margin-top:4px; font-weight:700; color:#fda4af">
                    Baris ${it.top_line_no || '-'} (${it.top_line_weight_pct}% Attention): <code>${escapeHtml(it.top_line_code)}</code>
                  </div>
                  <div style="margin-top:8px; color:var(--muted); font-size:0.74rem; text-transform:uppercase; font-weight:700">Miskonsepsi Mahasiswa Minggu Itu</div>
                  <div style="margin-top:2px; font-size:0.81rem">${escapeHtml(it.misconception)}</div>
                </div>

                <div>
                  <div style="color:var(--muted); font-size:0.74rem; text-transform:uppercase; font-weight:700; margin-bottom:6px">Top 3 Subtree AST yang Diperhatikan Model</div>
                  <div class="subtree-list">${subtreesHtml}</div>
                </div>
              </div>
            </div>
          </div>
        `;
      }).join('');
    }

    function renderTable(items, mode) {
      const container = document.getElementById('tableContainer');
      const rows = items.map(it => {
        const predCls = mode === 'pure_nn' ? it.model_predicted_class : it.predicted_class;
        const isMatch = mode === 'pure_nn' ? it.is_model_match : it.is_match;
        return `
          <tr>
            <td>${it.index}</td>
            <td><strong>${escapeHtml(it.nama)}</strong><br><small style="color:var(--muted)">${escapeHtml(it.sumber_code)}</small></td>
            <td>${escapeHtml(it.nama_soal)}</td>
            <td>${it.gt_label === 1 ? '<span class="badge badge-ok">correct</span>' : '<span class="badge badge-err">wrong</span>'}<br><small style="color:var(--muted)">${escapeHtml(it.status_asli_code)}</small></td>
            <td>${predCls === 1 ? '<span class="badge badge-ok">CORRECT (1)</span>' : '<span class="badge badge-err">INCORRECT (0)</span>'}</td>
            <td>${isMatch ? '<span class="badge badge-ok">✔ MATCH</span>' : '<span class="badge badge-warn">✖ MISMATCH</span>'}</td>
            <td><strong>${it.prob_correct}%</strong> / ${it.prob_incorrect}%</td>
            <td>Baris ${it.top_line_no || '-'} (${it.top_line_weight_pct}%)<br><code>${escapeHtml(it.top_line_code)}</code></td>
          </tr>
        `;
      }).join('');

      container.innerHTML = `
        <table class="summary-table">
          <thead>
            <tr>
              <th>#</th>
              <th>Mahasiswa & Sumber</th>
              <th>Soal</th>
              <th>Ground Truth</th>
              <th>Prediksi SANN</th>
              <th>Status</th>
              <th>Prob (Corr / Inc)</th>
              <th>Baris Fokus Utama (#1)</th>
            </tr>
          </thead>
          <tbody>${rows}</tbody>
        </table>
      `;
    }

    async function runLivePredict() {
      const code = document.getElementById('liveCodeInput').value;
      const box = document.getElementById('liveResultContainer');
      box.innerHTML = '<div class="stat-card">⏳ Memproses AST & menghitung Attention...</div>';
      const res = await fetch('/api/predict_custom', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code })
      });
      const data = await res.json();
      if (data.error) {
        box.innerHTML = `<div class="stat-card" style="border-color:var(--danger); color:#fda4af">Error: ${escapeHtml(data.error)}</div>`;
        return;
      }
      box.innerHTML = `
        <div class="case-card">
          <div class="case-head">
            <div class="case-title">Hasil Prediksi Live Sandbox</div>
            <div class="badges">
              ${data.predicted_class === 1
                ? '<span class="badge badge-ok">Prediksi Final: CORRECT (1)</span>'
                : '<span class="badge badge-err">Prediksi Final: INCORRECT (0)</span>'}
              <span class="badge badge-info">Prob. Correct: ${data.prob_correct}% | Prob. Incorrect: ${data.prob_incorrect}%</span>
            </div>
          </div>
          <div class="case-body">
            <div class="code-pane">${renderCodeHeatmap(data.line_breakdown)}</div>
            <div class="info-pane">
              <div style="font-size:0.86rem">
                <strong>Fokus Utama (#1):</strong> Baris ${data.top_line_no || '-'} (${data.top_line_weight_pct}%) — <code>${escapeHtml(data.top_line_code || '')}</code>
                ${data.syntax_error ? `<div style="margin-top:8px; color:#fda4af">⚠️ Syntax Error di Baris ${data.syntax_error.line}: ${escapeHtml(data.syntax_error.message)}</div>` : ''}
              </div>
            </div>
          </div>
        </div>
      `;
    }

    refreshEvaluation(false);
  </script>
</body>
</html>
"""


def run_custom_prediction(code_str):
    if not code_str.strip():
        return {"error": "Kode pseudocode tidak boleh kosong."}, 400
    try:
        model_dir = get_active_model_dir()
        res = predict_and_explain(code_str, model_dir=model_dir, print_report=False)
        line_scores = {int(k): float(v) for k, v in res["line_scores"].items()}
        ranked_lines = res["ranked_lines"]
        lines = code_str.splitlines()
        top_line_no = ranked_lines[0] if ranked_lines else None
        top_line_code = lines[top_line_no - 1].strip() if top_line_no and 1 <= top_line_no <= len(lines) else ""
        top_line_weight = line_scores.get(top_line_no, 0.0) if top_line_no else 0.0
        line_breakdown = []
        for ln_no, ln_text in enumerate(lines, start=1):
            w = line_scores.get(ln_no, 0.0)
            rank_pos = ranked_lines.index(ln_no) + 1 if ln_no in ranked_lines else None
            is_syn_err = bool(res.get("syntax_error") and res["syntax_error"].get("line") == ln_no)
            line_breakdown.append({
                "line_no": ln_no,
                "code": ln_text,
                "weight": round(w, 4),
                "weight_pct": round(w * 100, 1),
                "rank": rank_pos,
                "is_syntax_error": is_syn_err,
            })
        return {
            "predicted_class": int(res["predicted_class"]),
            "model_predicted_class": int(res.get("model_predicted_class", res["predicted_class"])),
            "predicted_label": res["predicted_label"],
            "prob_correct": round(res["probabilities"]["correct_1"] * 100, 2),
            "prob_incorrect": round(res["probabilities"]["incorrect_0"] * 100, 2),
            "syntax_error": res.get("syntax_error"),
            "top_line_no": top_line_no,
            "top_line_code": top_line_code,
            "top_line_weight_pct": round(top_line_weight * 100, 1),
            "line_breakdown": line_breakdown,
        }, 200
    except Exception as e:
        return {"error": str(e)}, 500


class MonitorRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def _send_json(self, data, status=200):
        raw = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        qs = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            raw = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        elif path == "/api/evaluate":
            force = qs.get("refresh", ["0"])[0] == "1"
            data = evaluate_monitored_codes(force_refresh=force)
            self._send_json(data, 200)
        elif path == "/api/download_results_csv":
            if not EVAL_RESULTS_CSV.exists():
                evaluate_monitored_codes(force_refresh=True)
            raw = EVAL_RESULTS_CSV.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/csv; charset=utf-8")
            self.send_header("Content-Disposition", 'attachment; filename="Monitored_Test_Evaluation_Results.csv"')
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/predict_custom":
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8") if length > 0 else "{}"
            payload = json.loads(body or "{}")
            resp, status = run_custom_prediction(str(payload.get("code", "")))
            self._send_json(resp, status)
        else:
            self._send_json({"error": "Not found"}, 404)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    print("Pre-computing evaluation on 38 monitored test cases...")
    evaluate_monitored_codes(force_refresh=False)
    print(f"Menjalankan SANN Monitoring App di http://127.0.0.1:{port} ...")
    server = ThreadingHTTPServer(("127.0.0.1", port), MonitorRequestHandler)
    server.serve_forever()

