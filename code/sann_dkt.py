"""
Implementasi SANN + LSTM (SANN-DKT) berdasarkan paper:
"Automated Identification of Logical Errors in Programs: Advancing Scalable Analysis of Student Misconceptions"
(Muntasir Hoq et al., EDM 2025 / arXiv:2505.10913v1)

Fitur utama:
1. Modified SANN (Section 4.1 & 5.1):
   - Ekstraksi seluruh subtree secara rekursif penuh hingga leaf TANPA batas max_depth (Figure 2).
   - Two-way embedding (Node Embedding + Subtree/Path Embedding) -> TimeDistributed(Dense(128, tanh)).
   - Sigmoid Attention: a_i = sigma(sv_i^T . av) + Entropy Regularization (lambda = 3.5e-5).
   - Menghasilkan Code Vector c_t (128-dim) secara langsung TANPA normalisasi L2.

2. Attempt-based Chronological Sequence untuk SANN-DKT (Section 4.3 & 5.2):
   - Mengurutkan pengerjaan mahasiswa berdasarkan Attempt kuis secara kronologis:
     Attempt 1 (Soal 1 -> selesai sesuai urutan waktu submit), diikuti Attempt 2 (Soal 1 -> selesai)
     di urutan waktu berikutnya, dst.
   - Seluruh mahasiswa pada Reviewed_Students_Status.csv dipastikan masuk ke dalam Data Test.
   - Hasil inference menampilkan prediksi soal berikutnya (SANN-DKT) sekaligus tampilan kode
     beserta bobot Attention untuk setiap baris kode (line-level explainability).
"""

import os
import re
import glob
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
from keras.models import Model
from keras.layers import (
    Input,
    Dense,
    Embedding,
    Concatenate,
    TimeDistributed,
    LSTM,
    Masking,
    Lambda,
)
from keras import callbacks
from tensorflow.keras.utils import to_categorical

from config import Config
from parserDAP import (
    _get_ast_disk_cache,
    program_parser,
    repair_dap_code,
    build_fallback_ast,
    extracting_AST,
)
from modelDAP import (
    _norm_code,
    preprocess_raw_paths_wordtoken,
    encode_paths,
    preprocess_raw_paths_pathtoken,
    encode_paths_path,
)
from inferenceDAP import (
    extract_subtrees_with_lines,
    compute_line_attention_from_subtrees,
)

SEED = 4
np.random.seed(SEED)
tf.random.set_seed(SEED)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ANALYZE_CLEAN_DIR = PROJECT_ROOT.parent / "Analyze data" / "clean"
REVIEWED_CSV_PATH = PROJECT_ROOT / "Reviewed_Students_Status.csv"
MONITORED_CSV_PATH = PROJECT_ROOT / "Monitored_Codes_Dipantau.csv"
SAVED_DKT_DIR = PROJECT_ROOT / "saved_model_dkt"
DKT_HTML_REPORT = PROJECT_ROOT / "sann_dkt_report.html"


# ==============================================================================
# BAGIAN 1: MODIFIED SANN (Sigmoid Attention + Entropy Regularization - Sec 4.1)
# ==============================================================================

def create_modified_sann_model(
    node_vocab_size,
    path_vocab_size,
    max_subtree_length=64,
    embedding_size=64,
    code_vector_size=128,
    num_classes=2,
    entropy_lambda=3.5e-5,
):
    """
    Membangun Modified SANN sesuai Section 4.1 & 5.1 paper (arXiv:2505.10913):
    - Two-way embedding (Node + Subtree/Path) -> TimeDistributed Dense(code_vector_size, tanh)
    - Sigmoid Attention: a_i = sigma(sv_i^T * av)
    - Entropy Regularization loss: H(a) = -lambda * sum(a_i * log(a_i + eps))
    - Output: correctness_output (2 kelas) & code_vectors (128-dim tanpa normalisasi L2)
    """
    node_input = Input(
        (max_subtree_length, max_subtree_length), dtype=tf.int32, name="node_input"
    )
    path_input = Input((max_subtree_length,), dtype=tf.int32, name="path_input")

    nodes_embedded = Embedding(
        node_vocab_size + 2, embedding_size, trainable=True, name="node_embedding"
    )(node_input)
    path_embedded = Embedding(
        path_vocab_size + 2, embedding_size, trainable=True, name="path_embedding"
    )(path_input)

    nodes_embedded_merged = Lambda(
        lambda x: K.sum(x, axis=2), name="nodes_merged"
    )(nodes_embedded)

    node_path_merged = Concatenate(name="two_way_concat")(
        [nodes_embedded_merged, path_embedded]
    )

    subtree_vectors = TimeDistributed(
        Dense(code_vector_size, use_bias=False, activation="tanh"),
        name="subtree_vectors",
    )(node_path_merged)

    # Sigmoid Attention: a_i = sigma(sv_i^T . av)
    attention_logits = Dense(1, use_bias=False, name="attention_vectors")(
        subtree_vectors
    )
    attention_weights = keras.layers.Activation("sigmoid", name="attention_weights")(
        attention_logits
    )

    # Entropy regularization: H(a) = -lambda * sum_i a_i * log(a_i + eps)
    if entropy_lambda and entropy_lambda > 0:
        eps = 1e-7
        entropy_loss = -entropy_lambda * K.mean(
            K.sum(
                attention_weights * K.log(K.clip(attention_weights, eps, 1.0)),
                axis=1,
            )
        )
    else:
        entropy_loss = None

    # Weighted aggregation menghasilkan Source Code Vector c_t (tanpa normalisasi L2)
    code_vectors = Lambda(
        lambda inputs: K.sum(inputs[0] * inputs[1], axis=1), name="code_vectors"
    )([subtree_vectors, attention_weights])

    output_class = Dense(
        num_classes, use_bias=False, activation="softmax", name="correctness_output"
    )(code_vectors)

    model = Model(
        inputs=[node_input, path_input], outputs=output_class, name="Modified_SANN"
    )
    if entropy_loss is not None:
        model.add_loss(entropy_loss)

    encoder_model = Model(
        inputs=[node_input, path_input],
        outputs=[code_vectors, attention_weights, output_class],
        name="SANN_CodeVector_Extractor",
    )
    return model, encoder_model


# ==============================================================================
# BAGIAN 2: PEMBENTUKAN SEQUENCE BERDASARKAN ATTEMPT (Attempt 1 -> Attempt 2...)
# ==============================================================================

def load_monitored_codes_metadata():
    """
    Membaca Monitored_Codes_Dipantau.csv untuk mendapatkan informasi kode yang dipantau,
    Jenis_Misconception_Minggu_Itu, Kategori_Pantauan, Sumber_Code, Hasilnya, dan Status_Asli_Code.
    """
    by_nim_q_code = {}
    by_nim_q_source = {}
    by_nim_q = {}
    by_nim_summary = {}
    monitored_nims = set()

    if MONITORED_CSV_PATH.exists():
        mon_df = pd.read_csv(MONITORED_CSV_PATH)
        for idx, r in mon_df.iterrows():
            nama_raw = str(r.get("Nama", "")).strip()
            m_nim = re.search(r"\((\d+)\)", nama_raw)
            nim = m_nim.group(1).strip() if m_nim else ""
            if not nim:
                continue
            monitored_nims.add(nim)

            soal_raw = str(r.get("Nama_Soal", "")).strip()
            m_q = re.search(r":\s*(.+)\)$", soal_raw)
            qname = m_q.group(1).strip() if m_q else soal_raw

            sumber = str(r.get("Sumber_Code", "")).strip()
            code_norm = _norm_code(r.get("Code_Yang_Disimpan", ""))
            misconception = str(r.get("Jenis_Misconception_Minggu_Itu", "")).strip()
            kategori = str(r.get("Kategori_Pantauan", "")).strip()
            hasil_gt = str(r.get("Hasilnya", "")).strip()
            status_asli = str(r.get("Status_Asli_Code", "")).strip()
            waktu_simpan = str(r.get("Waktu_Disimpan", "")).strip()
            raw_note = r.get("Misconception_Di_Dalam_Code_Notes", "")
            code_note = (
                str(raw_note).strip()
                if pd.notna(raw_note) and str(raw_note).strip() and str(raw_note).strip().lower() != "nan"
                else "-"
            )
            raw_rev = r.get("Direview_Oleh", "")
            reviewer = (
                str(raw_rev).strip()
                if pd.notna(raw_rev) and str(raw_rev).strip() and str(raw_rev).strip().lower() != "nan"
                else "-"
            )

            entry = {
                "monitored_row_no": int(idx + 1),
                "nama_label": nama_raw,
                "nim": nim,
                "kelas": str(r.get("Kelas", "")).strip(),
                "minggu": str(r.get("Minggu_Ke_Berapa", "")).strip(),
                "nama_soal_label": soal_raw,
                "question_name": qname,
                "misconception": misconception,
                "code_note": code_note,
                "reviewer": reviewer,
                "sumber_code": sumber,
                "code": code_norm,
                "hasil_gt": hasil_gt,
                "kategori_pantauan": kategori,
                "status_asli_code": status_asli,
                "waktu_disimpan": waktu_simpan,
            }

            by_nim_q_code[(nim, qname, code_norm)] = entry
            by_nim_q_source[(nim, qname, sumber)] = entry
            if (nim, qname) not in by_nim_q:
                by_nim_q[(nim, qname)] = []
            by_nim_q[(nim, qname)].append(entry)

            if nim not in by_nim_summary:
                m_all_mp = re.search(r"Semua MP Minggu Itu:\s*(.+)$", misconception)
                all_mp_str = m_all_mp.group(1).strip() if m_all_mp else misconception
                by_nim_summary[nim] = {
                    "all_week_misconceptions": all_mp_str,
                    "monitored_count": 0,
                    "entries": [],
                }
            by_nim_summary[nim]["monitored_count"] += 1
            by_nim_summary[nim]["entries"].append(entry)

    # Tambahkan juga info miskonsepsi dari Sequential_Misconception_and_FirstPseudocode.csv jika ada
    seq_pattern = str(
        ANALYZE_CLEAN_DIR / "*" / "Week_01-02_Sequential_Misconception_and_FirstPseudocode.csv"
    )
    for sf in sorted(glob.glob(seq_pattern)):
        try:
            sdf = pd.read_csv(sf)
            for _, sr in sdf.iterrows():
                snim = str(sr.get("NIM", "")).strip()
                mp_all = str(sr.get("Misconception_Detected_Semua_MP", "")).strip()
                if snim and mp_all and mp_all != "nan":
                    if snim not in by_nim_summary:
                        by_nim_summary[snim] = {
                            "all_week_misconceptions": mp_all,
                            "monitored_count": 0,
                            "entries": [],
                        }
                    elif not by_nim_summary[snim].get("all_week_misconceptions"):
                        by_nim_summary[snim]["all_week_misconceptions"] = mp_all
        except Exception:
            pass

    return {
        "by_nim_q_code": by_nim_q_code,
        "by_nim_q_source": by_nim_q_source,
        "by_nim_q": by_nim_q,
        "by_nim_summary": by_nim_summary,
        "monitored_nims": monitored_nims,
    }


def load_reviewed_student_nims():
    """
    Membaca daftar NIM mahasiswa dari Reviewed_Students_Status.csv (serta Monitored_Codes_Dipantau.csv)
    yang wajib masuk ke dalam Data Test.
    """
    nims = set()
    if REVIEWED_CSV_PATH.exists():
        rev_df = pd.read_csv(REVIEWED_CSV_PATH)
        nims.update(rev_df["NIM"].dropna().astype(str).str.strip().tolist())
    mon_meta = load_monitored_codes_metadata()
    nims.update(mon_meta["monitored_nims"])
    return nims


def load_week1_chronological_attempts(recursive_subtrees=True, max_depth=None):
    """
    Membaca seluruh file Week_01-02_Problem_Solving_Questions.csv (8 kelas kanonik, 19 soal).
    Menyusun urutan sekuensial per mahasiswa berdasarkan ATTEMPT:
      - Attempt 1 (Soal 1 -> selesai, terurut sesuai waktu submit First_Submit_Time & Slot_Number)
      - Attempt 2 (Soal 1 -> selesai, berada di urutan waktu berikutnya setelah Attempt 1)
      - dst. lintas kuis (Easy -> Challenger -> Ultimate) sesuai waktu Started_Time & Attempt_ID.
    Mengekstrak seluruh subtree AST secara rekursif penuh TANPA batas max_depth (max_depth=None).
    """
    pattern = str(ANALYZE_CLEAN_DIR / "*" / "Week_01-02_Problem_Solving_Questions.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        fallback = PROJECT_ROOT / "Week_01-02_Problem_Solving_Questions.csv"
        files = [str(fallback)] if fallback.exists() else []

    dfs = []
    for f in files:
        d = pd.read_csv(f)
        if d["Question_Name"].nunique() <= 20:
            dfs.append(d)

    raw_df = pd.concat(dfs, ignore_index=True)

    unique_problems = sorted(raw_df["Question_Name"].dropna().astype(str).unique())
    problem_to_ix = {q: idx for idx, q in enumerate(unique_problems)}
    ix_to_problem = {idx: q for idx, q in enumerate(unique_problems)}

    problem_category = {}
    for _, r in raw_df[["Question_Name", "Quiz_Name"]].drop_duplicates().iterrows():
        qname = str(r["Question_Name"])
        qz = str(r["Quiz_Name"])
        if "Easy" in qz:
            problem_category[qname] = "Easy"
        elif "Challenger" in qz:
            problem_category[qname] = "Challenger"
        elif "Ultimate" in qz:
            problem_category[qname] = "Ultimate"
        else:
            problem_category[qname] = "Other"

    mon_meta = load_monitored_codes_metadata()
    by_nim_q_code = mon_meta["by_nim_q_code"]
    by_nim_q_source = mon_meta["by_nim_q_source"]
    by_nim_q = mon_meta["by_nim_q"]
    by_nim_summary = mon_meta["by_nim_summary"]

    # Kumpulkan setiap pengerjaan soal di dalam setiap Attempt kuis
    events = []
    for _, r in raw_df.iterrows():
        qname = str(r.get("Question_Name", ""))
        if qname not in problem_to_ix:
            continue

        sc = _norm_code(r.get("Student_Code", ""))
        fc = _norm_code(r.get("First_Submit_Pseudocode", ""))
        if not sc and not fc:
            continue

        nim = str(r.get("NIM", "")).strip()
        class_name = str(r.get("Class_Name", "")).strip()
        student_key = f"{class_name}_{nim}"
        student_name = str(r.get("Student_Name", "")).strip()
        quiz_name = str(r.get("Quiz_Name", "")).strip()
        quiz_cat = problem_category.get(qname, "Other")

        attempt_id = int(r.get("Attempt_ID", 0)) if pd.notna(r.get("Attempt_ID")) else 0
        attempt_no = (
            int(r.get("Attempt_Number", 1)) if pd.notna(r.get("Attempt_Number")) else 1
        )
        slot_no = int(r.get("Slot_Number", 0)) if pd.notna(r.get("Slot_Number")) else 0
        q_no = int(r.get("Question_No", slot_no)) if pd.notna(r.get("Question_No")) else slot_no

        started_t = (
            str(r.get("Started_Time", "")).strip()
            if pd.notna(r.get("Started_Time"))
            else ""
        )
        completed_t = (
            str(r.get("Completed_Time", "")).strip()
            if pd.notna(r.get("Completed_Time"))
            else ""
        )
        submit_t = (
            str(r.get("First_Submit_Time", "")).strip()
            if pd.notna(r.get("First_Submit_Time")) and str(r.get("First_Submit_Time", "")).strip()
            else (started_t or completed_t)
        )

        is_corr = int(r["Is_Correct"]) if pd.notna(r.get("Is_Correct")) else 0
        f_state = str(r.get("First_Submit_State", "")).strip()
        f_corr = 1 if f_state == "Correct" else 0
        num_submits = int(r["Num_Submits"]) if pd.notna(r.get("Num_Submits")) else 1

        stu_mp_summary = by_nim_summary.get(nim, {}).get(
            "all_week_misconceptions", "Tidak ada data miskonsepsi"
        )
        q_mon_list = by_nim_q.get((nim, qname), [])
        default_q_misconception = (
            q_mon_list[0]["misconception"] if q_mon_list else f"Semua MP Minggu Itu: {stu_mp_summary}"
        )

        # Cek apakah ada kode 'First Submit First Attempt' yang dipantau di Monitored_Codes_Dipantau.csv
        mon_first = (
            by_nim_q_code.get((nim, qname, fc))
            or (by_nim_q_source.get((nim, qname, "First Submit First Attempt")) if attempt_no == 1 else None)
        )
        mon_final = (
            by_nim_q_code.get((nim, qname, sc))
            or (by_nim_q_source.get((nim, qname, "Final Submit")) if attempt_no == 1 else None)
        )

        # Jika Attempt 1 memiliki kode First Submit yang dipantau dan berbeda dari Final Submit,
        # tampilkan langkah First Submit First Attempt terlebih dahulu lalu diikuti Final Submit pada soal tersebut
        if attempt_no == 1 and mon_first and fc and sc and fc != sc:
            events.append(
                {
                    "student_key": student_key,
                    "NIM": nim,
                    "Student_Name": student_name,
                    "Class_Name": class_name,
                    "Quiz_Name": quiz_name,
                    "Quiz_Category": quiz_cat,
                    "Question_Name": qname,
                    "problem_id": problem_to_ix[qname],
                    "attempt_id": attempt_id,
                    "attempt_number": attempt_no,
                    "attempt_started_time": started_t,
                    "attempt_completed_time": completed_t,
                    "submit_time": submit_t,
                    "slot_no": slot_no,
                    "sub_order": 1,
                    "submit_stage": "First Submit First Attempt",
                    "question_no": q_no,
                    "num_submits": num_submits,
                    "code": mon_first["code"] or fc,
                    "correctness": 1 if mon_first["hasil_gt"].lower() == "correct" else f_corr,
                    "is_monitored": True,
                    "monitored_row_no": mon_first["monitored_row_no"],
                    "monitored_category": mon_first["kategori_pantauan"],
                    "monitored_source": mon_first["sumber_code"],
                    "monitored_status_asli": mon_first["status_asli_code"],
                    "monitored_hasil_gt": mon_first["hasil_gt"],
                    "monitored_code_note": mon_first.get("code_note", "-"),
                    "monitored_reviewer": mon_first.get("reviewer", "-"),
                    "misconception": mon_first["misconception"],
                    "student_all_misconceptions": stu_mp_summary,
                }
            )

        code_used = sc if sc else fc
        matched_mon = mon_final or (mon_first if (not sc or fc == sc) else None)
        events.append(
            {
                "student_key": student_key,
                "NIM": nim,
                "Student_Name": student_name,
                "Class_Name": class_name,
                "Quiz_Name": quiz_name,
                "Quiz_Category": quiz_cat,
                "Question_Name": qname,
                "problem_id": problem_to_ix[qname],
                "attempt_id": attempt_id,
                "attempt_number": attempt_no,
                "attempt_started_time": started_t,
                "attempt_completed_time": completed_t,
                "submit_time": submit_t,
                "slot_no": slot_no,
                "sub_order": 2,
                "submit_stage": matched_mon["sumber_code"] if matched_mon else ("Final Submit" if sc else "First Submit"),
                "question_no": q_no,
                "num_submits": num_submits,
                "code": matched_mon["code"] if (matched_mon and matched_mon.get("code")) else code_used,
                "correctness": is_corr,
                "is_monitored": bool(matched_mon is not None),
                "monitored_row_no": matched_mon["monitored_row_no"] if matched_mon else None,
                "monitored_category": matched_mon["kategori_pantauan"] if matched_mon else "",
                "monitored_source": matched_mon["sumber_code"] if matched_mon else "",
                "monitored_status_asli": matched_mon["status_asli_code"] if matched_mon else "",
                "monitored_hasil_gt": matched_mon["hasil_gt"] if matched_mon else "",
                "monitored_code_note": matched_mon.get("code_note", "-") if matched_mon else "-",
                "monitored_reviewer": matched_mon.get("reviewer", "-") if matched_mon else "-",
                "misconception": matched_mon["misconception"] if matched_mon else default_q_misconception,
                "student_all_misconceptions": stu_mp_summary,
            }
        )

    events_df = pd.DataFrame(events)

    # Urutkan secara ketat berdasarkan:
    # 1. Mahasiswa (student_key)
    # 2. Waktu mulai Attempt kuis & ID Attempt & Nomor Attempt (Attempt 1 selesai dulu, baru Attempt 2, dst.)
    # 3. Waktu submit soal di dalam Attempt tersebut (submit_time, slot_no, sub_order) -> Soal 1 s/d selesai
    events_df.sort_values(
        by=[
            "student_key",
            "attempt_started_time",
            "attempt_id",
            "attempt_number",
            "submit_time",
            "slot_no",
            "sub_order",
        ],
        inplace=True,
    )
    events_df.reset_index(drop=True, inplace=True)

    # Beri nomor urut sesi Attempt ke-berapa bagi mahasiswa tersebut & nomor urut soal di dalam Attempt
    attempt_seq_nums = []
    q_in_attempt_nums = []
    q_in_attempt_totals = []

    for skey, stu_grp in events_df.groupby("student_key", sort=False):
        seen_attempts = []
        att_counts = stu_grp.groupby("attempt_id", sort=False).size().to_dict()
        att_curr_idx = {}
        for _, row in stu_grp.iterrows():
            aid = row["attempt_id"]
            if aid not in seen_attempts:
                seen_attempts.append(aid)
            att_seq_no = seen_attempts.index(aid) + 1
            att_curr_idx[aid] = att_curr_idx.get(aid, 0) + 1
            attempt_seq_nums.append(att_seq_no)
            q_in_attempt_nums.append(att_curr_idx[aid])
            q_in_attempt_totals.append(att_counts[aid])

    events_df["student_attempt_session"] = attempt_seq_nums
    events_df["q_order_in_attempt"] = q_in_attempt_nums
    events_df["q_total_in_attempt"] = q_in_attempt_totals

    # Parse seluruh kode unik menggunakan disk cache + DAP parser + fallback AST
    # sehingga 100% soal di setiap Attempt tetap utuh dan tidak ada yang terbuang
    disk_cache = _get_ast_disk_cache()
    config = Config()
    leaf_flag = config.leaf_flag

    unique_codes = events_df["code"].unique().tolist()
    code_to_subtrees = {}
    for c in unique_codes:
        parsed = None
        if c in disk_cache and isinstance(disk_cache[c], dict):
            parsed = disk_cache[c]
        else:
            try:
                parsed = program_parser(c)
            except Exception:
                try:
                    parsed = program_parser(repair_dap_code(c))
                except Exception:
                    parsed = build_fallback_ast(c)

        subtrees = extracting_AST(
            parsed, leaf_flag, max_depth, recursive=recursive_subtrees
        )
        subtrees = [list(filter(None, st)) for st in subtrees if list(filter(None, st))]
        if not subtrees:
            subtrees = [["ListNode", " empty "]]
        code_to_subtrees[c] = subtrees

    return (
        events_df,
        code_to_subtrees,
        problem_to_ix,
        ix_to_problem,
        problem_category,
    )


# ==============================================================================
# BAGIAN 3: DKT & SANN-DKT LSTM MODEL (Section 4.3 & 5.2)
# ==============================================================================

def dkt_masked_bce_loss(y_true, y_pred):
    """
    Loss Binary Cross-Entropy ter-masking khusus untuk DKT (Section 4.3):
    - y_true[:, :, :M] adalah one-hot mask untuk soal berikutnya q_{t+1}
    - y_true[:, :, M]  adalah label kebenaran berikutnya a_{t+1} in {0, 1}
    - y_pred[:, :, :M] adalah prediksi probabilitas keberhasilan untuk masing-masing M soal
    """
    question_mask = y_true[:, :, :-1]
    target_correct = y_true[:, :, -1]

    pred_for_next_q = K.sum(y_pred * question_mask, axis=-1)
    active_step_mask = K.sum(question_mask, axis=-1)

    bce = K.binary_crossentropy(target_correct, pred_for_next_q)
    masked_bce = bce * active_step_mask
    return K.sum(masked_bce) / (K.sum(active_step_mask) + K.epsilon())


def create_dkt_lstm_model(
    input_dim,
    num_problems,
    lstm_units=200,
    dropout_rate=0.2,
    learning_rate=0.005,
    model_name="SANN_DKT",
):
    """
    Membangun model LSTM Deep Knowledge Tracing sesuai Section 4.3 & 5.2 paper:
    - Neurons LSTM = 200
    - Dropout = 0.2
    - Learning rate = 0.005
    - Output = vektor ukuran M dengan aktivasi sigmoid
    """
    seq_input = Input(shape=(None, input_dim), dtype=tf.float32, name="sequence_input")
    masked_input = Masking(mask_value=0.0, name="sequence_masking")(seq_input)
    lstm_out = LSTM(
        lstm_units,
        return_sequences=True,
        dropout=dropout_rate,
        name="dkt_lstm",
    )(masked_input)
    prob_out = TimeDistributed(
        Dense(num_problems, activation="sigmoid"), name="problem_predictions"
    )(lstm_out)

    model = Model(inputs=seq_input, outputs=prob_out, name=model_name)
    optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss=dkt_masked_bce_loss, optimizer=optimizer)
    return model


def build_dkt_sequences_for_students(
    events_df,
    student_keys,
    code_to_vec,
    num_problems,
    code_vec_dim,
    max_seq_len=None,
):
    """
    Membentuk tensor sequence untuk daftar mahasiswa sesuai urutan Attempt:
    Attempt 1 (Soal 1 -> selesai), dilanjutkan Attempt 2 (Soal 1 -> selesai), dst.
    Pada setiap timestep t = 0 .. T-2:
      - One-hot DKT x_t berukuran 2*M:
        Elemen ke-[q_t + M*(1 - a_t)] = 1, lainnya 0 (rumus Section 4.3 paper).
      - SANN code vector c_t berukuran code_vec_dim (TANPA normalisasi L2).
      - Input SANN-DKT z_t = [x_t || c_t] berukuran (2*M + code_vec_dim).
      - Target pada timestep t adalah pengerjaan soal berikutnya (t+1):
        soal q_{t+1} dan kebenaran a_{t+1}.
    """
    sub_df = events_df[events_df["student_key"].isin(student_keys)]
    grouped = {skey: grp for skey, grp in sub_df.groupby("student_key", sort=False)}

    student_records = []
    for skey in student_keys:
        if skey not in grouped:
            continue
        grp = grouped[skey]
        if len(grp) < 2:
            continue
        rows = grp.to_dict("records")
        if max_seq_len is not None and len(rows) > max_seq_len + 1:
            rows = rows[-(max_seq_len + 1) :]
        student_records.append((skey, rows))

    if not student_records:
        raise ValueError("Tidak ada mahasiswa dengan minimal 2 interaksi berurutan.")

    seq_len = (
        max_seq_len
        if max_seq_len is not None
        else max(len(r) - 1 for _, r in student_records)
    )

    num_students = len(student_records)
    dkt_only_dim = 2 * num_problems
    sann_dkt_dim = 2 * num_problems + code_vec_dim

    X_dkt = np.zeros((num_students, seq_len, dkt_only_dim), dtype=np.float32)
    X_sann_dkt = np.zeros((num_students, seq_len, sann_dkt_dim), dtype=np.float32)
    Y_target = np.zeros((num_students, seq_len, num_problems + 1), dtype=np.float32)
    meta_steps = []

    for i, (skey, rows) in enumerate(student_records):
        steps_info = []
        for t in range(len(rows) - 1):
            curr_ev = rows[t]
            next_ev = rows[t + 1]

            q_t = int(curr_ev["problem_id"])
            a_t = int(curr_ev["correctness"])

            # Rumus Section 4.3: indeks one-hot = q_t + M * (1 - a_t)
            onehot_idx = q_t + num_problems * (1 - a_t)
            x_t = np.zeros(dkt_only_dim, dtype=np.float32)
            x_t[onehot_idx] = 1.0

            c_t = code_to_vec[curr_ev["code"]]

            X_dkt[i, t, :] = x_t
            X_sann_dkt[i, t, :dkt_only_dim] = x_t
            X_sann_dkt[i, t, dkt_only_dim:] = c_t

            q_next = int(next_ev["problem_id"])
            a_next = float(next_ev["correctness"])
            Y_target[i, t, q_next] = 1.0
            Y_target[i, t, num_problems] = a_next

            steps_info.append(
                {
                    "student_key": skey,
                    "NIM": curr_ev["NIM"],
                    "Student_Name": curr_ev["Student_Name"],
                    "Class_Name": curr_ev["Class_Name"],
                    "student_all_misconceptions": curr_ev.get("student_all_misconceptions", ""),
                    "timestep": t + 1,
                    "curr_attempt_id": curr_ev["attempt_id"],
                    "curr_attempt_no": curr_ev["attempt_number"],
                    "curr_attempt_session": curr_ev["student_attempt_session"],
                    "curr_quiz_name": curr_ev["Quiz_Name"],
                    "curr_quiz_category": curr_ev["Quiz_Category"],
                    "curr_q_order": curr_ev["q_order_in_attempt"],
                    "curr_q_total": curr_ev["q_total_in_attempt"],
                    "curr_submit_time": curr_ev["submit_time"],
                    "curr_submit_stage": curr_ev.get("submit_stage", ""),
                    "curr_problem_id": q_t,
                    "curr_problem_name": curr_ev["Question_Name"],
                    "curr_correctness": a_t,
                    "curr_code": curr_ev["code"],
                    "curr_is_monitored": bool(curr_ev.get("is_monitored", False)),
                    "curr_monitored_row_no": curr_ev.get("monitored_row_no"),
                    "curr_monitored_category": curr_ev.get("monitored_category", ""),
                    "curr_monitored_source": curr_ev.get("monitored_source", ""),
                    "curr_monitored_status_asli": curr_ev.get("monitored_status_asli", ""),
                    "curr_monitored_hasil_gt": curr_ev.get("monitored_hasil_gt", ""),
                    "curr_monitored_code_note": curr_ev.get("monitored_code_note", "-"),
                    "curr_monitored_reviewer": curr_ev.get("monitored_reviewer", "-"),
                    "curr_misconception": curr_ev.get("misconception", ""),
                    "next_attempt_id": next_ev["attempt_id"],
                    "next_attempt_no": next_ev["attempt_number"],
                    "next_attempt_session": next_ev["student_attempt_session"],
                    "next_quiz_name": next_ev["Quiz_Name"],
                    "next_quiz_category": next_ev["Quiz_Category"],
                    "next_q_order": next_ev["q_order_in_attempt"],
                    "next_q_total": next_ev["q_total_in_attempt"],
                    "next_submit_time": next_ev["submit_time"],
                    "next_submit_stage": next_ev.get("submit_stage", ""),
                    "next_problem_id": q_next,
                    "next_problem_name": next_ev["Question_Name"],
                    "next_correctness": int(a_next),
                    "next_code": next_ev["code"],
                    "next_is_monitored": bool(next_ev.get("is_monitored", False)),
                    "next_monitored_row_no": next_ev.get("monitored_row_no"),
                    "next_monitored_category": next_ev.get("monitored_category", ""),
                    "next_monitored_source": next_ev.get("monitored_source", ""),
                    "next_monitored_status_asli": next_ev.get("monitored_status_asli", ""),
                    "next_monitored_hasil_gt": next_ev.get("monitored_hasil_gt", ""),
                    "next_monitored_code_note": next_ev.get("monitored_code_note", "-"),
                    "next_monitored_reviewer": next_ev.get("monitored_reviewer", "-"),
                    "next_misconception": next_ev.get("misconception", ""),
                }
            )
        meta_steps.append(steps_info)

    return X_dkt, X_sann_dkt, Y_target, meta_steps


def compute_code_line_explainability_map(
    unique_codes_list,
    extracted_attn,
    extracted_probs,
    max_subtree_length=64,
    max_depth=None,
    recursive=True,
):
    """
    Menghitung hasil prediksi SANN + bobot Attention per baris kode (line_scores & line_breakdown)
    untuk setiap kode unik sehingga setiap langkah pada sequence DKT dapat menampilkan
    kode mahasiswa beserta bobot attention per barisnya.
    """
    code_explain_map = {}
    for idx, code_str in enumerate(unique_codes_list):
        subtrees, subtree_line_lists, syntax_err_info = extract_subtrees_with_lines(
            code_str,
            max_depth=max_depth,
            leaf_flag=1,
            return_syntax_info=True,
            recursive=recursive,
        )
        raw_attn_1d = extracted_attn[idx, :, 0]
        probs = extracted_probs[idx]
        sann_pred_cls = int(np.argmax(probs))

        line_scores, ranked_lines, ranked_subtrees, _ = (
            compute_line_attention_from_subtrees(
                code_str,
                subtrees,
                subtree_line_lists,
                raw_attn_1d,
                max_subtree_length=max_subtree_length,
            )
        )

        lines = code_str.splitlines()
        top_ln = ranked_lines[0] if ranked_lines else 1
        top_code = lines[top_ln - 1].strip() if 1 <= top_ln <= len(lines) else ""
        top_weight = line_scores.get(top_ln, 0.0)

        line_breakdown = []
        for ln_no, ln_text in enumerate(lines, start=1):
            w = float(line_scores.get(ln_no, 0.0))
            rank_pos = (
                ranked_lines.index(ln_no) + 1 if ln_no in ranked_lines else None
            )
            line_breakdown.append(
                {
                    "line_no": ln_no,
                    "code": ln_text,
                    "weight": round(w, 4),
                    "weight_pct": round(w * 100, 1),
                    "rank": rank_pos,
                }
            )

        code_explain_map[code_str] = {
            "sann_pred_class": sann_pred_cls,
            "sann_prob_correct": round(float(probs[1]) * 100, 2),
            "sann_prob_incorrect": round(float(probs[0]) * 100, 2),
            "syntax_error": syntax_err_info,
            "top_line_no": top_ln,
            "top_line_code": top_code,
            "top_line_weight_pct": round(float(top_weight) * 100, 1),
            "line_scores": line_scores,
            "ranked_lines": ranked_lines,
            "line_breakdown": line_breakdown,
        }
    return code_explain_map


def evaluate_dkt_predictions(
    model, X_seq, Y_target, meta_steps, code_explain_map=None
):
    """
    Mengevaluasi prediksi model DKT pada seluruh timestep valid di data uji
    dan menyertakan informasi line-level attention untuk setiap kode pada langkah tersebut.
    """
    preds = model.predict(X_seq, verbose=0)
    y_true_all = []
    y_prob_all = []
    flat_records = []

    for i, steps_info in enumerate(meta_steps):
        for t, info in enumerate(steps_info):
            q_next = info["next_problem_id"]
            a_next = info["next_correctness"]
            prob = float(preds[i, t, q_next])
            y_true_all.append(a_next)
            y_prob_all.append(prob)

            rec = dict(info)
            rec["pred_prob"] = prob
            rec["pred_label"] = 1 if prob >= 0.5 else 0

            if code_explain_map is not None:
                curr_exp = code_explain_map.get(info["curr_code"], {})
                next_exp = code_explain_map.get(info["next_code"], {})
                rec["curr_explain"] = curr_exp
                rec["next_explain"] = next_exp
                rec["curr_top_line_no"] = curr_exp.get("top_line_no")
                rec["curr_top_line_weight_pct"] = curr_exp.get("top_line_weight_pct")
                rec["curr_top_line_code"] = curr_exp.get("top_line_code")

            flat_records.append(rec)

    y_true_arr = np.array(y_true_all, dtype=np.int32)
    y_prob_arr = np.array(y_prob_all, dtype=np.float32)
    y_pred_arr = (y_prob_arr >= 0.5).astype(np.int32)

    auc_score = (
        float(metrics.roc_auc_score(y_true_arr, y_prob_arr))
        if len(np.unique(y_true_arr)) > 1
        else 0.0
    )
    acc_score = float(metrics.accuracy_score(y_true_arr, y_pred_arr))
    prec_score = float(metrics.precision_score(y_true_arr, y_pred_arr, zero_division=0))
    rec_score = float(metrics.recall_score(y_true_arr, y_pred_arr, zero_division=0))
    f1_score = float(metrics.f1_score(y_true_arr, y_pred_arr, zero_division=0))

    df_res = pd.DataFrame(flat_records)
    category_auc = {}
    for cat in ["Easy", "Challenger", "Ultimate"]:
        sub = df_res[df_res["next_quiz_category"] == cat]
        if len(sub) > 0 and sub["next_correctness"].nunique() > 1:
            category_auc[cat] = round(
                float(metrics.roc_auc_score(sub["next_correctness"], sub["pred_prob"]))
                * 100,
                2,
            )

    return {
        "auc": round(auc_score * 100, 2),
        "accuracy": round(acc_score * 100, 2),
        "precision": round(prec_score * 100, 2),
        "recall": round(rec_score * 100, 2),
        "f1": round(f1_score * 100, 2),
        "total_predicted_steps": len(y_true_all),
        "category_auc": category_auc,
        "records": flat_records,
        "raw_matrix_preds": preds,
    }


# ==============================================================================
# BAGIAN 4: PIPELINE TRAINING & EVALUASI LENGKAP (SANN -> SANN-DKT vs DKT)
# ==============================================================================

def run_full_sann_dkt_pipeline(
    max_depth=None,
    max_subtree_length=64,
    embedding_size=64,
    code_vector_size=128,
    sann_epochs=40,
    dkt_epochs=60,
    lstm_units=200,
    dropout_rate=0.2,
    dkt_lr=0.005,
    entropy_lambda=3.5e-5,
    recursive_subtrees=True,
):
    print("=" * 85)
    print(" [TAHAP 1] MEMBANGUN DATASET & URUTAN ATTEMPT MAHASISWA MINGGU 1 (TANPA MAX_DEPTH)")
    print("=" * 85)
    (
        events_df,
        code_to_subtrees,
        problem_to_ix,
        ix_to_problem,
        problem_category,
    ) = load_week1_chronological_attempts(
        recursive_subtrees=recursive_subtrees, max_depth=max_depth
    )

    reviewed_nims = load_reviewed_student_nims()
    num_problems = len(problem_to_ix)

    # Hanya gunakan mahasiswa yang memiliki minimal 2 soal untuk sequence DKT
    stu_counts = events_df.groupby("student_key").size()
    all_students = sorted(stu_counts[stu_counts >= 2].index.tolist())

    # Pisahkan mahasiswa yang ada di Reviewed_Students_Status.csv agar PASTI masuk Test Set
    reviewed_student_keys = sorted(
        events_df[events_df["NIM"].isin(reviewed_nims)]["student_key"].unique().tolist()
    )
    reviewed_student_keys = [s for s in reviewed_student_keys if s in all_students]
    non_reviewed_students = [s for s in all_students if s not in reviewed_student_keys]

    # Target ukuran split 80:10:10 dari total mahasiswa; semua reviewed_student_keys wajib di test_students
    total_stu = len(all_students)
    target_val_size = max(1, int(round(0.10 * total_stu)))
    target_test_size = max(len(reviewed_student_keys), int(round(0.10 * total_stu)))
    extra_test_needed = max(0, target_test_size - len(reviewed_student_keys))

    if extra_test_needed > 0:
        train_val_pool, extra_test_students = train_test_split(
            non_reviewed_students, test_size=extra_test_needed, random_state=SEED
        )
        test_students = reviewed_student_keys + sorted(extra_test_students)
    else:
        train_val_pool = non_reviewed_students
        test_students = list(reviewed_student_keys)

    train_students, val_students = train_test_split(
        train_val_pool, test_size=target_val_size, random_state=SEED
    )

    print(f" Total Mahasiswa Unik (Minggu 1)        : {len(all_students)}")
    print(f" Mahasiswa Reviewed_Students_Status.csv : {len(reviewed_student_keys)} (100% masuk Data Test)")
    print(f" Total Soal Kanonik (M)                 : {num_problems}")
    print(f" Total Pengerjaan Soal (Events)         : {len(events_df)}")
    print(f" Total Kode Unik                        : {len(code_to_subtrees)}")
    print(
        f" Student-Level Split                    : Train={len(train_students)}, "
        f"Val={len(val_students)}, Test={len(test_students)} (Reviewed={len(reviewed_student_keys)})"
    )

    # --------------------------------------------------------------------------
    # TAHAP 2: MELATIH MODIFIED SANN UNTUK MENGHASILKAN CODE VECTORS (Sec 4.1)
    # --------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" [TAHAP 2] MELATIH MODIFIED SANN (Tanpa max_depth, Sigmoid Attention + Entropy Reg.)")
    print("=" * 85)

    train_events = events_df[events_df["student_key"].isin(train_students)]
    val_events = events_df[events_df["student_key"].isin(val_students)]
    test_events = events_df[events_df["student_key"].isin(test_students)]

    raw_paths_train = [code_to_subtrees[c] for c in train_events["code"]]
    raw_paths_val = [code_to_subtrees[c] for c in val_events["code"]]
    raw_paths_test = [code_to_subtrees[c] for c in test_events["code"]]

    raw_paths_train_str = [[" ".join(st) for st in p] for p in raw_paths_train]
    raw_paths_val_str = [[" ".join(st) for st in p] for p in raw_paths_val]
    raw_paths_test_str = [[" ".join(st) for st in p] for p in raw_paths_test]

    _, X_sann_train, node_vocab, wordtoix, ixtoword = preprocess_raw_paths_wordtoken(
        raw_paths_train, max_subtree_length
    )
    _, X_sann_val = encode_paths(raw_paths_val, wordtoix, max_subtree_length)
    _, X_sann_test = encode_paths(raw_paths_test, wordtoix, max_subtree_length)

    _, X_sann_train_p, path_vocab, pathtoix, ixtopath = preprocess_raw_paths_pathtoken(
        raw_paths_train_str, max_subtree_length
    )
    _, X_sann_val_p = encode_paths_path(raw_paths_val_str, pathtoix, max_subtree_length)
    _, X_sann_test_p = encode_paths_path(raw_paths_test_str, pathtoix, max_subtree_length)

    X_sann_train = np.array(X_sann_train, dtype=np.int32)
    X_sann_train_p = np.array(X_sann_train_p, dtype=np.int32)
    X_sann_val = np.array(X_sann_val, dtype=np.int32)
    X_sann_val_p = np.array(X_sann_val_p, dtype=np.int32)
    X_sann_test = np.array(X_sann_test, dtype=np.int32)
    X_sann_test_p = np.array(X_sann_test_p, dtype=np.int32)

    y_sann_train_raw = train_events["correctness"].values.astype(np.int32)
    y_sann_val_raw = val_events["correctness"].values.astype(np.int32)
    y_sann_test_raw = test_events["correctness"].values.astype(np.int32)

    Y_sann_train = to_categorical(y_sann_train_raw, num_classes=2)
    Y_sann_val = to_categorical(y_sann_val_raw, num_classes=2)
    Y_sann_test = to_categorical(y_sann_test_raw, num_classes=2)

    sann_model, sann_encoder = create_modified_sann_model(
        node_vocab_size=len(node_vocab),
        path_vocab_size=len(path_vocab),
        max_subtree_length=max_subtree_length,
        embedding_size=embedding_size,
        code_vector_size=code_vector_size,
        num_classes=2,
        entropy_lambda=entropy_lambda,
    )

    sann_opt = keras.optimizers.Adamax(learning_rate=0.001)
    sann_model.compile(
        loss="categorical_crossentropy", optimizer=sann_opt, metrics=["acc"]
    )

    cw_vals = compute_class_weight(
        "balanced", classes=np.unique(y_sann_train_raw), y=y_sann_train_raw
    )
    sann_cw = {int(c): float(w) for c, w in zip(np.unique(y_sann_train_raw), cw_vals)}

    sann_es = callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=10, restore_best_weights=True
    )
    sann_model.fit(
        [X_sann_train, X_sann_train_p],
        Y_sann_train,
        validation_data=([X_sann_val, X_sann_val_p], Y_sann_val),
        batch_size=32,
        epochs=sann_epochs,
        callbacks=[sann_es],
        class_weight=sann_cw,
        verbose=1,
    )

    sann_test_probs = sann_model.predict([X_sann_test, X_sann_test_p], verbose=0)
    sann_test_pred = np.argmax(sann_test_probs, axis=1)
    sann_acc = float(metrics.accuracy_score(y_sann_test_raw, sann_test_pred))
    sann_f1 = float(metrics.f1_score(y_sann_test_raw, sann_test_pred, zero_division=0))
    print(
        f" -> Modified SANN Test Accuracy: {sann_acc*100:.2f}% | F1-Score: {sann_f1*100:.2f}%"
    )

    # Ekstrak SANN Code Vectors c_t (TANPA normalisasi L2) & Attention Weights
    print("\n -> Mengekstrak SANN Code Vectors (c_t) TANPA normalisasi L2 + Line Attention...")
    unique_codes_list = list(code_to_subtrees.keys())
    all_unique_paths = [code_to_subtrees[c] for c in unique_codes_list]
    all_unique_paths_str = [[" ".join(st) for st in p] for p in all_unique_paths]

    _, X_all_nodes = encode_paths(all_unique_paths, wordtoix, max_subtree_length)
    _, X_all_paths = encode_paths_path(
        all_unique_paths_str, pathtoix, max_subtree_length
    )

    extracted_vecs, extracted_attn, extracted_probs = sann_encoder.predict(
        [
            np.array(X_all_nodes, dtype=np.int32),
            np.array(X_all_paths, dtype=np.int32),
        ],
        batch_size=64,
        verbose=0,
    )

    # TANPA normalisasi L2 sesuai paper asli
    code_to_vec = {
        c: extracted_vecs[idx] for idx, c in enumerate(unique_codes_list)
    }

    # Hitung bobot attention per baris untuk seluruh kode di test set
    test_unique_codes = test_events["code"].unique().tolist()
    test_code_indices = [unique_codes_list.index(c) for c in test_unique_codes]
    test_code_explain_map = compute_code_line_explainability_map(
        test_unique_codes,
        extracted_attn[test_code_indices],
        extracted_probs[test_code_indices],
        max_subtree_length=max_subtree_length,
        max_depth=max_depth,
        recursive=recursive_subtrees,
    )

    # --------------------------------------------------------------------------
    # TAHAP 3: MEMBANGUN TENSOR SEQUENCE & MELATIH LSTM SANN-DKT vs ORIGINAL DKT
    # --------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" [TAHAP 3] MELATIH LSTM DEEP KNOWLEDGE TRACING (SANN-DKT vs ORIGINAL DKT)")
    print("=" * 85)

    X_dkt_tr, X_sdkt_tr, Y_tr, meta_tr = build_dkt_sequences_for_students(
        events_df, train_students, code_to_vec, num_problems, code_vector_size
    )
    X_dkt_val, X_sdkt_val, Y_val, meta_val = build_dkt_sequences_for_students(
        events_df, val_students, code_to_vec, num_problems, code_vector_size
    )
    X_dkt_te, X_sdkt_te, Y_te, meta_te = build_dkt_sequences_for_students(
        events_df, test_students, code_to_vec, num_problems, code_vector_size
    )

    print(
        f" Dimensi Input Original DKT : {X_dkt_tr.shape[2]} (2 * M = 2 * {num_problems})"
    )
    print(
        f" Dimensi Input SANN-DKT     : {X_sdkt_tr.shape[2]} (2 * M + d_c = {2*num_problems} + {code_vector_size})"
    )
    print(
        f" Shape Sequence Train       : {X_sdkt_tr.shape} (Mahasiswa={X_sdkt_tr.shape[0]}, MaxSteps={X_sdkt_tr.shape[1]})"
    )
    print(
        f" Shape Sequence Test        : {X_sdkt_te.shape} (Mahasiswa={X_sdkt_te.shape[0]}, MaxSteps={X_sdkt_te.shape[1]})"
    )

    # 3A. Latih Baseline Original DKT
    print("\n--- Melatih Baseline Original DKT (Tanpa Code Vector) ---")
    orig_dkt_model = create_dkt_lstm_model(
        input_dim=2 * num_problems,
        num_problems=num_problems,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        learning_rate=dkt_lr,
        model_name="Original_DKT",
    )
    es_orig = callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=12, restore_best_weights=True
    )
    orig_dkt_model.fit(
        X_dkt_tr,
        Y_tr,
        validation_data=(X_dkt_val, Y_val),
        batch_size=16,
        epochs=dkt_epochs,
        callbacks=[es_orig],
        verbose=0,
    )
    orig_eval_val = evaluate_dkt_predictions(orig_dkt_model, X_dkt_val, Y_val, meta_val)
    orig_eval_test = evaluate_dkt_predictions(orig_dkt_model, X_dkt_te, Y_te, meta_te)

    # 3B. Latih SANN-DKT (One-Hot {q_t, a_t} + Raw SANN Code Vector c_t)
    print("--- Melatih SANN-DKT (One-Hot DKT + Raw SANN Code Vectors Tanpa L2 Norm) ---")
    sann_dkt_model = create_dkt_lstm_model(
        input_dim=2 * num_problems + code_vector_size,
        num_problems=num_problems,
        lstm_units=lstm_units,
        dropout_rate=dropout_rate,
        learning_rate=dkt_lr,
        model_name="SANN_DKT",
    )
    es_sdkt = callbacks.EarlyStopping(
        monitor="val_loss", mode="min", patience=12, restore_best_weights=True
    )
    sann_dkt_model.fit(
        X_sdkt_tr,
        Y_tr,
        validation_data=(X_sdkt_val, Y_val),
        batch_size=16,
        epochs=dkt_epochs,
        callbacks=[es_sdkt],
        verbose=1,
    )
    sdkt_eval_val = evaluate_dkt_predictions(sann_dkt_model, X_sdkt_val, Y_val, meta_val)
    sdkt_eval_test = evaluate_dkt_predictions(
        sann_dkt_model, X_sdkt_te, Y_te, meta_te, code_explain_map=test_code_explain_map
    )

    # --------------------------------------------------------------------------
    # TAHAP 4: CETAK PERBANDINGAN HASIL & SIMPAN MODEL + LAPORAN VISUAL LENGKAP
    # --------------------------------------------------------------------------
    print("\n" + "=" * 85)
    print(" HASIL EVALUASI DEEP KNOWLEDGE TRACING MINGGU 1 (SESUAI TABEL 3 PAPER)")
    print("=" * 85)
    print(
        f"{'Model':<18} | {'Test AUC (%)':>12} | {'Val AUC (%)':>11} | "
        f"{'Acc (%)':>8} | {'F1 (%)':>8} | {'Easy AUC':>8} | {'Chal AUC':>8} | {'Ult AUC':>8}"
    )
    print("-" * 85)
    print(
        f"{'Original DKT':<18} | {orig_eval_test['auc']:>11.2f}% | {orig_eval_val['auc']:>10.2f}% | "
        f"{orig_eval_test['accuracy']:>7.2f}% | {orig_eval_test['f1']:>7.2f}% | "
        f"{orig_eval_test['category_auc'].get('Easy', 0):>7.2f}% | "
        f"{orig_eval_test['category_auc'].get('Challenger', 0):>7.2f}% | "
        f"{orig_eval_test['category_auc'].get('Ultimate', 0):>7.2f}%"
    )
    print(
        f"{'SANN-DKT (Ours)':<18} | {sdkt_eval_test['auc']:>11.2f}% | {sdkt_eval_val['auc']:>10.2f}% | "
        f"{sdkt_eval_test['accuracy']:>7.2f}% | {sdkt_eval_test['f1']:>7.2f}% | "
        f"{sdkt_eval_test['category_auc'].get('Easy', 0):>7.2f}% | "
        f"{sdkt_eval_test['category_auc'].get('Challenger', 0):>7.2f}% | "
        f"{sdkt_eval_test['category_auc'].get('Ultimate', 0):>7.2f}%"
    )
    print("=" * 85)

    SAVED_DKT_DIR.mkdir(parents=True, exist_ok=True)
    sann_model.save_weights(str(SAVED_DKT_DIR / "modified_sann_weights.h5"))
    sann_dkt_model.save_weights(str(SAVED_DKT_DIR / "sann_dkt_lstm_weights.h5"))
    orig_dkt_model.save_weights(str(SAVED_DKT_DIR / "original_dkt_lstm_weights.h5"))

    summary_payload = {
        "paper_reference": "Automated Identification of Logical Errors in Programs: Advancing Scalable Analysis of Student Misconceptions (arXiv:2505.10913v1)",
        "num_problems": num_problems,
        "problems": problem_to_ix,
        "problem_categories": problem_category,
        "reviewed_test_nims": sorted(list(reviewed_nims)),
        "test_student_keys": test_students,
        "hyperparameters": {
            "max_depth": max_depth,
            "l2_normalization": False,
            "max_subtree_length": max_subtree_length,
            "sann_embedding_size": embedding_size,
            "code_vector_size": code_vector_size,
            "entropy_lambda": entropy_lambda,
            "recursive_subtrees": recursive_subtrees,
            "lstm_units": lstm_units,
            "dropout_rate": dropout_rate,
            "dkt_learning_rate": dkt_lr,
        },
        "modified_sann_metrics": {
            "test_accuracy": round(sann_acc * 100, 2),
            "test_f1": round(sann_f1 * 100, 2),
            "node_vocab_size": len(node_vocab),
            "path_vocab_size": len(path_vocab),
        },
        "original_dkt_test": {
            k: v
            for k, v in orig_eval_test.items()
            if k not in ("records", "raw_matrix_preds")
        },
        "sann_dkt_test": {
            k: v
            for k, v in sdkt_eval_test.items()
            if k not in ("records", "raw_matrix_preds")
        },
        "wordtoix": wordtoix,
        "pathtoix": pathtoix,
    }

    with open(SAVED_DKT_DIR / "dkt_metadata.json", "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)

    # Simpan CSV trajectory tanpa dict bersarang berat
    csv_records = []
    for r in sdkt_eval_test["records"]:
        rc = {
            k: v
            for k, v in r.items()
            if k not in ("curr_explain", "next_explain")
        }
        csv_records.append(rc)
    traj_df = pd.DataFrame(csv_records)
    traj_csv_path = SAVED_DKT_DIR / "test_student_trajectories.csv"
    traj_df.to_csv(traj_csv_path, index=False, encoding="utf-8-sig")

    _save_dkt_html_report(
        summary_payload,
        sdkt_eval_test,
        orig_eval_test,
        reviewed_nims,
        DKT_HTML_REPORT,
    )

    print(f"\nModel & Metadata SANN-DKT disimpan ke : {SAVED_DKT_DIR}")
    print(f"Data Trajectory Mahasiswa disimpan ke : {traj_csv_path}")
    print(f"Laporan Visual HTML disimpan ke       : {DKT_HTML_REPORT}")

    return summary_payload


def _render_code_lines_html(code_str, explain_dict):
    """Me-render tabel baris kode beserta bar & persentase bobot Attention per baris."""
    lines = code_str.splitlines()
    line_scores = explain_dict.get("line_scores", {}) if explain_dict else {}
    ranked_lines = explain_dict.get("ranked_lines", []) if explain_dict else []
    syn_err = explain_dict.get("syntax_error") if explain_dict else None

    max_score = max(line_scores.values()) if line_scores else 1.0
    if max_score <= 0:
        max_score = 1.0

    rows_html = []
    for ln_idx, text in enumerate(lines, start=1):
        score = float(line_scores.get(ln_idx, 0.0))
        intensity = score / max_score if max_score > 0 else 0.0
        alpha = round(0.08 + 0.50 * intensity, 2) if score > 0 else 0.0
        bg_color = f"rgba(255, 82, 82, {alpha})" if score > 0 else "transparent"

        badges = []
        if syn_err and syn_err.get("line") == ln_idx:
            badges.append(
                f'<span style="background:#b71c1c;color:#fff;padding:1px 6px;border-radius:4px;font-size:10px;margin-left:6px;">{syn_err.get("message", "SyntaxError")}</span>'
            )
        if ranked_lines and ln_idx == ranked_lines[0]:
            badges.append(
                '<span style="background:#d32f2f;color:#fff;padding:1px 6px;border-radius:4px;font-size:10px;margin-left:6px;">FOKUS #1</span>'
            )
        elif len(ranked_lines) > 1 and ln_idx == ranked_lines[1]:
            badges.append(
                '<span style="background:#f57c00;color:#fff;padding:1px 6px;border-radius:4px;font-size:10px;margin-left:6px;">FOKUS #2</span>'
            )
        elif len(ranked_lines) > 2 and ln_idx == ranked_lines[2]:
            badges.append(
                '<span style="background:#fbc02d;color:#222;padding:1px 6px;border-radius:4px;font-size:10px;margin-left:6px;">FOKUS #3</span>'
            )

        esc = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        bar_w = int(round(intensity * 60))
        bar_html = (
            f'<div style="display:inline-block;width:60px;height:8px;background:#eee;border-radius:4px;overflow:hidden;vertical-align:middle;margin-right:6px;">'
            f'<div style="width:{bar_w}px;height:8px;background:#d32f2f;"></div></div>'
        )
        rows_html.append(
            f'<tr style="background:{bg_color};">'
            f'<td style="padding:2px 8px;text-align:right;color:#666;border-right:1px solid #e0e0e0;width:32px;user-select:none;">{ln_idx}</td>'
            f'<td style="padding:2px 8px;text-align:right;font-weight:bold;color:#b71c1c;border-right:1px solid #e0e0e0;width:125px;">{bar_html}{score*100:>4.1f}%</td>'
            f'<td style="padding:2px 10px;white-space:pre-wrap;font-family:Consolas,monospace;font-size:12.5px;">{esc}{"".join(badges)}</td>'
            f"</tr>"
        )
    return (
        '<table style="width:100%;border-collapse:collapse;border:1px solid #e0e0e0;background:#fff;margin-top:6px;">'
        "<thead><tr style='background:#f5f5f5;border-bottom:1px solid #ddd;font-size:11.5px;color:#555;'>"
        "<th style='padding:3px 8px;text-align:right;'>Line</th>"
        "<th style='padding:3px 8px;text-align:right;'>Bobot Attention</th>"
        "<th style='padding:3px 10px;text-align:left;'>Kode Pseudocode Mahasiswa</th>"
        "</tr></thead>"
        f"<tbody>{''.join(rows_html)}</tbody></table>"
    )


def _has_valid_code_note(note_val):
    if note_val is None:
        return False
    s = str(note_val).strip()
    return bool(s and s != "-" and s.lower() != "nan")


def _save_dkt_html_report(
    summary, sdkt_eval, orig_eval, reviewed_nims, html_path
):
    """
    Membuat laporan HTML interaktif untuk seluruh mahasiswa di Reviewed_Students_Status.csv (Data Test):
    - Mengelompokkan riwayat pengerjaan berdasarkan Attempt (Attempt 1 Soal 1 - selesai, Attempt 2 Soal 1 - selesai, dst.)
    - Pada setiap soal menampilkan prediksi soal berikutnya (SANN-DKT) SEKALIGUS kode mahasiswa & bobot Attention per baris!
    - Dilengkapi filter interaktif untuk menyeleksi hanya kode yang memiliki Misconception Note.
    """
    records = sdkt_eval["records"]
    df_rec = pd.DataFrame(records)

    student_sections_html = []
    student_options_html = []
    all_note_tags = set()

    if not df_rec.empty:
        for _, r in df_rec.iterrows():
            for col_n in ("curr_monitored_code_note", "next_monitored_code_note"):
                nv = r.get(col_n, "-")
                if _has_valid_code_note(nv):
                    for part in str(nv).split(","):
                        p = part.strip()
                        if p:
                            all_note_tags.add(p)

        # Prioritaskan semua mahasiswa dari Reviewed_Students_Status.csv
        all_skeys = df_rec["student_key"].unique().tolist()
        rev_skeys = [
            sk
            for sk in all_skeys
            if str(df_rec[df_rec["student_key"] == sk]["NIM"].iloc[0]) in reviewed_nims
        ]
        other_skeys = [sk for sk in all_skeys if sk not in rev_skeys]
        ordered_skeys = rev_skeys + other_skeys

        for s_idx, skey in enumerate(ordered_skeys):
            grp = df_rec[df_rec["student_key"] == skey]
            first_row = grp.iloc[0]
            nim_str = str(first_row["NIM"])
            is_rev = nim_str in reviewed_nims
            rev_tag = " [Reviewed]" if is_rev else ""
            stu_acc = (grp["pred_label"] == grp["next_correctness"]).mean() * 100
            mon_count = int(
                grp["curr_is_monitored"].sum()
                + (1 if grp.iloc[-1].get("next_is_monitored") else 0)
            )
            note_count = int(
                sum(1 for _, rr in grp.iterrows() if _has_valid_code_note(rr.get("curr_monitored_code_note")))
                + (1 if _has_valid_code_note(grp.iloc[-1].get("next_monitored_code_note")) else 0)
            )
            stu_mp_all = first_row.get(
                "student_all_misconceptions", "Tidak ada data miskonsepsi"
            )

            student_options_html.append(
                f'<option value="stu_{s_idx}" data-note-count="{note_count}" data-mon-count="{mon_count}">'
                f'{first_row["Student_Name"]} ({nim_str}) - {first_row["Class_Name"]}{rev_tag} | '
                f'Ada Note: {note_count} kode | Monitored: {mon_count} kode | Acc: {stu_acc:.1f}%</option>'
            )

            # Kelompokkan berdasarkan sesi Attempt (Attempt 1, Attempt 2, ...)
            attempt_blocks_html = []
            for att_sess, att_grp in grp.groupby("curr_attempt_session", sort=False):
                att_first = att_grp.iloc[0]
                att_no = att_first["curr_attempt_no"]
                quiz_name = att_first["curr_quiz_name"]
                att_id = att_first["curr_attempt_id"]

                step_cards = []
                for _, r in att_grp.iterrows():
                    c_curr_badge = (
                        '<span style="background:#e8f5e9;color:#2e7d32;padding:2px 8px;border-radius:4px;font-weight:bold;">Correct (1)</span>'
                        if r["curr_correctness"] == 1
                        else '<span style="background:#ffebee;color:#c62828;padding:2px 8px;border-radius:4px;font-weight:bold;">Incorrect (0)</span>'
                    )
                    c_next_badge = (
                        '<span style="background:#e8f5e9;color:#2e7d32;padding:2px 8px;border-radius:4px;font-weight:bold;">Correct (1)</span>'
                        if r["next_correctness"] == 1
                        else '<span style="background:#ffebee;color:#c62828;padding:2px 8px;border-radius:4px;font-weight:bold;">Incorrect (0)</span>'
                    )
                    prob_pct = r["pred_prob"] * 100
                    match_ok = r["pred_label"] == r["next_correctness"]
                    pred_bg = "#e8f5e9" if match_ok else "#fff3e0"
                    pred_col = "#1b5e20" if match_ok else "#e65100"
                    match_txt = "MATCH" if match_ok else "MISMATCH"

                    curr_exp = r.get("curr_explain", {})
                    code_table_html = _render_code_lines_html(r["curr_code"], curr_exp)

                    next_att_info = (
                        f"Attempt {r['next_attempt_no']} ({r['next_quiz_category']} - Soal {r['next_q_order']}/{r['next_q_total']})"
                    )

                    is_mon = bool(r.get("curr_is_monitored", False))
                    mon_note = str(r.get("curr_monitored_code_note", "-")).strip()
                    has_note = _has_valid_code_note(mon_note)
                    mon_banner_html = ""
                    if is_mon:
                        mon_cat = r.get("curr_monitored_category", "")
                        mon_src = r.get("curr_monitored_source", "")
                        mon_stat = r.get("curr_monitored_status_asli", "")
                        mon_row = r.get("curr_monitored_row_no", "")
                        mon_rev = r.get("curr_monitored_reviewer", "-")
                        mon_bg = "#fff8e1" if "Wrong" in mon_cat else "#e3f2fd"
                        mon_border = "#f57c00" if "Wrong" in mon_cat else "#1976d2"
                        note_badge_style = (
                            "background:#b71c1c;color:#fff;padding:1px 7px;border-radius:4px;font-weight:bold;"
                            if has_note
                            else "background:#e0e0e0;color:#555;padding:1px 7px;border-radius:4px;"
                        )
                        mon_banner_html = (
                            f'<div style="margin-top:6px;padding:8px 12px;background:{mon_bg};border-left:4px solid {mon_border};border-radius:4px;font-size:12.5px;">'
                            f'<strong>[Monitored_Codes_Dipantau.csv #{mon_row}]</strong> &nbsp;|&nbsp; '
                            f'Kategori: <strong style="color:{mon_border};">{mon_cat}</strong> &nbsp;|&nbsp; '
                            f'Sumber: <code>{mon_src}</code> &nbsp;|&nbsp; Status Asli: <code>{mon_stat}</code> &nbsp;|&nbsp; '
                            f'Reviewer: <strong>{mon_rev}</strong><br>'
                            f'<strong>Note Misconception di Dalam Code (<code>Misconception_Di_Dalam_Code_Notes</code>):</strong> '
                            f'<span style="{note_badge_style}">{mon_note}</span><br>'
                            f'<strong>Jenis Misconception Minggu Itu:</strong> <span style="color:#b71c1c;font-weight:600;">{r.get("curr_misconception", "-")}</span>'
                            f'</div>'
                        )
                    else:
                        mon_banner_html = (
                            f'<div style="margin-top:6px;padding:5px 10px;background:#f5f7fa;border-left:3px solid #90a4ae;border-radius:4px;font-size:12px;color:#444;">'
                            f'<strong>Misconception Mahasiswa Minggu Ini:</strong> {r.get("curr_misconception", "-")}'
                            f'</div>'
                        )

                    card_border = (
                        "2px solid #d32f2f"
                        if has_note
                        else ("2px solid #f57c00" if is_mon else "1px solid #dce3ec")
                    )
                    mon_badge = (
                        f'<span style="background:#f57c00;color:#fff;padding:2px 7px;border-radius:4px;font-size:11px;margin-left:6px;">DIPANTAU #{r.get("curr_monitored_row_no")} ({r.get("curr_monitored_category")})</span>'
                        if is_mon
                        else ""
                    )
                    note_header_badge = (
                        f'<span style="background:#b71c1c;color:#fff;padding:2px 7px;border-radius:4px;font-size:11px;margin-left:6px;">NOTE: {mon_note}</span>'
                        if has_note
                        else ""
                    )

                    step_cards.append(
                        f"""
                        <div class="step-card" data-has-note="{'1' if has_note else '0'}" data-is-monitored="{'1' if is_mon else '0'}" data-note-val="{mon_note if has_note else ''}" style="border:{card_border};border-radius:6px;padding:12px;margin-bottom:12px;background:#fafbfd;">
                          <div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:8px;border-bottom:1px solid #e6ebf1;padding-bottom:8px;">
                            <div>
                              <strong style="color:#1565c0;">Step t={r['timestep']}</strong> &nbsp;|&nbsp;
                              <strong>Attempt {r['curr_attempt_no']} — Soal {r['curr_q_order']}/{r['curr_q_total']}: {r['curr_problem_name']}</strong>
                              <span style="color:#555;font-size:12px;">[{r.get('curr_submit_stage', '')} | {r['curr_submit_time']}]</span>
                              {mon_badge}{note_header_badge}
                            </div>
                            <div>
                              Hasil Saat Ini (a_t): {c_curr_badge}
                            </div>
                          </div>

                          {mon_banner_html}

                          <div style="margin-top:8px;padding:8px 12px;background:{pred_bg};border-radius:6px;display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;">
                            <div>
                              <strong>Prediksi Soal Berikutnya (t+1):</strong>
                              <span style="color:#333;">{next_att_info} &rarr; <strong>{r['next_problem_name']}</strong></span>
                            </div>
                            <div>
                              Prediksi SANN-DKT P(a_{{t+1}}=1): <strong style="color:{pred_col};font-size:14px;">{prob_pct:.1f}%</strong>
                              &nbsp;|&nbsp; Aktual (a_{{t+1}}): {c_next_badge}
                              &nbsp;<span style="font-size:11px;font-weight:bold;color:{pred_col};">[{match_txt}]</span>
                            </div>
                          </div>

                          <div style="margin-top:8px;">
                            <div style="font-size:12px;color:#444;font-weight:600;">
                              Kode Mahasiswa pada Step t={r['timestep']} &amp; Bobot Attention SANN per Baris
                              (Fokus #1: Baris {curr_exp.get('top_line_no', '-')} — {curr_exp.get('top_line_weight_pct', 0)}%):
                            </div>
                            {code_table_html}
                          </div>
                        </div>
                        """
                    )

                attempt_blocks_html.append(
                    f"""
                    <div class="attempt-block" style="margin-top:14px;border:2px solid #1976d2;border-radius:8px;overflow:hidden;background:#fff;">
                      <div style="background:#1976d2;color:#fff;padding:8px 14px;font-weight:bold;display:flex;justify-content:space-between;">
                        <span>Sesi Attempt #{att_sess} &mdash; {quiz_name} (Attempt Ke-{att_no} | Attempt_ID: {att_id})</span>
                        <span>Urutan Soal 1 s/d Selesai</span>
                      </div>
                      <div style="padding:12px;">
                        {''.join(step_cards)}
                      </div>
                    </div>
                    """
                )

            # Tampilkan juga kode soal terakhir (Step T) pada Attempt terakhir mahasiswa tersebut
            last_r = grp.iloc[-1]
            next_exp = last_r.get("next_explain", {})
            last_code_html = _render_code_lines_html(last_r["next_code"], next_exp)
            c_last_badge = (
                '<span style="background:#e8f5e9;color:#2e7d32;padding:2px 8px;border-radius:4px;font-weight:bold;">Correct (1)</span>'
                if last_r["next_correctness"] == 1
                else '<span style="background:#ffebee;color:#c62828;padding:2px 8px;border-radius:4px;font-weight:bold;">Incorrect (0)</span>'
            )
            last_is_mon = bool(last_r.get("next_is_monitored", False))
            last_note = str(last_r.get("next_monitored_code_note", "-")).strip()
            last_has_note = _has_valid_code_note(last_note)
            last_mon_banner = ""
            if last_is_mon:
                l_cat = last_r.get("next_monitored_category", "")
                l_src = last_r.get("next_monitored_source", "")
                l_stat = last_r.get("next_monitored_status_asli", "")
                l_row = last_r.get("next_monitored_row_no", "")
                l_rev = last_r.get("next_monitored_reviewer", "-")
                l_bg = "#fff8e1" if "Wrong" in l_cat else "#e3f2fd"
                l_border = "#f57c00" if "Wrong" in l_cat else "#1976d2"
                l_note_style = (
                    "background:#b71c1c;color:#fff;padding:1px 7px;border-radius:4px;font-weight:bold;"
                    if last_has_note
                    else "background:#e0e0e0;color:#555;padding:1px 7px;border-radius:4px;"
                )
                last_mon_banner = (
                    f'<div style="margin-top:6px;padding:8px 12px;background:{l_bg};border-left:4px solid {l_border};border-radius:4px;font-size:12.5px;">'
                    f'<strong>[Monitored_Codes_Dipantau.csv #{l_row}]</strong> &nbsp;|&nbsp; '
                    f'Kategori: <strong style="color:{l_border};">{l_cat}</strong> &nbsp;|&nbsp; '
                    f'Sumber: <code>{l_src}</code> &nbsp;|&nbsp; Status Asli: <code>{l_stat}</code> &nbsp;|&nbsp; '
                    f'Reviewer: <strong>{l_rev}</strong><br>'
                    f'<strong>Note Misconception di Dalam Code (<code>Misconception_Di_Dalam_Code_Notes</code>):</strong> '
                    f'<span style="{l_note_style}">{last_note}</span><br>'
                    f'<strong>Jenis Misconception Minggu Itu:</strong> <span style="color:#b71c1c;font-weight:600;">{last_r.get("next_misconception", "-")}</span>'
                    f'</div>'
                )

            attempt_blocks_html.append(
                f"""
                <div class="attempt-block" style="margin-top:12px;">
                  <div class="step-card" data-has-note="{'1' if last_has_note else '0'}" data-is-monitored="{'1' if last_is_mon else '0'}" data-note-val="{last_note if last_has_note else ''}" style="border:1px solid #90a4ae;border-radius:8px;padding:12px;background:#fff;">
                    <div style="display:flex;justify-content:space-between;align-items:center;">
                      <strong>Step Terakhir t={last_r['timestep']+1} &mdash; Attempt {last_r['next_attempt_no']} (Soal {last_r['next_q_order']}/{last_r['next_q_total']}): {last_r['next_problem_name']}</strong>
                      <span>Hasil Aktual (a_T): {c_last_badge}</span>
                    </div>
                    {last_mon_banner}
                    <div style="margin-top:8px;">
                      <div style="font-size:12px;color:#444;font-weight:600;">
                        Kode Mahasiswa pada Step Terakhir &amp; Bobot Attention SANN per Baris:
                      </div>
                      {last_code_html}
                    </div>
                  </div>
                </div>
                """
            )

            disp_style = "block" if s_idx == 0 else "none"
            student_sections_html.append(
                f"""
                <div id="stu_{s_idx}" class="student-panel" data-note-count="{note_count}" data-mon-count="{mon_count}" style="display:{disp_style};">
                  <h3 style="margin:12px 0 4px 0;">
                    {first_row['Student_Name']} ({nim_str}) &mdash; Kelas {first_row['Class_Name']}
                    {'<span style="background:#2e7d32;color:#fff;padding:2px 8px;border-radius:4px;font-size:12px;margin-left:8px;">Reviewed_Students_Status.csv (Data Test)</span>' if is_rev else ''}
                  </h3>
                  <p style="margin:4px 0 12px 0;color:#555;font-size:13.5px;">
                    Total Langkah Prediksi: <strong>{len(grp)}</strong> &nbsp;|&nbsp;
                    Kode Ada Note Misconception: <strong style="color:#b71c1c;">{note_count} kode</strong> &nbsp;|&nbsp;
                    Kode Dipantau (<code>Monitored_Codes_Dipantau.csv</code>): <strong style="color:#e65100;">{mon_count} kode</strong> &nbsp;|&nbsp;
                    Semua Misconception Minggu Ini: <strong style="color:#b71c1c;">{stu_mp_all}</strong> &nbsp;|&nbsp;
                    Akurasi Prediksi Soal Berikutnya (SANN-DKT): <strong>{stu_acc:.2f}%</strong>
                  </p>
                  {''.join(attempt_blocks_html)}
                </div>
                """
            )

    note_tag_options = ['<option value="ALL">Semua Kode Note Misconception</option>'] + [
        f'<option value="{t}">{t}</option>' for t in sorted(all_note_tags)
    ]

    html = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<title>SANN-DKT Deep Knowledge Tracing &amp; Line Attention Report</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 20px; background: #f4f6f9; color: #222; }}
  .container {{ max-width: 1180px; margin: 0 auto; }}
  .card {{ background: #fff; border: 1px solid #dce3ec; border-radius: 8px; padding: 20px; margin-bottom: 20px; box-shadow: 0 2px 6px rgba(0,0,0,0.04); }}
  table.metric-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
  table.metric-table th, table.metric-table td {{ padding: 10px 12px; border-bottom: 1px solid #eee; text-align: center; }}
  table.metric-table th {{ background: #f0f4f8; }}
  select.stu-select {{ padding: 8px 12px; font-size: 13.5px; border: 1px solid #1976d2; border-radius: 6px; background: #fff; }}
  .filter-box {{ background: #f0f4f8; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px 16px; margin-bottom: 16px; display: flex; flex-wrap: wrap; gap: 12px; align-items: flex-end; }}
  .filter-item {{ display: flex; flex-direction: column; gap: 4px; flex: 1; min-width: 210px; }}
  .filter-item label {{ font-size: 12px; font-weight: 700; color: #334155; text-transform: uppercase; }}
  .btn-filter {{ padding: 8px 14px; border-radius: 6px; border: none; font-weight: 700; font-size: 13px; cursor: pointer; }}
  .btn-note-only {{ background: #b71c1c; color: #fff; }}
  .btn-reset {{ background: #475569; color: #fff; }}
</style>
<script>
  function applyFilters() {{
    var stuVal = document.getElementById('stu_dropdown').value;
    var modeVal = document.getElementById('filter_mode').value;
    var noteTagVal = document.getElementById('filter_note_tag').value;

    var panels = document.getElementsByClassName('student-panel');
    var totalVisibleSteps = 0;
    var totalVisibleStudents = 0;

    for (var i = 0; i < panels.length; i++) {{
      var panel = panels[i];
      var stuMatch = (stuVal === 'ALL_STUDENTS') || (panel.id === stuVal);
      if (!stuMatch) {{
        panel.style.display = 'none';
        continue;
      }}

      var attBlocks = panel.getElementsByClassName('attempt-block');
      var panelVisibleSteps = 0;

      for (var j = 0; j < attBlocks.length; j++) {{
        var attBlock = attBlocks[j];
        var cards = attBlock.getElementsByClassName('step-card');
        var attVisibleCards = 0;

        for (var k = 0; k < cards.length; k++) {{
          var card = cards[k];
          var hasNote = card.getAttribute('data-has-note') === '1';
          var isMon = card.getAttribute('data-is-monitored') === '1';
          var noteVal = card.getAttribute('data-note-val') || '';

          var showCard = true;
          if (modeVal === 'only_with_note' && !hasNote) {{
            showCard = false;
          }} else if (modeVal === 'only_monitored' && !isMon) {{
            showCard = false;
          }}

          if (showCard && noteTagVal !== 'ALL') {{
            if (!hasNote || noteVal.indexOf(noteTagVal) === -1) {{
              showCard = false;
            }}
          }}

          card.style.display = showCard ? 'block' : 'none';
          if (showCard) {{
            attVisibleCards++;
            panelVisibleSteps++;
            totalVisibleSteps++;
          }}
        }}
        attBlock.style.display = (attVisibleCards > 0) ? 'block' : 'none';
      }}

      if (stuVal === 'ALL_STUDENTS') {{
        panel.style.display = (panelVisibleSteps > 0) ? 'block' : 'none';
      }} else {{
        panel.style.display = 'block';
      }}
      if (panelVisibleSteps > 0) totalVisibleStudents++;
    }}

    var infoEl = document.getElementById('filter_status_info');
    if (infoEl) {{
      infoEl.innerHTML = 'Menampilkan <strong>' + totalVisibleSteps + '</strong> kode/soal pada <strong>' + totalVisibleStudents + '</strong> mahasiswa.';
    }}
  }}

  function quickFilterNotesOnly() {{
    document.getElementById('stu_dropdown').value = 'ALL_STUDENTS';
    document.getElementById('filter_mode').value = 'only_with_note';
    document.getElementById('filter_note_tag').value = 'ALL';
    applyFilters();
  }}

  function resetAllFilters() {{
    var dd = document.getElementById('stu_dropdown');
    if (dd.options.length > 1) dd.selectedIndex = 1;
    document.getElementById('filter_mode').value = 'all';
    document.getElementById('filter_note_tag').value = 'ALL';
    applyFilters();
  }}
</script>
</head>
<body>
<div class="container">
  <div class="card">
    <h2 style="margin-top:0;">SANN-DKT: Deep Knowledge Tracing + Line-Level Code Attention (Minggu 1)</h2>
    <p style="margin-bottom:8px;">
      Implementasi sesuai paper <em>"Automated Identification of Logical Errors in Programs: Advancing Scalable Analysis of Student Misconceptions"</em> (Muntasir Hoq et al., EDM 2025):
      <strong>Tanpa batas max_depth</strong>, <strong>Tanpa normalisasi L2</strong>, urutan pengerjaan per <strong>Attempt (Attempt 1 Soal 1-selesai &rarr; Attempt 2 Soal 1-selesai)</strong>, dan seluruh mahasiswa di <code>Reviewed_Students_Status.csv</code> berada di <strong>Data Test</strong>.
    </p>
    <table class="metric-table">
      <thead>
        <tr>
          <th style="text-align:left;">Model</th>
          <th>Overall Test AUC (%)</th>
          <th>Accuracy (%)</th>
          <th>F1-Score (%)</th>
          <th>Easy AUC (%)</th>
          <th>Challenger AUC (%)</th>
          <th>Ultimate AUC (%)</th>
        </tr>
      </thead>
      <tbody>
        <tr>
          <td style="text-align:left;"><strong>Original DKT</strong> (Hanya One-Hot {{q_t, a_t}})</td>
          <td>{orig_eval['auc']:.2f}%</td>
          <td>{orig_eval['accuracy']:.2f}%</td>
          <td>{orig_eval['f1']:.2f}%</td>
          <td>{orig_eval['category_auc'].get('Easy', 0):.2f}%</td>
          <td>{orig_eval['category_auc'].get('Challenger', 0):.2f}%</td>
          <td>{orig_eval['category_auc'].get('Ultimate', 0):.2f}%</td>
        </tr>
        <tr style="background:#e8f5e9;font-weight:bold;">
          <td style="text-align:left;"><strong>SANN-DKT</strong> (One-Hot + Raw SANN Code Vector c_t)</td>
          <td>{sdkt_eval['auc']:.2f}%</td>
          <td>{sdkt_eval['accuracy']:.2f}%</td>
          <td>{sdkt_eval['f1']:.2f}%</td>
          <td>{sdkt_eval['category_auc'].get('Easy', 0):.2f}%</td>
          <td>{sdkt_eval['category_auc'].get('Challenger', 0):.2f}%</td>
          <td>{sdkt_eval['category_auc'].get('Ultimate', 0):.2f}%</td>
        </tr>
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2 style="margin-top:0;">Inference Trajectory per Attempt &amp; Visualisasi Bobot Attention per Baris Kode</h2>

    <div class="filter-box">
      <div class="filter-item" style="flex:2;min-width:280px;">
        <label for="stu_dropdown">1. Pilih Mahasiswa Data Test:</label>
        <select id="stu_dropdown" class="stu-select" onchange="applyFilters()">
          <option value="ALL_STUDENTS">🌟 Semua Mahasiswa Data Test Sekaligus</option>
          {''.join(student_options_html)}
        </select>
      </div>
      <div class="filter-item">
        <label for="filter_mode">2. Filter Seleksi Kode:</label>
        <select id="filter_mode" class="stu-select" onchange="applyFilters()">
          <option value="all">Tampilkan Semua Kode / Soal</option>
          <option value="only_with_note">Hanya Kode yang Ada Misconception Note</option>
          <option value="only_monitored">Hanya Kode di Monitored_Codes_Dipantau.csv</option>
        </select>
      </div>
      <div class="filter-item">
        <label for="filter_note_tag">3. Filter Kode Note Spesifik:</label>
        <select id="filter_note_tag" class="stu-select" onchange="applyFilters()">
          {''.join(note_tag_options)}
        </select>
      </div>
      <div style="display:flex;gap:8px;">
        <button type="button" class="btn-filter btn-note-only" onclick="quickFilterNotesOnly()">🔍 Hanya Kode Ada Misconception Note</button>
        <button type="button" class="btn-filter btn-reset" onclick="resetAllFilters()">🔄 Reset</button>
      </div>
    </div>
    <div id="filter_status_info" style="font-size:13px;color:#334155;margin-bottom:10px;"></div>

    {''.join(student_sections_html)}
  </div>
</div>
<script>
  // Pilih mahasiswa pertama secara default saat halaman dibuka
  var ddInit = document.getElementById('stu_dropdown');
  if (ddInit && ddInit.options.length > 1) ddInit.selectedIndex = 1;
  applyFilters();
</script>
</body>
</html>"""

    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)


def _print_step_code_and_line_weights(code_str, explain_dict):
    """Mencetak kode mahasiswa beserta bar dan bobot attention untuk setiap baris di terminal."""
    lines = code_str.splitlines()
    line_scores = explain_dict.get("line_scores", {}) if explain_dict else {}
    ranked_lines = explain_dict.get("ranked_lines", []) if explain_dict else []
    syn_err = explain_dict.get("syntax_error") if explain_dict else None
    max_score = max(line_scores.values()) if line_scores else 0.0

    print(f"    {'Line':>4} | {'Attention Bar':<12} | {'Bobot':>6} | Kode Pseudocode")
    print("    " + "-" * 76)
    for ln_idx, text in enumerate(lines, start=1):
        score = float(line_scores.get(ln_idx, 0.0))
        bar_len = int(round((score / max_score) * 10)) if max_score > 0 else 0
        bar = "#" * bar_len + "." * (10 - bar_len)

        marker = ""
        if syn_err and syn_err.get("line") == ln_idx:
            marker = f"  <<< [FOKUS #1 & {syn_err.get('message', 'SyntaxError')}]"
        elif ranked_lines and ln_idx == ranked_lines[0]:
            marker = "  <<< [FOKUS UTAMA #1]"
        elif len(ranked_lines) > 1 and ln_idx == ranked_lines[1]:
            marker = "  <<  [FOKUS #2]"
        elif len(ranked_lines) > 2 and ln_idx == ranked_lines[2]:
            marker = "  <   [FOKUS #3]"

        print(f"    {ln_idx:>4} | [{bar}] | {score*100:>5.1f}% | {text}{marker}")
    print("    " + "-" * 76)


def _note_matches_filter(note_val, only_with_note=False, note_tag=None):
    has_n = _has_valid_code_note(note_val)
    if only_with_note and not has_n:
        return False
    if note_tag:
        if not has_n:
            return False
        tags = [t.strip().upper() for t in str(note_val).split(",") if t.strip()]
        if str(note_tag).strip().upper() not in tags and str(note_tag).strip().upper() not in str(note_val).upper():
            return False
    return True


def predict_student_trajectory(
    target_nim, show_code_lines=True, only_with_note=False, note_tag=None
):
    """
    Melakukan inference Knowledge Tracing (SANN-DKT) berdasarkan urutan Attempt:
    Attempt 1 (Soal 1 -> selesai), diikuti Attempt 2 (Soal 1 -> selesai), dst.,
    serta menampilkan kode mahasiswa beserta bobot Attention untuk setiap barisnya.
    Jika only_with_note=True atau note_tag diberikan, hanya menampilkan step kode
    yang memiliki Misconception_Di_Dalam_Code_Notes.
    """
    meta_path = SAVED_DKT_DIR / "dkt_metadata.json"
    sann_w_path = SAVED_DKT_DIR / "modified_sann_weights.h5"
    dkt_w_path = SAVED_DKT_DIR / "sann_dkt_lstm_weights.h5"

    if not meta_path.exists() or not sann_w_path.exists() or not dkt_w_path.exists():
        raise FileNotFoundError(
            f"Model SANN-DKT belum ditemukan di {SAVED_DKT_DIR}. Jalankan sann_dkt.py terlebih dahulu."
        )

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    hp = meta["hyperparameters"]
    num_problems = meta["num_problems"]
    wordtoix = meta["wordtoix"]
    pathtoix = meta["pathtoix"]
    node_vocab_size = meta["modified_sann_metrics"]["node_vocab_size"]
    path_vocab_size = meta["modified_sann_metrics"]["path_vocab_size"]

    _, sann_encoder = create_modified_sann_model(
        node_vocab_size=node_vocab_size,
        path_vocab_size=path_vocab_size,
        max_subtree_length=hp["max_subtree_length"],
        embedding_size=hp["sann_embedding_size"],
        code_vector_size=hp["code_vector_size"],
        num_classes=2,
        entropy_lambda=hp["entropy_lambda"],
    )
    sann_encoder.load_weights(str(sann_w_path), by_name=True)

    sann_dkt_model = create_dkt_lstm_model(
        input_dim=2 * num_problems + hp["code_vector_size"],
        num_problems=num_problems,
        lstm_units=hp["lstm_units"],
        dropout_rate=hp["dropout_rate"],
        learning_rate=hp["dkt_learning_rate"],
    )
    sann_dkt_model.load_weights(str(dkt_w_path))

    events_df, code_to_subtrees, _, _, _ = load_week1_chronological_attempts(
        recursive_subtrees=hp["recursive_subtrees"], max_depth=hp["max_depth"]
    )

    stu_df = events_df[events_df["NIM"].astype(str) == str(target_nim).strip()]
    if stu_df.empty:
        raise ValueError(
            f"Mahasiswa dengan NIM {target_nim} tidak ditemukan di dataset Minggu 1."
        )

    target_key = stu_df["student_key"].iloc[0]
    stu_codes = stu_df["code"].unique().tolist()
    paths = [code_to_subtrees[c] for c in stu_codes]
    paths_str = [[" ".join(st) for st in p] for p in paths]

    _, X_n = encode_paths(paths, wordtoix, hp["max_subtree_length"])
    _, X_p = encode_paths_path(paths_str, pathtoix, hp["max_subtree_length"])
    vecs, attn, probs = sann_encoder.predict(
        [np.array(X_n, dtype=np.int32), np.array(X_p, dtype=np.int32)], verbose=0
    )

    # TANPA normalisasi L2
    code_to_vec = {c: vecs[i] for i, c in enumerate(stu_codes)}

    code_explain_map = compute_code_line_explainability_map(
        stu_codes,
        attn,
        probs,
        max_subtree_length=hp["max_subtree_length"],
        max_depth=hp["max_depth"],
        recursive=hp["recursive_subtrees"],
    )

    _, X_sdkt, Y_t, meta_steps = build_dkt_sequences_for_students(
        events_df, [target_key], code_to_vec, num_problems, hp["code_vector_size"]
    )
    eval_res = evaluate_dkt_predictions(
        sann_dkt_model, X_sdkt, Y_t, meta_steps, code_explain_map=code_explain_map
    )

    first = eval_res["records"][0]
    mon_total = sum(1 for r in eval_res["records"] if r.get("curr_is_monitored")) + (
        1 if eval_res["records"][-1].get("next_is_monitored") else 0
    )
    note_total = sum(
        1
        for r in eval_res["records"]
        if _has_valid_code_note(r.get("curr_monitored_code_note"))
    ) + (
        1
        if _has_valid_code_note(eval_res["records"][-1].get("next_monitored_code_note"))
        else 0
    )
    filter_banner = ""
    if only_with_note or note_tag:
        filter_banner = f" | FILTER AKTIF: Hanya Kode dengan Misconception Note{f' ({note_tag})' if note_tag else ''}"
    print("\n" + "#" * 98)
    print(
        f" INFERENCE SANN-DKT & LINE ATTENTION MAHASISWA: {first['Student_Name']} ({first['NIM']}) | Kelas: {first['Class_Name']}{filter_banner}"
    )
    print(
        f" Semua Misconception Minggu Ini : {first.get('student_all_misconceptions', '-')}"
    )
    print(
        f" Jumlah Kode Dipantau : {mon_total} kode | Kode Memiliki Misconception Note : {note_total} kode"
    )
    print("#" * 98)

    curr_session = None
    shown_count = 0
    for r in eval_res["records"]:
        if not _note_matches_filter(
            r.get("curr_monitored_code_note"),
            only_with_note=only_with_note,
            note_tag=note_tag,
        ):
            continue
        shown_count += 1
        if r["curr_attempt_session"] != curr_session:
            curr_session = r["curr_attempt_session"]
            print(
                f"\n==================== [SESI ATTEMPT #{curr_session}: {r['curr_quiz_name']} — ATTEMPT KE-{r['curr_attempt_no']} (ID: {r['curr_attempt_id']})] ===================="
            )

        status_curr = "CORRECT (1)" if r["curr_correctness"] == 1 else "INCORRECT (0)"
        status_next = "CORRECT (1)" if r["next_correctness"] == 1 else "INCORRECT (0)"
        mon_tag = (
            f" *** [DIPANTAU di Monitored_Codes_Dipantau.csv #{r['curr_monitored_row_no']}: {r['curr_monitored_category']}] ***"
            if r.get("curr_is_monitored")
            else ""
        )
        print(
            f"\n[Step t={r['timestep']}] Attempt {r['curr_attempt_no']} (Soal {r['curr_q_order']}/{r['curr_q_total']}): {r['curr_problem_name']} [{r.get('curr_submit_stage', '')}]{mon_tag}"
        )
        print(
            f"  -> Waktu Submit  : {r['curr_submit_time']} | Hasil Saat Ini (a_t): {status_curr}"
        )
        if r.get("curr_is_monitored"):
            print(
                f"  -> Info Pantauan : Kategori = {r.get('curr_monitored_category')} | Sumber = {r.get('curr_monitored_source')} | Status Asli = {r.get('curr_monitored_status_asli')} | Reviewer = {r.get('curr_monitored_reviewer', '-')}"
            )
            print(
                f"  -> Note Misconception di Dalam Code : {r.get('curr_monitored_code_note', '-')}"
            )
        print(
            f"  -> Misconception : {r.get('curr_misconception', '-')}"
        )
        print(
            f"  -> Prediksi SANN-DKT untuk Soal Berikutnya (t={r['timestep']+1} | Attempt {r['next_attempt_no']} Soal {r['next_q_order']}/{r['next_q_total']} - {r['next_problem_name']} [{r.get('next_submit_stage', '')}]):"
        )
        print(
            f"     Probabilitas Benar P(a_{{t+1}}=1) = {r['pred_prob']*100:.1f}% | Hasil Aktual (a_{{t+1}}) = {status_next}"
        )
        if show_code_lines:
            _print_step_code_and_line_weights(r["curr_code"], r.get("curr_explain", {}))

    # Tampilkan juga kode pada soal paling akhir (Step T) bila lolos filter
    last_r = eval_res["records"][-1]
    if _note_matches_filter(
        last_r.get("next_monitored_code_note"),
        only_with_note=only_with_note,
        note_tag=note_tag,
    ):
        shown_count += 1
        if last_r["next_attempt_session"] != curr_session:
            print(
                f"\n==================== [SESI ATTEMPT #{last_r['next_attempt_session']}: {last_r['next_quiz_name']} — ATTEMPT KE-{last_r['next_attempt_no']} (ID: {last_r['next_attempt_id']})] ===================="
            )
        status_last = "CORRECT (1)" if last_r["next_correctness"] == 1 else "INCORRECT (0)"
        mon_last_tag = (
            f" *** [DIPANTAU di Monitored_Codes_Dipantau.csv #{last_r['next_monitored_row_no']}: {last_r['next_monitored_category']}] ***"
            if last_r.get("next_is_monitored")
            else ""
        )
        print(
            f"\n[Step Terakhir t={last_r['timestep']+1}] Attempt {last_r['next_attempt_no']} (Soal {last_r['next_q_order']}/{last_r['next_q_total']}): {last_r['next_problem_name']} [{last_r.get('next_submit_stage', '')}]{mon_last_tag}"
        )
        print(
            f"  -> Waktu Submit  : {last_r['next_submit_time']} | Hasil Aktual (a_T): {status_last}"
        )
        if last_r.get("next_is_monitored"):
            print(
                f"  -> Info Pantauan : Kategori = {last_r.get('next_monitored_category')} | Sumber = {last_r.get('next_monitored_source')} | Status Asli = {last_r.get('next_monitored_status_asli')} | Reviewer = {last_r.get('next_monitored_reviewer', '-')}"
            )
            print(
                f"  -> Note Misconception di Dalam Code : {last_r.get('next_monitored_code_note', '-')}"
            )
        print(
            f"  -> Misconception : {last_r.get('next_misconception', '-')}"
        )
        if show_code_lines:
            _print_step_code_and_line_weights(
                last_r["next_code"], last_r.get("next_explain", {})
            )

    if (only_with_note or note_tag) and shown_count == 0:
        print(
            f"\n  [INFO] Tidak ada kode untuk mahasiswa {target_nim} yang sesuai dengan filter Misconception Note."
        )

    print("\n" + "=" * 98)
    print(
        f" Ditampilkan: {shown_count} kode | Total Akurasi Prediksi Soal Berikutnya (SANN-DKT) pada Mahasiswa Ini: {eval_res['accuracy']:.2f}%"
    )
    print("=" * 98 + "\n")
    return eval_res


def refresh_saved_dkt_report():
    """
    Memperbarui sann_dkt_report.html, test_student_trajectories.csv, dan
    monitored_with_notes_trajectories.csv menggunakan model yang sudah tersimpan
    di saved_model_dkt/ (tanpa melatih ulang dari awal).
    """
    meta_path = SAVED_DKT_DIR / "dkt_metadata.json"
    sann_w_path = SAVED_DKT_DIR / "modified_sann_weights.h5"
    dkt_w_path = SAVED_DKT_DIR / "sann_dkt_lstm_weights.h5"
    orig_w_path = SAVED_DKT_DIR / "original_dkt_lstm_weights.h5"

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    hp = meta["hyperparameters"]
    num_problems = meta["num_problems"]
    wordtoix = meta["wordtoix"]
    pathtoix = meta["pathtoix"]
    test_students = meta["test_student_keys"]
    reviewed_nims = load_reviewed_student_nims()
    node_vocab_size = meta["modified_sann_metrics"]["node_vocab_size"]
    path_vocab_size = meta["modified_sann_metrics"]["path_vocab_size"]

    _, sann_encoder = create_modified_sann_model(
        node_vocab_size=node_vocab_size,
        path_vocab_size=path_vocab_size,
        max_subtree_length=hp["max_subtree_length"],
        embedding_size=hp["sann_embedding_size"],
        code_vector_size=hp["code_vector_size"],
        num_classes=2,
        entropy_lambda=hp["entropy_lambda"],
    )
    sann_encoder.load_weights(str(sann_w_path), by_name=True)

    sann_dkt_model = create_dkt_lstm_model(
        input_dim=2 * num_problems + hp["code_vector_size"],
        num_problems=num_problems,
        lstm_units=hp["lstm_units"],
        dropout_rate=hp["dropout_rate"],
        learning_rate=hp["dkt_learning_rate"],
    )
    sann_dkt_model.load_weights(str(dkt_w_path))

    orig_dkt_model = create_dkt_lstm_model(
        input_dim=2 * num_problems,
        num_problems=num_problems,
        lstm_units=hp["lstm_units"],
        dropout_rate=hp["dropout_rate"],
        learning_rate=hp["dkt_learning_rate"],
    )
    orig_dkt_model.load_weights(str(orig_w_path))

    events_df, code_to_subtrees, _, _, _ = load_week1_chronological_attempts(
        recursive_subtrees=hp["recursive_subtrees"], max_depth=hp["max_depth"]
    )
    test_events = events_df[events_df["student_key"].isin(test_students)]
    test_codes = test_events["code"].unique().tolist()
    paths = [code_to_subtrees[c] for c in test_codes]
    paths_str = [[" ".join(st) for st in p] for p in paths]

    _, X_n = encode_paths(paths, wordtoix, hp["max_subtree_length"])
    _, X_p = encode_paths_path(paths_str, pathtoix, hp["max_subtree_length"])
    vecs, attn, probs = sann_encoder.predict(
        [np.array(X_n, dtype=np.int32), np.array(X_p, dtype=np.int32)],
        batch_size=64,
        verbose=0,
    )
    code_to_vec = {c: vecs[i] for i, c in enumerate(test_codes)}
    code_explain_map = compute_code_line_explainability_map(
        test_codes,
        attn,
        probs,
        max_subtree_length=hp["max_subtree_length"],
        max_depth=hp["max_depth"],
        recursive=hp["recursive_subtrees"],
    )

    X_dkt_te, X_sdkt_te, Y_te, meta_te = build_dkt_sequences_for_students(
        events_df, test_students, code_to_vec, num_problems, hp["code_vector_size"]
    )
    orig_eval_test = evaluate_dkt_predictions(orig_dkt_model, X_dkt_te, Y_te, meta_te)
    sdkt_eval_test = evaluate_dkt_predictions(
        sann_dkt_model, X_sdkt_te, Y_te, meta_te, code_explain_map=code_explain_map
    )

    csv_records = [
        {k: v for k, v in r.items() if k not in ("curr_explain", "next_explain")}
        for r in sdkt_eval_test["records"]
    ]
    traj_csv_path = SAVED_DKT_DIR / "test_student_trajectories.csv"
    df_traj = pd.DataFrame(csv_records)
    df_traj.to_csv(traj_csv_path, index=False, encoding="utf-8-sig")

    notes_only_records = [
        r
        for r in csv_records
        if _has_valid_code_note(r.get("curr_monitored_code_note"))
        or _has_valid_code_note(r.get("next_monitored_code_note"))
    ]
    notes_csv_path = SAVED_DKT_DIR / "monitored_with_notes_trajectories.csv"
    pd.DataFrame(notes_only_records).to_csv(
        notes_csv_path, index=False, encoding="utf-8-sig"
    )

    _save_dkt_html_report(
        meta, sdkt_eval_test, orig_eval_test, reviewed_nims, DKT_HTML_REPORT
    )
    print(f"Laporan HTML & CSV berhasil diperbarui: {DKT_HTML_REPORT}")
    print(
        f"CSV khusus kode dengan Misconception Note tersimpan di: {notes_csv_path} ({len(notes_only_records)} baris transisi)"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Latih dan evaluasi SANN + LSTM (SANN-DKT) berdasarkan paper EDM 2025."
    )
    parser.add_argument(
        "--predict-nim",
        type=str,
        default=None,
        help="Jalankan inference trajectory & line attention langsung untuk NIM mahasiswa tertentu",
    )
    parser.add_argument(
        "--only-with-note",
        action="store_true",
        help="Filter hanya menampilkan kode yang memiliki Misconception_Di_Dalam_Code_Notes",
    )
    parser.add_argument(
        "--note-tag",
        type=str,
        default=None,
        help="Filter hanya menampilkan kode dengan kode Misconception Note tertentu (misal: OP-01, VA-12, IO-05)",
    )
    parser.add_argument(
        "--refresh-report",
        action="store_true",
        help="Perbarui laporan HTML & CSV dari bobot model yang sudah tersimpan tanpa melatih ulang",
    )
    parser.add_argument("--max-subtree-length", type=int, default=64)
    parser.add_argument("--embedding-size", type=int, default=64)
    parser.add_argument("--code-vector-size", type=int, default=128)
    parser.add_argument("--sann-epochs", type=int, default=40)
    parser.add_argument("--dkt-epochs", type=int, default=60)
    parser.add_argument("--lstm-units", type=int, default=200)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.005)
    args = parser.parse_args()

    if args.predict_nim:
        predict_student_trajectory(
            args.predict_nim,
            only_with_note=args.only_with_note,
            note_tag=args.note_tag,
        )
    elif args.refresh_report:
        refresh_saved_dkt_report()
    elif args.only_with_note or args.note_tag:
        # Jika dijalankan tanpa --predict-nim, tampilkan seluruh mahasiswa reviewed yang memiliki Misconception Note
        for nim in sorted(load_reviewed_student_nims()):
            predict_student_trajectory(
                nim,
                only_with_note=True,
                note_tag=args.note_tag,
            )
    else:
        run_full_sann_dkt_pipeline(
            max_depth=None,
            max_subtree_length=args.max_subtree_length,
            embedding_size=args.embedding_size,
            code_vector_size=args.code_vector_size,
            sann_epochs=args.sann_epochs,
            dkt_epochs=args.dkt_epochs,
            lstm_units=args.lstm_units,
            dropout_rate=args.dropout,
            dkt_lr=args.lr,
        )
