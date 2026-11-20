import os
import re
import sys
import glob
import json
import html
import copy
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from parserDAP import (
    _get_ast_disk_cache,
    program_parser,
    normalize_dap_ast,
    repair_dap_code,
    build_fallback_ast,
)

RAW_DIR = PROJECT_ROOT.parent / "Analyze data" / "Scrapping-20261109T100658Z-1-001" / "Scrapping"
OUT_CSV = PROJECT_ROOT / "Week_01-04_All_Submits.csv"
AST_CACHE_W1_W4 = PROJECT_ROOT / "saved_model_dap" / "ast_cache_w1_w4_dap_latest.json"

TARGET_WEEKS = ("Week_01-02", "Week_03", "Week_04")
KC_PREFIXES = ["CO", "VA", "OP", "EX", "IO", "CD", "LO", "SQ"]

COURSE_MAP_FALLBACK = {
    588:  {"prodi": "PRODI S1 INFORMATIKA (FIF)", "shortname": "CAK1BAB3-BS1IF-50-REG01", "fullname": "ALGORITMA DAN PEMROGRAMAN 1 BS1IF-50-REG01 [PEY]", "class_name": "BS1IF-50-REG01", "lecturer": "PEY"},
    589:  {"prodi": "PRODI S1 INFORMATIKA (FIF)", "shortname": "CAK1BAB3-BS1IF-50-REG02", "fullname": "ALGORITMA DAN PEMROGRAMAN 1 BS1IF-50-REG02 [MDS]", "class_name": "BS1IF-50-REG02", "lecturer": "MDS"},
    591:  {"prodi": "PRODI S1 INFORMATIKA (FIF)", "shortname": "CAK1BAB3-BS1IF-50-REG04", "fullname": "ALGORITMA DAN PEMROGRAMAN 1 BS1IF-50-REG04 [UAI]", "class_name": "BS1IF-50-REG04", "lecturer": "UAI"},
    592:  {"prodi": "PRODI S1 INFORMATIKA (FIF)", "shortname": "CAK1BAB3-BS1IF-50-REG03", "fullname": "ALGORITMA DAN PEMROGRAMAN 1 BS1IF-50-REG03 [IGR]", "class_name": "BS1IF-50-REG03", "lecturer": "IGR"},
    593:  {"prodi": "PRODI S1 INFORMATIKA (FIF)", "shortname": "CAK1BAB3-BS1IF-50-REG06", "fullname": "ALGORITMA DAN PEMROGRAMAN 1 BS1IF-50-REG06 [PPV]", "class_name": "BS1IF-50-REG06", "lecturer": "PPV"},
    594:  {"prodi": "PRODI S1 INFORMATIKA (FIF)", "shortname": "CAK1BAB3-BS1IF-50-REG05", "fullname": "ALGORITMA DAN PEMROGRAMAN 1 BS1IF-50-REG05 [UAI]", "class_name": "BS1IF-50-REG05", "lecturer": "UAI"},
    595:  {"prodi": "PRODI S1 INFORMATIKA (FIF)", "shortname": "CAK1BAB3-BS1IF-50-REG07", "fullname": "ALGORITMA DAN PEMROGRAMAN 1 BS1IF-50-REG07 [PPV]", "class_name": "BS1IF-50-REG07", "lecturer": "PPV"},
    597:  {"prodi": "PRODI S1 INFORMATIKA (FIF)", "shortname": "CAK1BAB3-BS1IF-50-REG09", "fullname": "ALGORITMA DAN PEMROGRAMAN 1 BS1IF-50-REG09 [LDS]", "class_name": "BS1IF-50-REG09", "lecturer": "LDS"},
    598:  {"prodi": "PRODI S1 INFORMATIKA (FIF)", "shortname": "CAK1BAB3-BS1IF-50-REG08", "fullname": "ALGORITMA DAN PEMROGRAMAN 1 BS1IF-50-REG08 [FFS]", "class_name": "BS1IF-50-REG08", "lecturer": "FFS"},
    599:  {"prodi": "PRODI S1 INFORMATIKA (FIF)", "shortname": "CAK1BAB3-BS1IF-50-REG10", "fullname": "ALGORITMA DAN PEMROGRAMAN 1 BS1IF-50-REG10 [LDS]", "class_name": "BS1IF-50-REG10", "lecturer": "LDS"},
    601:  {"prodi": "PRODI S1 INFORMATIKA (FIF)", "shortname": "CAK1BAB3-BS1IF-50-REG12", "fullname": "ALGORITMA DAN PEMROGRAMAN 1 BS1IF-50-REG12 [UFP]", "class_name": "BS1IF-50-REG12", "lecturer": "UFP"},
    602:  {"prodi": "PRODI S1 INFORMATIKA (FIF)", "shortname": "CAK1BAB3-BS1IF-50-REG11", "fullname": "ALGORITMA DAN PEMROGRAMAN 1 BS1IF-50-REG11 [BMG]", "class_name": "BS1IF-50-REG11", "lecturer": "BMG"},
    603:  {"prodi": "PRODI S1 INFORMATIKA (FIF)", "shortname": "CAK1BAB3-BS1IF-50-REG13", "fullname": "ALGORITMA DAN PEMROGRAMAN 1 BS1IF-50-REG13 [UFP]", "class_name": "BS1IF-50-REG13", "lecturer": "UFP"},
    5373: {"prodi": "S1 Data Sains", "shortname": "CDK1CAB3-BS1DS-50-REG-01", "fullname": "PENGENALAN PEMROGRAMAN BS1DS-50-REG-01 [UAI]", "class_name": "BS1DS-50-REG-01", "lecturer": "UAI"},
    5374: {"prodi": "S1 Data Sains", "shortname": "CDK1CAB3-BS1DS-50-REG-02", "fullname": "PENGENALAN PEMROGRAMAN BS1DS-50-REG-02 [GHK]", "class_name": "BS1DS-50-REG-02", "lecturer": "GHK"},
    5375: {"prodi": "S1 Data Sains", "shortname": "CDK1CAB3-BS1DS-50-REG-03", "fullname": "PENGENALAN PEMROGRAMAN BS1DS-50-REG-03 [IGR]", "class_name": "BS1DS-50-REG-03", "lecturer": "IGR"},
    5376: {"prodi": "S1 Data Sains", "shortname": "CDK1CAB3-BS1DS-50-REG-04", "fullname": "PENGENALAN PEMROGRAMAN BS1DS-50-REG-04 [THF]", "class_name": "BS1DS-50-REG-04", "lecturer": "THF"},
    6515: {"prodi": "S1 Sains Data - Kampus Purwokerto", "shortname": "CDK1CAB3-PS1DS-07-REG-01", "fullname": "PENGENALAN PEMROGRAMAN PS1DS-07-REG-01 [AGW]", "class_name": "PS1DS-07-REG-01", "lecturer": "AGW"},
}


def get_course_metadata(jsonl_path: str):
    m = re.search(r"course(\d+)", jsonl_path, re.IGNORECASE)
    if m:
        cid = int(m.group(1))
        if cid in COURSE_MAP_FALLBACK:
            return cid, COURSE_MAP_FALLBACK[cid]
    if "BS1IF-50-REG02" in jsonl_path:
        return 589, COURSE_MAP_FALLBACK[589]
    raise ValueError(f"Tidak dapat menentukan kelas untuk file: {jsonl_path}")


def clean_text(text, multiline=True, preserve_indent=False):
    if text is None:
        return ""
    s = str(text)
    s = html.unescape(s)
    s = s.replace("\xa0", " ").replace("\r\n", "\n").replace("\r", "\n")
    if not preserve_indent:
        s = re.sub(r"<br\s*/?>", "\n", s, flags=re.IGNORECASE)
        s = re.sub(r"</p>", "\n", s, flags=re.IGNORECASE)
        s = re.sub(r"<[^>]+>", "", s)
    if multiline:
        if preserve_indent:
            lines = [line.rstrip() for line in s.split("\n")]
            return "\n".join(lines).strip()
        lines = [re.sub(r"[ \t]+", " ", line).strip() for line in s.split("\n")]
        s = "\n".join(lines)
        s = re.sub(r"\n{3,}", "\n\n", s).strip()
    else:
        s = re.sub(r"\s+", " ", s).strip()
    return s


def parse_datetime(dt_str):
    if not dt_str or str(dt_str).strip() in ("-", "Never", ""):
        return ""
    s = clean_text(dt_str, multiline=False)
    s = re.sub(r"^[A-Za-z]+,\s*", "", s)
    s = s.replace(",", "")
    s = re.sub(r"\s+", " ", s).strip()
    for fmt in ("%d %B %Y %I:%M %p", "%d %B %Y %H:%M", "%d/%m/%y %H:%M:%S"):
        try:
            return datetime.strptime(s, fmt).strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            continue
    return s


def parse_float(val):
    if val is None or str(val).strip() in ("", "-", "Not yet graded"):
        return np.nan
    m = re.search(r"[-+]?\d*\.?\d+", str(val).replace(",", "."))
    return float(m.group(0)) if m else np.nan


def parse_marks(marks_str):
    if not marks_str:
        return np.nan, np.nan
    m = re.search(r"([\d.]+)\s*/\s*([\d.]+)", str(marks_str))
    if m:
        return float(m.group(1)), float(m.group(2))
    return np.nan, np.nan


def extract_week(quiz_dict, file_path=""):
    qname = str(quiz_dict.get("name") or "")
    sec = str(quiz_dict.get("section_name") or "")
    label = str(quiz_dict.get("label") or "")
    combined = f"{qname} | {sec} | {label} | {os.path.basename(file_path)}".lower()

    if any(k in combined for k in ["week 1-2", "pekan 02", "tugas 01", "tugas_01", "tugas 02", "tugas_02", "kelompok 2"]):
        return "Week_01-02"
    if any(k in combined for k in ["week 3", "pekan 03", "tugas 03", "tugas_03"]):
        return "Week_03"
    if any(k in combined for k in ["week 4", "pekan 04", "tugas 04", "tugas_04"]):
        return "Week_04"
    if any(k in combined for k in ["week 5", "pekan 05", "tugas 05", "tugas_05"]):
        return "Week_05-06"
    return "Week_Unknown"


def extract_quiz_category(quiz_dict, questions_list):
    qname = str(quiz_dict.get("name") or "").lower()
    if "misconception" in qname:
        return "Misconception_Probing"
    if "problem solving" in qname or "python code" in qname:
        return "Problem_Solving"
    qtypes = {q.get("qtype") for q in (questions_list or [])}
    if "coderunner" in qtypes:
        return "Problem_Solving"
    return "Misconception_Probing"


def extract_kc_and_tags(q_dict):
    raw_tags = list(q_dict.get("tags") or [])
    meta_tags = list((q_dict.get("meta") or {}).get("tags") or [])
    qname = str(q_dict.get("question_name") or (q_dict.get("meta") or {}).get("name") or "")
    cleaned_tags = []
    for t in raw_tags + meta_tags:
        t_clean = str(t).strip()
        if not t_clean:
            continue
        t_up = t_clean.upper()
        if t_up == "I0-03":
            t_clean = "IO-03"
        elif t_clean.lower() in ("easy", "basic", "intermediate", "challenger", "ultimate"):
            t_clean = t_clean.capitalize()
        elif t_clean.lower() == "chall":
            t_clean = "Challenger"
        elif t_clean.lower() == "ult":
            t_clean = "Ultimate"
        if t_clean not in cleaned_tags:
            cleaned_tags.append(t_clean)
    combined_for_kc = " ".join(cleaned_tags + [qname]).upper()
    kc_flags = {}
    for kc in KC_PREFIXES:
        pattern = rf"\b{kc}(?:-\d+|\b)"
        kc_flags[f"KC_{kc}"] = 1 if re.search(pattern, combined_for_kc) else 0
    return ", ".join(cleaned_tags), kc_flags


def _norm_code_cmp(code_str):
    return "\n".join(line.rstrip() for line in str(code_str or "").replace("\r\n", "\n").split("\n")).strip()


def extract_all_submits_csv():
    t0 = time.time()
    jsonl_files = sorted(glob.glob(str(RAW_DIR / "**" / "*.jsonl"), recursive=True))
    print(f"Membaca {len(jsonl_files)} file JSONL dari {RAW_DIR}...")

    raw_by_attempt = {}
    for fpath in jsonl_files:
        cid, cmeta = get_course_metadata(fpath)
        rel_path = os.path.relpath(fpath, str(RAW_DIR))
        with open(fpath, "r", encoding="utf-8") as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                aid = str((rec.get("row") or {}).get("attempt_id", "")).strip()
                raw_by_attempt[aid] = (cid, cmeta, rel_path, rec)

    print(f"Total attempt unik: {len(raw_by_attempt)}")

    rows = []
    for aid_str, (cid, cmeta, rel_path, rec) in raw_by_attempt.items():
        row = rec.get("row") or {}
        if str(row.get("status", "")).strip().lower() != "finished":
            continue

        quiz = rec.get("quiz") or {}
        questions = rec.get("questions") or []
        week = extract_week(quiz, rel_path)
        if week not in TARGET_WEEKS:
            continue

        category = extract_quiz_category(quiz, questions)
        if category != "Problem_Solving":
            continue

        attempt_id = int(aid_str) if aid_str.isdigit() else aid_str
        user_id = row.get("user_id", "")
        nim = str(row.get("id_number") or row.get("nim") or row.get("username") or "").strip()
        student_name = clean_text(row.get("name") or f"{row.get('firstname', '')} {row.get('lastname', '')}", multiline=False)
        email = clean_text(row.get("email"), multiline=False)
        attempt_no = row.get("attempt_number")
        try:
            attempt_no = int(attempt_no)
        except Exception:
            attempt_no = 1

        summary = rec.get("summary") or {}
        s_fields = summary.get("fields") or {}
        started_time = parse_datetime(s_fields.get("Started") or row.get("started"))
        completed_time = parse_datetime(s_fields.get("Completed") or row.get("completed"))
        quiz_name = clean_text(quiz.get("name"), multiline=False)
        quiz_cmid = quiz.get("cmid", "")

        for q in questions:
            if q.get("qtype") != "coderunner":
                continue

            qname = clean_text(q.get("question_name") or (q.get("meta") or {}).get("name"), multiline=False)
            q_tags_str, kc_flags = extract_kc_and_tags(q)
            slot_no = q.get("slot")
            q_no = q.get("qno")
            q_id = q.get("question_id")
            final_code = clean_text(q.get("code"), multiline=True, preserve_indent=True)
            final_state = clean_text(q.get("state"), multiline=False)
            final_state_cls = clean_text(q.get("state_class"), multiline=False)
            mark_val, max_mark_val = parse_marks(q.get("grade_text") or q.get("marks"))
            if np.isnan(mark_val):
                mark_val = parse_float(q.get("mark"))
            if np.isnan(max_mark_val):
                max_mark_val = parse_float(q.get("max_mark"))
            final_is_correct = 1 if (
                final_state_cls.lower() == "correct"
                or final_state.lower() == "correct"
                or (not np.isnan(mark_val) and not np.isnan(max_mark_val) and max_mark_val > 0 and mark_val >= max_mark_val)
            ) else 0

            history_list = q.get("history") or []
            # Ambil SELURUH langkah submit ('submit') yang berisi kode non-kosong
            submit_steps = []
            for h in history_list:
                at = str(h.get("action_type", "")).strip().lower()
                code_h = clean_text(h.get("response_in_step"), multiline=True, preserve_indent=True)
                if at == "submit" and code_h:
                    st_h = clean_text(h.get("state"), multiline=False)
                    mk_h = parse_float(h.get("marks"))
                    tm_h = parse_datetime(h.get("time"))
                    corr_h = 1 if (
                        st_h.lower() == "correct"
                        or (not np.isnan(mk_h) and not np.isnan(max_mark_val) and max_mark_val > 0 and mk_h >= max_mark_val)
                    ) else 0
                    submit_steps.append({
                        "code": code_h,
                        "state": st_h,
                        "marks": mk_h,
                        "time": tm_h,
                        "is_correct": corr_h,
                        "action_type": at,
                    })

            # Jika ada final_code yang belum tercatat sebagai step terakhir (misal hanya 'saved' atau diubah sebelum finish)
            if final_code:
                if not submit_steps or _norm_code_cmp(submit_steps[-1]["code"]) != _norm_code_cmp(final_code):
                    submit_steps.append({
                        "code": final_code,
                        "state": final_state,
                        "marks": mark_val,
                        "time": completed_time or started_time,
                        "is_correct": final_is_correct,
                        "action_type": "final_finish",
                    })

            total_subs = len(submit_steps)
            for idx_s, st in enumerate(submit_steps, start=1):
                is_first_overall = (idx_s == 1 and attempt_no == 1)
                is_last_in_q = (idx_s == total_subs)
                if is_first_overall and is_last_in_q:
                    stage = "First Submit First Attempt"
                elif is_first_overall:
                    stage = f"Submit #{idx_s}/{total_subs} (First Submit First Attempt)"
                elif is_last_in_q:
                    stage = f"Submit #{idx_s}/{total_subs} (Final Submit)"
                else:
                    stage = f"Submit #{idx_s}/{total_subs}"

                rows.append({
                    "Course_ID": cid,
                    "Class_Name": cmeta["class_name"],
                    "Week": week,
                    "Quiz_CMID": quiz_cmid,
                    "Quiz_Name": quiz_name,
                    "Attempt_ID": attempt_id,
                    "Attempt_Number": attempt_no,
                    "User_ID": user_id,
                    "NIM": nim,
                    "Student_Name": student_name,
                    "Email": email,
                    "Started_Time": started_time,
                    "Completed_Time": completed_time,
                    "Slot_Number": slot_no,
                    "Question_No": q_no,
                    "Question_ID": q_id,
                    "Question_Name": qname,
                    "Question_Tags": q_tags_str,
                    "Submit_Index": idx_s,
                    "Total_Submits_In_Question": total_subs,
                    "Submit_Stage": stage,
                    "Action_Type": st["action_type"],
                    "Submit_Time": st["time"] or started_time or completed_time,
                    "Submit_State": st["state"],
                    "Submit_Marks": st["marks"],
                    "Max_Mark": max_mark_val,
                    "Is_Correct": st["is_correct"],
                    "Student_Code": st["code"],
                    **kc_flags,
                })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    elapsed = time.time() - t0
    print(f"\nSelesai mengekstrak seluruh submit Minggu 1-4 dalam {elapsed:.1f} detik!")
    print(f"File tersimpan: {OUT_CSV}")
    print(f"Total baris submit : {len(df)}")
    print(f"Per Week           : {df['Week'].value_counts().to_dict()}")
    print(f"Total siswa unik   : {df['NIM'].nunique()}")
    print(f"Total kode unik    : {df['Student_Code'].nunique()}")
    print(f"Total soal unik    : {df['Question_Name'].nunique()}")
    seq_lens = df.groupby(["Class_Name", "NIM"]).size()
    print(f"Panjang sekuens per siswa -> min={seq_lens.min()}, median={int(seq_lens.median())}, p95={int(seq_lens.quantile(0.95))}, max={seq_lens.max()}")
    return df


def build_parallel_ast_cache(df: pd.DataFrame, max_workers: int = 16):
    t0 = time.time()
    old_cache = _get_ast_disk_cache()
    new_cache = {}
    if AST_CACHE_W1_W4.exists():
        try:
            with open(AST_CACHE_W1_W4, "r", encoding="utf-8") as f:
                new_cache = json.load(f)
            print(f"Memuat cache W1-W4 existing: {len(new_cache)} entri")
        except Exception:
            new_cache = {}

    unique_codes = df["Student_Code"].dropna().unique().tolist()
    to_parse = []
    reused_old = 0
    for c in unique_codes:
        if c in new_cache:
            continue
        if c in old_cache and isinstance(old_cache[c], dict):
            new_cache[c] = normalize_dap_ast(old_cache[c])
            reused_old += 1
        else:
            to_parse.append(c)

    print(
        f"Total kode unik W1-W4: {len(unique_codes)} | Sudah di cache baru: {len(new_cache) - reused_old} | "
        f"Normalisasi dari cache lama: {reused_old} | Perlu compile DAP baru: {len(to_parse)}"
    )

    def _compile_one(code_str):
        try:
            return code_str, program_parser(code_str, normalize=True), "ok"
        except Exception:
            try:
                return code_str, program_parser(repair_dap_code(code_str), normalize=True), "repaired"
            except Exception:
                return code_str, normalize_dap_ast(build_fallback_ast(code_str)), "fallback"

    if to_parse:
        counts = {"ok": 0, "repaired": 0, "fallback": 0}
        done = 0
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_compile_one, c) for c in to_parse]
            for fut in as_completed(futures):
                c_str, ast_obj, status = fut.result()
                new_cache[c_str] = ast_obj
                counts[status] += 1
                done += 1
                if done % 5000 == 0 or done == len(to_parse):
                    print(
                        f"  Progress compile DAP terbaru: {done}/{len(to_parse)} "
                        f"({done*100/len(to_parse):.1f}%) | ok={counts['ok']}, repaired={counts['repaired']}, fallback={counts['fallback']} | {time.time()-t0:.1f}s"
                    )

    AST_CACHE_W1_W4.parent.mkdir(parents=True, exist_ok=True)
    with open(AST_CACHE_W1_W4, "w", encoding="utf-8") as f:
        json.dump(new_cache, f, ensure_ascii=False)
    print(f"Cache AST W1-W4 (DAP terbaru) tersimpan di {AST_CACHE_W1_W4} ({len(new_cache)} entri) dalam {time.time()-t0:.1f}s.")


if __name__ == "__main__":
    df_all = extract_all_submits_csv()
    build_parallel_ast_cache(df_all, max_workers=16)
