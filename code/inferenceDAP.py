import os
import re
import json
import argparse
import functools
from pathlib import Path
import numpy as np
import tensorflow as tf
from keras.models import Model
from anytree import Node, PreOrderIter, findall_by_attr

from config import Config
from parserDAP import program_parser, get_token, get_children, repair_dap_code, build_fallback_ast
from modelDAP import create_model, encode_paths, encode_paths_path


DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "saved_model_dap"


@functools.lru_cache(maxsize=2)
def load_saved_sann_model(model_dir=None):
    """Memuat model SANN DAP beserta metadata vocabulary yang telah disimpan."""
    if model_dir is None:
        model_dir = DEFAULT_MODEL_DIR
    else:
        model_dir = Path(model_dir)

    metadata_path = model_dir / "metadata.json"
    weights_path = model_dir / "sann_dap_weights.h5"

    if not metadata_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"Model belum ditemukan di {model_dir}.\n"
            f"Pastikan sudah menjalankan training (train_test_evaluate di modelDAP.py) "
            f"agar file 'sann_dap_weights.h5' dan 'metadata.json' tersimpan."
        )

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    max_subtree_length = metadata["max_subtree_length"]
    node_vocab_size = metadata["node_vocab_size"]
    path_vocab_size = metadata["path_vocab_size"]
    embedding_size = metadata.get("embedding_size", 256)

    dummy_weights = np.zeros((node_vocab_size + 2, embedding_size))
    model = create_model(
        dummy_weights, node_vocab_size, path_vocab_size, max_subtree_length
    )
    model.load_weights(str(weights_path))

    attention_layer = model.get_layer("attention_weights")
    inference_model = Model(
        inputs=model.input,
        outputs=[model.output, attention_layer.output],
    )

    return inference_model, metadata


def _collect_statement_keywords(ast_node):
    """Mengambil token-token kunci dari sebuah node statement AST untuk pencocokan baris."""
    if not isinstance(ast_node, dict):
        return []
    node_type = ast_node.get("type", "")
    tokens = []

    if node_type in ("VarDeclNode", "VarAssignNode", "VarAccessToken"):
        name = ast_node.get("name")
        if name:
            tokens.append(str(name).lower())
    elif node_type == "NumberNode":
        val = ast_node.get("value")
        if val is not None:
            tokens.append(str(val).lower())
    elif node_type == "BinOpNode":
        op = ast_node.get("operator")
        if op:
            tokens.append(str(op).lower())
    elif node_type == "IfNode":
        tokens.append("if")
    elif node_type == "WhileNode":
        tokens.append("while")
    elif node_type == "RepeatNode":
        tokens.append("repeat")
    elif node_type == "ForNode":
        tokens.append("for")

    for k, v in ast_node.items():
        if k not in ("type", "name", "program", "variables", "cases", "else_case", "body"):
            if isinstance(v, dict):
                tokens.extend(_collect_statement_keywords(v))
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        tokens.extend(_collect_statement_keywords(item))
    return tokens


def annotate_ast_with_lines(ast_root, code_str):
    """Memetakan setiap node pada JSON AST DAP ke nomor baris (1-based) pada kode sumber."""
    lines = code_str.splitlines()
    kamus_start = 1
    algo_start = 1

    for idx, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip().lower()
        if re.match(r"^kamus\b", stripped):
            kamus_start = idx
        elif re.match(r"^algoritma\b", stripped):
            algo_start = idx

    def find_best_line(tokens, start_line, end_line, current_cursor):
        if not tokens:
            return current_cursor
        best_line = None
        best_score = 0
        for line_no in range(max(start_line, current_cursor), end_line + 1):
            line_text = lines[line_no - 1].strip().lower()
            if not line_text or line_text in ("kamus", "algoritma", "endprogram", "endif", "endwhile"):
                continue
            score = sum(1 for t in tokens if t in line_text)
            if score > best_score:
                best_score = score
                best_line = line_no
                if score == len(tokens):
                    break
        if best_line is not None:
            return best_line
        for line_no in range(start_line, end_line + 1):
            line_text = lines[line_no - 1].strip().lower()
            if not line_text or line_text in ("kamus", "algoritma", "endprogram", "endif", "endwhile"):
                continue
            score = sum(1 for t in tokens if t in line_text)
            if score > best_score:
                best_score = score
                best_line = line_no
        return best_line if best_line is not None else current_cursor

    def propagate_line(node, line_no):
        if isinstance(node, dict):
            if "_line" not in node:
                node["_line"] = line_no
            for k, v in node.items():
                if k != "_line":
                    propagate_line(v, node["_line"])
        elif isinstance(node, list):
            for item in node:
                propagate_line(item, line_no)

    def walk_block(block_node, cursor_line):
        if not isinstance(block_node, dict):
            return cursor_line
        for stmt in block_node.get("program") or []:
            if not isinstance(stmt, dict):
                continue
            stype = stmt.get("type", "")
            if stype == "DictionaryNode":
                stmt["_line"] = kamus_start
                var_cursor = kamus_start
                kamus_end = algo_start if algo_start > kamus_start else len(lines)
                for var_decl in stmt.get("variables") or []:
                    toks = _collect_statement_keywords(var_decl)
                    matched = find_best_line(toks, kamus_start, kamus_end, var_cursor)
                    propagate_line(var_decl, matched)
                    var_cursor = matched
            elif stype == "IfNode":
                cases = stmt.get("cases") or []
                for c in cases:
                    cond = c.get("condition")
                    toks = ["if"] + _collect_statement_keywords(cond)
                    matched = find_best_line(toks, algo_start, len(lines), cursor_line)
                    stmt["_line"] = matched
                    propagate_line(cond, matched)
                    cursor_line = walk_block(c.get("body"), matched + 1)
                else_case = stmt.get("else_case") or {}
                if else_case.get("body"):
                    cursor_line = walk_block(else_case.get("body"), cursor_line)
            elif stype in ("WhileNode", "RepeatNode", "ForNode"):
                cond = stmt.get("condition")
                toks = _collect_statement_keywords(stmt) + _collect_statement_keywords(cond)
                matched = find_best_line(toks, algo_start, len(lines), cursor_line)
                stmt["_line"] = matched
                if cond:
                    propagate_line(cond, matched)
                cursor_line = walk_block(stmt.get("body"), matched + 1)
            else:
                toks = _collect_statement_keywords(stmt)
                matched = find_best_line(toks, algo_start, len(lines), cursor_line)
                propagate_line(stmt, matched)
                cursor_line = matched
        return cursor_line

    walk_block(ast_root, algo_start)
    return ast_root


def _get_children_preserve_line(root):
    if isinstance(root, dict):
        children = [
            v
            for k, v in root.items()
            if k not in ("type", "_line") and not isinstance(v, bool)
        ]
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    yield sub_item
            elif isinstance(item, dict) and "type" not in item:
                for sub_item in expand(
                    [v for k, v in item.items() if k != "_line" and not isinstance(v, bool)]
                ):
                    yield sub_item
            elif item is not None and not isinstance(item, bool) and item != "" and item != []:
                yield str(item) if isinstance(item, (int, float)) else item

    return list(expand(children))


def _try_parse_with_syntax_fallback(code_str):
    """
    Mencoba mem-parsing kode DAP. Jika gagal karena kesalahan sintaks umum mahasiswa
    (misal penggunaan '=' tunggal di dalam ekspresi kurung seperti '(N mod d = 0)'),
    mendeteksi baris error dari pesan compiler DAP dan melakukan perbaikan minimal
    agar pohon AST tetap dapat diekstrak untuk analisis Attention.
    """
    import copy
    import subprocess
    import tempfile
    from parserDAP import get_dap_path

    try:
        return copy.deepcopy(program_parser(code_str)), None
    except Exception:
        pass

    # Cek pesan error dari dap.exe untuk mengambil nomor baris SyntaxError
    syntax_err_info = None
    dap_bin = get_dap_path()
    if dap_bin:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".dap", delete=False, encoding="utf-8"
        ) as tmp_f:
            tmp_f.write(code_str)
            tmp_path = tmp_f.name
        try:
            res = subprocess.run(
                [dap_bin, tmp_path, "--show-ast-json"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            out = (res.stdout or "") + "\n" + (res.stderr or "")
            m_line = re.search(r"line\s+(\d+)", out)
            m_err = re.search(r"(SyntaxError:.*)", out)
            if m_line:
                syntax_err_info = {
                    "line": int(m_line.group(1)),
                    "message": m_err.group(1).strip() if m_err else "SyntaxError",
                }
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    repaired_code = repair_dap_code(code_str)
    if repaired_code != code_str:
        try:
            return copy.deepcopy(program_parser(repaired_code)), syntax_err_info
        except Exception:
            pass

    return build_fallback_ast(code_str), (
        syntax_err_info or {"line": 1, "message": "SyntaxError (Fallback AST)"}
    )


def extract_subtrees_with_lines(
    code_str, max_depth=None, leaf_flag=1, return_syntax_info=False, recursive=False
):
    """Mengekstrak subtree AST sekaligus daftar nomor baris asal untuk setiap node di subtree."""
    raw_tree, syntax_err_info = _try_parse_with_syntax_fallback(code_str)
    annotated_tree = annotate_ast_with_lines(raw_tree, code_str)

    head = Node(["1", get_token(annotated_tree)], order="1", line_no=annotated_tree.get("_line"))

    def build_tree(current_node, parent_node, order):
        token = get_token(current_node)
        children = _get_children_preserve_line(current_node)
        line_no = (
            current_node.get("_line", parent_node.line_no)
            if isinstance(current_node, dict)
            else parent_node.line_no
        )
        node = Node([order, token], parent=parent_node, order=order, line_no=line_no)
        for child_order, child in enumerate(children):
            build_tree(child, node, order + str(child_order + 1))

    for child_order, child in enumerate(_get_children_preserve_line(annotated_tree)):
        build_tree(child, head, "1" + str(child_order + 1))

    if leaf_flag != 1:
        leaf_nodes = findall_by_attr(head, name="is_leaf", value=True)
        for leaf in leaf_nodes:
            if leaf.parent is None:
                continue
            p = leaf.parent
            p.children = [c for c in p.children if c != leaf]

    subtrees = []
    subtree_line_lists = []

    if recursive:
        que = [head]
        while que:
            node = que.pop(0)
            if node.children:
                if max_depth is not None and max_depth > 0:
                    iter_nodes = [
                        n for n in PreOrderIter(node, maxlevel=max_depth) if n.name[1]
                    ]
                else:
                    iter_nodes = [n for n in PreOrderIter(node) if n.name[1]]
                if len(iter_nodes) > 1:
                    subtrees.append([n.name[1] for n in iter_nodes])
                    specific_lines = [
                        n.line_no
                        for n in iter_nodes
                        if n.line_no is not None
                        and n.name[1] not in ("ListNode", "DictionaryNode")
                    ]
                    if not specific_lines and node.line_no is not None:
                        specific_lines = [node.line_no]
                    subtree_line_lists.append(specific_lines)
                for child in node.children:
                    que.append(child)
        if not subtrees and head.name[1]:
            subtrees.append([head.name[1]])
            subtree_line_lists.append([head.line_no] if head.line_no else [1])
    else:
        eff_depth = max_depth if (max_depth is not None and max_depth > 0) else 3
        que = [(head, 1)]
        while que:
            node, depth = que.pop(0)
            if depth == len(node.name[0]):
                iter_nodes = [
                    n for n in PreOrderIter(node, maxlevel=eff_depth) if n.name[1]
                ]
                subtrees.append([n.name[1] for n in iter_nodes])
                specific_lines = [
                    n.line_no
                    for n in iter_nodes
                    if n.line_no is not None
                    and n.name[1] not in ("ListNode", "DictionaryNode")
                ]
                if not specific_lines and node.line_no is not None:
                    specific_lines = [node.line_no]
                subtree_line_lists.append(specific_lines)
                for child in node.children:
                    que.append((child, depth + eff_depth))
            else:
                for child in node.children:
                    que.append((child, depth))

    if return_syntax_info:
        return subtrees, subtree_line_lists, syntax_err_info
    return subtrees, subtree_line_lists


def compute_line_attention_from_subtrees(
    code_str, subtrees, subtree_line_lists, raw_attn_1d, max_subtree_length=64
):
    """
    Menghitung bobot attention per baris kode (line_scores) dan urutan baris terpenting (ranked_lines)
    dari daftar subtree, daftar baris per subtree, serta vektor attention model SANN.
    """
    num_actual_subtrees = min(len(subtrees), max_subtree_length, len(raw_attn_1d))
    attn_scores = np.array(raw_attn_1d[:num_actual_subtrees], dtype=float)
    attn_sum = float(np.sum(attn_scores))
    if attn_sum > 0:
        attn_scores_norm = attn_scores / attn_sum
    else:
        attn_scores_norm = attn_scores

    lines = code_str.splitlines()
    line_scores = {i: 0.0 for i in range(1, len(lines) + 1)}
    line_subtrees = {i: [] for i in range(1, len(lines) + 1)}

    subtree_details = []
    for idx in range(num_actual_subtrees):
        weight = float(attn_scores_norm[idx])
        st_lines = subtree_line_lists[idx]
        unique_lines = sorted(set(st_lines))
        subtree_details.append(
            {
                "subtree_index": idx + 1,
                "weight": weight,
                "raw_weight": float(attn_scores[idx]),
                "tokens": subtrees[idx],
                "lines": unique_lines,
            }
        )
        is_root_overview = (
            idx == 0
            and len(subtrees[idx]) > 0
            and subtrees[idx][0] == "ListNode"
            and len(unique_lines) > 1
            and num_actual_subtrees > 1
        )
        if unique_lines and not is_root_overview:
            is_var_decl_subtree = st_lines and any(
                t in ("VarDeclNode", "integer", "real", "boolean", "string")
                for t in subtrees[idx]
            )
            share = weight / len(unique_lines)
            for ln in unique_lines:
                if 1 <= ln <= len(lines):
                    if is_var_decl_subtree:
                        line_scores[ln] = max(line_scores[ln], share)
                    else:
                        line_scores[ln] += share
                    if (idx + 1) not in line_subtrees[ln]:
                        line_subtrees[ln].append(idx + 1)

    total_line_score = sum(line_scores.values())
    if total_line_score > 0:
        for ln in line_scores:
            line_scores[ln] /= total_line_score

    ranked_lines = sorted(
        [ln for ln in line_scores if line_scores[ln] > 0],
        key=lambda ln: line_scores[ln],
        reverse=True,
    )
    specific_subtrees = [
        st
        for st in subtree_details
        if not (
            st["subtree_index"] == 1
            and st["tokens"]
            and st["tokens"][0] == "ListNode"
            and len(st["lines"]) > 1
            and len(subtree_details) > 1
        )
    ]
    spec_total = sum(st["raw_weight"] for st in specific_subtrees)
    if spec_total > 0:
        for st in specific_subtrees:
            st["weight"] = st["raw_weight"] / spec_total
    ranked_subtrees = sorted(
        specific_subtrees, key=lambda x: x["weight"], reverse=True
    )
    return line_scores, ranked_lines, ranked_subtrees, line_subtrees


def predict_and_explain(code_str, model_dir=None, html_output_path=None, print_report=True):
    """
    Melakukan inference pada sebuah kode pseudocode DAP dan mendeteksi
    subtree serta nomor baris kode yang paling diperhatikan model (Attention).
    """
    inference_model, metadata = load_saved_sann_model(model_dir)

    max_depth = metadata["max_depth"]
    max_subtree_length = metadata["max_subtree_length"]
    leaf_flag = metadata.get("leaf_flag", 1)
    wordtoix = metadata["wordtoix"]
    pathtoix = metadata["pathtoix"]

    subtrees, subtree_line_lists, syntax_err_info = extract_subtrees_with_lines(
        code_str, max_depth=max_depth, leaf_flag=leaf_flag, return_syntax_info=True
    )
    path_strings = [" ".join(st) for st in subtrees]

    _, X_node = encode_paths([subtrees], wordtoix, max_subtree_length)
    _, X_path = encode_paths_path([path_strings], pathtoix, max_subtree_length)

    X_node = np.array(X_node)
    X_path = np.array(X_path)

    pred_probs, raw_attn = inference_model.predict([X_node, X_path], verbose=0)
    probs = pred_probs[0]
    predicted_class = int(np.argmax(probs))
    confidence = float(probs[predicted_class])

    num_actual_subtrees = min(len(subtrees), max_subtree_length)
    attn_scores = raw_attn[0, :num_actual_subtrees, 0].astype(float)
    attn_sum = float(np.sum(attn_scores))
    if attn_sum > 0:
        attn_scores_norm = attn_scores / attn_sum
    else:
        attn_scores_norm = attn_scores

    lines = code_str.splitlines()
    line_scores = {i: 0.0 for i in range(1, len(lines) + 1)}
    line_subtrees = {i: [] for i in range(1, len(lines) + 1)}

    subtree_details = []
    for idx in range(num_actual_subtrees):
        weight = float(attn_scores_norm[idx])
        st_lines = subtree_line_lists[idx]
        unique_lines = sorted(set(st_lines))
        subtree_details.append(
            {
                "subtree_index": idx + 1,
                "weight": weight,
                "raw_weight": float(attn_scores[idx]),
                "tokens": subtrees[idx],
                "lines": unique_lines,
            }
        )
        # Gunakan subtree spesifik (bukan root ListNode yang mencakup seluruh blok program)
        # agar perhitungan fokus per baris merepresentasikan bagian/statement spesifik
        is_root_overview = (
            idx == 0
            and len(subtrees[idx]) > 0
            and subtrees[idx][0] == "ListNode"
            and len(unique_lines) > 1
            and num_actual_subtrees > 1
        )
        if unique_lines and not is_root_overview:
            is_var_decl_subtree = (
                st_lines
                and any(t in ("VarDeclNode", "integer", "real", "boolean", "string") for t in subtrees[idx])
            )
            share = weight / len(unique_lines)
            for ln in unique_lines:
                if 1 <= ln <= len(lines):
                    line_scores[ln] += share
                    if is_var_decl_subtree:
                        # Gunakan nilai maksimum (bukan penjumlahan berulang) jika 1 baris kamus mendeklarasikan banyak variabel
                        line_scores[ln] = max(line_scores[ln], share)
                    else:
                        line_scores[ln] += share
                    if (idx + 1) not in line_subtrees[ln]:
                        line_subtrees[ln].append(idx + 1)

    total_line_score = sum(line_scores.values())
    if total_line_score > 0:
        for ln in line_scores:
            line_scores[ln] /= total_line_score

    ranked_lines = sorted(
        [ln for ln in line_scores if line_scores[ln] > 0],
        key=lambda ln: line_scores[ln],
        reverse=True,
    )
    specific_subtrees = [
        st for st in subtree_details
        if not (st["subtree_index"] == 1 and st["tokens"] and st["tokens"][0] == "ListNode" and len(st["lines"]) > 1 and len(subtree_details) > 1)
    ]
    spec_total = sum(st["raw_weight"] for st in specific_subtrees)
    if spec_total > 0:
        for st in specific_subtrees:
            st["weight"] = st["raw_weight"] / spec_total
    ranked_subtrees = sorted(specific_subtrees, key=lambda x: x["weight"], reverse=True)

    result = {
        "predicted_class": 0 if syntax_err_info else predicted_class,
        "model_predicted_class": predicted_class,
        "predicted_label": (
            f"INCORRECT (Salah - {syntax_err_info['message']} di Baris {syntax_err_info['line']})"
            if syntax_err_info
            else ("CORRECT (Benar)" if predicted_class == 1 else "INCORRECT (Salah)")
        ),
        "syntax_error": syntax_err_info,
        "probabilities": {
            "incorrect_0": float(probs[0]),
            "correct_1": float(probs[1]),
        },
        "confidence": confidence,
        "line_scores": line_scores,
        "ranked_lines": ranked_lines,
        "ranked_subtrees": ranked_subtrees,
    }

    if print_report:
        _print_visual_report(code_str, result, line_subtrees)

    if html_output_path:
        _save_html_report(code_str, result, html_output_path)

    return result


def _print_visual_report(code_str, result, line_subtrees):
    lines = code_str.splitlines()
    line_scores = result["line_scores"]
    ranked_lines = result["ranked_lines"]
    max_line_score = max(line_scores.values()) if line_scores else 0.0

    print("\n" + "=" * 78)
    print("                 HASIL INFERENCE & EXPLAINABILITY SANN-DAP")
    print("=" * 78)
    print(f" Prediksi Model : {result['predicted_label']}")
    print(
        f" Probabilitas   : Correct (1) = {result['probabilities']['correct_1']*100:.2f}% | "
        f"Incorrect (0) = {result['probabilities']['incorrect_0']*100:.2f}%"
    )
    if ranked_lines:
        top_ln = ranked_lines[0]
        print(
            f" Baris Fokus #1 : Baris {top_ln} ({line_scores[top_ln]*100:.1f}% Attention) "
            f"-> `{lines[top_ln - 1].strip()}`"
        )
    print("=" * 78)

    print("\n[1] VISUALISASI ATTENTION PER BARIS KODE:")
    print("-" * 78)
    print(f"{'No':>3} | {'Attention Bar':<14} | {'Bobot':>6} | {'Kode Pseudocode'}")
    print("-" * 78)

    for ln_idx, text in enumerate(lines, start=1):
        score = line_scores.get(ln_idx, 0.0)
        bar_len = int(round((score / max_line_score) * 12)) if max_line_score > 0 else 0
        bar = "#" * bar_len + "." * (12 - bar_len)

        marker = ""
        if result.get("syntax_error") and ln_idx == result["syntax_error"]["line"]:
            marker = f"  <<< [FOKUS UTAMA #1 & {result['syntax_error']['message']}]"
        elif ranked_lines and ln_idx == ranked_lines[0]:
            marker = "  <<< [FOKUS UTAMA #1]"
        elif len(ranked_lines) > 1 and ln_idx == ranked_lines[1]:
            marker = "  <<  [FOKUS #2]"
        elif len(ranked_lines) > 2 and ln_idx == ranked_lines[2]:
            marker = "  <   [FOKUS #3]"

        print(f"{ln_idx:>3} | [{bar}] | {score*100:>5.1f}% | {text}{marker}")
    print("-" * 78)

    print("\n[2] TOP SUBTREE AST DENGAN BOBOT ATTENTION TERTINGGI:")
    print("-" * 78)
    for rank, st in enumerate(result["ranked_subtrees"][:5], start=1):
        ln_str = ", ".join(str(l) for l in st["lines"]) if st["lines"] else "-"
        tokens_preview = " -> ".join(st["tokens"][:10])
        if len(st["tokens"]) > 10:
            tokens_preview += f" ... (+{len(st['tokens']) - 10} token)"
        print(
            f" Rank #{rank} | Subtree #{st['subtree_index']:<2} | "
            f"Attention: {st['weight']*100:>5.1f}% | Baris: [{ln_str}]"
        )
        print(f"          Tokens: {tokens_preview}")
    print("=" * 78 + "\n")


def _save_html_report(code_str, result, html_output_path):
    lines = code_str.splitlines()
    line_scores = result["line_scores"]
    max_score = max(line_scores.values()) if line_scores else 1.0
    if max_score == 0:
        max_score = 1.0

    rows_html = []
    for ln_idx, text in enumerate(lines, start=1):
        score = line_scores.get(ln_idx, 0.0)
        intensity = score / max_score
        alpha = round(0.08 + 0.55 * intensity, 2) if score > 0 else 0.0
        bg_color = f"rgba(255, 82, 82, {alpha})" if score > 0 else "transparent"
        badge = ""
        if result["ranked_lines"] and ln_idx == result["ranked_lines"][0]:
            badge = '<span style="background:#d32f2f;color:#fff;padding:2px 6px;border-radius:4px;font-size:11px;margin-left:8px;">FOKUS UTAMA #1</span>'
        elif len(result["ranked_lines"]) > 1 and ln_idx == result["ranked_lines"][1]:
            badge = '<span style="background:#f57c00;color:#fff;padding:2px 6px;border-radius:4px;font-size:11px;margin-left:8px;">FOKUS #2</span>'

        escaped_text = (
            text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        )
        rows_html.append(
            f'<tr style="background:{bg_color};">'
            f'<td style="padding:4px 10px;text-align:right;color:#666;border-right:1px solid #ddd;user-select:none;">{ln_idx}</td>'
            f'<td style="padding:4px 10px;text-align:right;font-weight:bold;color:#b71c1c;border-right:1px solid #ddd;">{score*100:.1f}%</td>'
            f'<td style="padding:4px 12px;white-space:pre;font-family:Consolas,monospace;">{escaped_text}{badge}</td>'
            f"</tr>"
        )

    html_content = f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8">
<title>SANN-DAP Attention Report</title>
<style>
  body {{ font-family: system-ui, -apple-system, sans-serif; margin: 24px; background: #fafafa; color: #222; }}
  .card {{ background: #fff; border: 1px solid #e0e0e0; border-radius: 8px; padding: 20px; max-width: 900px; box-shadow: 0 2px 6px rgba(0,0,0,0.04); }}
  table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }}
</style>
</head>
<body>
  <div class="card">
    <h2 style="margin-top:0;">Hasil Inference &amp; Line Attention SANN-DAP</h2>
    <p><strong>Prediksi:</strong> {result['predicted_label']} &nbsp;|&nbsp;
       <strong>Probabilitas Correct (1):</strong> {result['probabilities']['correct_1']*100:.2f}% &nbsp;|&nbsp;
       <strong>Probabilitas Incorrect (0):</strong> {result['probabilities']['incorrect_0']*100:.2f}%</p>
    <table>
      <thead>
        <tr style="background:#f0f0f0;border-bottom:2px solid #ccc;">
          <th style="padding:6px 10px;text-align:right;width:40px;">Baris</th>
          <th style="padding:6px 10px;text-align:right;width:80px;">Attention</th>
          <th style="padding:6px 12px;text-align:left;">Kode Pseudocode</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows_html)}
      </tbody>
    </table>
  </div>
</body>
</html>"""

    with open(html_output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Visualisasi HTML disimpan ke: {html_output_path}")


if __name__ == "__main__":
    import pandas as pd

    parser = argparse.ArgumentParser(
        description="Inference dan deteksi baris (Attention) untuk pseudocode DAP menggunakan model SANN."
    )
    parser.add_argument(
        "--file", "-f", type=str, default=None, help="Path ke file .dap yang ingin diuji"
    )
    parser.add_argument(
        "--code", "-c", type=str, default=None, help="String pseudocode DAP langsung"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="Week_05-06_Problem_Solving_Questions.csv",
        help="Path atau nama file CSV dataset (default: Week_05-06_Problem_Solving_Questions.csv)",
    )
    parser.add_argument(
        "--row",
        "-r",
        type=int,
        default=524,
        help="Indeks baris (pandas 0-based) pada CSV (default: 524 - Hazel Argyanto Memeriksa Faktor)",
    )
    parser.add_argument(
        "--col",
        type=str,
        default="First_Submit_Pseudocode",
        help="Nama kolom kode di CSV (default: First_Submit_Pseudocode)",
    )
    parser.add_argument(
        "--html",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "inference_report.html"),
        help="Path output laporan visual HTML",
    )
    args = parser.parse_args()

    if args.file:
        with open(args.file, "r", encoding="utf-8") as f:
            sample_code = f.read()
    elif args.code:
        sample_code = args.code
    else:
        csv_path = Path(args.csv)
        if not csv_path.exists():
            csv_path = Path(__file__).resolve().parent.parent / args.csv
        df = pd.read_csv(csv_path)
        row_data = df.loc[args.row]
        sample_code = str(row_data[args.col])
        print("\n" + "#" * 78)
        print(f" DATA ASLI DARI DATASET ({csv_path.name})")
        print("#" * 78)
        print(f" Baris CSV (Index) : {args.row} (Kolom: {args.col})")
        print(f" Mahasiswa         : {row_data.get('Student_Name')} ({row_data.get('NIM')})")
        print(f" Nama Soal         : {row_data.get('Question_Name')}")
        if args.col == "First_Submit_Pseudocode":
            print(
                f" Status First Subm : State = {row_data.get('First_Submit_State')} | "
                f"Marks = {row_data.get('First_Submit_Marks')}"
            )
        else:
            print(
                f" Status / Skor     : State = {row_data.get('State')} | "
                f"Is_Correct = {row_data.get('Is_Correct')}"
            )
        print("#" * 78)

    predict_and_explain(sample_code, html_output_path=args.html)

