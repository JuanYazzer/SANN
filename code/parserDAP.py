import pandas as pd
import json
from anytree import Node, RenderTree, AsciiStyle, PreOrderIter, findall_by_attr
import numpy as np
import argparse
import random
import os
import subprocess
import tempfile
import functools
from pathlib import Path
from config import Config
from anytree.exporter import DotExporter
import re

@functools.lru_cache(maxsize=1)
def get_dap_path():
    """Finds the absolute path to the DAP compiler executable."""
    env_path = os.environ.get("DAP_PATH")
    if env_path and Path(env_path).exists():
        return env_path

    dap_exe = Path.home() / ".dap" / "bin" / "dap.exe"
    if dap_exe.exists():
        return str(dap_exe)

    try:
        subprocess.run(
            ["dap", "-h"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=1
        )
        return "dap"
    except Exception:
        pass

    return None

def get_subtree(root, text_paths, depth, MAX_DEPTH):
    config = Config()
    que = []
    que.append((root, 1))
    while(que):
        node, depth = que.pop(0)
        #print(node.name[1], depth)
        if depth==len(node.name[0]):
            subtree = [node.name[1] for node in PreOrderIter(node, maxlevel=MAX_DEPTH)] #PreOrderIter(node, maxlevel=8)]
            #subtree_string = ' '.join(subtree)
            text_paths.append(subtree)
            for child in node.children:
                que.append((child, depth+MAX_DEPTH))
        #print(subtree)
        else:
            for child in node.children:
                que.append((child, depth))
    
    return text_paths


def get_subtree_recursive(root, text_paths, MAX_DEPTH=None):
    """
    Mengekstrak seluruh subtree secara rekursif dari setiap internal node AST
    sesuai paper 'Automated Identification of Logical Errors in Programs' (Section 4.1, Fig. 2).
    """
    que = [root]
    while que:
        node = que.pop(0)
        if node.children:
            if MAX_DEPTH is not None and MAX_DEPTH > 0:
                subtree = [n.name[1] for n in PreOrderIter(node, maxlevel=MAX_DEPTH) if n.name[1]]
            else:
                subtree = [n.name[1] for n in PreOrderIter(node) if n.name[1]]
            if len(subtree) > 1:
                text_paths.append(subtree)
            for child in node.children:
                que.append(child)
    if not text_paths and root.name[1]:
        text_paths.append([root.name[1]])
    return text_paths


def get_token(node):
    token = ''
    if isinstance(node, str):
        token = node
    elif isinstance(node, set):
        token = 'Modifier'  # node.pop()
    elif isinstance(node, dict):
        token = str(node.get('type', ''))

    return token

def get_children(root):
    if isinstance(root, dict):
        children = [v for k, v in root.items() if k != 'type' and not isinstance(v, bool)]
    elif isinstance(root, set):
        children = list(root)
    else:
        children = []

    def expand(nested_list):
        for item in nested_list:
            if isinstance(item, list):
                for sub_item in expand(item):
                    yield sub_item
            elif isinstance(item, dict) and 'type' not in item:
                for sub_item in expand([v for v in item.values() if not isinstance(v, bool)]):
                    yield sub_item
            elif item is not None and not isinstance(item, bool) and item != '' and item != []:
                yield str(item) if isinstance(item, (int, float)) else item

    return list(expand(children))

def get_trees(current_node, parent_node, order):
    
    token, children = get_token(current_node), get_children(current_node)
    node = Node([order,token], parent=parent_node, order=order)

    for child_order in range(len(children)):
        get_trees(children[child_order], node, order+str(int(child_order)+1))

def extracting_AST(c_code, leaves, MAX_DEPTH, recursive=False):
    """Extracting paths for a given json code.
    Input:
    json_code: json object. The json object of a snap program to be extracted.
    max_length: int. Max length of the path to be restained.
    max_width: int. Max width of the path to be restained.
    hash_path: boolean. if true, MD5 hashed path will be returned to save space.
    hashing_table: Dict. Hashing table for path.
    Return:
    walk_paths: list of AST paths from the json code.
    """
    
    config = Config()
    depth = 1
    
    # Initialize head node of the code.
    head = Node(["1",get_token(c_code)])
    
    # Recursively construct AST tree.
    
    for child_order in range(len(get_children(c_code))):

        get_trees(get_children(c_code)[child_order], head, "1"+str(int(child_order)+1))
        
    #print(RenderTree(head, style=AsciiStyle()).by_attr())
    #DotExporter(head).to_picture("udo.png")
    
    text_paths = []
    
    # leaves unpruned
    if leaves == 1:
        if recursive:
            text_paths = get_subtree_recursive(head, text_paths, MAX_DEPTH)
        else:
            text_paths = get_subtree(head, text_paths, depth, MAX_DEPTH)
    # leaves pruned
    else:
        leaf_nodes = findall_by_attr(head, name="is_leaf", value=True)
        for leaf in leaf_nodes:
            if(leaf.parent is None):
                continue
            p = leaf.parent
            #print(leaf.name)
            children = []
            for child in p.children:
                if child!=leaf:
                    children.append(child)
            p.children = children
        if recursive:
            text_paths = get_subtree_recursive(head, text_paths, MAX_DEPTH)
        else:
            text_paths = get_subtree(head, text_paths, depth, MAX_DEPTH)
    return text_paths

@functools.lru_cache(maxsize=None)
def repair_dap_code(code_str):
    """Perbaikan minimal untuk kesalahan sintaks umum mahasiswa agar AST tetap dapat diekstrak."""
    if not isinstance(code_str, str):
        return ""
    s = code_str.replace("\r\n", "\n").replace("\r", "\n")
    s = s.replace("←", "<-")
    lines = s.split("\n")
    new_lines = []
    has_program = False
    has_kamus = False
    has_endprogram = False
    for line in lines:
        stripped = line.strip()
        low = stripped.lower()
        m_prog = re.match(
            r"^(program|pemograman|algoritma_program)\s+(.*)$",
            stripped,
            flags=re.IGNORECASE,
        )
        if m_prog and not has_program and not low.startswith("algoritma "):
            has_program = True
            prog_name = re.sub(r"[^A-Za-z0-9_]", "_", m_prog.group(2).strip())
            if (
                not prog_name
                or prog_name.lower() in ("algoritma", "kamus", "program", "endprogram")
                or prog_name[0].isdigit()
            ):
                prog_name = "prog_" + (prog_name or "main")
            new_lines.append(f"program {prog_name}")
            continue
        if re.match(r"^kamus\b", low):
            has_kamus = True
            new_lines.append("kamus")
            continue
        if re.match(r"^algoritma+\b", low):
            new_lines.append("algoritma")
            continue
        if re.match(r"^end\s*program\b", low):
            has_endprogram = True
            new_lines.append("endprogram")
            continue
        line_fixed = line
        line_fixed = re.sub(r"\bDIV\b", "div", line_fixed)
        line_fixed = re.sub(r"\bMOD\b", "mod", line_fixed)
        line_fixed = re.sub(r"&([A-Za-z_][A-Za-z0-9_]*)", r"\1", line_fixed)
        line_fixed = re.sub(
            r"\boutput\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*:[0-9:]+\s*\)",
            r"output(\1)",
            line_fixed,
        )
        line_fixed = re.sub(
            r"^(\s*)output\s+(?!\()(.+)$",
            lambda m: f'{m.group(1)}output("{m.group(2).strip()}")',
            line_fixed,
        )
        if re.search(r"\bif\b.*\bthen\b", line_fixed, flags=re.IGNORECASE):
            line_fixed = re.sub(r"(?<![<>!=])=(?!=)", "==", line_fixed)
        new_lines.append(line_fixed)
    if not has_program:
        if not has_kamus:
            new_lines.insert(0, "kamus")
        new_lines.insert(0, "program prog_main")
    if not has_endprogram:
        new_lines.append("endprogram")
    res = "\n".join(new_lines)
    res = re.sub(r"(\([^\n()]*?)(?<![<>!=])=(?!=)([^\n()]*?\))", r"\1==\2", res)
    res = re.sub(r"\boutput\(\s*([A-Za-z0-9_]+\s*:)\s*\)", r'output("\1")', res)
    return res


def build_fallback_ast(code_str):
    """
    Membangun pohon AST JSON fallback berbasis baris & token apabila kode mahasiswa
    memiliki kesalahan sintaks berat yang tidak dapat dikompilasi langsung oleh dap.exe.
    Memastikan tidak ada attempt/soal mahasiswa yang terlewati pada sekuens DKT maupun line attention.
    """
    if not isinstance(code_str, str) or not code_str.strip():
        return {"type": "ListNode", "program": []}

    lines = code_str.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    in_kamus = False
    in_algo = False
    variables = []
    program_stmts = []

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("{"):
            continue
        low = stripped.lower()
        if re.match(r"^(program|pemograman)\b", low):
            continue
        if re.match(r"^kamus\b", low):
            in_kamus = True
            in_algo = False
            continue
        if re.match(r"^algoritma+\b", low):
            in_algo = True
            in_kamus = False
            continue
        if low in ("endprogram", "end program", "endif", "endwhile", "endfor"):
            continue

        if in_kamus and ":" in stripped:
            parts = stripped.split(":", 1)
            var_names = [v.strip() for v in parts[0].split(",") if v.strip()]
            vtype = parts[1].strip().split()[0] if parts[1].strip() else "integer"
            for vn in var_names:
                variables.append({
                    "type": "VarDeclNode",
                    "name": vn,
                    "is_const": False,
                    "is_declaration": True,
                    "value": {"type": "VarAccessToken", "name": vtype},
                })
        else:
            m_call = re.match(r"^(input|output)\s*\((.*)\)$", stripped, flags=re.IGNORECASE)
            if m_call:
                fn_name = m_call.group(1).lower()
                arg_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?", m_call.group(2))
                args = [
                    {"type": "NumberNode", "value": t} if t[0].isdigit() else {"type": "VarAccessToken", "name": t}
                    for t in arg_tokens
                ]
                program_stmts.append({
                    "type": "CallNode",
                    "call": {"type": "VarAccessToken", "name": fn_name},
                    "args": args,
                })
            elif "<-" in stripped or "=" in stripped:
                sep = "<-" if "<-" in stripped else "="
                lhs, rhs = stripped.split(sep, 1)
                lhs_name = lhs.strip()
                rhs_tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|[+\-*/<>]=?|==", rhs)
                if len(rhs_tokens) >= 3:
                    val_node = {
                        "type": "BinOpNode",
                        "left": {"type": "NumberNode", "value": rhs_tokens[0]} if rhs_tokens[0][0].isdigit() else {"type": "VarAccessToken", "name": rhs_tokens[0]},
                        "operator": rhs_tokens[1],
                        "right": {"type": "NumberNode", "value": rhs_tokens[2]} if rhs_tokens[2][0].isdigit() else {"type": "VarAccessToken", "name": rhs_tokens[2]},
                    }
                elif rhs_tokens:
                    t0 = rhs_tokens[0]
                    val_node = {"type": "NumberNode", "value": t0} if t0[0].isdigit() else {"type": "VarAccessToken", "name": t0}
                else:
                    val_node = {"type": "VarAccessToken", "name": rhs.strip()}
                program_stmts.append({
                    "type": "VarAssignNode",
                    "name": lhs_name,
                    "is_const": False,
                    "is_declaration": False,
                    "value": val_node,
                })
            else:
                toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?", stripped)
                program_stmts.append({
                    "type": "CallNode",
                    "call": {"type": "VarAccessToken", "name": toks[0] if toks else stripped},
                    "args": [{"type": "VarAccessToken", "name": t} for t in toks[1:]],
                })

    prog = []
    if variables:
        prog.append({"type": "DictionaryNode", "variables": variables})
    prog.extend(program_stmts)
    if not prog:
        prog.append({"type": "VarAccessToken", "name": "empty"})
    return {"type": "ListNode", "program": prog}


_AST_DISK_CACHE = None


def _get_ast_disk_cache():
    global _AST_DISK_CACHE
    if _AST_DISK_CACHE is None:
        cache_file = Path(__file__).resolve().parent.parent / "saved_model_dap" / "ast_cache.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as f:
                    _AST_DISK_CACHE = json.load(f)
            except Exception:
                _AST_DISK_CACHE = {}
        else:
            _AST_DISK_CACHE = {}
    return _AST_DISK_CACHE


@functools.lru_cache(maxsize=None)
def program_parser(func):
    if not isinstance(func, str) or not func.strip():
        raise ValueError("Empty DAP code")

    dap_bin = get_dap_path()
    if not dap_bin:
        raise FileNotFoundError(
            "DAP compiler binary not found. Please install DAP first."
        )

    temp_path = None
    try:
        if os.path.isfile(func):
            target_path = func
        else:
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".dap", delete=False, encoding="utf-8"
            ) as tmp_file:
                tmp_file.write(func)
                temp_path = tmp_file.name
            target_path = temp_path

        cmd = [dap_bin, target_path, "--show-ast-json"]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr)

        stdout = result.stdout.strip()
        tree = json.loads(stdout)
        if not isinstance(tree, dict) or "type" not in tree:
            raise ValueError("Invalid DAP AST JSON")
        return tree
    finally:
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except OSError:
                pass


def parser_main(code_list, Y, MAX_DEPTH, recursive=False):
    config = Config()
    
    leaf_flag = config.leaf_flag # 1: with leaves, else: pruned
    
    print("Parsing Start")
    
    disk_cache = _get_ast_disk_cache()
    parsed_code = []
    
    for c in code_list:
        if c in disk_cache:
            parsed = disk_cache[c]
        else:
            try:
                parsed = program_parser(c)
            except Exception:
                try:
                    parsed = program_parser(repair_dap_code(c))
                except Exception:
                    parsed = "Uncompilable"
        parsed_code.append(parsed)
        
    print("----Parsing finished. Extracting subtrees----")
    
    # Extracting paths for all programs in the csv file. Output is [["start,path_hash/path,end",...,...],...].
    raw_paths = [extracting_AST(c_code, leaf_flag, MAX_DEPTH, recursive=recursive) for c_code in parsed_code]
    print("----Extracting subtrees finished----")
    #print(raw_paths)
    
    concatanated = []
    for problem in raw_paths:
        problem_con = []
        for subtree in problem:
            subtree = list(filter(None, subtree))
            subtree = '$$@$$'.join(subtree)
            problem_con.append(subtree)
        problem_con = '@@$@@'.join(problem_con)
        concatanated.append(problem_con)
        
    main_df = pd.DataFrame(list(
        zip(code_list, concatanated, Y)),
               columns =['Code', 'raw_ast', 'Score'])
    
    main_df = main_df[main_df["raw_ast"] != "Uncompilable"]
    main_df = main_df[main_df["raw_ast"] != "n"]
    
    return main_df

