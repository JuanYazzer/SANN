"""AST builder for DAP pseudocode files.

Thin wrapper around the DAP compiler binary (see
https://github.com/rafiathallah3/dap) that turns a .dap source file into a
JSON AST, with light normalization (single-use variable inlining + removal
of unused dictionary declarations) so the resulting tree is easier to
consume downstream (program embedding, diffing, etc.).
"""
from __future__ import annotations

import copy
import functools
import json
import os
import subprocess
from pathlib import Path
from typing import Any


@functools.cache
def get_dap_path() -> str | None:
    """Finds the absolute path to the DAP compiler executable.

    Resolution order: DAP_PATH environment variable, ~/.dap/bin/dap.exe,
    then a `dap` binary on PATH.
    """
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


def count_var_references(node: Any, name: str) -> int:
    """Recursively counts how many times the variable name is referenced."""
    if not isinstance(node, dict):
        if isinstance(node, list):
            return sum(count_var_references(item, name) for item in node)
        return 0

    count = 0
    t = node.get("type")
    if t == "VarAccessToken" and node.get("name") == name:
        count += 1
    elif t == "VarAssignNode" and node.get("name") == name:
        count += 1

    for k, v in node.items():
        if k not in ["type", "name"]:
            count += count_var_references(v, name)
    return count


def collect_var_counts(node: Any, counts: dict[str, int] | None = None) -> dict[str, int]:
    """Counts references of every variable name in one subtree walk."""
    if counts is None:
        counts = {}
    if isinstance(node, list):
        for item in node:
            collect_var_counts(item, counts)
        return counts
    if not isinstance(node, dict):
        return counts

    t = node.get("type")
    if t in ("VarAccessToken", "VarAssignNode"):
        name = node.get("name")
        if name:
            counts[name] = counts.get(name, 0) + 1

    for k, v in node.items():
        if k not in ("type", "name"):
            collect_var_counts(v, counts)
    return counts


def replace_var_reference(node: Any, name: str, replacement_value: dict[str, Any]) -> None:
    """Recursively replaces VarAccessToken with replacement_value."""
    if not isinstance(node, dict):
        if isinstance(node, list):
            for idx, item in enumerate(node):
                if isinstance(item, dict) and item.get("type") == "VarAccessToken" and item.get("name") == name:
                    node[idx] = copy.deepcopy(replacement_value)
                else:
                    replace_var_reference(item, name, replacement_value)
        return

    for k, v in list(node.items()):
        if isinstance(v, dict) and v.get("type") == "VarAccessToken" and v.get("name") == name:
            node[k] = copy.deepcopy(replacement_value)
        else:
            replace_var_reference(v, name, replacement_value)


def normalize_ast(node: Any) -> Any:
    """Recursively normalizes the AST (e.g., inlining single-use variables)."""
    if not isinstance(node, dict):
        if isinstance(node, list):
            for item in node:
                normalize_ast(item)
        return node

    # First recursively normalize children
    for k, v in list(node.items()):
        if k != "type":
            normalize_ast(v)

    t = node.get("type")
    if t == "ListNode":
        # DAP emits "program": null (not []) when a block has zero
        # statements/declarations (e.g. a "kamus" section that's only a
        # comment) -- .get(key, []) would NOT catch that since the key is
        # present, just null, so it silently returns None and blows up the
        # loop below with "'NoneType' object is not iterable".
        program = node.get("program") or []
        # One counting walk per statement (DictionaryNode declarations excluded),
        # instead of re-walking all other statements for every assignment.
        counters: list[dict[str, int]] = []
        for stmt in program:
            if isinstance(stmt, dict) and stmt.get("type") != "DictionaryNode":
                counters.append(collect_var_counts(stmt))
            else:
                counters.append({})
        total: dict[str, int] = {}
        for c in counters:
            for name, cnt in c.items():
                total[name] = total.get(name, 0) + cnt

        new_program = []
        prev: dict[str, int] = {}
        for i, stmt in enumerate(program):
            cur = counters[i]
            if isinstance(stmt, dict) and stmt.get("type") == "VarAssignNode":
                var_name = stmt.get("name")
                ref_count = (
                    total.get(var_name, 0) - prev.get(var_name, 0) - cur.get(var_name, 0)
                )
                if ref_count == 1 and prev.get(var_name, 0) == 0:
                    # Inline: replace the single later reference with the value
                    for j in range(i + 1, len(program)):
                        if counters[j].get(var_name, 0):
                            replace_var_reference(program[j], var_name, stmt.get("value"))
                            # The target statement changed; refresh its counts
                            new_counts = collect_var_counts(program[j])
                            for name in set(counters[j]) | set(new_counts):
                                total[name] = (
                                    total.get(name, 0)
                                    - counters[j].get(name, 0)
                                    + new_counts.get(name, 0)
                                )
                            counters[j] = new_counts
                            break
                    # Drop the declaration and its contribution to the totals
                    for name, cnt in cur.items():
                        total[name] = total.get(name, 0) - cnt
                    continue
            new_program.append(stmt)
            for name, cnt in cur.items():
                prev[name] = prev.get(name, 0) + cnt
        node["program"] = new_program

    return node


def cleanup_dictionary_node(root_node: dict[str, Any]) -> None:
    """Removes variables from the DictionaryNode that are never used in the algorithm."""
    if not isinstance(root_node, dict) or root_node.get("type") != "ListNode":
        return

    # Find DictionaryNode and other statements
    dict_node = None
    algo_nodes = []
    for stmt in (root_node.get("program") or []):
        if isinstance(stmt, dict) and stmt.get("type") == "DictionaryNode":
            dict_node = stmt
        else:
            algo_nodes.append(stmt)

    if not dict_node:
        return

    # Count references of each declared variable in the algorithm nodes
    # (see note above -- "variables" can be present-but-null too)
    variables = dict_node.get("variables") or []
    new_variables = []
    for var in variables:
        var_name = var.get("name")
        is_const = var.get("is_const", False)
        count = sum(count_var_references(node, var_name) for node in algo_nodes)
        if count > 0 or is_const:
            new_variables.append(var)
    dict_node["variables"] = new_variables


_reference_ast_cache: dict[tuple[str, bool], dict[str, Any]] = {}


def get_ast(filepath: str | Path, normalize: bool = True) -> dict[str, Any]:
    """Executes the DAP compiler on the file and returns parsed JSON AST."""
    filepath_str = str(filepath)
    filename = Path(filepath_str).name
    cache_key = (filepath_str, normalize)
    if filename.startswith("c_") and cache_key in _reference_ast_cache:
        return copy.deepcopy(_reference_ast_cache[cache_key])

    dap_bin = get_dap_path()
    if not dap_bin:
        raise FileNotFoundError(
            "DAP compiler binary not found. Please install DAP first."
        )

    cmd = [dap_bin, filepath_str, "--show-ast-json"]
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Failed to generate AST for {filepath_str}. Error:\n{result.stderr}"
        )

    # Extract only the JSON portion from the stdout
    stdout = result.stdout.strip()
    try:
        ast_data = json.loads(stdout)
        if normalize:
            ast_data = normalize_ast(ast_data)
            cleanup_dictionary_node(ast_data)
        if filename.startswith("c_"):
            _reference_ast_cache[cache_key] = copy.deepcopy(ast_data)
        return ast_data
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Could not parse DAP AST output as JSON. Output was:\n{stdout}\nError: {e}"
        )
