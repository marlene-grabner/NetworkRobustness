"""
Every script in scripts/ must only import things from NoiseEffect that exist.

The scripts are not run (most need the data and take hours); their
`from NoiseEffect... import ...` lines are read and resolved against the
package. This catches the usual way a script breaks: a function was renamed,
moved or removed in src/ and a script still uses the old name.
"""

import ast
import importlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = sorted(
    p.relative_to(ROOT).as_posix()
    for p in (ROOT / "scripts").rglob("*.py")
    if ".ipynb_checkpoints" not in p.parts
)

# Scripts that are known to be broken (they use functions that no longer
# exist). Fix or delete the script, then remove it from this list: the test
# fails if a listed script works again or is gone, so the list stays current.
KNOWN_BROKEN = {
    "scripts/global_properties/gcc/calculate_giant_connected_component.py",
    "scripts/global_properties/global_efficiency/run_baseline_task.py",
    "scripts/global_properties/global_efficiency/run_perturbed_task.py",
    "scripts/local_structure/similarity_perturbation/test_community_pertrubed.py",
}


def unresolved_imports(script: str) -> list[str]:
    problems = []
    tree = ast.parse((ROOT / script).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [a.name for a in node.names if a.name.split(".")[0] == "NoiseEffect"]
            names = []
        elif (
            isinstance(node, ast.ImportFrom)
            and node.level == 0
            and (node.module or "").split(".")[0] == "NoiseEffect"
        ):
            modules, names = [node.module], [a.name for a in node.names]
        else:
            continue
        for module_name in modules:
            try:
                module = importlib.import_module(module_name)
            except Exception as error:
                problems.append(f"line {node.lineno}: import {module_name}: {error}")
                continue
            for name in names:
                if name == "*" or hasattr(module, name):
                    continue
                try:  # a submodule that the package does not import itself
                    importlib.import_module(f"{module_name}.{name}")
                except Exception:
                    problems.append(f"line {node.lineno}: {module_name} has no '{name}'")
    return problems


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_imports_exist(script):
    problems = unresolved_imports(script)
    if script in KNOWN_BROKEN:
        assert problems, f"{script} works again: remove it from KNOWN_BROKEN"
        pytest.xfail("; ".join(problems))
    assert not problems, "\n".join(problems)


def test_known_broken_list_is_current():
    gone = KNOWN_BROKEN - set(SCRIPTS)
    assert not gone, f"remove from KNOWN_BROKEN (script no longer exists): {sorted(gone)}"
