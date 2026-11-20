"""Register the pip-installed NVIDIA DLL folders so TensorFlow can find them.

Run once with the venv's Python after installing requirements-gpu.txt:

    .venv\\Scripts\\python.exe setup_gpu_path.py

It writes a small .pth file into the venv's site-packages that prepends every
site-packages\\nvidia\\*\\bin folder to PATH whenever Python starts.
"""
import sysconfig
from pathlib import Path

site = Path(sysconfig.get_paths()["purelib"])
pth = site / "nvidia_dll_path.pth"
pth.write_text(
    "import os, glob; "
    "os.environ['PATH'] = os.pathsep.join("
    "glob.glob(os.path.join(" + repr(str(site)) + ", 'nvidia', '*', 'bin'))"
    " + [os.environ.get('PATH', '')])\n"
)
print(f"Wrote {pth}")

