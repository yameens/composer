#!/usr/bin/env python3
"""
Render assets/conductor_brand.png using LaTeX Dürer Informal (\\usepackage{duerer}, \\duinfamily).

The catalogue face has no official OTF/TTF; this uses TeX Live’s METAFONT output.
Requires: pdflatex with package `duerer`, and pdftoppm (Poppler) for PDF→PNG.

  python3 tools/export_conductor_brand.py
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"

TEX = r"""
\documentclass[border=12pt]{standalone}
\usepackage[T1]{fontenc}
\usepackage{duerer}
\begin{document}
\duinfamily\fontsize{30}{34}\selectfont COMPOSER
\end{document}
"""


def main() -> int:
    if not shutil.which("pdflatex"):
        print("pdflatex not found. Install MacTeX or TeX Live, then retry.", file=sys.stderr)
        return 1
    if not shutil.which("pdftoppm"):
        print(
            "pdftoppm not found. Install Poppler (brew install poppler) for PDF→PNG.",
            file=sys.stderr,
        )
        return 1

    ASSETS.mkdir(parents=True, exist_ok=True)
    out_png = ASSETS / "conductor_brand.png"

    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        tex_path = tdir / "conductor_brand.tex"
        tex_path.write_text(TEX.strip() + "\n", encoding="utf-8")
        r = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", f"-output-directory={tdir}", str(tex_path)],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            print(r.stdout + r.stderr, file=sys.stderr)
            print("pdflatex failed. Is the `duerer` package installed (tlmgr / TeX Live)?", file=sys.stderr)
            return 1
        pdf = tdir / "conductor_brand.pdf"
        if not pdf.is_file():
            print("Missing PDF output.", file=sys.stderr)
            return 1
        # -singlefile → exactly conductor_brand.png in ASSETS
        prefix = ASSETS / "conductor_brand"
        subprocess.run(
            ["pdftoppm", "-png", "-r", "300", "-singlefile", str(pdf), str(prefix)],
            check=True,
        )

    if not out_png.is_file():
        print(f"Expected {out_png}", file=sys.stderr)
        return 1
    print(f"Wrote {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
