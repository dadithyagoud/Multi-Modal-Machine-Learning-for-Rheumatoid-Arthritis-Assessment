"""
utils.py — Shared utilities for the FusionRA pipeline
======================================================
Classes:
    Logger      — Prints section headers to stdout
    FigureSaver — Saves matplotlib figures to the outputs directory
"""

import os
import matplotlib.pyplot as plt


class Logger:
    """Prints formatted section headers to stdout."""

    @staticmethod
    def section(title: str) -> None:
        print("\n" + "=" * 60)
        print(f"  {title}")
        print("=" * 60)


class FigureSaver:
    """Saves and closes matplotlib figures to a specified output directory."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

    def save(self, name: str, dpi: int = 150) -> None:
        path = os.path.join(self.output_dir, name)
        plt.tight_layout()
        plt.savefig(path, dpi=dpi)
        plt.close()
        print(f"  [saved] {path}")
