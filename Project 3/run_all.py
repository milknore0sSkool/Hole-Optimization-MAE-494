from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
subprocess.run([sys.executable, str(ROOT / "src" / "run_experiments.py")], cwd=ROOT, check=True)
subprocess.run([sys.executable, str(ROOT / "src" / "make_plots.py")], cwd=ROOT, check=True)
print("Done. See results/ and figures/.")
