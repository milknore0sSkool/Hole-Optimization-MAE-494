from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

# Canonical data used by the original PINN Burgers example.
URLS = [
    "https://raw.githubusercontent.com/maziarraissi/PINNs/master/appendix/Data/burgers_shock.mat",
    "https://github.com/XinyuanLiao/Adaptive-optimization-of-PINN/raw/refs/heads/main/burgers_shock.mat",
]


def main():
    parser = argparse.ArgumentParser(description="Download the public Burgers PINN reference data.")
    parser.add_argument("--out", default="data/burgers_shock.mat")
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    last_error = None
    for url in URLS:
        try:
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, out)
            print(f"Saved {out} ({out.stat().st_size / 1024:.1f} KiB)")
            return
        except Exception as exc:
            last_error = exc
            print(f"Failed: {exc}")

    raise SystemExit(
        "Automatic download failed. See the optional public-data validation section in README.md "
        f"and save the file as {out}. Last error: {last_error}"
    )


if __name__ == "__main__":
    main()
