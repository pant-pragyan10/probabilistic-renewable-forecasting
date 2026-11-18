"""Download the GEFCom2014 solar and wind CSVs into data/raw and record a manifest.

Works on a laptop, Google Colab and Kaggle (needs internet access). If the download is blocked,
place solar_new.csv and wind.csv into data/raw manually and re-run script 01.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from probfc.config import load_config, resolve_path  # noqa: E402
from probfc.data import download_raw, sha256  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-download even if files exist")
    args = ap.parse_args()

    cfg = load_config()
    paths = download_raw(cfg, force=args.force)

    manifest = {"downloaded_on": date.today().isoformat(), "source": cfg["data"]["base_url"], "files": {}}
    for track, p in paths.items():
        manifest["files"][track] = {"file": p.name, "bytes": p.stat().st_size, "sha256": sha256(p)}
        print(f"{track:5s} {p.name:15s} {p.stat().st_size:>10,d} bytes  sha256={manifest['files'][track]['sha256'][:16]}...")

    out = resolve_path(cfg, "results")
    out.mkdir(parents=True, exist_ok=True)
    (out / "raw_data_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest -> {out / 'raw_data_manifest.json'}")


if __name__ == "__main__":
    main()
