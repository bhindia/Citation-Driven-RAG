import tarfile
import urllib.request
import logging
import json
from pathlib import Path
from typing import List

log = logging.getLogger("cd_rag_optionc.utils")

def download_url(url: str, save_path: Path, retries: int = 3):
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if save_path.exists():
        log.info(f"✓ File already exists: {save_path}")
        return save_path
    for attempt in range(retries):
        try:
            log.info(f"⬇️ Downloading {url} (attempt {attempt+1})")
            urllib.request.urlretrieve(url, save_path)
            log.info("✓ Download complete")
            return save_path
        except Exception as e:
            log.warning(f"Attempt {attempt+1} failed: {e}")
    raise RuntimeError(f"Failed to download {url}")

def safe_extract_tar_gz(tar_path: Path, out_dir: Path):
    log.info(f"📦 Extracting {tar_path} -> {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(out_dir)
    log.info("✓ Extracted")

def read_jsonl(path: Path) -> List[dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out
