"""Download and verify the routing-fragility raw data (per-question JSONL).

    python research/fetch_data.py            # into research/data/
    python research/fetch_data.py --dest DIR # elsewhere
    python research/fetch_data.py --check    # verify what is already on disk, download nothing

The JSONL (one row per question per arm, ~175 MB raw) is not in git. It is
published as gzipped assets of a GitHub Release; research/data/SHA256SUMS
lists the SHA-256 of every .gz asset (checked on download) and
research/data/SHA256SUMS.decompressed the SHA-256 of every decompressed file
(checked after decompression and by --check; gzip bytes are not portable
across zlib versions, the decompressed bytes are). Each asset is named after its path
under research/data with "/" replaced by "__", e.g.
p2A__L256_b32_s256_fixed_B32__control.jsonl.gz -> p2A/L256_b32_s256_fixed_B32/control.jsonl.

Standard library only. A file whose hash does not match is not written.
"""

import argparse
import gzip
import hashlib
import os
import sys
import urllib.request

REPO = "Ahmed-Hedhili-se/DMInfr"
TAG = "routing-fragility-data-v1"
URL = f"https://github.com/{REPO}/releases/download/{TAG}/{{name}}"
HERE = os.path.dirname(os.path.abspath(__file__))
SUMS = os.path.join(HERE, "data", "SHA256SUMS")
RAW_SUMS = os.path.join(HERE, "data", "SHA256SUMS.decompressed")


def entries():
    raw = {}
    with open(RAW_SUMS, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                d, rel = line.split()
                raw[rel] = d
    with open(SUMS, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                digest, name = line.split()
                rel = name[: -len(".gz")].replace("__", "/")
                yield digest, raw[rel], name, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.join(HERE, "data"))
    ap.add_argument("--check", action="store_true", help="only verify files already on disk")
    ap.add_argument("--force", action="store_true", help="re-download files that already exist")
    a = ap.parse_args()
    bad = done = 0
    for digest, raw_digest, name, rel in entries():
        target = os.path.join(a.dest, rel)
        if a.check or (os.path.exists(target) and not a.force):
            if not os.path.exists(target):
                print("missing  ", rel)
                bad += 1
                continue
            ok = hashlib.sha256(open(target, "rb").read()).hexdigest() == raw_digest
            print(("ok       " if ok else "MISMATCH ") + rel)
            bad += not ok
            continue
        with urllib.request.urlopen(URL.format(name=name)) as r:
            gz = r.read()
        if hashlib.sha256(gz).hexdigest() != digest:
            print("MISMATCH ", name, "(not written)")
            bad += 1
            continue
        raw = gzip.decompress(gz)
        if hashlib.sha256(raw).hexdigest() != raw_digest:
            print("MISMATCH ", rel, "after decompression (not written)")
            bad += 1
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with open(target, "wb") as f:
            f.write(raw)
        done += 1
        print("fetched  ", rel)
    print(f"\n{done} downloaded, {bad} problem(s)")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
