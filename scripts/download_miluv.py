"""Download the MILUV sensor CSVs without the camera images and rosbags.

Each MILUV experiment is one large zip on Figshare (186 GB in total), almost all
of it images. This reads each zip's central directory with HTTP range requests
and fetches only the .csv/.yaml members (~0.74 GB), plus the devkit's config/
folder (anchor and tag positions, IMU calibration).

    python scripts/download_miluv.py                  # all experiments -> data/miluv
    python scripts/download_miluv.py --only default_1_random3_0
"""

import argparse
import io
import json
import struct
import tarfile
import time
import urllib.error
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ARTICLE = "https://api.figshare.com/v2/articles/28386041"
DEVKIT = "https://codeload.github.com/decargroup/miluv/tar.gz/refs/heads/main"
KEEP = (".csv", ".yaml")


def fetch(url, start=None, end=None, attempts=5):
    """GET a byte range. Figshare redirects to S3 links that expire after 10 s, so
    every request goes through the redirect again rather than reusing a resolved link."""
    headers = {} if start is None else {"Range": f"bytes={start}-{'' if end is None else end}"}
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
                return r.read()
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts - 1:
                raise
            time.sleep(2**attempt)


def zip64_values(extra, *fields):
    """Replace 0xFFFFFFFF placeholders with the values from the zip64 extra field."""
    out, pos = list(fields), 0
    while pos + 4 <= len(extra):
        tag, size = struct.unpack("<HH", extra[pos:pos + 4])
        if tag == 1:
            vals = list(struct.unpack(f"<{size // 8}Q", extra[pos + 4:pos + 4 + size]))
            out = [vals.pop(0) if f == 0xFFFFFFFF else f for f in fields]
            break
        pos += 4 + size
    return out


def members(url, size):
    tail = fetch(url, size - 65536)
    i = tail.rfind(b"PK\x05\x06")
    cd_size, cd_off = struct.unpack("<II", tail[i + 12:i + 20])
    j = tail.rfind(b"PK\x06\x06")
    if j >= 0:
        cd_size, cd_off = struct.unpack("<QQ", tail[j + 40:j + 56])
    cd = fetch(url, cd_off, cd_off + cd_size - 1)
    pos, out = 0, []
    while cd[pos:pos + 4] == b"PK\x01\x02":
        method, = struct.unpack("<H", cd[pos + 10:pos + 12])
        crc, comp, usize = struct.unpack("<III", cd[pos + 16:pos + 28])
        n, m, k = struct.unpack("<HHH", cd[pos + 28:pos + 34])
        offset, = struct.unpack("<I", cd[pos + 42:pos + 46])
        name = cd[pos + 46:pos + 46 + n].decode()
        usize, comp, offset = zip64_values(cd[pos + 46 + n:pos + 46 + n + m], usize, comp, offset)
        out.append(dict(name=name, method=method, crc=crc, comp=comp, size=usize, offset=offset))
        pos += 46 + n + m + k
    return out


def extract(url, member, dest):
    header = fetch(url, member["offset"], member["offset"] + 29)
    n, m = struct.unpack("<HH", header[26:30])
    start = member["offset"] + 30 + n + m
    raw = fetch(url, start, start + member["comp"] - 1) if member["comp"] else b""
    data = zlib.decompress(raw, -15) if member["method"] == 8 else raw
    if zlib.crc32(data) != member["crc"] or len(data) != member["size"]:
        raise IOError(f"CRC/size mismatch for {member['name']}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return len(raw)


def download_experiment(entry, out):
    url, size = entry["download_url"], entry["size"]
    wanted = [m for m in members(url, size) if m["name"].endswith(KEEP)]
    fetched = 0
    for m in wanted:
        dest = out / m["name"]
        if dest.exists() and dest.stat().st_size == m["size"]:
            continue
        fetched += extract(url, m, dest)
    return entry["name"], len(wanted), fetched


def download_devkit_config(out):
    with tarfile.open(fileobj=io.BytesIO(fetch(DEVKIT)), mode="r:gz") as tar:
        for info in tar.getmembers():
            parts = Path(info.name).parts
            if info.isfile() and len(parts) > 2 and parts[1] == "config":
                dest = out / "config" / Path(*parts[2:])
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(tar.extractfile(info).read())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("data/miluv"))
    parser.add_argument("--only", nargs="*", help="experiment names (default: all except calibration)")
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    entries = [f for f in json.loads(fetch(ARTICLE))["files"] if "calib" not in f["name"]]
    if args.only:
        entries = [f for f in entries if f["name"].removesuffix(".zip") in args.only]
    args.out.mkdir(parents=True, exist_ok=True)
    download_devkit_config(args.out)
    print(f"devkit config -> {args.out / 'config'}", flush=True)
    total = 0
    with ThreadPoolExecutor(args.threads) as pool:
        for name, count, fetched in pool.map(lambda e: download_experiment(e, args.out), entries):
            total += fetched
            print(f"{name:45s} {count:3d} files  {fetched / 1e6:7.1f} MB", flush=True)
    print(f"done: {total / 1e9:.2f} GB downloaded")


if __name__ == "__main__":
    main()
