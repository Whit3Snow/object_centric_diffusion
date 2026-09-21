"""Download PerAct's pre-generated RLBench demonstrations from Google Drive.

The dataset folder is an old-style Drive folder that needs a `resourceKey`,
which `gdown --folder` does not send - it just returns 401. This script lists
the folder through the Drive v3 API (using the public web API key embedded in
the folder page, plus the resource key) and downloads the zips through the same
API with `alt=media`.

Why not gdown for the files: gdown goes through the interactive
"can't scan for viruses" web page, whose per-IP budget is small - a handful of
parallel multi-GB downloads gets the whole IP throttled for hours, and then
even previously-working files fail. The API path takes HTTP Range requests
instead, so it is both resumable and far less trigger-happy.

Resumable: a zip that already has the expected size is skipped, a partial one
continues from where it stopped.

Examples
--------
    # the 13 SPOT tasks, train + test (~77 GB)
    python tools/download_peract.py --out /home/nas_main/<user>/data/spot/peract/raw

    # one task, then unzip in place
    python tools/download_peract.py --out ... --tasks insert_onto_square_peg --extract

    # just show what would be downloaded
    python tools/download_peract.py --out ... --list
"""

if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).resolve().parent.parent)
    sys.path.append(ROOT_DIR)

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

# PerAct's "rlbench" folder (see peract.github.io / the SPOT README)
FOLDER_ID = "0B2LlLwoO3nfZfkFqMEhXWkxBdjJNNndGYl9uUDQwS1pfNkNHSzFDNGwzd1NnTmlpZXR1bVE"
RESOURCE_KEY = "0-jRw5RaXEYRLe2W6aNrNFEQ"
FOLDER_URL = (f"https://drive.google.com/drive/folders/{FOLDER_ID}"
              f"?resourcekey={RESOURCE_KEY}")

# the task suite SPOT trains on (diffusion_policy_3d/dataset/rlbench_dataset_list.py)
SPOT_TASKS = [
    "meat_off_grill", "put_money_in_safe", "place_wine_at_rack_location",
    "reach_and_drag", "stack_blocks", "close_jar", "light_bulb_in",
    "put_groceries_in_cupboard", "place_shape_in_shape_sorter",
    "insert_onto_square_peg", "stack_cups", "place_cups", "turn_tap",
]

FOLDER_MIME = "application/vnd.google-apps.folder"


UA = "Mozilla/5.0"
_API_KEY = None


def _scrape_api_keys():
    """The folder page embeds the web client's API keys; one of them is allowed
    to call drive.files.list without a referer."""
    req = urllib.request.Request(FOLDER_URL, headers={"User-Agent": UA})
    html = urllib.request.urlopen(req, timeout=60).read().decode("utf-8", "ignore")
    seen, keys = set(), []
    for k in re.findall(r'"(AIza[A-Za-z0-9_\-]{30,40})"', html):
        if k not in seen:
            seen.add(k)
            keys.append(k)
    return keys


def _media_ok(key, file_id):
    """Does this key allow drive.files.get?alt=media (not all of them do)?"""
    req = urllib.request.Request(
        f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={key}",
        headers={"Range": "bytes=0-99", "User-Agent": UA})
    try:
        urllib.request.urlopen(req, timeout=45).read()
        return True
    except Exception:
        return False


def _list(folder_id, key, resource_keys):
    files, page = [], None
    while True:
        q = {"q": f"'{folder_id}' in parents", "key": key, "pageSize": "1000",
             "fields": "nextPageToken,files(id,name,mimeType,size,resourceKey)"}
        if page:
            q["pageToken"] = page
        req = urllib.request.Request(
            "https://www.googleapis.com/drive/v3/files?" + urllib.parse.urlencode(q),
            headers={"X-Goog-Drive-Resource-Keys": resource_keys})
        data = json.load(urllib.request.urlopen(req, timeout=60))
        files += data.get("files", [])
        page = data.get("nextPageToken")
        if not page:
            return files


def build_index():
    """{split: {task: (file_id, size)}} for the whole dataset folder."""
    global _API_KEY
    root_rk = f"{FOLDER_ID}/{RESOURCE_KEY}"
    last_err = None
    for key in _scrape_api_keys():
        try:
            splits = _list(FOLDER_ID, key, root_rk)
        except Exception as e:  # key rejected (referer / method blocked)
            last_err = e
            continue
        _API_KEY = key

        index = {}
        for s in splits:
            if s["mimeType"] != FOLDER_MIME:
                continue
            rk = root_rk + (f",{s['id']}/{s['resourceKey']}" if s.get("resourceKey") else "")
            index[s["name"]] = {
                f["name"][:-4]: (f["id"], int(f.get("size", 0)))
                for f in _list(s["id"], key, rk)
                if f["name"].endswith(".zip")
            }
        return index
    raise RuntimeError(f"no usable Drive API key found on the folder page: {last_err}")


def fetch(file_id, path, size, chunk=8 << 20, retries=8):
    """Download `file_id` to `path`, resuming with a Range request."""
    for attempt in range(retries):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        if have == size:
            return
        if have > size:
            os.remove(path)
            have = 0
        req = urllib.request.Request(
            f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={_API_KEY}",
            headers={"Range": f"bytes={have}-", "User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "ab") as f:
                while True:
                    buf = r.read(chunk)
                    if not buf:
                        break
                    f.write(buf)
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, OSError) as e:
            wait = min(300, 15 * 2 ** attempt)
            print(f"  [retry {attempt+1}/{retries}] {os.path.basename(path)}: {e} "
                  f"- sleeping {wait}s")
            time.sleep(wait)
    got = os.path.getsize(path) if os.path.exists(path) else 0
    if got != size:
        raise RuntimeError(f"{os.path.basename(path)}: got {got} bytes, expected {size}")


def download(task, split, file_id, size, out_root, extract=False):
    out_dir = os.path.join(out_root, split)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{task}.zip")

    if os.path.exists(path) and os.path.getsize(path) == size:
        print(f"[skip] {split}/{task}.zip already complete")
    else:
        print(f"[get ] {split}/{task}.zip ({size/1e9:.2f} GB)", flush=True)
        fetch(file_id, path, size)

    if extract:
        target = os.path.join(out_dir)
        marker = os.path.join(out_dir, task, "all_variations")
        if os.path.isdir(marker):
            print(f"[skip] {split}/{task} already extracted")
        else:
            print(f"[unzip] {split}/{task}")
            with zipfile.ZipFile(path) as z:
                z.extractall(target)
    return f"{split}/{task}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="e.g. <...>/peract/raw")
    ap.add_argument("--tasks", nargs="*", default=SPOT_TASKS)
    ap.add_argument("--splits", nargs="*", default=["train", "test"],
                    help="SPOT uses train + test; val is unused")
    ap.add_argument("--jobs", type=int, default=3,
                    help="keep this low; Drive throttles the whole IP")
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--list", action="store_true", help="only print what would be fetched")
    args = ap.parse_args()

    index = build_index()
    if not _media_ok(_API_KEY, index["test"]["turn_tap"][0]):
        raise SystemExit("the Drive API key found on the folder page cannot download "
                         "files (alt=media blocked); try again later")

    todo, total = [], 0
    for split in args.splits:
        if split not in index:
            raise SystemExit(f"unknown split {split!r}; available: {sorted(index)}")
        for task in args.tasks:
            if task not in index[split]:
                raise SystemExit(f"{split}: no zip for task {task!r}")
            fid, size = index[split][task]
            todo.append((task, split, fid, size))
            total += size

    print(f"{len(todo)} archives, {total/1e9:.1f} GB total")
    if args.list:
        for task, split, fid, size in todo:
            print(f"  {split:5s} {task:32s} {size/1e9:6.2f} GB  {fid}")
        return

    failed = []
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        futs = {ex.submit(download, t, s, f, z, args.out, args.extract): (s, t)
                for t, s, f, z in todo}
        for fut in as_completed(futs):
            split, task = futs[fut]
            try:
                fut.result()
                print(f"[done] {split}/{task}")
            except Exception as e:
                print(f"[FAIL] {split}/{task}: {e}")
                failed.append(f"{split}/{task}")

    if failed:
        raise SystemExit("failed: " + ", ".join(failed))
    print("all downloads complete")


if __name__ == "__main__":
    main()
