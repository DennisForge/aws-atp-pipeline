import os
import io
import csv
import json
import tempfile
import zipfile
from datetime import datetime
from typing import Tuple, Optional, List

import boto3
import requests
from botocore.exceptions import ClientError

# -----------------------------
# Config (environment variables)
# -----------------------------
# S3_BUCKET: target S3 bucket name
# RAW_PREFIX: prefix where original/raw files live (e.g., "raw/")
# RESULTS_PREFIX: prefix where processed outputs are stored (e.g., "results/")
# RAW_OBJECT: expected CSV filename inside RAW_PREFIX (used to detect if we already have data)
# SECRET_ARN: AWS Secrets Manager ARN that stores Kaggle API credentials
# DATASET_SLUG: Kaggle dataset slug, used to download the ZIP when RAW_OBJECT is missing
S3_BUCKET       = os.environ.get("S3_BUCKET", "atp-analysis-your-bucket-name")
RAW_PREFIX      = os.environ.get("RAW_PREFIX", "raw/")
RESULTS_PREFIX  = os.environ.get("RESULTS_PREFIX", "results/")
RAW_OBJECT      = os.environ.get("RAW_OBJECT", "atp_tennis.csv")
SECRET_ARN      = os.environ["SECRET_ARN"]  # arn:aws:secretsmanager:...:secret:kaggle/credentials-...
DATASET_SLUG    = os.environ.get("DATASET_SLUG", "dissfya/atp-tennis-2000-2023daily-pull")

# -----------------------------
# AWS/Kaggle clients and session
# -----------------------------
# s3: used to read/write objects
# sm: used to read secrets (Kaggle credentials)
# SESSION: shared HTTP session for Kaggle API calls
# TIMEOUT: (connect_timeout, read_timeout) for HTTP requests
s3 = boto3.client("s3")
sm = boto3.client("secretsmanager")
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "lambda-atp-pipeline/1.0"})
TIMEOUT = (10, 120)

# -----------------------------
# Tournament classification aids
# -----------------------------
# These sets/lookup maps help classify tournament level (GS, M1000, ATP500)
# even if the input columns differ slightly across datasets.
_GRAND_SLAMS = {
    "australian open", "roland garros", "french open", "wimbledon", "us open"
}
_MASTERS_HINTS = {
    "indian wells", "miami", "monte-carlo", "madrid", "rome", "italian open",
    "canadian open", "montreal", "toronto", "cincinnati", "shanghai", "paris"
}
_LEVEL_CODE_MAP = {"g": "GS", "m": "M1000", "a": "ATP500", "500": "ATP500", "1000": "M1000"}

# -----------------------------
# Secrets / Kaggle helpers
# -----------------------------
def load_kaggle_creds(secret_arn: str) -> Tuple[str, str]:
    """
    Read Kaggle credentials from Secrets Manager.
    The secret JSON must contain "KAGGLE_USERNAME" and "KAGGLE_KEY".
    Returns (username, key). Raises if missing or Secrets Manager fails.
    """
    try:
        resp = sm.get_secret_value(SecretId=secret_arn)
        data = json.loads(resp["SecretString"])
        username = data["KAGGLE_USERNAME"]
        key = data["KAGGLE_KEY"]
        if not username or not key:
            raise KeyError("Empty KAGGLE_USERNAME/KAGGLE_KEY")
        return username, key
    except ClientError as e:
        raise RuntimeError(f"SecretsManager error: {e}") from e
    except KeyError as e:
        raise RuntimeError("Secret must contain KAGGLE_USERNAME and KAGGLE_KEY") from e


def kaggle_download_zip(dataset_slug: str, auth: Tuple[str, str]) -> str:
    """
    Download the Kaggle dataset ZIP to a temporary file using Basic Auth.
    Returns the local path to the downloaded ZIP.
    Raises PermissionError for 403 (license not accepted / wrong account).
    """
    url = f"https://www.kaggle.com/api/v1/datasets/download/{dataset_slug}"
    tmp_zip = os.path.join(tempfile.gettempdir(), "kaggle_dataset.zip")

    with SESSION.get(url, auth=auth, stream=True, timeout=TIMEOUT) as r:
        if r.status_code == 403:
            # Most common cause: dataset terms not accepted for that account or mismatched token.
            raise PermissionError("Kaggle 403: accept dataset terms or align token with the account.")
        r.raise_for_status()
        with open(tmp_zip, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    return tmp_zip


def unzip_upload_select_csv(zip_path: str, bucket: str, prefix: str, preferred_name: str) -> str:
    """
    Extract CSV files from the ZIP and upload them into S3 under 'prefix'.
    If a file named 'preferred_name' exists, prefer that one; otherwise pick the first CSV.
    Returns the S3 key of the chosen CSV for downstream processing.
    """
    uploaded: List[str] = []
    chosen_key: Optional[str] = None
    with zipfile.ZipFile(zip_path) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            if not name.lower().endswith(".csv"):
                continue
            key = f"{prefix}{os.path.basename(name)}"
            with zf.open(info) as src:
                s3.upload_fileobj(src, bucket, key)
            uploaded.append(key)
            if os.path.basename(name) == preferred_name:
                chosen_key = key

    if chosen_key:
        return chosen_key
    if uploaded:
        return uploaded[0]
    raise RuntimeError("ZIP did not contain any CSV files.")

# -----------------------------
# S3 helpers
# -----------------------------
def s3_key_exists(bucket: str, key: str) -> bool:
    """
    Lightweight check if an S3 object exists (HEAD request).
    Returns True if present, False if not found.
    """
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def get_or_download_raw_csv() -> str:
    """
    Ensure we have a raw CSV in S3.
    - If RAW_OBJECT already exists under RAW_PREFIX, return its key.
    - Otherwise, download the Kaggle ZIP, upload CSV(s), and return the chosen CSV key.
    """
    wanted_key = f"{RAW_PREFIX}{RAW_OBJECT}"
    if s3_key_exists(S3_BUCKET, wanted_key):
        return wanted_key
    auth = load_kaggle_creds(SECRET_ARN)
    zip_path = kaggle_download_zip(DATASET_SLUG, auth)
    chosen_key = unzip_upload_select_csv(zip_path, S3_BUCKET, RAW_PREFIX, RAW_OBJECT)
    return chosen_key


def presign(key: str, seconds: int = 7*24*3600) -> str:
    """
    Generate a pre-signed GET URL for an S3 object.
    Default expiry is 7 days (604800 seconds).
    """
    return s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=seconds,
    )

# -----------------------------
# CSV schema detection & parsing
# -----------------------------
def _lc(x): 
    """Lowercase helper that handles None and trims whitespace."""
    return (x or "").strip().lower()


def detect_cols(header: list) -> dict:
    """
    Map flexible header names from the dataset to canonical names we use.
    Returns a dict with original column names for:
    'winner', 'tourney_name', 'tourney_level', 'date'
    Some are optional; 'winner' and 'date' are required for counting and timeline.
    """
    h = {c.lower(): c for c in header}

    def pick(*cands):
        for c in cands:
            if c.lower() in h:
                return h[c.lower()]
        return None

    return {
        "winner":       pick("winner_name", "winner", "winner_player_name", "winner_full_name"),
        "tourney_name": pick("tourney_name", "tournament_name", "event_name"),
        "tourney_level":pick("tourney_level", "level", "tournament_level", "tournament_category", "series"),
        "date":         pick("match_date", "date", "tourney_date", "tournament_date"),
    }


def parse_date(raw: str):
    """
    Try multiple common formats:
    - 8-digit YYYYMMDD
    - ISO date YYYY-MM-DD
    - common day-first or month-first formats
    - year-only (YYYY)
    Returns a date or None if parsing fails.
    """
    s = (raw or "").strip()
    if not s:
        return None
    if s.isdigit() and len(s) == 8:
        try:
            return datetime.strptime(s, "%Y%m%d").date()
        except ValueError:
            pass
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%m/%d/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    if s.isdigit() and len(s) == 4:
        return datetime(int(s), 1, 1).date()
    return None


def classify_level(level_val: str, tname_val: str) -> str:
    """
    Normalize tournament level into one of:
    - "GS" (Grand Slam)
    - "M1000" (Masters 1000)
    - "ATP500"
    - "OTHER"
    Uses both the level column and tournament name hints.
    """
    lv, tn = _lc(level_val), _lc(tname_val)
    if lv in _LEVEL_CODE_MAP:
        return _LEVEL_CODE_MAP[lv]
    if "grand" in lv and "slam" in lv:
        return "GS"
    if "masters" in lv or "1000" in lv:
        return "M1000"
    if "500" in lv:
        return "ATP500"
    if any(g in tn for g in _GRAND_SLAMS):
        return "GS"
    if any(m in tn for m in _MASTERS_HINTS):
        return "M1000"
    if "500" in tn:
        return "ATP500"
    return "OTHER"

# -----------------------------
# Transform: compute Top 50 by wins
# -----------------------------
def transform_top50_from_s3_csv(key: str) -> List[List[str]]:
    """
    Read the raw CSV from S3, detect required columns, and aggregate per-player wins.
    For each player we count:
      - total wins
      - GS wins
      - Masters 1000 wins
      - ATP 500 wins
      - first win date
      - last win date
    Return the top 50 rows sorted by total wins desc, then player_name asc.
    """
    obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
    body = io.TextIOWrapper(obj["Body"], encoding="utf-8", newline="")
    reader = csv.reader(body)

    header = next(reader)
    cols = detect_cols(header)

    # winner + date are required to build correct stats and timeline
    missing = [k for k, v in cols.items() if v is None and k in ("winner", "date")]
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}; header={header}")

    # Build index map: canonical_name -> column index
    idx = {name: header.index(col) if col else None for name, col in cols.items() if col}

    # Aggregation map: player -> stats dict
    agg = {}
    def ensure(p):
        if p not in agg:
            agg[p] = {
                "total_wins": 0,
                "grand_slem_wins": 0,
                "atp1000_wins": 0,
                "atp500_wins": 0,
                "first_win": None,
                "last_win": None
            }
        return agg[p]

    # Iterate dataset rows, be tolerant to bad lines
    for row in reader:
        try:
            winner = (row[idx["winner"]] or "").strip()
            if not winner:
                continue

            d = parse_date(row[idx["date"]]) if "date" in idx and idx["date"] is not None else None
            tname = row[idx["tourney_name"]].strip() if "tourney_name" in idx and idx["tourney_name"] is not None else ""
            tlevel = row[idx["tourney_level"]].strip() if "tourney_level" in idx and idx["tourney_level"] is not None else ""
            tag = classify_level(tlevel, tname)

            a = ensure(winner)
            a["total_wins"] += 1

            if d:
                if a["first_win"] is None or d < a["first_win"]:
                    a["first_win"] = d
                if a["last_win"] is None or d > a["last_win"]:
                    a["last_win"] = d

            if tag == "GS":
                a["grand_slem_wins"] += 1
            elif tag == "M1000":
                a["atp1000_wins"] += 1
            elif tag == "ATP500":
                a["atp500_wins"] += 1
        except Exception:
            # Defensive: skip malformed lines instead of failing the whole run.
            continue

    # Build final rows for CSV output
    items = []
    for player, a in agg.items():
        items.append([
            player,
            a["total_wins"],
            a["grand_slem_wins"],
            a["atp1000_wins"],
            a["atp500_wins"],
            a["first_win"].isoformat() if a["first_win"] else "",
            a["last_win"].isoformat() if a["last_win"] else "",
        ])

    # Sort by total wins (desc), then player name (asc), and keep top 50
    items.sort(key=lambda r: (-r[1], r[0]))
    return items[:50]

# -----------------------------
# Persist processed results
# -----------------------------
def write_results(rows: List[List[str]]) -> str:
    """
    Write the Top 50 table to S3 as CSV under RESULTS_PREFIX.
    Filename is date-stamped: atp-top-50-DD-MM-YYYY.csv
    Returns the S3 key of the written object.
    """
    out = io.StringIO()
    w = csv.writer(out)
    w.writerow(["player_name","total_wins","grand_slem_wins","atp1000_wins","atp500_wins","first_win","last_win"])
    for r in rows:
        w.writerow(r)

    stamp = datetime.utcnow().strftime("%d-%m-%Y")
    out_key = f"{RESULTS_PREFIX}atp-top-50-{stamp}.csv"
    s3.put_object(Bucket=S3_BUCKET, Key=out_key, Body=out.getvalue().encode("utf-8"))
    return out_key

# -----------------------------
# Lambda entry point
# -----------------------------
def lambda_handler(event, context):
    """
    Orchestrates the full flow:
      1) Ensure raw CSV is present in S3 (download from Kaggle if missing)
      2) Transform to Top 50 table
      3) Write results to S3 under RESULTS_PREFIX
      4) Return a 7-day pre-signed URL for convenience
    """
    # 1) Acquire or download the raw CSV
    csv_key = get_or_download_raw_csv()

    # 2) Compute Top 50 from the raw CSV
    top50 = transform_top50_from_s3_csv(csv_key)

    # 3) Persist the result CSV into S3
    out_key = write_results(top50)

    # 4) Convenience URL for easy sharing/download (expires in 7 days)
    return {
        "status": "ok",
        "raw_key": csv_key,
        "output_key": out_key,
        "rows": len(top50),
        "presigned_url_7d": presign(out_key),
        # If needed, also expose a link to the raw CSV:
        # "raw_presigned_7d": presign(csv_key),
    }
