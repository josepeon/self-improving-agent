#!/usr/bin/env python3
"""
Sync SIA interaction logs from Railway to local Mac.

Railway's filesystem is ephemeral — logs are lost on redeploy.
The companion backend prints SIA logs to stdout with a "SIA_LOG:" prefix.
This script pulls those lines via `railway service logs` and stores them
locally as JSONL for training.

Usage:
    # Sync last 24 hours (default)
    python scripts/sync_logs.py

    # Sync last 7 days
    python scripts/sync_logs.py --since 7d

    # Sync specific time range
    python scripts/sync_logs.py --since 2026-02-20T00:00:00Z --until 2026-02-26T00:00:00Z

    # Dry run — show what would be synced
    python scripts/sync_logs.py --dry-run

Requirements:
    - Railway CLI installed and authenticated (`railway login`)
    - CLI linked to oh-companion project (`railway link` in companion dir)
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# Default output directory (gitignored)
DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "data" / "companion_logs"

# Prefix used by sia_logging.py to tag stdout lines
SIA_LOG_PREFIX = "SIA_LOG:"

# Railway project details (service name must match Railway dashboard)
RAILWAY_SERVICE = "oh-companion-backend"
RAILWAY_ENVIRONMENT = "production"


def run_railway_logs(
    *,
    since: str = "24h",
    until: str | None = None,
    lines: int | None = None,
    companion_dir: str | None = None,
) -> list[str]:
    """Pull raw log lines from Railway via CLI.

    Args:
        since: How far back to fetch (e.g., '24h', '7d', ISO timestamp).
        until: Optional end time for the window.
        lines: If set, fetch exactly N lines (overrides since/until).
        companion_dir: Path to oh-companion-backend repo (for railway context).

    Returns:
        List of raw log line strings.
    """
    cmd = [
        "railway",
        "service",
        "logs",
        "--service",
        RAILWAY_SERVICE,
        "--environment",
        RAILWAY_ENVIRONMENT,
        "--json",
    ]

    if lines:
        cmd.extend(["--lines", str(lines)])
    else:
        cmd.extend(["--since", since])
        if until:
            cmd.extend(["--until", until])
        # When no --until and no --lines, use --until=now to disable streaming
        # Railway needs either --lines or bounded time range to not stream
        if not until:
            cmd.extend(["--until", "0s"])

    # Run from companion dir if provided (for railway link context)
    cwd = companion_dir or os.environ.get(
        "RAILWAY_PROJECT_DIR",
        str(Path(__file__).parent.parent.parent / "oh-companion-backend"),
    )

    print(f"  Running: {' '.join(cmd)}")
    print(f"  Working dir: {cwd}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=120,
        )
    except FileNotFoundError:
        print("ERROR: Railway CLI not found. Install with: brew install railway")
        sys.exit(1)
    except subprocess.TimeoutExpired:
        print("ERROR: Railway logs command timed out (120s)")
        sys.exit(1)

    if result.returncode != 0:
        stderr = result.stderr.strip()
        if "Unauthorized" in stderr or "login" in stderr.lower():
            print(
                "ERROR: Railway CLI not authenticated. Run: railway login --browserless"
            )
        elif "No linked project" in stderr:
            print(f"ERROR: Railway not linked. Run: cd {cwd} && railway link")
        else:
            print(f"ERROR: Railway CLI failed (exit {result.returncode})")
            print(f"  stderr: {stderr}")
        sys.exit(1)

    return result.stdout.strip().split("\n") if result.stdout.strip() else []


def extract_sia_entries(raw_lines: list[str]) -> list[dict]:
    """Extract SIA interaction entries from Railway JSON log lines.

    Railway --json output gives one JSON object per log line with a 'message' field.
    We look for lines where the message starts with SIA_LOG_PREFIX.

    Args:
        raw_lines: Raw JSON log lines from Railway CLI.

    Returns:
        List of parsed SIA interaction dictionaries.
    """
    entries = []
    parse_errors = 0

    for line in raw_lines:
        if not line.strip():
            continue

        try:
            log_obj = json.loads(line)
        except json.JSONDecodeError:
            # Some lines might not be JSON (e.g., Railway metadata)
            # Try direct SIA_LOG: prefix match on raw text
            if SIA_LOG_PREFIX in line:
                idx = line.index(SIA_LOG_PREFIX)
                json_str = line[idx + len(SIA_LOG_PREFIX) :]
                try:
                    entry = json.loads(json_str)
                    entries.append(entry)
                except json.JSONDecodeError:
                    parse_errors += 1
            continue

        # Railway JSON format: {"message": "...", "timestamp": "...", ...}
        message = log_obj.get("message", "")

        if isinstance(message, str) and message.startswith(SIA_LOG_PREFIX):
            json_str = message[len(SIA_LOG_PREFIX) :]
            try:
                entry = json.loads(json_str)
                entries.append(entry)
            except json.JSONDecodeError:
                parse_errors += 1

    if parse_errors:
        print(f"  Warning: {parse_errors} SIA log lines failed to parse")

    return entries


def deduplicate_entries(
    new_entries: list[dict],
    existing_file: Path,
) -> list[dict]:
    """Remove entries that already exist in the local file.

    Uses (timestamp, user_message) as dedup key.

    Args:
        new_entries: Newly fetched entries.
        existing_file: Path to the existing JSONL file.

    Returns:
        List of entries not already present locally.
    """
    if not existing_file.exists():
        return new_entries

    existing_keys = set()
    with open(existing_file) as f:
        for line in f:
            if line.strip():
                try:
                    entry = json.loads(line)
                    key = (entry.get("timestamp", ""), entry.get("user_message", ""))
                    existing_keys.add(key)
                except json.JSONDecodeError:
                    continue

    deduped = []
    for entry in new_entries:
        key = (entry.get("timestamp", ""), entry.get("user_message", ""))
        if key not in existing_keys:
            deduped.append(entry)

    skipped = len(new_entries) - len(deduped)
    if skipped:
        print(f"  Skipped {skipped} duplicate entries")

    return deduped


def save_entries(entries: list[dict], output_dir: Path) -> Path:
    """Append entries to a date-partitioned JSONL file.

    Args:
        entries: SIA interaction entries.
        output_dir: Directory for output files.

    Returns:
        Path to the output file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Group by date from timestamp
    by_date: dict[str, list[dict]] = {}
    for entry in entries:
        ts = entry.get("timestamp", "")
        date_str = ts[:10] if len(ts) >= 10 else datetime.now(UTC).strftime("%Y-%m-%d")
        by_date.setdefault(date_str, []).append(entry)

    files_written = []
    for date_str, date_entries in sorted(by_date.items()):
        output_file = output_dir / f"{date_str}.jsonl"

        # Deduplicate against existing file
        date_entries = deduplicate_entries(date_entries, output_file)

        if not date_entries:
            continue

        with open(output_file, "a") as f:
            for entry in date_entries:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        files_written.append((output_file, len(date_entries)))

    return files_written


def print_summary(entries: list[dict]) -> None:
    """Print a summary of synced interactions."""
    if not entries:
        print("\n  No SIA interactions found in the time window.")
        return

    models = {}
    latencies = []
    fallbacks = 0

    for e in entries:
        model = e.get("model", "unknown")
        models[model] = models.get(model, 0) + 1
        if e.get("latency_ms"):
            latencies.append(e["latency_ms"])
        if e.get("is_fallback"):
            fallbacks += 1

    print(f"\n  Interactions found: {len(entries)}")
    print(f"  Models used: {models}")
    if latencies:
        avg_lat = sum(latencies) / len(latencies)
        print(
            f"  Avg latency: {avg_lat:.0f}ms (min: {min(latencies):.0f}, max: {max(latencies):.0f})"
        )
    print(f"  Fallbacks: {fallbacks}")

    # Time range
    timestamps = [e.get("timestamp", "") for e in entries if e.get("timestamp")]
    if timestamps:
        print(f"  Time range: {min(timestamps)} → {max(timestamps)}")


def main():
    parser = argparse.ArgumentParser(
        description="Sync SIA interaction logs from Railway to local Mac",
    )
    parser.add_argument(
        "--since",
        default="24h",
        help="How far back to fetch (e.g., 24h, 7d, 2026-02-20T00:00:00Z). Default: 24h",
    )
    parser.add_argument(
        "--until",
        default=None,
        help="End of time window (same formats as --since). Default: now",
    )
    parser.add_argument(
        "--lines",
        type=int,
        default=None,
        help="Fetch exactly N log lines (overrides --since/--until)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory for JSONL files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--companion-dir",
        default=None,
        help="Path to oh-companion-backend repo (for Railway CLI context)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be synced without writing files",
    )
    args = parser.parse_args()

    print(f"{'=' * 60}")
    print("SIA Log Sync — Railway → Local")
    print(f"{'=' * 60}")
    print(f"  Since: {args.since}")
    if args.until:
        print(f"  Until: {args.until}")
    print(f"  Output: {args.output_dir}")
    print()

    # Step 1: Pull raw logs from Railway
    print("[1/3] Fetching logs from Railway...")
    raw_lines = run_railway_logs(
        since=args.since,
        until=args.until,
        lines=args.lines,
        companion_dir=args.companion_dir,
    )
    print(f"  Retrieved {len(raw_lines)} raw log lines")

    # Step 2: Extract SIA entries
    print("\n[2/3] Extracting SIA interactions...")
    entries = extract_sia_entries(raw_lines)
    print_summary(entries)

    if not entries:
        print("\nDone — nothing to sync.")
        return

    # Step 3: Save locally
    if args.dry_run:
        print(
            f"\n[3/3] DRY RUN — would save {len(entries)} entries to {args.output_dir}/"
        )
        print("\nSample entry:")
        print(json.dumps(entries[0], indent=2, ensure_ascii=False))
    else:
        print(f"\n[3/3] Saving to {args.output_dir}/...")
        files_written = save_entries(entries, args.output_dir)
        if files_written:
            for fpath, count in files_written:
                print(f"  {fpath.name}: +{count} entries")
        else:
            print("  All entries already synced (duplicates skipped)")

    print(f"\n{'=' * 60}")
    print("Sync complete!")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
