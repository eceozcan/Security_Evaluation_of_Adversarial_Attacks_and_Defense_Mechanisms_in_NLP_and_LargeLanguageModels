import argparse
import re
from pathlib import Path

import pandas as pd


DEFAULT_BASELINE_DIR = Path(r"C:\Users\HUAWEI\Desktop\log results\internml")
DEFAULT_DEFENSE_DIR = Path(r"C:\Users\HUAWEI\Desktop\Capstone\Defense\Defense Results\internlm-def")
DEFAULT_OUTPUT_CSV = Path(r"C:\Users\HUAWEI\Desktop\capstone project\result\log_folder_comparison.csv")

KNOWN_DATASETS = ["sms_spam", "sst2", "mrpc", "hsol", "rte", "jfleg"]
MODEL_PREFIX = "internlm"


METRIC_PATTERNS = {
    "ASR": re.compile(r"(?:ASV|ASR)\s*=\s*([\d.]+)"),
    "TSR": re.compile(r"(?:PNA-T|TSR)\s*=\s*([\d.]+)"),
    "IRR": re.compile(r"(?:PNA-I|IRR)\s*=\s*([\d.]+)"),
    "MR": re.compile(r"MR\s*=\s*([\d.]+)"),
}


def extract_metrics_from_log(log_path: Path):
    if not log_path.exists():
        return None

    try:
        content = log_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None

    metrics = {}
    for metric_name, pattern in METRIC_PATTERNS.items():
        match = pattern.search(content)
        if match:
            metrics[metric_name] = float(match.group(1))

    return metrics if metrics else None


def parse_log_metadata(log_path: Path):
    """
    Best-effort metadata extraction from the log filename.
    Expected pattern is often:
    model_target_injected_data_num_strategy_defense.txt
    """
    stem = log_path.stem
    model = MODEL_PREFIX if stem.startswith(MODEL_PREFIX) else "unknown"

    remainder = stem[len(MODEL_PREFIX):] if stem.startswith(MODEL_PREFIX) else stem
    remainder = remainder.lstrip("_")

    defense_match = re.search(r"_(adaptive|no)$", remainder)
    defense = defense_match.group(1) if defense_match else "unknown"
    body = remainder[: defense_match.start()] if defense_match else remainder
    body = body.strip("_")

    # Body formats seen in this repo:
    # - hsoljfleg100combine_no
    # - mrpc_hsol_100_combine_no
    # Normalize the trailing task markers first, then extract target/injected.
    strategy = "combine" if body.endswith("_combine") or body.endswith("combine") else "unknown"
    if strategy != "unknown":
        body = body[: -len("_combine")] if body.endswith("_combine") else body[: -len("combine")]
    body = body.strip("_")

    data_num = "100" if body.endswith("_100") or body.endswith("100") else "unknown"
    if data_num != "unknown":
        body = body[: -len("_100")] if body.endswith("_100") else body[: -len("100")]
    body = body.strip("_")

    target = "unknown"
    injected = "unknown"
    search_space = remainder

    for candidate in sorted(KNOWN_DATASETS, key=len, reverse=True):
        if body.startswith(candidate):
            target = candidate
            remainder_after_target = body[len(candidate):].lstrip("_")
            for injected_candidate in sorted(KNOWN_DATASETS, key=len, reverse=True):
                if remainder_after_target.startswith(injected_candidate):
                    injected = injected_candidate
                    break
            if injected == "unknown" and remainder_after_target:
                injected = remainder_after_target
            break

    if injected == "unknown" and search_space:
        for injected_candidate in sorted(KNOWN_DATASETS, key=len, reverse=True):
            if search_space.startswith(injected_candidate):
                injected = injected_candidate
                break
        if injected == "unknown":
            injected = search_space

    return {
        "file_name": log_path.name,
        "file_key": stem,
        "model": model,
        "target": target,
        "injected": injected,
        "data_num": data_num,
        "strategy": strategy,
        "defense": defense,
    }


def collect_logs(folder_path: Path):
    records = []

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    for log_path in sorted(folder_path.rglob("*.txt")):
        metrics = extract_metrics_from_log(log_path)
        if metrics is None:
            continue

        meta = parse_log_metadata(log_path)
        meta["pair_key"] = f"{meta['model']}|{meta['target']}|{meta['injected']}|{meta['data_num']}|{meta['strategy']}"
        meta["folder"] = folder_path.name
        records.append({**meta, **metrics, "log_path": str(log_path)})

    return pd.DataFrame(records)


def build_comparison_table(baseline_df: pd.DataFrame, defense_df: pd.DataFrame):
    if baseline_df.empty:
        raise ValueError("No completed baseline logs found.")
    if defense_df.empty:
        raise ValueError("No completed defense logs found.")

    baseline = baseline_df.rename(
        columns={
            "ASR": "baseline_ASR",
            "TSR": "baseline_TSR",
            "IRR": "baseline_IRR",
            "MR": "baseline_MR",
            "defense": "baseline_defense",
            "folder": "baseline_folder",
        }
    ).copy()

    defense = defense_df.rename(
        columns={
            "ASR": "defense_ASR",
            "TSR": "defense_TSR",
            "IRR": "defense_IRR",
            "MR": "defense_MR",
            "defense": "defense_defense",
            "folder": "defense_folder",
        }
    ).copy()

    baseline_grouped = (
        baseline.groupby("pair_key", as_index=False)
        .agg(
            {
                "model": "first",
                "target": "first",
                "injected": "first",
                "data_num": "first",
                "strategy": "first",
                "baseline_defense": "first",
                "baseline_folder": "first",
                "baseline_ASR": "mean",
                "baseline_TSR": "mean",
                "baseline_IRR": "mean",
                "baseline_MR": "mean",
            }
        )
    )
    baseline_grouped = baseline_grouped.rename(
        columns={
            "model": "model_baseline",
            "target": "target_baseline",
            "injected": "injected_baseline",
            "data_num": "data_num_baseline",
            "strategy": "strategy_baseline",
        }
    )

    defense_grouped = (
        defense.groupby("pair_key", as_index=False)
        .agg(
            {
                "model": "first",
                "target": "first",
                "injected": "first",
                "data_num": "first",
                "strategy": "first",
                "defense_defense": "first",
                "defense_folder": "first",
                "defense_ASR": "mean",
                "defense_TSR": "mean",
                "defense_IRR": "mean",
                "defense_MR": "mean",
            }
        )
    )
    defense_grouped = defense_grouped.rename(
        columns={
            "model": "model_defense",
            "target": "target_defense",
            "injected": "injected_defense",
            "data_num": "data_num_defense",
            "strategy": "strategy_defense",
        }
    )

    merged = baseline_grouped.merge(
        defense_grouped,
        on="pair_key",
        how="outer",
        suffixes=("_baseline", "_defense"),
        indicator=True,
    )

    merged["attack"] = merged.apply(lambda row: f"{row['target_baseline']}→{row['injected_baseline']}", axis=1)

    merged["delta_ASR"] = merged["defense_ASR"] - merged["baseline_ASR"]
    merged["delta_TSR"] = merged["defense_TSR"] - merged["baseline_TSR"]
    merged["delta_IRR"] = merged["defense_IRR"] - merged["baseline_IRR"]
    merged["delta_MR"] = merged["defense_MR"] - merged["baseline_MR"]

    def _reduction(row):
        base = row.get("baseline_ASR")
        defended = row.get("defense_ASR")
        if pd.notna(base) and pd.notna(defended) and base > 0:
            return ((base - defended) / base) * 100
        return None

    merged["ASR_reduction_%"] = merged.apply(_reduction, axis=1)
    return merged


def print_summary_table(comparison_df: pd.DataFrame, output_csv: Path):
    display_cols = [
        "attack",
        "baseline_ASR",
        "defense_ASR",
        "baseline_TSR",
        "defense_TSR",
        "baseline_IRR",
        "defense_IRR",
        "baseline_MR",
        "defense_MR",
        "_merge",
    ]

    available_cols = [col for col in display_cols if col in comparison_df.columns]
    pretty_df = comparison_df[available_cols].copy()

    for col in [c for c in pretty_df.columns if any(tag in c for tag in ["ASR", "TSR", "IRR", "MR", "reduction"])]:
        pretty_df[col] = pd.to_numeric(pretty_df[col], errors="coerce").round(4)

    pretty_df = pretty_df.sort_values([c for c in ["attack"] if c in pretty_df.columns])
    pretty_df = pretty_df[[c for c in [
        "attack",
        "baseline_ASR",
        "defense_ASR",
        "baseline_TSR",
        "defense_TSR",
        "baseline_IRR",
        "defense_IRR",
        "baseline_MR",
        "defense_MR",
    ] if c in pretty_df.columns]]

    print("\n" + "=" * 140)
    print("BASELINE vs DEFENSE COMPARISON")
    print("=" * 140 + "\n")
    # Replace arrow with ASCII-safe version for Windows console
    display_df = pretty_df.copy()
    if "attack" in display_df.columns:
        display_df["attack"] = display_df["attack"].str.replace("→", "->")
    print(display_df.to_string(index=False))
    print("\n" + "=" * 140)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison_df[[c for c in [
        "attack",
        "baseline_ASR",
        "defense_ASR",
        "baseline_TSR",
        "defense_TSR",
        "baseline_IRR",
        "defense_IRR",
        "baseline_MR",
        "defense_MR",
    ] if c in comparison_df.columns]].to_csv(output_csv, index=False)
    print(f"Saved CSV report to: {output_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare two folders of txt logs and generate a baseline vs defense table."
    )
    parser.add_argument(
        "--baseline-dir",
        default=str(DEFAULT_BASELINE_DIR),
        help=f"Folder containing undefended/baseline .txt logs. Default: {DEFAULT_BASELINE_DIR}",
    )
    parser.add_argument(
        "--defense-dir",
        default=str(DEFAULT_DEFENSE_DIR),
        help=f"Folder containing defended .txt logs. Default: {DEFAULT_DEFENSE_DIR}",
    )
    parser.add_argument(
        "--output-csv",
        default=str(DEFAULT_OUTPUT_CSV),
        help="Path to save the merged CSV report.",
    )
    parser.add_argument(
        "--file1",
        default=None,
        help="Path to a single baseline log file to compare (optional)",
    )
    parser.add_argument(
        "--file2",
        default=None,
        help="Path to a single defense log file to compare (optional)",
    )

    args = parser.parse_args()

    baseline_dir = Path(args.baseline_dir).expanduser()
    defense_dir = Path(args.defense_dir).expanduser()
    output_csv = Path(args.output_csv).expanduser()
    file1 = Path(args.file1).expanduser() if args.file1 else None
    file2 = Path(args.file2).expanduser() if args.file2 else None

    # If the filenames in the folders start with a model prefix (e.g. 'llama3'),
    # use the baseline folder name as MODEL_PREFIX so parsing recognizes files.
    global MODEL_PREFIX
    try:
        candidate = baseline_dir.name
        if candidate:
            MODEL_PREFIX = candidate
    except Exception:
        pass

    # If two individual files provided, create a one-row comparison and exit
    if file1 and file2:
        m1 = extract_metrics_from_log(file1)
        m2 = extract_metrics_from_log(file2)
        if m1 is None and m2 is None:
            raise SystemExit(f"No metrics found in either file: {file1}, {file2}")

        meta1 = parse_log_metadata(file1)
        meta2 = parse_log_metadata(file2)

        # Build a single-row comparison DataFrame
        row = {
            "attack": f"{meta1.get('target','unknown')}→{meta1.get('injected','unknown')}",
            "baseline_ASR": m1.get("ASR") if m1 else None,
            "defense_ASR": m2.get("ASR") if m2 else None,
            "baseline_TSR": m1.get("TSR") if m1 else None,
            "defense_TSR": m2.get("TSR") if m2 else None,
            "baseline_IRR": m1.get("IRR") if m1 else None,
            "defense_IRR": m2.get("IRR") if m2 else None,
            "baseline_MR": m1.get("MR") if m1 else None,
            "defense_MR": m2.get("MR") if m2 else None,
            "_merge": "both_files",
        }
        import pandas as _pd

        comparison_df = _pd.DataFrame([row])
        print_summary_table(comparison_df, output_csv)
        return

    baseline_df = collect_logs(baseline_dir)
    defense_df = collect_logs(defense_dir)

    if baseline_df.empty:
        raise SystemExit(f"No completed baseline logs found in: {baseline_dir}")
    if defense_df.empty:
        raise SystemExit(f"No completed defense logs found in: {defense_dir}")

    comparison_df = build_comparison_table(baseline_df, defense_df)
    print_summary_table(comparison_df, output_csv)


if __name__ == "__main__":
    main()