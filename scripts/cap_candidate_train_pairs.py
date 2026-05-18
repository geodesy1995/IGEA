import argparse
import csv
import os
from collections import defaultdict


def is_true(value: str) -> bool:
    return str(value).lower() in {"true", "1", "yes"}


def cap_train_pairs(input_path: str, output_path: str, max_false_per_wkid: int) -> dict:
    false_counts = defaultdict(int)
    metrics = {
        "input_rows": 0,
        "output_rows": 0,
        "true_rows": 0,
        "false_rows": 0,
        "dropped_false_rows": 0,
    }

    with open(input_path, "r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{input_path} has no header")

        with open(output_path, "w", encoding="utf-8", newline="") as output_file:
            writer = csv.DictWriter(output_file, fieldnames=reader.fieldnames, delimiter="\t")
            writer.writeheader()

            for row in reader:
                metrics["input_rows"] += 1
                wkid = row.get("wkid", "")
                if is_true(row.get("match", "")):
                    writer.writerow(row)
                    metrics["true_rows"] += 1
                    metrics["output_rows"] += 1
                    continue

                if max_false_per_wkid < 0 or false_counts[wkid] < max_false_per_wkid:
                    writer.writerow(row)
                    false_counts[wkid] += 1
                    metrics["false_rows"] += 1
                    metrics["output_rows"] += 1
                else:
                    metrics["dropped_false_rows"] += 1

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keep all positive candidate pairs and cap false training pairs per KG entity."
    )
    parser.add_argument("input", help="Input train pairs TSV.")
    parser.add_argument("--output", help="Output train pairs TSV. Required unless --in-place is used.")
    parser.add_argument("--max-false-per-wkid", type=int, default=20)
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument("--backup-suffix", default=".uncapped.tsv")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input)
    if args.in_place:
        output_path = input_path + ".tmp"
    elif args.output:
        output_path = os.path.abspath(args.output)
    else:
        raise ValueError("Pass --output or --in-place")

    metrics = cap_train_pairs(input_path, output_path, args.max_false_per_wkid)

    if args.in_place:
        backup_path = input_path + args.backup_suffix
        os.replace(input_path, backup_path)
        os.replace(output_path, input_path)
        metrics["backup_path"] = backup_path

    print(metrics)


if __name__ == "__main__":
    main()
