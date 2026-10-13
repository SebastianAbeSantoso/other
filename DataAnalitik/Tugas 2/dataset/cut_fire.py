import csv
import os
import pandas as pd

def ifExists(filename):
    return filename if os.path.exists(filename) else None

original_file = ifExists("fire.csv")

part1_file = "fire_part1.csv"
part2_file = "fire_part2.csv"

combined_file = "fire_combined.csv"

if original_file:
    print("=" * 70)
    print("SPLITTING FIRE.CSV")
    print("=" * 70)

    with open(original_file, "r", encoding="utf-8", newline="") as f:
        total_rows = sum(1 for _ in f) - 1  
    split_at = total_rows // 2

    print(f"Total rows : {total_rows:,}")
    print(f"Part 1     : {split_at:,}")
    print(f"Part 2     : {total_rows - split_at:,}")

    with open(original_file, "r", encoding="utf-8", newline="") as infile:
        reader = csv.reader(infile)
        header = next(reader)

        with open(part1_file, "w", encoding="utf-8", newline="") as f1, \
            open(part2_file, "w", encoding="utf-8", newline="") as f2:

            writer1 = csv.writer(f1)
            writer2 = csv.writer(f2)

            writer1.writerow(header)
            writer2.writerow(header)

            for i, row in enumerate(reader):
                if i < split_at:
                    writer1.writerow(row)
                else:
                    writer2.writerow(row)

    print("\nSplit complete.")

    print(
        f"{part1_file}: "
        f"{os.path.getsize(part1_file) / (1024**2):.2f} MiB"
    )

    print(
        f"{part2_file}: "
        f"{os.path.getsize(part2_file) / (1024**2):.2f} MiB"
    )

if not original_file:
    print("\n" + "=" * 70)
    print("COMBINING PARTS")
    print("=" * 70)

    part1 = pd.read_csv(part1_file)
    part2 = pd.read_csv(part2_file)

    if list(part1.columns) != list(part2.columns):
        raise ValueError("Column names/order do not match between the two files.")

    combined = pd.concat([part1, part2], ignore_index=True)

    combined.to_csv(combined_file, index=False)

    print(f"Combined rows   : {len(combined):,}")
    print(f"Combined columns: {len(combined.columns)}")
    print(f"Saved as        : {combined_file}")

    print(
        f"Combined size   : "
        f"{os.path.getsize(combined_file) / (1024**2):.2f} MiB"
    )

    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)

    original = pd.read_csv(original_file)

    print("Original shape :", original.shape)
    print("Combined shape :", combined.shape)

    if original.shape == combined.shape:
        print("✓ Row/column count matches.")
    else:
        print("✗ Shape does not match.")

    if original.equals(combined):
        print("✓ Combined dataset matches the original dataset.")
    else:
        print("⚠ Combined dataset differs from the original dataset.")