import csv
import os

input_path = os.path.join(os.path.dirname(
    __file__), "..", "data", "YCSEP_static.csv")
tdk_output_path = os.path.join(os.path.dirname(
    __file__), "..", "data", "TDK_subset.csv")
non_tdk_output_path = os.path.join(os.path.dirname(
    __file__), "..", "data", "non_TDK_subset.csv")

with (
    open(input_path, "r", newline="") as infile,
    open(tdk_output_path, "w", newline="") as tdk_outfile,
    open(non_tdk_output_path, "w", newline="") as non_tdk_outfile,
):
    reader = csv.reader(infile)
    tdk_writer = csv.writer(tdk_outfile)
    non_tdk_writer = csv.writer(non_tdk_outfile)
    header = next(reader)
    tdk_writer.writerow(header)
    non_tdk_writer.writerow(header)
    channel_idx = header.index("channel")
    tdk_count = 0
    non_tdk_count = 0
    for row in reader:
        if row[channel_idx] == "The_Daily_Ketchup_Podcast":
            tdk_writer.writerow(row)
            tdk_count += 1
        else:
            non_tdk_writer.writerow(row)
            non_tdk_count += 1

print(f"Wrote {tdk_count} rows to {tdk_output_path}")
print(f"Wrote {non_tdk_count} rows to {non_tdk_output_path}")
