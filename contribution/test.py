import csv
import os
from pathlib import Path

csv_file = "results/aggregated_results.csv"

with open(csv_file, newline="") as f:
    reader = csv.reader(f)
    for row in reader:
        first_three = row[:3]
        print(first_three)