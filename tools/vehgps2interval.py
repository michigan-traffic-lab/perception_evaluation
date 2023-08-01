from pathlib import Path
import argparse
import re
import json

args = argparse.ArgumentParser(
    description='Convert the GPS data to standard format.')
args.add_argument('-i', '--input', type=Path, help='the input folder.')
args.add_argument('-o', '--output', type=Path,
                  help='the output file.', default=Path("./timestamps.csv"))
args.add_argument('--file-prefix', type=str,
                  help='the prefix of the vehicle 1 file.', default="mkz_data")

args = args.parse_args()

def find_file_by_prefix(folder, prefix):
    for f in folder.iterdir():
        if f.is_file() and f.name.startswith(prefix):
            return f
    return None

input_folder_path = args.input
for d in input_folder_path.iterdir():
    if d.is_dir():
        veh1_file = find_file_by_prefix(d, args.file_prefix)
        if veh1_file is None:
            continue
        with open(veh1_file, 'r') as f:
            data = f.readlines()
        first_timestamp = data[1].strip().split(',')[1]
        last_timestamp = data[-1].strip().split(',')[1]
        with open(args.output, 'a') as f:
            f.write(f'{d.name},{first_timestamp},{last_timestamp}\n')
