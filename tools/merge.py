import argparse
from pathlib import Path
import json

argparser = argparse.ArgumentParser(description='Merge the raw data into a single JSON file.')
argparser.add_argument('-i', '--input', type=Path, help='the input folder.', required=True)
argparser.add_argument('-o', '--output', type=Path, help='the output file.', default='./output.json')
args = argparser.parse_args()

args.output.parent.mkdir(parents=True, exist_ok=True)

data = []

for f in args.input.iterdir():
    if f.is_file():
        with open(f, 'r') as f:
            ds = f.readlines()
            for d in ds:
                data.append(json.loads(d))

with open(args.output, 'w') as f:
    json.dump(data, f, indent=4)
