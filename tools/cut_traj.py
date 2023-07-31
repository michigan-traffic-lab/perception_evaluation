import argparse
import json
from pathlib import Path

argparser = argparse.ArgumentParser()
argparser.add_argument('-i', '--input-file', type=Path, required=True, help='the input file.')
argparser.add_argument('-o', '--output-file', type=Path, default='output.json', help='the output file.')
argparser.add_argument('-c', '--cut-ids', type=str, nargs='+', required=True, help='the ids to be cut.')
argparser.add_argument('-t', '--times', type=float, nargs='+', required=True, help='the times to be cut.')
args = argparser.parse_args()

with open(args.input_file, 'r') as f:
    data = json.load(f)

cut_ids = args.cut_ids
times = args.times
assert len(cut_ids) == len(times), 'the length of cut ids and times should be the same.'
for frame in data:
    t = float(frame['timestamp'])
    for i in range(len(cut_ids)):
        for oid in frame['objs']:
            if str(oid['id']) == cut_ids[i] and t >= times[i]:
                oid['id'] = str(oid['id'])+'-cut'

with open(args.output_file, 'w') as f:
    json.dump(data, f, indent=4)
