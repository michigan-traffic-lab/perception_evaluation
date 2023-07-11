import json
from pathlib import Path
import argparse

argparser = argparse.ArgumentParser(description='Truncate a JSON file into pieces according to each trials start and end time.')
argparser.add_argument('-c', '--config', type=str, help='the configuration of start and end timestamp.')
argparser.add_argument('-i', '--input', type=str, help='the input JSON file.')
argparser.add_argument('-o', '--output', type=str, help='the output folder.')
args = argparser.parse_args()

timestamps = {}
with open(args.config, 'r') as f:
    for line in f:
        tmp = line.strip().split(',')
        # print(tmp)
        timestamps[tmp[0]] = (float(tmp[1]), float(tmp[2]))

with open(args.input, 'r') as f:
    data = json.load(f)

truncated_data = {}

def find_within_timestamp(data):
    timestamp = data['timestamp']
    for k, v in timestamps.items():
        if v[0] <= timestamp and timestamp <= v[1]:
            if k not in truncated_data:
                truncated_data[k] = [data]
            truncated_data[k].append(data)

for d in data:
    find_within_timestamp(d)

Path(args.output).mkdir(parents=True, exist_ok=True)
for k, v in truncated_data.items():
    with open(args.output + '/' + str(k) + '.json', 'w') as f:
        json.dump(v, f, indent=4)
