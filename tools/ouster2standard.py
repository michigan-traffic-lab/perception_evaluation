import argparse
from pathlib import Path
import json
from datetime import datetime, timezone
import time

argparser = argparse.ArgumentParser(description='Merge the raw data into a single JSON file.')
argparser.add_argument('-i', '--input', type=Path, help='the JSON file', required=True)
argparser.add_argument('-o', '--output', type=Path, help='the output JSON file', default='./output.json')
argparser.add_argument('--output-prefix', type=str, help='the prefix of the output file.', default="ped_gps_data_")
args = argparser.parse_args()


def time_string_to_timestamp(time_string):
    dt = datetime.fromisoformat(time_string)
    dt = dt.replace(tzinfo=timezone.utc)
    return datetime.timestamp(dt)


with open(args.input, 'r') as f:
    raw_data = json.load(f)

converted_data = []
for d in raw_data:
    cd = {}
    cd['timestamp'] = time_string_to_timestamp(d['sensing_timestamp'])
    cd['info'] = d['info']
    cd['objs'] = []
    for obj in d['data']:
        cd['objs'].append({
            'id': obj['id'],
            'classType': obj['classType'],
            'lon': obj['centerLon'],
            'lat': obj['centerLat'],
            'speed': obj['speed'],
            'rotation': obj['rotation'],
            'accuracy': obj['accuracy'],
        })
    converted_data.append(cd)
with open(args.output, 'w') as f:
    json.dump(converted_data, f, indent=4)
