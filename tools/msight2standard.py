import argparse
from pathlib import Path
import json

argparser = argparse.ArgumentParser(description='convert the msight raw data into a standard JSON file.')
argparser.add_argument('-i', '--input', type=Path, help='the input file.', required=True)
argparser.add_argument('-o', '--output', type=Path, help='the output file.', default='./output.json')
args = argparser.parse_args()

data = []

with open(args.input, 'r') as f:
    raw_data = json.load(f)

for d in raw_data:
    t = d['event_timestamp']
    info = d['info']
    objs = []
    for obj in d['results']['fusion']:
        cat = obj['category']
        if int(cat) == 0:
            classType = '2'
        elif int(cat) == 2:
            classType = '10'
        objs.append({
            'id': obj['id'],
            'classType': classType,
            'lon': obj['lon'],
            'lat': obj['lat'],
            'speed': obj['speed'],
        })
    data.append({
        'timestamp': t,
        'info': info,
        'objs': objs,
    })
with open(args.output, 'w') as f:
    json.dump(data, f, indent=4)
