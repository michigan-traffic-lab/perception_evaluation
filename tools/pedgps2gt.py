import argparse
from pathlib import Path
import json
from datetime import datetime
import pytz

argparser = argparse.ArgumentParser()
argparser.add_argument('-i', '--input', type=Path, help='the input folder.', required=True)
argparser.add_argument('-o', '--output', type=Path,
                  help='the output folder.', default=Path("./output"))
argparser.add_argument('--output-prefix', type=str, help='the prefix of the output file.', default="ped_gps_data_")
argparser.add_argument('--year', type=int, required=True, help='year of the exp.',)
argparser.add_argument('--month', type=int, required=True, help='month of the exp.',)
argparser.add_argument('--day', type=int, required=True, help='day of the exp.',)
args = argparser.parse_args()

args.output.mkdir(parents=True, exist_ok=True)

def timestamp_to_epoch(timestamp, year, month, day):
    # Function to split timestamp into components
    def split_timestamp(timestamp):
        timestamp_str = str(timestamp)
        hours = int(timestamp_str[:2])
        minutes = int(timestamp_str[2:4])
        seconds = float(timestamp_str[4:])
        return hours, minutes, seconds

    # Split the timestamp into components
    hours, minutes, seconds = split_timestamp(timestamp)

    # Create a datetime object, timezone-aware
    dt = datetime(year, month, day, hours, minutes, int(seconds), int((seconds % 1) * 1_000_000), tzinfo=pytz.UTC)

    # Convert to epoch (Unix time)
    epoch_time = dt.timestamp()

    return epoch_time

def convert_file(file):
    obj_list = []
    with open(file, 'r') as f:
        data = f.readlines()
    for l in data:
        tmp = l.strip().split(',')
        timestamp = float(tmp[0])
        t = timestamp_to_epoch(timestamp, args.year, args.month, args.day)
        obj_list.append({
            'timestamp': t,
            'info': 'ground_truth',
            'objs': [{
                'lon': float(tmp[2]),
                'lat': float(tmp[1]),
                'id': 0,
                'classType': '10',
            }]
        })
    return obj_list


for f in args.input.iterdir():
    if f.is_file():
        objs = convert_file(f)
        with open(args.output / (args.output_prefix + f.stem + '.json'), 'w') as f:
            json.dump(objs, f, indent=4)