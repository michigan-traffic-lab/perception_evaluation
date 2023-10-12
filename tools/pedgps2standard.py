import argparse
from pathlib import Path
import json
from datetime import datetime, timezone

argparser = argparse.ArgumentParser(description='convert the ped GPS data into a standard JSON file.')
argparser.add_argument('-i', '--input', type=Path, help='the input folder.', required=True)
argparser.add_argument('-o', '--output', type=Path, help='the output folder.', default='./output')
argparser.add_argument('--input-prefix', type=str, help='the prefix of the input file.', default="gps_box_data")
argparser.add_argument('--output-prefix', type=str, help='the prefix of the output file.', default="ped_gps_data_")
args = argparser.parse_args()
input_folder_path = args.input

args.output.mkdir(parents=True, exist_ok=True)

def find_file_by_prefix(folder, prefix):
    for f in folder.iterdir():
        if f.is_file() and f.name.startswith(prefix):
            return f
    return None

def time_string_to_timestamp(time_string):
    dt = datetime.fromisoformat(time_string)
    dt = dt.replace(tzinfo=timezone.utc)
    return datetime.timestamp(dt)

def convert_ped_file(f):
    with open(f, 'r') as f:
        data = f.readlines()
    data = data[1:]
    converted_data = []
    for line in data:
        tmp = line.strip().split(',')
        timestamp = time_string_to_timestamp(tmp[1])
        lon = float(tmp[2])
        lat = float(tmp[3])
        converted_data.append({
            "timestamp": timestamp,
            "info": "ground_truth",
            'objs': [{"lon": lon,
                      "lat": lat,
                      'id': '0',
                      "classType": "10",}]
        })
    return converted_data


for d in input_folder_path.iterdir():
    if d.is_dir():
        ped_file = find_file_by_prefix(d, args.input_prefix)
        if ped_file is None:
            continue
        data = convert_ped_file(ped_file)
        output_file = args.output_prefix + d.name + '.json'
        with open(args.output / output_file, 'w') as f:
            json.dump(data, f, indent=4)
