import argparse
from pathlib import Path
import json

argparser = argparse.ArgumentParser()
argparser.add_argument('-f', '--file', type=Path, required=True)
argparser.add_argument('-t', '--time-file', type=Path, required=True)
argparser.add_argument('-o', '--output', type=Path, default=Path('./output.json'))
args = argparser.parse_args()

def interpolate(t1, t2, t2_next, lat2, lon2, lat2_next, lon2_next):
    # Calculate the fraction of the total time that has passed since t2
    fraction = (t1 - t2) / (t2_next - t2)

    # Use this fraction to calculate the interpolated latitude and longitude
    lat1 = lat2 + fraction * (lat2_next - lat2)
    lon1 = lon2 + fraction * (lon2_next - lon2)

    return lat1, lon1

def interpolate_data(data, times):
    p = 0
    converted_data = []
    for t in times:
        if p + 1 > len(data) - 1:
            break
        if data[p]['timestamp'] > t:
            continue
        while data[p+1]['timestamp'] < t:
            p += 1
            if p + 1 > len(data) - 1:
                break
        if p + 1 > len(data) - 1:
            break
        # print(p)
        
        d2 = data[p]
        t2 = d2['timestamp']
        lat2 = d2['objs'][0]['lat']
        lon2 = d2['objs'][0]['lon']
        # if t1 < t2:
        #     continue
        t2_next = data[p+1]['timestamp']
        print(t2, t,  t2_next)
        lat2_next = data[p+1]['objs'][0]['lat']
        lon2_next = data[p+1]['objs'][0]['lon']
        lat2_interpolated, lon2_interpolated = interpolate(
            t, t2, t2_next, lat2, lon2, lat2_next, lon2_next)
        objs = []
        objs.append({
            "lon": lon2_interpolated,
            "lat": lat2_interpolated,
            'id': '3',
            "classType": "10",
        })
        converted_data.append({
            "timestamp": t,
            "info": "ped_ground_truth",
            'objs': objs
        })
        # print(objs)
    return converted_data

with open(args.file, 'r') as f:
    data = json.load(f)

with open(args.time_file, 'r') as f:
    time_to_interpolate = json.load(f)

times = list(map(lambda x: x['timestamp'], time_to_interpolate))

data2 = interpolate_data(data, times)
with open(args.output, 'w') as f:
    json.dump(data2, f, indent=2)
