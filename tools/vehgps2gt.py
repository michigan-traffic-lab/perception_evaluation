from pathlib import Path
import argparse
import re
import json

args = argparse.ArgumentParser(
    description='Convert the GPS data to standard format.')
args.add_argument('-i', '--input', type=Path, help='the input folder.')
args.add_argument('-o', '--output', type=Path,
                  help='the output folder.', default=Path("./output"))
args.add_argument('--veh1-file-prefix', type=str,
                  help='the prefix of the vehicle 1 file.', default="mkz_data")
args.add_argument('--veh2-file-prefix', type=str,
                  help='the prefix of the vehicle 2 file.', default="gps_data")
args.add_argument('--output-prefix', type=str, help='the prefix of the output file.', default="gps_data_")
args = args.parse_args()

args.output.mkdir(parents=True, exist_ok=True)


def find_file_by_prefix(folder, prefix):
    for f in folder.iterdir():
        if f.is_file() and f.name.startswith(prefix):
            return f
    return None


def interpolate(t1, t2, t2_next, lat2, lon2, lat2_next, lon2_next):
    # Calculate the fraction of the total time that has passed since t2
    fraction = (t1 - t2) / (t2_next - t2)

    # Use this fraction to calculate the interpolated latitude and longitude
    lat1 = lat2 + fraction * (lat2_next - lat2)
    lon1 = lon2 + fraction * (lon2_next - lon2)

    return lat1, lon1


def convert_veh1_file(file):
    # read csv file
    with open(file, 'r') as f:
        data = f.readlines()
    # remove the first line
    data = data[1:]
    new_data = []
    # convert to standard format
    for line in data:
        tmp = line.strip().split(',')
        timestamp = float(tmp[1])
        lon = float(tmp[2])
        lat = float(tmp[3])
        new_data.append({
            "timestamp": timestamp,
            "info": "ground_truth",
            'objs': [{"lon": lon,
                     "lat": lat,
                      'id': '0',
                      "classType": "2",}]
        })
    # sort data by timestamp
    new_data = sorted(new_data, key=lambda x: x['timestamp'])
    return new_data


def convert_veh2_file(file):
    data = []
    with open(file, 'r') as file:
        lines = file.readlines()
    sec = None
    nsec = None
    lat = None
    lon = None

    for line in lines:
        if re.search(r'\bsec:\s\d+', line):
            sec = re.findall(r'\d+', line)[0]
        elif re.search(r'\bnanosec:\s\d+', line):
            nsec = re.findall(r'\d+', line)[0]

        elif 'latitude' in line:
            lat = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
        elif 'longitude' in line:
            lon = float(re.findall(r"[-+]?\d*\.\d+|\d+", line)[0])
        elif '---' in line:
            # timestamp = float(str(sec) + '.' + str(nsec))
            timestamp = int(str(sec)) + int(str(nsec)) / 1e9
            data.append({'timestamp': timestamp, 'lat': lat, 'lon': lon})

    # sort data by timestamp
    data = sorted(data, key=lambda x: x['timestamp'])
    # adding the last set of data because file does not end with '---'
    # timestamp = int(str(sec) + str(nsec))
    # data.append({'timestamp': timestamp, 'lat': lat, 'lon': lon})
    return data


def merge_veh1_veh2_data(veh1_data, veh2_data):
    p2 = 0
    data = []
    for d1 in veh1_data:
        t1 = d1['timestamp']
        if p2 + 1 > len(veh2_data) - 1:
            break
        while veh2_data[p2+1]['timestamp'] < t1:
            p2 += 1
            if p2 + 1 > len(veh2_data) - 1:
                break
        if p2 + 1 > len(veh2_data) - 1:
            break
        # print(p2)
        
        d2 = veh2_data[p2]
        t2 = d2['timestamp']
        lat2 = d2['lat']
        lon2 = d2['lon']
        # if t1 < t2:
        #     continue
        t2_next = veh2_data[p2 + 1]['timestamp']
        # print(t2, t1,  t2_next)
        lat2_next = veh2_data[p2+1]['lat']
        lon2_next = veh2_data[p2+1]['lon']
        lat2_interpolated, lon2_interpolated = interpolate(
            t1, t2, t2_next, lat2, lon2, lat2_next, lon2_next)
        objs = d1['objs']
        objs.append({
            "lon": lon2_interpolated,
            "lat": lat2_interpolated,
            'id': 1,
            "classType": "2",
        })
        data.append({
            "timestamp": t1,
            "info": "ground_truth",
            'objs': objs
        })
        # print(objs)
    return data

def merge_data(veh1_data, veh2_data):
    p2 = 0
    data = []
    for d1 in veh1_data:
        t1 = d1['timestamp']
        if p2 + 1 > len(veh2_data) - 1:
            break
        while veh2_data[p2+1]['timestamp'] < t1:
            p2 += 1
            if p2 + 1 > len(veh2_data) - 1:
                break
        if p2 + 1 > len(veh2_data) - 1:
            break
        # print(p2)
        
        d2 = veh2_data[p2]
        t2 = d2['timestamp']
        lat2 = d2['objs'][0]['lat']
        lon2 = d2['objs'][0]['lon']
        # if t1 < t2:
        #     continue
        t2_next = veh2_data[p2 + 1]['timestamp']
        print(t2, t1,  t2_next)
        lat2_next = veh2_data[p2+1]['objs'][0]['lat']
        lon2_next = veh2_data[p2+1]['objs'][0]['lon']
        lat2_interpolated, lon2_interpolated = interpolate(
            t1, t2, t2_next, lat2, lon2, lat2_next, lon2_next)
        objs = d1['objs']
        objs.append({
            "lon": lon2_interpolated,
            "lat": lat2_interpolated,
            'id': '1',
            "classType": "2",
        })
        data.append({
            "timestamp": t1,
            "info": "ground_truth",
            'objs': objs
        })
        # print(objs)
    return data


input_folder_path = args.input
for d in input_folder_path.iterdir():
    if d.is_dir():
        veh1_file = find_file_by_prefix(d, args.veh1_file_prefix)
        veh2_file = find_file_by_prefix(d, args.veh2_file_prefix)
        data1 = convert_veh1_file(veh1_file)
        if veh2_file is not None:
            # data2 = convert_veh2_file(veh2_file)
            # data = merge_veh1_veh2_data(data1, data2)
            data2 = convert_veh1_file(veh2_file)
            data = merge_data(data1, data2)
        else:
            data = data1
        with open(args.output / (args.output_prefix + d.name + '.json'), 'w') as f:
            json.dump(data, f, indent=4)
