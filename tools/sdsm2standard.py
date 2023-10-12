# from pycrate_sdsm.SDSMEncoder import sdsm_encoder
from pycrate_sdsm.SDSMDecoder import sdsm_decoder
import json
import argparse
import re
from geopy.distance import distance as geodistance
from datetime import datetime, timezone
from tqdm import tqdm

argparser = argparse.ArgumentParser(
    description='Decode the SDSM Messages from and generate new JSON file')
argparser.add_argument('-i', '--input', type=str,
                       help='the input JSON file.', required=True)
argparser.add_argument('-o', '--output', type=str,
                       help='the output JSON file.', default='./output.json')
argparser.add_argument('-t', '--type', type=str,
                       help='the type of the SDSM message.', default='msight')
args = argparser.parse_args()


def convert_to_timestamp(time_dict):
    dt = datetime(
        year=time_dict['year'],
        month=time_dict['month'],
        day=time_dict['day'],
        hour=time_dict['hour'],
        minute=time_dict['minute'],
        second=int(time_dict['second']),
        microsecond=int((time_dict['second'] % 1) * 1e6),
        tzinfo=timezone.utc
    )
    return dt.timestamp()


def calculate_final_position(center_lat, center_long, x_offset, y_offset):
    start_point = (center_lat, center_long)
    # Calculate new coordinates based on offsets
    # new_lat_lon = geodistance(meters=x_offset).destination(point=start_point, bearing=0)  # 0 degrees is North for latitude change
    # new_lat_lon = geodistance(meters=y_offset).destination(point=new_lat_lon, bearing=90)  # 90 degrees is East for longitude change
    new_lat_lon = geodistance(meters=y_offset).destination(point=start_point, bearing=180)  # 180 degrees is South for latitude change
    new_lat_lon = geodistance(meters=x_offset).destination(point=new_lat_lon, bearing=90)  # 90 degrees is East for longitude change
    return new_lat_lon.latitude, new_lat_lon.longitude


with open(args.input, 'r') as f:
    data = json.load(f)

decoded_data = []
for d in tqdm(data):
    encoded_data = d['data']
    info = d['info']
    
    # print(decoded_bytes)
    if args.type == 'derq':
        encoded_bytes = bytes.fromhex(encoded_data)
        decoded_bytes = encoded_bytes.decode('utf-8')
        match = re.search(r'Payload=(.*?)(\n|$)', decoded_bytes)
        if match:
            payload_value = match.group(1).strip()
            # print(payload_value)
    elif args.type == 'ouster':
        payload_value = encoded_data
    elif args.type == 'msight':
        payload_value = decoded_bytes
    else:
        raise Exception('Unknown type of SDSM message.')
    try:
        result = sdsm_decoder(payload_value)
    except:
        print('Failed to decode the SDSM message.')
        print(payload_value)
        print(d["event_timestamp"])
        continue
    t = result["sDSMTimeStamp"]
    timestamp = convert_to_timestamp(t)
    center_lat = result["refPos"]['lat']
    center_long = result["refPos"]['long']
    decoded_frame = {
        "timestamp": timestamp,
        "info": info,
        "objs": []
    }
    for obj_ in result["objects"]:
        obj = obj_["detObjCommon"]
        id = obj['objectID']
        if id == 0:
            continue
        if obj['objType'] == 'vehicle':
            classType = '2'
        elif obj['objType'] == 'vru':
            classType = '10'
        else:
            print(f'WARNING: unknown classType {obj["objType"]}, set to 0')
            classType = '0'
        rotation = obj["heading"]
        speed = obj["speed"]
        x_offset = obj["pos"]["offsetX"]
        y_offset = obj["pos"]["offsetY"]
        lat, lon = calculate_final_position(
            center_lat, center_long, x_offset, y_offset)
        # print('id', id, 'classType', classType, 'rotation', rotation, 'speed', speed, 'lat', lat, 'lon', lon)
        decoded_frame['objs'].append({
            'id': str(id),
            'classType': classType,
            'lon': lon,
            'lat': lat,
            'speed': speed,
            'rotation': rotation,
        })
    decoded_data.append(decoded_frame)

with open(args.output, 'w') as f:
    json.dump(decoded_data, f, indent=4)

