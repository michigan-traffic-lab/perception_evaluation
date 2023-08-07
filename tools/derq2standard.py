import SDSMEncoder
import json
import argparse
import os
from geopy import Point
from geopy import distance as geodistance
from pathlib import Path
from datetime import datetime, timedelta
import pytz

argparser = argparse.ArgumentParser(description='Decode the SDSM Messages from DERQ')
argparser.add_argument('-i', '--input', type=str, help='the input folder of JSON file.')
argparser.add_argument('-o', '--output', type=str, help='the output folder.')
argparser.add_argument('-m', '--month', type=int, help='month')
argparser.add_argument('-y', '--year', type=int, help='year')
args = argparser.parse_args()

def calculate_final_position(center_lat, center_long, x_offset, y_offset):
    final_latitude = []
    final_longitude = []
    for i in range(0,len(x_offset)):
        # print(center_lat, center_long, x_offset, y_offset)
        # Calculate the destination point in the X-axis direction

        # switched x for y because that works
        x_destination = geodistance.distance(meters=y_offset[i]).destination((center_lat, center_long), 90)

        # Calculate the destination point in the Y-axis direction
        y_destination = geodistance.distance(meters=x_offset[i]).destination((center_lat, center_long), 0)

        # Extract the latitude and longitude of the final position
        # print(y_destination.latitude, x_destination.longitude)
        final_latitude.append(y_destination.latitude)
        final_longitude.append(x_destination.longitude)

    return final_latitude, final_longitude
# {
# "event_timestamp": 1688685104.6418185,
# "info": "derq_raw",
# "data": "56657273696f6e3d302e370a547970653d42534d0a505349443d307832300a5072696f726974793d350a54784d6f64653d434f4e540a54784368616e6e656c3d3138330a5478496e74657276616c3d300a44656c697665727953746172743d0a44656c697665727953746f703d0a5369676e61747572653d46616c73650a456e6372797074696f6e3d46616c73650a5061796c6f61643d34343433343234313430323664636230303263363930323736646330326431636233333339326666666666666666383030303061653031346136333032346666343930313439323030303037383936303030",
# "time": "2023-07-06 19:11:44.641821",
# "timestamp": 1688685104.6418185
# },

def create_object_from_decoded(message):
    objs = []
    final_lat, final_lon = calculate_final_position(message[9],message[10],message[20],message[21])

    for i in range(0,len(message[17])):
        classType = "0"
        if message[15][i]==1:
            classType = "2"
        elif message[15][i]==2:
            classType = "10"
        object = {"id": message[17][i],
                  "centerX": "",
                  "centerY": "",
                  "width": "",
                  "length": "",
                  "rotation": "",
                  "classType": classType,
                  "speed": message[24][i],
                  "lat": final_lat[i],
                  "lon": final_lon[i]}
        objs.append(object)
    return objs
# "objs": [
# {
# "id": "488298714",
# "centerX": "-20.49983787536621",
# "centerY": "38.220611572265625",
# "width": "0.4403114318847656",
# "length": "0.44771432876586914",
# "rotation": "1.5460433959960938",
# "classType": "10",
# "speed": "0.38331180810928345",
# "lat": 42.30048876305199,
# "lon": -83.69875494019689
# },

def decode_message(data):
    bytes = bytearray.fromhex(data)
    bytes = bytes.decode()
    x = bytes.split("Payload=")
    return SDSMEncoder.decode(bytearray.fromhex(x[-1]))

if __name__ == "__main__":
    file_list = os.listdir(args.input)
    print(file_list)
    file_count = len(file_list)

    # file_list = ["detection_1.json"]
    Path(args.output).mkdir(parents=True, exist_ok=True)
    for filename in file_list:
        if filename == "combined.json":
            continue

        result = []
        with open(Path(args.input) / filename) as f:
            lines = json.load(f)
            for x in lines:
                # print(x)
                message = decode_message(x["data"])

                day = message[4]
                hour = message[5]
                minute = message[6]
                second = message[7]
                offset = message[8]
                # print(day, hour, minute, second, offset)
                # print(int((second % 1) * 1_000_000))

                # Create a datetime object, timezone-aware
                dt = datetime(args.year, args.month, day, hour, minute, int(second), int((second % 1) * 1_000_000), tzinfo=pytz.UTC)
                # print(dt)

                # Convert to epoch (Unix time)
                epoch_time = dt.timestamp()

                result.append({"event_timestamp": x["event_timestamp"],
                               "info": x["info"],
                               "objs": create_object_from_decoded(message),
                               "time": x["time"],
                               "timestamp": epoch_time})


        with open(Path(args.output) / filename, "w") as f:
            f.write(json.dumps(result, indent=2))
