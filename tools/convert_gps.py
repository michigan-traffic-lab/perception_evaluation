import os
import pandas as pd
import argparse
from pathlib import Path


def format_gts_from_csv(input_folder_path, output_folder_path):
    dir_list = [i for i in os.listdir(input_folder_path) if os.path.isdir(os.path.join(input_folder_path, i))]
    # dir_list = [int(i) for i in dir_list]
    # dir_list = sorted(dir_list)
    # print(dir_list)
    # fix_data = [i for i in file_list if not "vel" in i]
    # vel_data = [i for i in file_list if "vel" in i]
    # print(fix_data)
    # print(vel_data)

    for dirname in dir_list:
        data = pd.read_csv(str(Path(input_folder_path) / str(dirname) / "mkz_data.csv"))
        with open(Path(output_folder_path) / f"gps_fix_{dirname}.txt", "w") as f:
            f.write("%time,field.header.seq,field.header.stamp,field.header.frame_id,field.status.status,field.status.service,field.latitude,field.longitude,field.altitude,field.position_covariance0,field.position_covariance1,field.position_covariance2,field.position_covariance3,field.position_covariance4,field.position_covariance5,field.position_covariance6,field.position_covariance7,field.position_covariance8,field.position_covariance_type\n")
            for index, row in data.iterrows():
                # print(row)
                f.write(f"{int(float(row['UTC Timestamp']) * 1e9)},,{int(float(row['UTC Timestamp']) * 1e9)},,,,{row['lat']},{row['lon']},,,,,,,,,,,\n")

        data = pd.read_csv(str(Path(input_folder_path) / str(dirname) / "mkz_vel_data.csv"))
        with open(Path(output_folder_path) / f"gps_vel_{dirname}.txt", "w") as f:
            f.write("%time,field.header.seq,field.header.stamp,field.header.frame_id,field.twist.twist.linear.x,field.twist.twist.linear.y,field.twist.twist.linear.z,field.twist.twist.angular.x,field.twist.twist.angular.y,field.twist.twist.angular.z,field.twist.covariance0,field.twist.covariance1,field.twist.covariance2,field.twist.covariance3,field.twist.covariance4,field.twist.covariance5,field.twist.covariance6,field.twist.covariance7,field.twist.covariance8,field.twist.covariance9,field.twist.covariance10,field.twist.covariance11,field.twist.covariance12,field.twist.covariance13,field.twist.covariance14,field.twist.covariance15,field.twist.covariance16,field.twist.covariance17,field.twist.covariance18,field.twist.covariance19,field.twist.covariance20,field.twist.covariance21,field.twist.covariance22,field.twist.covariance23,field.twist.covariance24,field.twist.covariance25,field.twist.covariance26,field.twist.covariance27,field.twist.covariance28,field.twist.covariance29,field.twist.covariance30,field.twist.covariance31,field.twist.covariance32,field.twist.covariance33,field.twist.covariance34,field.twist.covariance35\n")
            for index, row in data.iterrows():
                f.write(f"{int(float(row['UTC Timestamp']) * 1e9)},,{int(float(row['UTC Timestamp']) * 1e9)},,{row['vel_x']},{row['vel_y']},{row['vel_z']},,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,\n")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Convert the GPS data from CSV that can be directlly read from evaluator.')
    argparser.add_argument('-i', '--input', type=str, help='the input folder.')
    argparser.add_argument('-o', '--output', type=str, help='the output folder.')
    args = argparser.parse_args()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    format_gts_from_csv(args.input, args.output)

