from pathlib import Path
import pickle
import json
import argparse

def convert_instances(objs):
    new_objs = []
    # print(objs)
    for obj in objs:
        # print(obj)
        new_obj = {}
        new_obj['id'] = str(obj['id'])
        # 'veh' type 2
        # 'ped' type 10
        # print(int(obj['type']))
        assert int(obj['type']) in [1, 2], f'unknown type {obj["type"]}' # 1 for vehicle, 0 for pedestrian
        new_obj['classType'] = '2' if int(obj['type']) == 1 else '10'
        new_obj['lat'] = obj['lonlat'][0]
        new_obj['lon'] = obj['lonlat'][1]
        new_obj['speed'] = obj['speed']
        new_obj['rotation'] = obj['heading']
        new_objs.append(new_obj)
    return new_objs



def process_instances(obj):
    # print(obj.keys())
    timestamp = obj['timestamp']
    roadside_detection = obj['roadside_detection']
    onboard_detection = obj['onboard_detection']
    fused_objects = obj['fused_objects']
    ground_truth = obj['groundtruth']
    print(f'processing ground truth')
    new_gt_objs = {
        'objs':convert_instances(ground_truth),
        'timestamp':timestamp,
        'info':'ground truth'
    }
    print(f'processing roadside detection')
    new_rd_objs = {
        'objs':convert_instances(roadside_detection),
        'timestamp':timestamp,
        'info':'roadside detection'
    }
    print(f'processing onboard detection')
    new_obd_objs = {
        'objs':convert_instances(onboard_detection),
        'timestamp':timestamp,
        'info':'onboard detection'
    }
    print(f'processing fusion')
    new_f_objs = {
        'objs':convert_instances(fused_objects),
        'timestamp':timestamp,
        'info':'fusion'
    }
    
    return new_gt_objs, new_rd_objs, new_obd_objs, new_f_objs

# data_path = Path("../data/cooperative_perception_sim/raw")

if __name__ == "__main__":
    argp = argparse.ArgumentParser()
    argp.add_argument('-o', '--output-path', type=Path, default=Path('./output'))
    argp.add_argument('-d', '--data-path', type=Path)
    argp.add_argument("--ground-truth-folder-name", type=str, default="ground_truth")
    argp.add_argument("--roadside-detection-folder-name", type=str, default="roadside_detection")
    argp.add_argument("--onboard-detection-folder-name", type=str, default="onboard_detection")
    argp.add_argument("--fusion-folder-name", type=str, default="fusion")
    args = argp.parse_args()
    args.output_path.mkdir(exist_ok=True, parents=True)
    roadside_detection_objs = []
    onboard_detection_objs = []
    fusion_objs = []
    ground_truth_objs = []
    data_path = args.data_path
    for folder in data_path.iterdir():
        # print(file)
        # read pkl file
        if folder.is_dir():
            case_name = folder.name
            print(f'Processing {case_name}')
            for file in sorted(folder.iterdir()):
                if file.suffix == '.pkl':
                    print(f'Processing {file} ... ')
                    with open(file, 'rb') as f:
                        data = pickle.load(f)
                        gt_objs, rd_objs, obd_objs, f_objs = process_instances(data)
                        ground_truth_objs.append(gt_objs)
                        roadside_detection_objs.append(rd_objs)
                        onboard_detection_objs.append(obd_objs)
                        fusion_objs.append(f_objs)
            # write output
            Path(args.output_path / args.ground_truth_folder_name).mkdir(exist_ok=True, parents=True)
            with open(args.output_path / args.ground_truth_folder_name / f'{case_name}.json', 'w') as f:
                json.dump(ground_truth_objs, f, indent=4)
            Path(args.output_path / args.roadside_detection_folder_name).mkdir(exist_ok=True, parents=True)
            with open(args.output_path / args.roadside_detection_folder_name / f'{case_name}.json', 'w') as f:
                json.dump(roadside_detection_objs, f, indent=4)
            Path(args.output_path / args.onboard_detection_folder_name).mkdir(exist_ok=True, parents=True)
            with open(args.output_path / args.onboard_detection_folder_name / f'{case_name}.json', 'w') as f:
                json.dump(onboard_detection_objs, f, indent=4)
            Path(args.output_path / args.fusion_folder_name).mkdir(exist_ok=True, parents=True)
            with open(args.output_path / args.fusion_folder_name / f'{case_name}.json', 'w') as f:
                json.dump(fusion_objs, f, indent=4)
                
        # print(data['timestamp'])
        # print(data['fused_objects'])
        # break