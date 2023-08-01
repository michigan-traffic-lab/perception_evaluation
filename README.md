# Perception Sysmtem Evaluation
This repository contains code and resources for evaluating the performance of a perception system. This document only shows basic usages. For an in-depth understanding, check our [developer document](docs/README-developer.md). 

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites
Before you begin, make sure you have the following installed on your machine:
- [Python 3.7+](https://www.python.org/downloads/)

The code is tested in Linux operation system only. We recommend you
to run this code in Linux or WSL (Windows Subsystem for Linux).

### Installing
Follow these steps to set up the repository on your local machine:

1. Clone the repository:
```
git clone git@github.com:michigan-traffic-lab/perception_evaluation.git
```
2. Navigate to the repository:
```
cd perception_evaluation
```
3. Install the required packages:
```
pip install -r requirements.txt
```

## Data Preparation
The testing data should be formatted by:
```
data_root/
|---- your-data-name/    # the experiment date of testing 1
|     |---- dets/
|     |     |---- detection_1.json     # the detection results
|     |     |---- detection_2.json     # the detection results
|     |     └ ...
|     |---- gts/
|     |     |---- gps_1.json     # gps results for position
|     |     |---- gps_2.json
|     |     | ...
|     |     |---- ped_11.json      # pedestrian gps data
|     |     └ ...
|     |---- config.yaml     # configuration of this testing
| ...
| ...
|---- another-data-name/    # the experiment date of testing 2
└ ...
```
Each testing should contain a testing configuration file `config.yaml`. The configuration should include all the global parameters, as well as the specification of each trial. Here is an example
```
GROUND_TRUTH_FREQUENCY_VEH: 50
GROUND_TRUTH_FREQUENCY_PED: 10
DET_FREQUENCY: 10
TP_DISTANCE_THRESHOLD: 1.5
CENTER_COR: [42.300947, -83.698649]
ROI_RADIUS: 50
trials:
  - id: 1
    type: 'latency'
    det_file: 'dets/detection_1.json'
    gt_file: 'gts/gps_data_1.json'
    roi_radius: 20
  - id: 2
    type: 'latency'
    det_file: 'dets/detection_2.json'
    gt_file: 'gts/gps_data_2.json'
    roi_radius: 20
  - id: 3
    type: 'vehicle_evaluation'
    det_file: 'dets/detection_3.json'
    gt_file: 'gts/gps_data_3.json'
  - id: 4
    type: 'vehicle_evaluation'
    det_file: 'dets/detection_4.json'
    gt_file: 'gts/gps_data_4.json'
  - id: 5
    type: 'vehicle_evaluation'
    det_file: 'dets/detection_5.json'
    gt_file: 'gts/gps_data_5.json'
  - id: 6
    type: 'vehicle_evaluation'
    det_file: 'dets/detection_6.json'
    gt_file: 'gts/gps_data_6.json'
  - id: 7
    type: 'vehicle_evaluation'
    det_file: 'dets/detection_7.json'
    gt_file: 'gts/gps_data_7.json'
  - id: 8
    type: 'vehicle_evaluation'
    det_file: 'dets/detection_8.json'
    gt_file: 'gts/gps_data_8.json'
  - id: 9
    type: 'vehicle_evaluation'
    det_file: 'dets/detection_9.json'
    gt_file: 'gts/gps_data_9.json'
  - id: 10
    type: 'vehicle_evaluation'
    det_file: 'dets/detection_10.json'
    gt_file: 'gts/gps_data_10.json'
  - id: 11
    type: 'vehicle_evaluation'
    det_file: 'dets/detection_11_c.json'
    gt_file: 'gts/gps_data_11_c.json'
  - id: 12
    type: 'vehicle_evaluation'
    det_file: 'dets/detection_12.json'
    gt_file: 'gts/gps_data_12_c.json'
  - id: 13
    type: 'vehicle_evaluation'
    det_file: 'dets/detection_13_c.json'
    gt_file: 'gts/gps_data_13_c.json'
  - id: 14
    type: 'vehicle_evaluation'
    det_file: 'dets/detection_14.json'
    gt_file: 'gts/gps_data_14_c.json'
  - id: 15
    type: 'vehicle_evaluation'
    det_file: 'dets/detection_15_c.json'
    gt_file: 'gts/gps_data_15_c.json'
  - id: 11_ped
    type: 'pedestrian_evaluation'
    det_file: 'dets/detection_11.json'
    gt_file: 'gts/ped_gps_data_11.json'
    roi_radius: 25
  - id: 12_ped
    type: 'pedestrian_evaluation'
    det_file: 'dets/detection_12.json'
    gt_file: 'gts/ped_gps_data_12.json'
    roi_radius: 25
  - id: 13_ped
    type: 'pedestrian_evaluation'
    det_file: 'dets/detection_13.json'
    gt_file: 'gts/ped_gps_data_13.json'
    roi_radius: 25
  - id: 14_ped
    type: 'pedestrian_evaluation'
    det_file: 'dets/detection_14.json'
    gt_file: 'gts/ped_gps_data_14.json'
    roi_radius: 25
  - id: 15_ped
    type: 'pedestrian_evaluation'
    det_file: 'dets/detection_15.json'
    gt_file: 'gts/ped_gps_data_15.json'
    roi_radius: 25

DATE: '2023-07-06'
```

Please prepare the data in the above format to conduct the evaluation.

The trajectory data, both groundtruth and the detection data must be in the required JSON format, please look at [example data](example_data/data-1/dets/detection_1.json) for reference.

We also provide a couple of [tools](docs/tools.md) to convert and operate the data.

## Evaluation

To evaluate a perception system, run
```
python main.py --d </path/to/data/root> -n <name-of-the-testing> [--latency <estimated-latency>]

## Developers
Depu Meng, Zihao Wang, Rusheng Zhang, Shengyin Shen

## Contact
Henry Liu - henryliu@umich.edu

## License
TBD
