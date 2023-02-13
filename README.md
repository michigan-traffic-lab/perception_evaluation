# Perception Sysmtem Evaluation
This repository contains code and resources for evaluating the performance of a perception system. The evaluations for Derq, Bluecity, and MSight systems are supported.

TODO:
- [ ] Unify the data format for different perception systems
- [ ] Design better configuration file for testing
- [ ] Support multi-object evaluation

## Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

## Prerequisites
Before you begin, make sure you have the following installed on your machine:
- [Python 3.x](https://www.python.org/downloads/)

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
|---- mm-dd-yyyy/    # the experiment date of testing 1
|     |---- dets/
|     |     |---- bluecity_detection_1.json     # the detection results (Bluecity)
|     |     |---- bluecity_detection_2.json     # the detection results (Bluecity)
|     |     | ...
|     |     |---- derq_detection_1.json     # the detection results (Derq)
|     |     |---- derq_detection_2.json     # the detection results (Derq)
|     |     | ...
|     |     |---- msight_detection_1.json     # the detection results (MSight)
|     |     └ ...
|     |---- gts/
|     |     |---- gps_fix_1.txt     # gps results for position
|     |     |---- gps_vel_1.txt     # gps results for velocity
|     |     |---- gps_fix_2.txt
|     |     |---- gps_vel_2.txt
|     |     | ...
|     |     |---- ped_11.txt      # pedestrian gps data
|     |     └ ...
|     |---- config.yaml     # configuration of this testing
| ...
| ...
|---- mm-dd-yyyy/    # the experiment date of testing 2
└ ...
```
Each testing should contain a testing configuration file `config.yaml`. The configuration should include:
```
GROUND_TRUTH_FREQUENCY_VEH1: 50     # the GPS frequency for vehicle 1
GROUND_TRUTH_FREQUENCY_VEH2: 100    # the GPS frequency for vehicle 2
DET_FREQUENCY: 10    # the expected frequency for the detection system
CENTER_COR: [42.300947, -83.698649]     # the center coordinate of the testing (Mcity: [42.300947, -83.698649], Roundabout: [42.229394, -83.739015])
LATENCY_TRIAL_IDS: [1, 2]     # which trials you want to use for latency estimation
SINGLE_VEH_TRIAL_IDS: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]   # trial ids for single-vehicle testings
TWO_VEH_TRIAL_IDS: [13, 14, 15]     # trial ids for two-vehicle testing
PED_TRIAL_IDS: [11]     # trial ids for pedestrian testing
GROUND_TRUTH_FREQUENCY_PED: 10     # the GPS frequency for pedeestrians
DATE: '2023-01-05'      # the date of the testing
```

Please prepare the data in the above format to conduct the evaluation.

## Evaluation

To evaluate a perception system, run
```
python main.py --data-root </path/to/data/root> --data-name <name-of-the-testing> --system <perception-system-for-test> [--latency <estimated-latency>] [--threshold <lateral-error-threshold>]
```
For example, to evaluate a testing on `2023-01-05` for Bluecity system, run
```
python main.py --data-root example_data --data-name 2023-01-05 --system Bluecity
```

## Developers
Depu Meng, Zihao Wang, Rusheng Zhang, Shengyin Shen

## Contact
Henry Liu - henryliu@umich.edu

## License
TBD
