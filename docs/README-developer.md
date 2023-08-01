# This Document gives Developer guide for the repo
## Data Class
We have four data class:
### DataPoint
The fundamental data element. It represents one point. It also stores intermediate results during evaluation, for example its matched groundtruth point.

The rest of the three classes are the container classes that contains the DataPoint. They have dp_list attribute which is a list of DataPoint.

### DataFrame
All the points with the same timestamp.

### Trajectory
All the points with same ID.

### TrajectorySet
This is the class operated by the end customer. It organize the datapoint both in Trajectories and DataFrames. It contains two dict, the `.trajectories` maps id to Trajectory. The `.dataframes` map timestamp to DataFrame.


## Evaluation Process
The evaluation process contains mainly two major step, a point matching step and a trajectory matching step. The point matching step produce all necessary results using point matching, such as **tp**, **fn**, **fp**, **ids**, and **fn_freq**. The trajectory matching produce the trajectory matching results such as **tpa**, **fna**, **fpa**. Then, all the metrics are calculated based on these results.

## Results formating
The Result class will handle all the results, in terms of grouping them amd calculate statistics. use the `add_trial` function to add new result entity to it. It takes a dict as argument and must have a 'type' key.