
## Requirements
- Python 3.9+
- Cuda 11.3



## Setup
```bash
$ cd ./tools
$ make
```


## Recipe
- [dcase2022-task2](https://dcase.community/challenge2022/task-unsupervised-anomalous-sound-detection-for-machine-condition-monitoring): The main challenge of this task is to detect unknown anomalous sounds under the condition that only normal sound samples have been provided as training data.

To run the recipe, please follow the below instruction.

```bash
# Let us move on the recipe directory
$ cd scripts

#all models are trained, to do the inference, 
$ ./job.sh --stage 2
# You can see the results at exp/all/**/score*.csv
```
