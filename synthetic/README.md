# RIFCA

RIFCA is a Python library adapted from [IFCA](https://arxiv.org/abs/2006.04088)
 for robust federated learning. 

This part is for synthetic experiments. 

## Requirements
```bash 
    Python 3.9.16
    pip install -r requirements.txt 
```

## Config.json
```json
    "m" : 100,            # number of machines
    "p" : 2,              # number of cluster groups (expects m % p == 0)
    "n" : 100,            # number of datapoints for each machine
    "d" : 1000,           # dimension of the datapoint
    "r" : 1.0,            # separation parameter for synthetic data generation
    "noise_scale": 0.001, # noise parameter (\sigma)  for synthetic data generation
    "num_epochs": 300,    # number of data pass
    "score":"set",        # scoring method ( only 'set' is used)
    "update_method":"median", # update method used by the server
    "lr":0.1,             # learning rate
    "data_seed":0,        # random seed for generating data
    "train_seed":0        # random seed for weight initiailization and training
```

## Running the experiments

* To run the single instance of experiment (with data_seed=0 and train_seed=0):
```bash
python3 run_all.py
```

[//]: # (* To reproduce results in the paper:)

[//]: # (```bash)

[//]: # (python run_all.py   # p=2, m vs n)

[//]: # (python run_p4_m_n.py   # p=4, m vs n)

[//]: # (python run_p2_r_noise.py   # p=2, r vs noise)

[//]: # (python run_p4_r_noise.py   # p=4, r vs noise)

[//]: # (```)
[//]: # (Each script runs all possible combinations of experiments in cfg in main&#40;&#41; &#40; by overwriting configuration data in config.json&#41;, and 4&#40;=max_procs&#41; processes of experiments are run concurrently. After all experiments are done, all result files are read and checks if the hyparameter combination had successful convergence.)
[//]: # ()
[//]: # (## Notes)

[//]: # (* It takes several days to run all the experiment. We recommend running this script in cluster with many CPU cores, with max_procs tuned to match the number of cores.)

[//]: # ()
