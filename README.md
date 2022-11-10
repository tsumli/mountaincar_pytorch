# Mountain Car (gym) using Q-Learning, PyTorch DQN
## Steps 
1. Change `agent_name` in config.yaml (now supporting `QAgent` and `DQNAgent`).
2. Training.
```bash
python train.py
```
3. (Optional) Visualization.
```bash
python visualize.py --path path/to/pth
```

4. Validation.
```bash
python validate.py --path path/to/pth
```

## Results
Following table shows the validation results (mean and std of 100 experiments).
The larger the reward is, the better.
|  | mean reward| std reward|
| ------------- | ------------- | ---- |
| Q-Learning  | -139.0  | 21.0 |
| DQN  | -287.8 | 24.6 |
| DoubleDQN  | -167.3 | 20.0|

In this experiment, 300 steps on all methods, and 30000 episodes on Q-Learning and 500 episodes on DQN, because DQN takes longer to process than Q-Learning.

## Docker
```bash
cd docker
docker build . -t foobar
cd ..
docker run -it --rm -v $PWD:/workspace/ foobar bash
```
