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
|  | mean | std |
| ------------- | ------------- | ---- |
| Q-Learning  | -141.35  | 26.834 |
| DQN  | -140.81 | 34.515|

In this experiment, 30000 episodes on Q-Learning and 1000 episodes on DQN, because DQN takes longer to process than Q-Learning.

## Docker
```bash
cd docker
docker build . -t foobar
cd ..
docker run -it --rm -v $PWD:/workspace/ foobar bash
```
