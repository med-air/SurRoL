# Training RL Policy with Demonstrations
We use the [DEX](https://github.com/med-air/DEX/tree/main) repository, which implements multiple state-of-the-art demonstration-guided RL algorithms, to train our RL policy. To run the training process, you should first install the dependencies by following the installation instructions from DEX. Then start the training by running the following command:
- Train **DDPGBC** with demonstrations:
```bash
python3 rl/train.py task=NeedlePick-v0 agent=ddpgbc use_wb=True demo_path=your_demo_path
```

Note that you should specify the path of demonstration data you would like to provide to the RL agent, which could be collected by both human demonstrators or scripted controllers. For the latter, please refer to [here](../surrol/data/) for more details. 