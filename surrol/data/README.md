## Demonstration Data

We provide a way to generate the demonstration dataset or record the video,
similar to [Hindsight Experience Replay with Demonstrations](https://github.com/openai/baselines/tree/master/baselines/her#hindsight-experience-replay-with-demonstrations).
This is done via the scripted policy in the individual task file.

To generate the demonstrations for *env_name*, you can use

```shell
python data_generation.py --env env_name
```

And the generated data is stored in *./demo*.
It will record the data in 100 randomly initially env resets by default.

Alternatively, you can record the video for one run by using

```shell
python data_generation.py --env env_name --video
```

And the generated video and data are stored in *./video*.
