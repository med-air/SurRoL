python -m baselines.run \
--num_env=2 --alg=her --env=NeedlePick-v0 --num_timesteps=1e5 --policy_save_interval=5 \
--demo_file=../surrol/data/demo/data_NeedlePick-v0_random_100.npz \
--save_path=../policies/her/NeedlePick-demo1e5_0 \
--bc_loss=1 --q_filter=1 --num_demo=100 --demo_batch_size=128 --prm_loss_weight=0.001 --aux_loss_weight=0.0078 --n_cycles=20 --batch_size=1024 --random_eps=0.1 --noise_eps=0.1 \
--log_path=../logs/her/NeedlePick-demo1e5_0
