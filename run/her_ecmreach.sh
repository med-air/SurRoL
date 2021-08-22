python -m baselines.run --num_env=2 --alg=her --env=ECMReach-v0 --num_timesteps=1e5 \
--save_path=../policies/her/ECMReach-1e5_0 \
--log_path=../logs/her/ECMReach-1e5_0 \
--n_cycles=20
