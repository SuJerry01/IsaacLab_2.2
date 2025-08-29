

# locomotion -Qmini-


## #train

```
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-Qmini-v0 --headless
```


## #play
```
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-Qmini-v0 --num_envs 32 --load_run qmini_flat   --checkpoint logs/rsl_rl/qmini_flat/2025-08-26_23-46-27/model_999.pt

```


# locomotion -BDX_Go-

```

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-BDX-Go-v0 --headless
```


```
# BDX
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-BDX-Go-v0 --num_envs 32 --load_run bdx_flat --checkpoint logs/rsl_rl/bdx_flat/2025-08-27_00-07-45/model_999.pt

# BDX_Go
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-BDX-Go-v0 --num_envs 32 --load_run bdx_go_flat --checkpoint logs/rsl_rl/bdx_go_flat/2025-08-27_11-47-29/model_999.pt

```




# locomotion -BDX_A1-

```

# 平地環境訓練
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Flat-BDX_A1-v0 --headless

# 崎嶇地形環境訓練  
python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Velocity-Rough-BDX_A1-v0 --headless


```


```

# 平地環境測試
python scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Flat-BDX_A1-Play-v0 --num_envs 32 --load_run bdx_a1_flat --checkpoint logs/rsl_rl/bdx_a1_flat/2025-08-27_14-15-45/model_999.pt

# 崎嶇地形環境測試
python scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Velocity-Rough-BDX_A1-Play-v0 --num_envs 32 --load_run bdx_a1_flat --checkpoint logs/rsl_rl/bdx_a1_flat/2025-08-27_11-47-29/model_999.pt
```
