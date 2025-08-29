
# loco_manipu -Digit-

## A 當前loco-manip實現主要是架構組合，而不是真正的協調性設計~缺乏：

### 1. 動態耦合機制：手臂和腿部的實時協調
### 2. 預測性控制：基於未來狀態的協調決策
### 3. 工作空間優化：基於可達性的移動策略
### 4. 時序協調：不同階段的任務優先級調整
### 5. 全身動力學：整體動量和穩定性優化

## B 缺失的協調機制

### 現有實現缺乏以下關鍵協調機制：
### ❌ 動態重心補償
### ❌ 實時步態調整
### ❌ 預測性平衡控制
### ❌ 角動量管理
### ❌ 支撐多邊形適應


## C 逐步實現策略
### 階段一：實現基本的重心補償機制
### 階段二：加入角動量協調
### 階段三：實現預測性控制
### 階段四：整合工作空間約束




## #train

```
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Tracking-LocoManip-Digit-v0 --headless

python scripts\reinforcement_learning\rsl_rl\train.py --task Isaac-Tracking-LocoManip-Digit-v0 --headless


```


## #play
```
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py   --task Isaac-Tracking-LocoManip-Digit-v0   --num_envs 32   --load_run digit_loco_manip   --checkpoint logs/rsl_rl/digit_loco_manip/2025-08-26_12-38-10/model_1999.pt


python  scripts/reinforcement_learning/rsl_rl/play.py   --task Isaac-Tracking-LocoManip-Digit-v0   --num_envs 32   --load_run digit_loco_manip   --checkpoint logs/rsl_rl/digit_loco_manip/2025-08-28_16-58-35/model_1999.pt
```