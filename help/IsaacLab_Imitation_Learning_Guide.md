# IsaacLab 模仿學習完整指南

## 目錄

- [概覽](#概覽)
- [1. Isaac Lab Mimic](#1-isaac-lab-mimic)
  - [1.1 架構設計](#11-架構設計)
  - [1.2 核心組件](#12-核心組件)
  - [1.3 工作流程](#13-工作流程)
  - [1.4 技術特點](#14-技術特點)
- [2. AMP (Adversarial Motion Priors)](#2-amp-adversarial-motion-priors)
  - [2.1 架構設計](#21-架構設計)
  - [2.2 運動數據處理](#22-運動數據處理)
  - [2.3 環境實現](#23-環境實現)
- [3. Robomimic 集成](#3-robomimic-集成)
  - [3.1 支持的算法](#31-支持的算法)
  - [3.2 訓練流程](#32-訓練流程)
  - [3.3 評估系統](#33-評估系統)
- [4. 實用腳本和工具](#4-實用腳本和工具)
- [5. 支持的環境和任務](#5-支持的環境和任務)
- [6. 使用教程](#6-使用教程)
- [7. 最佳實踐](#7-最佳實踐)
- [8. 常見問題](#8-常見問題)

---

## 概覽

IsaacLab 提供了三種主要的模仿學習方法，每種都針對不同的應用場景設計：

| 方法 | 主要用途 | 核心特點 | 適用場景 |
|------|----------|----------|----------|
| **Isaac Lab Mimic** | 操作任務數據生成 | 對象中心轉換、少量演示擴充 | 機械臂操作、多臂協調 |
| **AMP** | 自然運動學習 | 對抗訓練、參考運動模仿 | 人形機器人運動控制 |
| **Robomimic** | 通用模仿學習 | 多算法支持、成熟框架 | 精確行為克隆 |

---

## 1. Isaac Lab Mimic

### 1.1 架構設計

Isaac Lab Mimic 基於 MimicGen 系統設計，旨在從少量人工演示自動生成大量高質量的訓練數據。

```
源演示數據 → 子任務分割 → 場景適應 → 軌跡生成 → 新數據集
```

**核心目錄結構：**
```
source/isaaclab_mimic/
├── isaaclab_mimic/
│   ├── datagen/              # 數據生成核心邏輯
│   │   ├── data_generator.py # 主要生成器類
│   │   ├── selection_strategy.py # 演示選擇策略
│   │   ├── waypoint.py       # 軌跡表示
│   │   └── utils.py          # 工具函數
│   ├── envs/                 # Mimic 環境實現
│   │   ├── franka_stack_*.py # Franka 堆疊任務
│   │   └── pinocchio_envs/   # Pinocchio 集成環境
│   └── ui/                   # 用戶界面
├── scripts/
│   └── imitation_learning/
│       └── isaaclab_mimic/
│           ├── annotate_demos.py    # 演示標註
│           ├── generate_dataset.py  # 數據生成
│           └── consolidated_demo.py # 數據整合
```

### 1.2 核心組件

#### 1.2.1 DataGenerator 類

```python
class DataGenerator:
    """主要的數據生成器，負責從源演示生成新軌跡"""
    
    def __init__(self, env, src_demo_datagen_info_pool, dataset_path):
        self.env = env                           # 仿真環境
        self.src_demo_datagen_info_pool = ...   # 源演示數據池
        
    def generate_trajectory(self, env_id, eef_name, subtask_ind):
        """為指定子任務生成軌跡"""
        # 1. 選擇源演示
        # 2. 轉換場景座標
        # 3. 生成新軌跡
        
    async def generate(self, env_id, success_term):
        """生成完整的演示軌跡"""
        # 異步生成新的完整軌跡
```

#### 1.2.2 座標轉換系統

**對象中心轉換：**
```python
def transform_source_data_segment_using_object_pose(
    obj_pose: torch.Tensor,      # 當前場景對象姿態 [4x4]
    src_eef_poses: torch.Tensor, # 源演示末端姿態序列 [T, 4, 4]
    src_obj_pose: torch.Tensor,  # 源演示對象姿態 [4x4]
) -> torch.Tensor:
    """
    保持末端執行器相對於對象的相對姿態關係
    適應新場景的對象位置和方向
    """
    # 計算相對變換
    src_eef_poses_rel_obj = PoseUtils.pose_in_A_to_pose_in_B(
        pose_in_A=src_eef_poses,
        pose_A_in_B=PoseUtils.pose_inv(src_obj_pose[None])
    )
    
    # 應用到新場景
    transformed_eef_poses = PoseUtils.pose_in_A_to_pose_in_B(
        pose_in_A=src_eef_poses_rel_obj,
        pose_A_in_B=obj_pose[None]
    )
    return transformed_eef_poses
```

**協調轉換方案：**
```python
class SubTaskConstraintCoordinationScheme(Enum):
    TRANSFORM = "transform"  # 完整 6DOF 轉換
    TRANSLATE = "translate"  # 僅平移轉換
    REPLAY = "replay"       # 直接重放
```

#### 1.2.3 子任務約束系統

```python
class SubTaskConstraintType(Enum):
    COORDINATION = "coordination"      # 多臂協調約束
    _SEQUENTIAL_FORMER = "seq_former"  # 順序約束（前置）
    _SEQUENTIAL_LATTER = "seq_latter"  # 順序約束（後置）
```

### 1.3 工作流程

#### 階段 1：演示標註
```bash
# 自動標註
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --device cpu --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 --auto \
  --input_file ./datasets/raw_demos.hdf5 \
  --output_file ./datasets/annotated_demos.hdf5

# 手動標註（交互式）
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --device cpu --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 \
  --input_file ./datasets/raw_demos.hdf5 \
  --output_file ./datasets/annotated_demos.hdf5
```

#### 階段 2：數據生成
```bash
# 生成新數據集
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --device gpu --num_envs 10 --generation_num_trials 1000 \
  --input_file ./datasets/annotated_demos.hdf5 \
  --output_file ./datasets/generated_demos.hdf5 \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0
```

#### 階段 3：數據整合（可選）
```bash
# 合併多個數據集
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/consolidated_demo.py \
  --input_files ./datasets/dataset1.hdf5 ./datasets/dataset2.hdf5 \
  --output_file ./datasets/consolidated.hdf5
```

### 1.4 技術特點

#### 對象中心分割
- **子任務邊界檢測**：自動或手動標註關鍵操作邊界
- **對象關聯**：每個子任務關聯特定的操作對象
- **相對座標表示**：保持末端執行器與對象的相對關係

#### 場景適應能力
- **隨機化支持**：對象位置、子任務時序的隨機化
- **約束處理**：支持複雜的多臂協調約束
- **噪聲注入**：動作和姿態的可配置噪聲

#### 選擇策略
```python
class SelectionStrategy:
    def select_source_demo(self, eef_pose, object_pose, src_subtask_datagen_infos):
        """選擇最適合的源演示"""
        # 基於當前狀態選擇源演示
        # 支持隨機選擇、距離基礎選擇等策略
```

---

## 2. AMP (Adversarial Motion Priors)

### 2.1 架構設計

AMP 使用對抗訓練學習自然的人形機器人運動，包含策略網絡、價值網絡和判別器網絡。

**目錄結構：**
```
source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/
├── humanoid_amp_env.py        # AMP 環境實現
├── humanoid_amp_env_cfg.py    # 環境配置
├── motions/                   # 運動數據
│   ├── motion_loader.py       # 運動數據加載器
│   ├── humanoid_walk.npz      # 走路運動數據
│   ├── humanoid_run.npz       # 跑步運動數據
│   └── humanoid_dance.npz     # 舞蹈運動數據
└── agents/                    # 訓練配置
    ├── skrl_walk_amp_cfg.yaml # 走路訓練配置
    ├── skrl_run_amp_cfg.yaml  # 跑步訓練配置
    └── skrl_dance_amp_cfg.yaml# 舞蹈訓練配置
```

### 2.2 運動數據處理

#### 2.2.1 MotionLoader 類

```python
class MotionLoader:
    """運動數據加載器，支持 NPZ 格式的參考運動"""
    
    def __init__(self, motion_file: str, device: torch.device):
        # 加載運動數據
        data = np.load(motion_file)
        self.dof_positions = torch.tensor(data["dof_positions"])
        self.body_positions = torch.tensor(data["body_positions"])
        self.body_rotations = torch.tensor(data["body_rotations"])
        # ... 其他運動屬性
        
    def sample(self, num_samples: int, times: np.ndarray = None):
        """採樣運動數據，支持時間插值"""
        # 線性插值和球面插值
        return (dof_pos, dof_vel, body_pos, body_rot, body_lin_vel, body_ang_vel)
        
    def sample_times(self, num_samples: int) -> np.ndarray:
        """隨機採樣運動時間點"""
        return self.duration * np.random.uniform(0.0, 1.0, num_samples)
```

#### 2.2.2 插值算法

**線性插值：**
```python
def _interpolate(self, a, b=None, blend=None, start=None, end=None):
    """線性插值，支持多維數據"""
    return (1.0 - blend) * a + blend * b
```

**球面線性插值 (SLERP)：**
```python
def _slerp(self, q0, q1=None, blend=None):
    """四元數球面線性插值，保證旋轉連續性"""
    # 計算角度差
    cos_half_theta = torch.sum(q0 * q1, dim=-1)
    # 選擇最短路徑
    q1 = torch.where(cos_half_theta < 0, -q1, q1)
    # 球面插值計算
    return interpolated_quaternion
```

### 2.3 環境實現

#### 2.3.1 HumanoidAmpEnv 類

```python
class HumanoidAmpEnv(DirectRLEnv):
    """AMP 人形機器人環境"""
    
    def __init__(self, cfg: HumanoidAmpEnvCfg):
        super().__init__(cfg)
        
        # 加載參考運動
        self._motion_loader = MotionLoader(
            motion_file=self.cfg.motion_file, 
            device=self.device
        )
        
        # AMP 觀察緩衝區
        self.amp_observation_buffer = torch.zeros(
            (self.num_envs, self.cfg.num_amp_observations, self.cfg.amp_observation_space)
        )
        
    def collect_reference_motions(self, num_samples: int) -> torch.Tensor:
        """收集參考運動數據用於判別器訓練"""
        # 採樣參考運動時間序列
        times = self._motion_loader.sample_times(num_samples)
        # 獲取運動數據
        motion_data = self._motion_loader.sample(num_samples, times)
        # 計算 AMP 觀察
        return compute_obs(*motion_data)
```

#### 2.3.2 重置策略

```python
def _reset_strategy_random(self, env_ids: torch.Tensor, start: bool = False):
    """基於參考運動的隨機重置"""
    num_samples = env_ids.shape[0]
    
    # 採樣運動時間（如果 start=True 則從開始）
    times = np.zeros(num_samples) if start else self._motion_loader.sample_times(num_samples)
    
    # 獲取運動狀態
    dof_pos, dof_vel, body_pos, body_rot, body_lin_vel, body_ang_vel = \
        self._motion_loader.sample(num_samples, times)
    
    # 設置機器人狀態
    root_state = self.robot.data.default_root_state[env_ids].clone()
    root_state[:, 0:3] = body_pos[:, torso_idx] + self.scene.env_origins[env_ids]
    root_state[:, 3:7] = body_rot[:, torso_idx]
    # ... 設置速度和關節狀態
    
    # 更新 AMP 觀察歷史
    amp_observations = self.collect_reference_motions(num_samples, times)
    self.amp_observation_buffer[env_ids] = amp_observations.view(num_samples, -1, obs_dim)
```

#### 2.3.3 觀察計算

```python
@torch.jit.script
def compute_obs(dof_positions, dof_velocities, root_positions, root_rotations,
                root_linear_velocities, root_angular_velocities, key_body_positions):
    """計算 AMP 觀察向量"""
    obs = torch.cat((
        dof_positions,                    # 關節位置
        dof_velocities,                   # 關節速度
        root_positions[:, 2:3],           # 根部高度
        quaternion_to_tangent_and_normal(root_rotations), # 根部方向
        root_linear_velocities,           # 根部線速度
        root_angular_velocities,          # 根部角速度
        (key_body_positions - root_positions.unsqueeze(-2)).view(batch_size, -1) # 關鍵點相對位置
    ), dim=-1)
    return obs
```

---

## 3. Robomimic 集成

### 3.1 支持的算法

Isaac Lab 集成了 Robomimic 框架，支持多種先進的模仿學習算法：

| 算法 | 描述 | 主要特點 |
|------|------|----------|
| **BC** | Behavioral Cloning | 基礎行為克隆 |
| **BCO** | BC with Observation Noise | 抗噪聲訓練 |
| **HBC** | Hierarchical BC | 分層行為克隆 |
| **IRIS** | Imitation with Relative Importance Sampling | 重要性採樣 |

### 3.2 訓練流程

#### 3.2.1 配置設置

訓練配置通過 JSON 文件管理，支持靈活的參數調整：

```json
{
    "algo_name": "bc",
    "experiment": {
        "name": "bc_training",
        "validate": true,
        "logging": {
            "terminal_output_to_txt": true,
            "log_tb": true
        }
    },
    "train": {
        "data": "./datasets/training_data.hdf5",
        "output_dir": "./logs/robomimic",
        "num_epochs": 2000,
        "batch_size": 100,
        "seed": 1
    }
}
```

#### 3.2.2 訓練腳本

```python
# scripts/imitation_learning/robomimic/train.py 核心流程

def train(config: Config, device: str, log_dir: str):
    """主訓練函數"""
    
    # 1. 設置隨機種子
    np.random.seed(config.train.seed)
    torch.manual_seed(config.train.seed)
    
    # 2. 加載數據集
    trainset, validset = TrainUtils.load_data_for_training(
        config, obs_keys=shape_meta["all_obs_keys"]
    )
    
    # 3. 創建模型
    model = algo_factory(
        algo_name=config.algo_name,
        config=config,
        obs_key_shapes=shape_meta["all_shapes"],
        ac_dim=shape_meta["ac_dim"],
        device=device
    )
    
    # 4. 訓練循環
    for epoch in range(1, config.train.num_epochs + 1):
        step_log = TrainUtils.run_epoch(model, train_loader, epoch)
        # 驗證和檢查點保存
```

#### 3.2.3 動作正規化

```python
def normalize_hdf5_actions(config: Config, log_dir: str) -> str:
    """將動作正規化到 [-1, 1] 範圍"""
    
    with h5py.File(dataset_path, "r+") as f:
        # 計算全局最大最小值
        all_actions = []
        for demo_key in f["data"].keys():
            actions = np.array(f[f"/data/{demo_key}/actions"])
            all_actions.append(actions.flatten())
        
        dataset = np.concatenate(all_actions)
        max_val, min_val = np.max(dataset), np.min(dataset)
        
        # 正規化所有動作
        for demo_key in f["data"].keys():
            actions = np.array(f[f"/data/{demo_key}/actions"])
            normalized = 2 * ((actions - min_val) / (max_val - min_val)) - 1
            f[f"/data/{demo_key}/actions"] = normalized
```

### 3.3 評估系統

#### 3.3.1 策略評估

```python
# scripts/imitation_learning/robomimic/play.py 核心功能

def rollout(policy, env, success_term, horizon, device):
    """執行策略評估"""
    
    policy.start_episode()
    obs_dict, _ = env.reset()
    trajectory = dict(actions=[], obs=[], next_obs=[])
    
    for step in range(horizon):
        # 處理觀察
        obs = prepare_observations(obs_dict["policy"])
        
        # 策略推理
        actions = policy(obs)
        
        # 反正規化動作（如果需要）
        if norm_factor_min is not None and norm_factor_max is not None:
            actions = unnormalize_actions(actions, norm_factor_min, norm_factor_max)
        
        # 執行動作
        obs_dict, _, terminated, truncated, _ = env.step(actions)
        
        # 檢查成功條件
        if success_term.func(env, **success_term.params)[0]:
            return True, trajectory
            
    return False, trajectory
```

#### 3.3.2 圖像觀察處理

```python
def process_image_observations(obs_dict, env):
    """處理圖像觀察用於 Robomimic 推理"""
    
    if hasattr(env.cfg, "image_obs_list"):
        for image_name in env.cfg.image_obs_list:
            if image_name in obs_dict["policy"]:
                # CHW uint8 -> HWC normalized float
                image = torch.squeeze(obs_dict["policy"][image_name])
                image = image.permute(2, 0, 1).float() / 255.0
                image = image.clip(0.0, 1.0)
                obs_dict["policy"][image_name] = image
    
    return obs_dict
```

---

## 4. 實用腳本和工具

### 4.1 Isaac Lab Mimic 工具鏈

#### 4.1.1 演示標註工具

**自動標註模式：**
```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --device cpu \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 \
  --auto \
  --input_file ./datasets/raw_demos.hdf5 \
  --output_file ./datasets/annotated_demos.hdf5
```

**手動標註模式：**
```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --device cpu \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 \
  --input_file ./datasets/raw_demos.hdf5 \
  --output_file ./datasets/annotated_demos.hdf5
```

**手動標註操作：**
- `N`: 開始播放
- `B`: 暫停
- `S`: 標記子任務邊界
- `Q`: 跳過當前演示

#### 4.1.2 數據生成工具

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --device gpu \
  --num_envs 16 \                      # 並行環境數量
  --generation_num_trials 1000 \       # 生成軌跡數量
  --input_file ./datasets/annotated_demos.hdf5 \
  --output_file ./datasets/generated_demos.hdf5 \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 \
  --rendering_mode performance         # 渲染模式：performance/balanced/quality
```

**參數說明：**
- `--num_envs`: 並行環境數量（建議根據 GPU 記憶體調整）
- `--generation_num_trials`: 目標生成的軌跡數量
- `--rendering_mode`: 渲染質量模式
- `--pause_subtask`: 調試模式，每個子任務後暫停

#### 4.1.3 數據整合工具

```bash
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/consolidated_demo.py \
  --input_files ./datasets/dataset1.hdf5 ./datasets/dataset2.hdf5 \
  --output_file ./datasets/consolidated.hdf5 \
  --filter_success_only                # 僅保留成功的軌跡
```

### 4.2 Robomimic 工具鏈

#### 4.2.1 訓練工具

```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
  --algo bc \                          # 算法選擇：bc, bco, hbc, iris
  --dataset ./datasets/training_data.hdf5 \
  --name experiment_name \             # 實驗名稱
  --log_dir robomimic \               # 日誌目錄
  --epochs 2000 \                     # 訓練輪數
  --normalize_training_actions        # 是否正規化動作
```

#### 4.2.2 評估工具

```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
  --checkpoint ./logs/robomimic/model.pth \
  --num_rollouts 50 \                 # 評估次數
  --horizon 800 \                     # 最大步數
  --seed 101 \                        # 隨機種子
  --norm_factor_min -1.0 \            # 動作反正規化參數
  --norm_factor_max 1.0
```

#### 4.2.3 魯棒性評估

```bash
./isaaclab.sh -p scripts/imitation_learning/robomimic/robust_eval.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
  --checkpoint ./logs/robomimic/model.pth \
  --num_rollouts 100 \
  --noise_levels 0.0 0.1 0.2 0.3 \    # 不同噪聲水平
  --randomize_objects                  # 隨機化對象位置
```

### 4.3 AMP 工具

#### 4.3.1 AMP 訓練

```bash
# 使用 SKRL 框架訓練 AMP
./isaaclab.sh -p scripts/rl_training/skrl/train.py \
  --task Isaac-Humanoid-Direct-v0 \
  --algorithm AMP \                    # 指定 AMP 算法
  --agent_cfg ./agents/skrl_walk_amp_cfg.yaml \
  --num_envs 4096 \
  --max_iterations 1500
```

#### 4.3.2 運動數據查看器

```bash
./isaaclab.sh -p source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/motions/motion_viewer.py \
  --file ./motions/humanoid_walk.npz
```

---

## 5. 支持的環境和任務

### 5.1 Isaac Lab Mimic 環境

#### 5.1.1 Franka 堆疊任務

**狀態基礎任務：**
- `Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0`
- `Isaac-Stack-Cube-Franka-IK-Abs-Mimic-v0`

**視覺運動任務：**
- `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0`
- `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Cosmos-Mimic-v0`

**Pinocchio 集成任務：**
- `Isaac-Pickplace-GR1T2-Mimic-v0`
- `Isaac-Nutpour-GR1T2-Mimic-v0`
- `Isaac-Exhaustpipe-GR1T2-Mimic-v0`

#### 5.1.2 環境配置選項

```python
@configclass
class MimicEnvCfg:
    """Mimic 環境基礎配置"""
    
    # 子任務配置
    subtask_configs: dict[str, list[SubTaskConfig]] = ...
    
    # 任務約束配置
    task_constraint_configs: list[TaskConstraintConfig] = ...
    
    # 數據生成配置
    datagen_config: DataGenConfig = DataGenConfig()
    
    # 場景配置
    scene: InteractiveSceneCfg = ...
```

**子任務配置示例：**
```python
SubTaskConfig(
    object_ref="cube",                    # 參考對象
    subtask_term_signal="cube_grasped",   # 終止信號
    selection_strategy="closest",         # 選擇策略
    num_interpolation_steps=50,           # 插值步數
    num_fixed_steps=10,                   # 固定步數
    action_noise=0.01,                    # 動作噪聲
    subtask_term_offset_range=[0, 10],    # 終止偏移範圍
)
```

### 5.2 AMP 環境

#### 5.2.1 人形機器人任務

**基礎 AMP 任務：**
- `Isaac-Humanoid-Direct-v0`：通用人形機器人 AMP 任務

**支持的運動類型：**
- 走路運動 (`humanoid_walk.npz`)
- 跑步運動 (`humanoid_run.npz`)
- 舞蹈運動 (`humanoid_dance.npz`)

#### 5.2.2 AMP 配置參數

```yaml
# SKRL AMP 配置示例
models:
  separate: True
  policy:
    class: GaussianMixin
    network:
      layers: [1024, 512]
      activations: relu
  value:
    class: DeterministicMixin
    network:
      layers: [1024, 512]
      activations: relu
  discriminator:
    class: DeterministicMixin
    network:
      layers: [1024, 512]
      activations: relu

# AMP 特定內存
motion_dataset:
  class: RandomMemory
  memory_size: 200000

reply_buffer:
  class: RandomMemory
  memory_size: 1000000

agent:
  class: AMP
  rollouts: 16
  learning_epochs: 6
  mini_batches: 2
```

### 5.3 其他直接環境

IsaacLab 還支持多種其他機器人平台的直接環境：

| 環境類別 | 環境名稱 | 主要特點 |
|----------|----------|----------|
| **四足機器人** | Isaac-Anymal-C-* | 地形適應、運動控制 |
| **機械手** | Isaac-Shadow-Hand-* | 靈巧操作、物體操控 |
| **蟻群機器人** | Isaac-Ant-v0 | 多腿協調運動 |
| **無人機** | Isaac-Quadcopter-v0 | 飛行控制、懸停 |
| **工業機器人** | Isaac-Factory-* | 工業自動化任務 |

---

## 6. 使用教程

### 6.1 Isaac Lab Mimic 完整工作流程

#### 步驟 1：數據收集

首先使用遙操作收集原始演示數據：

```bash
# 收集遙操作演示
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
  --num_envs 1 \
  --device keyboard
```

#### 步驟 2：演示標註

對收集的演示進行子任務標註：

```bash
# 自動標註（推薦）
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --device cpu \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0 \
  --auto \
  --input_file ./datasets/raw_demos.hdf5 \
  --output_file ./datasets/annotated_demos.hdf5
```

#### 步驟 3：數據生成

從標註演示生成大量訓練數據：

```bash
# 生成 1000 條新軌跡
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --device gpu \
  --num_envs 16 \
  --generation_num_trials 1000 \
  --input_file ./datasets/annotated_demos.hdf5 \
  --output_file ./datasets/mimic_dataset.hdf5 \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Mimic-v0
```

#### 步驟 4：策略訓練

使用生成的數據訓練策略：

```bash
# 使用 Robomimic BC 算法訓練
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
  --algo bc \
  --dataset ./datasets/mimic_dataset.hdf5 \
  --epochs 2000
```

#### 步驟 5：策略評估

評估訓練好的策略：

```bash
# 評估策略性能
./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-v0 \
  --checkpoint ./logs/robomimic/model.pth \
  --num_rollouts 50
```

### 6.2 AMP 人形機器人訓練

#### 步驟 1：環境設置

確保安裝了 SKRL 庫（AMP 唯一支持的框架）：

```bash
./isaaclab.sh -i skrl
```

#### 步驟 2：訓練 AMP 策略

```bash
# 訓練走路行為
./isaaclab.sh -p scripts/rl_training/skrl/train.py \
  --task Isaac-Humanoid-Direct-v0 \
  --algorithm AMP \
  --agent_cfg ./agents/skrl_walk_amp_cfg.yaml \
  --num_envs 4096 \
  --max_iterations 1500
```

#### 步驟 3：測試訓練結果

```bash
# 測試訓練好的策略
./isaaclab.sh -p scripts/rl_training/skrl/play.py \
  --task Isaac-Humanoid-Direct-v0 \
  --checkpoint ./logs/skrl/model.pt \
  --num_envs 64
```

### 6.3 視覺運動策略訓練

#### 步驟 1：收集視覺演示

```bash
# 收集包含攝像頭觀察的演示
./isaaclab.sh -p scripts/environments/teleoperation/teleop_se3_agent.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 \
  --enable_cameras \
  --num_envs 1
```

#### 步驟 2：標註視覺演示

```bash
# 標註視覺演示
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
  --device cpu \
  --enable_cameras \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0 \
  --auto \
  --input_file ./datasets/visual_demos.hdf5 \
  --output_file ./datasets/annotated_visual_demos.hdf5
```

#### 步驟 3：生成視覺數據

```bash
# 生成視覺運動數據
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --device gpu \
  --enable_cameras \
  --num_envs 10 \
  --generation_num_trials 1000 \
  --input_file ./datasets/annotated_visual_demos.hdf5 \
  --output_file ./datasets/visual_mimic_dataset.hdf5 \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-Mimic-v0
```

#### 步驟 4：訓練視覺策略

```bash
# 訓練視覺運動策略
./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
  --task Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor-v0 \
  --algo bc \
  --dataset ./datasets/visual_mimic_dataset.hdf5 \
  --epochs 3000
```

---

## 7. 最佳實踐

### 7.1 數據收集建議

#### 7.1.1 演示質量

**高質量演示的特徵：**
- **動作平滑性**：避免突然的動作變化
- **任務完成度**：確保每個演示都成功完成任務
- **多樣性**：覆蓋不同的初始條件和對象配置
- **一致性**：相似情況下的動作應該一致

**收集策略：**
```python
# 建議的演示收集參數
collection_params = {
    "num_demos": 10-20,              # 初始演示數量
    "max_episode_length": 500,       # 最大步數
    "success_rate_threshold": 0.9,   # 成功率閾值
    "action_smoothing": True,        # 動作平滑
    "diverse_initial_conditions": True # 多樣化初始條件
}
```

#### 7.1.2 標註策略

**自動標註 vs 手動標註：**

| 標註方式 | 優點 | 缺點 | 適用場景 |
|----------|------|------|----------|
| **自動標註** | 快速、一致、可重複 | 需要實現檢測函數 | 明確定義的任務 |
| **手動標註** | 靈活、適用性廣 | 耗時、主觀性強 | 複雜或模糊的任務 |

**子任務邊界選擇原則：**
- 選擇語意上重要的時刻（如抓取、放置）
- 避免過於細粒度的分割
- 保證每個子任務有明確的目標

### 7.2 數據生成優化

#### 7.2.1 性能調優

**並行環境配置：**
```python
# 根據硬件調整並行環境數量
gpu_memory_gb = 24  # GPU 記憶體大小
recommended_envs = min(gpu_memory_gb * 2, 32)  # 經驗公式

# 生成參數建議
generation_params = {
    "num_envs": recommended_envs,
    "generation_num_trials": 1000,
    "rendering_mode": "performance",  # 性能優先
    "enable_cameras": False,          # 如不需要圖像則關閉
}
```

**內存優化：**
- 使用 `performance` 渲染模式
- 合理設置 batch size
- 及時清理不需要的數據

#### 7.2.2 質量控制

**數據篩選策略：**
```python
# 建議的數據篩選條件
quality_filters = {
    "success_rate": "> 0.95",        # 高成功率
    "trajectory_length": "< 600",     # 合理長度
    "action_smoothness": "> 0.8",     # 動作平滑度
    "object_displacement": "< 0.1",   # 對象位移合理性
}
```

### 7.3 訓練策略

#### 7.3.1 超參數調優

**Robomimic BC 推薦參數：**
```json
{
    "train": {
        "batch_size": 100,
        "num_epochs": 2000,
        "learning_rate": 1e-4,
        "weight_decay": 1e-4
    },
    "model": {
        "hidden_dimensions": [1024, 1024],
        "dropout": 0.1,
        "activation": "relu"
    }
}
```

**AMP 推薦參數：**
```yaml
agent:
  learning_rate_policy: 1e-4
  learning_rate_critic: 1e-3
  learning_rate_discriminator: 1e-4
  discount_factor: 0.95
  gae_lambda: 0.95
  clip_ratio: 0.2
```

#### 7.3.2 訓練監控

**關鍵指標監控：**
- **訓練損失**：應穩定下降
- **驗證損失**：避免過擬合
- **成功率**：定期評估任務成功率
- **動作分佈**：檢查動作範圍合理性

### 7.4 調試技巧

#### 7.4.1 常見問題診斷

**數據生成失敗：**
```bash
# 啟用調試模式
./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
  --pause_subtask \  # 每個子任務後暫停
  --num_envs 1 \     # 單環境調試
  --headless false   # 可視化調試
```

**訓練不收斂：**
1. 檢查數據質量和標註正確性
2. 調整學習率和 batch size
3. 增加訓練數據量
4. 檢查動作空間範圍

#### 7.4.2 性能分析

**內存使用監控：**
```python
import psutil
import os

def monitor_memory():
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"Memory usage: {memory_mb:.1f} MB")
```

**GPU 利用率監控：**
```bash
# 監控 GPU 使用情況
nvidia-smi -l 1
```

---

## 8. 常見問題

### 8.1 安裝和環境問題

#### Q1: SKRL 安裝失敗
**問題**: AMP 需要 SKRL，但安裝時出現依賴衝突

**解決方案**:
```bash
# 使用 Isaac Lab 提供的安裝腳本
./isaaclab.sh -i skrl

# 如果仍有問題，手動安裝
pip install skrl[torch]
```

#### Q2: Robomimic 導入錯誤
**問題**: 訓練時出現 robomimic 模組導入錯誤

**解決方案**:
```bash
# 安裝必要的依賴
sudo apt install cmake build-essential
./isaaclab.sh -i robomimic

# 驗證安裝
python -c "import robomimic; print('OK')"
```

### 8.2 數據相關問題

#### Q3: 演示標註失敗
**問題**: 自動標註報告子任務信號未檢測到

**解決方案**:
1. 檢查環境是否實現了 `get_subtask_term_signals()` 方法
2. 使用手動標註模式作為備選
3. 調整子任務檢測閾值

```python
# 檢查環境實現
if env.get_subtask_term_signals.__func__ is ManagerBasedRLMimicEnv.get_subtask_term_signals:
    print("需要實現 get_subtask_term_signals 方法")
```

#### Q4: 數據生成內存不足
**問題**: 生成數據時 GPU 內存溢出

**解決方案**:
```bash
# 減少並行環境數量
--num_envs 4  # 從 16 減少到 4

# 使用性能模式
--rendering_mode performance

# 關闉不必要的感測器
--disable_cameras  # 如果不需要視覺
```

### 8.3 訓練問題

#### Q5: BC 訓練不收斂
**問題**: 訓練損失不下降或策略性能差

**解決方案**:
1. 檢查數據質量和數量
2. 調整超參數
3. 增加正規化

```json
{
    "train": {
        "batch_size": 50,          // 減小 batch size
        "learning_rate": 5e-5,     // 降低學習率
        "weight_decay": 1e-3       // 增加正規化
    }
}
```

#### Q6: AMP 訓練不穩定
**問題**: AMP 訓練過程中判別器損失震盪

**解決方案**:
```yaml
# 調整學習率比例
learning_rate_policy: 1e-4
learning_rate_discriminator: 5e-5  # 降低判別器學習率

# 增加回放緩衝區
reply_buffer:
  memory_size: 2000000  # 加倍緩衝區大小
```

### 8.4 性能優化問題

#### Q7: 訓練速度過慢
**問題**: 訓練時間過長，GPU 利用率低

**解決方案**:
1. 增加 batch size（在內存允許的情況下）
2. 使用多 GPU 訓練
3. 減少不必要的記錄和可視化

```json
{
    "train": {
        "batch_size": 200,         // 增加 batch size
        "num_data_workers": 4      // 增加數據加載線程
    },
    "experiment": {
        "logging": {
            "log_tb": false        // 關閉 tensorboard（訓練時）
        }
    }
}
```

#### Q8: 評估時策略表現不一致
**問題**: 同一策略在不同運行中表現差異很大

**解決方案**:
1. 固定隨機種子
2. 增加評估次數
3. 檢查環境隨機化設置

```bash
# 固定種子評估
./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
  --seed 42 \              # 固定種子
  --num_rollouts 100       # 增加評估次數
```

### 8.5 部署問題

#### Q9: 策略在真實機器人上表現差
**問題**: 仿真中訓練的策略在真實環境中失效

**解決方案**:
1. 增加訓練時的隨機化
2. 收集更多樣化的演示數據
3. 使用領域適應技術

```python
# 增加仿真隨機化
env_cfg.randomization = {
    "object_position_noise": 0.02,
    "action_noise": 0.01,
    "observation_noise": 0.005
}
```

#### Q10: 部署時動作範圍問題
**問題**: 策略輸出的動作超出機器人安全範圍

**解決方案**:
1. 訓練時使用動作正規化
2. 部署時添加安全限制
3. 重新標定動作空間

```python
# 動作限制
def safe_action_clipping(actions, action_limits):
    return torch.clamp(actions, action_limits.low, action_limits.high)
```

---

## 總結

IsaacLab 提供了完整的模仿學習工具鏈，涵蓋從數據收集到策略部署的全流程。每種方法都有其特定的優勢和適用場景：

- **Isaac Lab Mimic**: 最適合操作任務，特別是需要處理對象交互的場景
- **AMP**: 專門用於學習自然流暢的運動模式
- **Robomimic**: 提供成熟的算法實現和廣泛的兼容性

選擇合適的方法並遵循最佳實踐，可以大大提高模仿學習的效果和效率。建議根據具體任務需求，結合硬件條件和時間預算，選擇最適合的方法和配置。
