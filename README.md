# raceformer
pretraining transformers for autonomous racing-based RL

## 1. Project Overview
**Objective:** Improve autonomous driving policy robustness in simulation (`racecar_gym`) by pretraining a multimodal representation model on real-world sensor data (CMHT dataset).

**Hypothesis:** Latent representations learned from noisy, asynchronous real-world data (LiDAR, RGB, Radar, IR) provide better perception priors than training from scratch in a clean simulator.

**Stack:**
- **Framework:** JAX
- **Neural Nets:** Flax NNX (Experimental functional API)
- **Optimizer:** Optax
- **Checkpointing:** Orbax
- **Metrics:** Metrax
- **Data Loading:** Grain
- **Compute:** Rutgers Amarel (4x A100 Nodes)

---

## 2. Environment Setup
The codebase requires a specific environment to handle `flax.nnx` and the `racecar_gym` physics engine.

### Dependencies
Create a `pyproject.toml` using uv with:
- `jax[cuda13]`, `jaxlib`
- `flax>=0.8.4` (Ensure NNX support)
- `optax`, `orbax-checkpoint`, `grain`
- `dm-tree`, `numpy`
- `gym`, `pybullet`
- `racecar_gym` (Install from source: https://github.com/axelbr/racecar_gym)

### GitHub Actions
The `.github/workflows/lint.yml` must execute `ruff check .` on every push.

---

## 3. Data Pipelines (`src/data`)

### Real-World: CMHT (Pretraining)
**Source:** [CMHT Dataset](https://www.frdr-dfdr.ca/repo/dataset/88b29aa4-b77a-4500-8812-9bec4bae9a16)
**File:** `src/data/cmht_loader.py`
**Task:**
1.  Implement a **Grain** data source that reads the raw CMHT files.
2.  **Synchronization:** The loader must align timestamps. LiDAR (10Hz) is the master clock. Interpolate GPS/IMU and select nearest neighbor frames for RGB/IR/Radar.
3.  **Preprocessing:**
    -   *LiDAR/Radar:* Voxelize or project to Bird's Eye View (BEV) images for ViT consumption, OR keep as point patches.
    -   *RGB/IR:* Resize to 224x224 (or similar ViT friendly resolution).
    -   *Normalization:* Standard ImageNet stats for RGB; Min/Max for Range data.

### Simulation: RacecarGym (Finetuning)
**Source:** [racecar_gym](https://github.com/axelbr/racecar_gym)
**File:** `src/data/gym_wrapper.py`
**Task:**
1.  Wrap the Gym environment to output dictionaries matching the CMHT structure.
2.  *Note:* The sim lacks Radar/IR. The model must handle **missing modalities** (Masking) during finetuning.
3.  Implement a JAX-compatible step function (interfacing host-callback if necessary for PyBullet).

---

## 4. Model Architecture (`src/models`)
**Framework:** Use `flax.nnx.Module` exclusively. Do not use `flax.linen`.

### Encoders (`encoders.py`)
Implement a `ModalityViT(nnx.Module)`:
-   **Inputs:** Specific sensor tensor.
-   **Structure:** PatchEmbed -> PositionalEmbedding -> N x TransformerEncoderBlocks.
-   **Output:** Sequence of latent tokens $Z_{modality}$.

### Fusion (`fusion.py`)
Implement `MultimodalFusion(nnx.Module)`:
-   **Input:** Concatenated tokens $[Z_{lidar}, Z_{rgb}, Z_{radar}, Z_{ir}]$.
-   **Structure:** Deep stack of Transformer blocks (Self-Attention).
-   **Mechanism:** Allow Cross-Attention if fusing differently, but concatenation + Self-Attention is preferred for simplicity.

### Heads (`heads.py`)
1.  **PretrainHead:** Reconstructs masked inputs (MAE style) or Contrastive Projection.
2.  **PolicyHead:** `Dense -> Tanh` (Continuous Action Space).
3.  **ValueHead:** `Dense -> Scalar` (Critic).

---

## 5. Training Workflow

### Phase 1: Representation Learning (Pretraining)
**Script:** `src/training/pretrainer.py`
**Config:** `configs/pretrain_cmht.toml`
**Strategy:** Masked Multimodal Modeling.
1.  Load CMHT batch.
2.  Randomly mask patches across all modalities.
3.  Forward pass unmasked patches.
4.  Predict masked patches.
5.  **Loss:** MSE on reconstruction.
6.  **Checkpoint:** Save `orbax` checkpoint to `runs/{run_name}/ckpt`.

### Phase 2: RL Finetuning
**Script:** `src/training/rl_agent.py`
**Config:** `configs/rl_racecar.toml`
**Strategy:** PPO (Proximal Policy Optimization).
1.  Load pretrained weights from Phase 1.
2.  Freeze encoders (optional, controllable via config).
3.  Interact with `racecar_gym`.
4.  **Observation:** RGB + LiDAR + GPS (Mask Radar/IR inputs as zeroes or learnable tokens).
5.  **Reward:** Progress - Collision penalty.
6.  **Loss:** PPO Actor-Critic loss.

---

## 6. Configuration & Logging

### Config Structure (TOML)
Example `configs/pretrain_cmht.toml`:
```toml
[model]
embed_dim = 256
depth = 6
num_heads = 8

[training]
batch_size = 64
learning_rate = 3e-4
steps = 100000
mask_ratio = 0.5

[data]
path = "/path/to/cmht"