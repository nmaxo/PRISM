<h1 align="center">
  Counterfactual Video Generation Enables Scalable Humanoid Loco-Manipulation
</h1>

<p align="center">
  <a href="https://arxiv.org/abs/2609.38172"><img src="https://img.shields.io/badge/arXiv-2609.38172-b31b1b.svg" alt="arXiv"></a>
  <a href="https://prism-real2sim2real.github.io/"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
  <a href="https://huggingface.co/datasets/Amazon-FAR/far-prism-data"><img src="https://img.shields.io/badge/Dataset-Hugging%20Face-yellow.svg" alt="Dataset on Hugging Face"></a>
</p>

<p align="center">
  This repository provides simulation training and real-robot deployment for PRISM, based on <a href="https://github.com/amazon-far/holosoma">HoloSoma</a>.
</p>

## Code branches

| Branch | Use |
|---|---|
| [`main`](https://github.com/amazon-far/PRISM/tree/main) | Simulation: teacher training, rollout collection and student distillation |
| [`sim2real`](https://github.com/amazon-far/PRISM/tree/sim2real) | Real-robot deployment with FastFoundationStereo |

For real-robot deployment, switch an existing clone to `sim2real` and follow that branch's README:

```bash
git fetch origin
git switch sim2real
git submodule update --init --recursive
```

## Installation

Linux, Python 3.11 and a compatible NVIDIA GPU are required.
See [system requirements and environment details](docs/installation.md).

```bash
git clone --branch main https://github.com/nmaxo/PRISM.git
cd PRISM
python3.11 -m venv .venv
source .venv/bin/activate
bash install.sh
```

## Data

Download and prepare the [Hugging Face dataset](https://huggingface.co/datasets/Amazon-FAR/far-prism-data) under `data/`:

```bash
bash download_data.sh
```

Includes object meshes and prepares the student training shards.
[Dataset details](docs/data.md).

Both training scripts use all visible GPUs on this machine. Set
`CUDA_VISIBLE_DEVICES=0` for one GPU or `CUDA_VISIBLE_DEVICES=0,1` for a subset.
Use `--envs-per-gpu` to lower memory use. [Training details](docs/training.md).

## Evaluation / pretrained policy demo

This fork adds [`eval.sh`](eval.sh) for the released depth student on published
box and ball scenes, and for the earlier box initializer. Install the environment
and download the data above first; the checkpoints are already in `_ckpts/`.
Evaluation uses one GPU and one environment. W&B login is not required for eval;
run `wandb login` before training.

```bash
source .venv/bin/activate
bash eval.sh --check                     # Assets, checkpoint SHA256 and CPU actor inference
bash eval.sh                            # Isaac Sim window, default box scene
bash eval.sh --headless --steps 100      # Bounded GPU smoke run
bash eval.sh --demo                      # Unitree materials, live policy depth and goal marker
```

Select a small box, a ball, or the earlier box policy:

```bash
bash eval.sh --demo --clip prism_cf_box_m3_v26 --box-color 1 0.25 0.04
bash eval.sh --demo --clip prism_cf_ball_m1_v10
bash eval.sh --checkpoint box_23000.pt --headless --steps 100
```

Clip IDs come from `data/far-prism-data/clips.csv`. This launcher supports box
and ball clips; `box_23000.pt` and `--box-color` require a box. GUI runs continue
until the window is closed unless `--steps` is set. `--demo`, `--box-color` and
`--record` require a display and cannot be combined with `--headless`.

Record a demo with DLAA, the depth panel and simulation-time playback
(requires `ffmpeg` on `PATH`, for example `sudo apt install ffmpeg`):

```bash
bash eval.sh --record outputs/eval/small_box.mp4 --clip prism_cf_box_m3_v26 --box-color 1 0.25 0.04
bash eval.sh --record outputs/eval/ball.mp4 --clip prism_cf_ball_m1_v10 --steps 450
```

Recording defaults to 450 control steps: 450 frames at 50 FPS, or 9 seconds of
1920x1080 H.264 video. A matching JSON sidecar stores simulation timestamps and
render settings. Existing video or sidecar files are rejected. Unitree material
attribution and license are in [`scripts/demo_assets/README.md`](scripts/demo_assets/README.md).

The launcher works from any working directory and uses the active environment's
`python3`; `PRISM_PYTHON` can select a different interpreter. For an existing data
download or a separate output directory:

```bash
PRISM_PYTHON=/path/to/venv/bin/python bash eval.sh --headless --steps 100 \
  --data-dir /path/to/far-prism-data --output-dir /path/to/eval-output
bash eval.sh --help
python3 tests/test_eval_cli.py            # Lightweight CLI regression check; no simulator needed
```

Logs, USD caches and the student's `student_geometry_audit.json` go under
`outputs/eval/` by default; data and generated results stay Git-ignored.
The launched simulator accepts NVIDIA's EULA through `OMNI_KIT_ACCEPT_EULA=1`.

These are pretrained policy rollouts, not a benchmark success-rate evaluation
or an exact training resume. The published object visual meshes differ from
the student's authenticated training geometry, so student runs explicitly use
HoloSoma's evaluation-only OOD geometry mode and save its audit. Actor, observation,
action and depth-preprocessing checks remain enabled. Each scene starts at frame
zero without initial pose noise; training rewards are disabled because the public
data omits some reward references. Physics/control settings come from the
checkpoint, with PhysX buffer capacities reduced for this single-environment run.
The launcher uses the pinned HoloSoma environment/PPO APIs to preserve the released
actor configuration; generic eval presets would rewrite its legacy CNN flags.

## Teacher Training

Train a privileged motion-tracking policy with PPO.
Teacher-training data is still under review; supply a prepared teacher bank.

```bash
bash train_teacher.sh --motion-bank /path/to/teacher_bank --entity YOUR_WANDB_ENTITY
```

## Rollout

Collect teacher trajectories and contact sidecars into `outputs/rollout/`.

```bash
bash rollout.sh --motion-bank /path/to/teacher_bank
```

## Student Distillation

Distill the teacher into a depth policy using the prepared data under `data/`.

```bash
bash train_student.sh --entity YOUR_WANDB_ENTITY
```

## Citation

If you use PRISM in your research, please cite:

```bibtex
@article{wang2026counterfactual,
  title={Counterfactual Video Generation Enables Scalable Humanoid Loco-Manipulation},
  author={Wang, Zihan and Wu, Zhen and Abbeel, Pieter and Duan, Rocky and Malik, Jitendra and Sferrazza, Carmelo and Liu, C. Karen and Shi, Guanya and Kanazawa, Angjoo},
  journal={arXiv preprint arXiv:2609.38172},
  year={2026}
}
```

## License and security

See [LICENSE](LICENSE), [NOTICE](docs/legal/NOTICE) and [THIRD_PARTY_LICENSES](docs/legal/THIRD_PARTY_LICENSES).

Report security issues through [AWS Vulnerability Reporting](https://aws.amazon.com/security/vulnerability-reporting/).
