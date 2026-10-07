import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

parser = argparse.ArgumentParser(prog="eval.sh", description="Visualize a released PRISM policy in Isaac Sim.")
parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data/far-prism-data", help="Extracted dataset directory containing clips.csv")
parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs/eval", help="Logs, geometry audit and USD caches")
parser.add_argument("--checkpoint", choices=["student_28000.pt", "box_23000.pt"], default="student_28000.pt")
parser.add_argument("--clip", default="prism_cf_box_m0_v0", help="Published box or ball clip ID")
parser.add_argument("--headless", action="store_true", help="Run without a window")
parser.add_argument("--steps", type=int, help="Stop after this many control steps")
parser.add_argument("--check", action="store_true", help="Check assets and actor on CPU")
parser.add_argument("--demo", action="store_true", help="Standard Isaac floor, Unitree materials, policy depth and reference goal")
parser.add_argument("--record", type=Path, help="Record DLAA demo to MP4 at simulation speed (default: 450 steps)")
parser.add_argument("--box-color", type=float, nargs=3, metavar=("R", "G", "B"), help="Display-only box color, RGB values from 0 to 1")
args = parser.parse_args()
if args.box_color is not None:
    if not all(0 <= value <= 1 for value in args.box_color):
        parser.error("--box-color values must be between 0 and 1")
    args.demo = True
if args.record:
    args.record = args.record.expanduser().resolve()
    if args.record.exists() or args.record.with_suffix(".json").exists():
        parser.error(f"Recording already exists: {args.record}")
    import shutil
    if shutil.which("ffmpeg") is None:
        parser.error("--record requires ffmpeg on PATH")
    args.demo = True
    if args.steps is None:
        args.steps = 450
if args.demo and args.headless:
    parser.error("--demo requires a window; omit --headless")
if args.steps is not None and args.steps < 1:
    parser.error("--steps must be positive")

repo = Path(__file__).resolve().parents[1]
dataset = args.data_dir.expanduser().resolve()
output = args.output_dir.expanduser().resolve()
if not (dataset / "clips.csv").is_file():
    parser.error(f"Dataset index not found: {dataset / 'clips.csv'}; run bash download_data.sh or pass --data-dir")
with (dataset / "clips.csv").open() as stream:
    clips = {row["clip_id"]: row for row in csv.DictReader(stream)}
if args.clip not in clips or clips[args.clip]["category"] not in {"box", "ball"}:
    parser.error("--clip must name a published box or ball clip from data/far-prism-data/clips.csv")
if clips[args.clip]["category"] != "box" and (args.checkpoint == "box_23000.pt" or args.box_color is not None):
    parser.error("Ball clips require student_28000.pt and no --box-color")
motion = dataset / clips[args.clip]["student_motion"]
bank = motion.parent
contacts = dataset / "data/train-student/data/contact_sidecars"
assets = dataset / "data/train-student/data/robot_assets"
objects = json.loads((bank / "_clip_object_urdf_map.json").read_text())["clips"]
urdf = (bank / objects[args.clip]["object_urdf_path"]).resolve()
checkpoint = repo / "_ckpts" / args.checkpoint
entry = next(item for item in json.loads((repo / "_ckpts/manifest.json").read_text())["checkpoints"]
             if item["file"] == checkpoint.name)
if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != entry["sha256"]:
    raise ValueError("Released checkpoint SHA256 mismatch")
for path in (motion, urdf, contacts, assets):
    if not path.exists():
        raise FileNotFoundError(path)

cache = output / "cache"
os.environ.update({
    "WANDB_MODE": "disabled", "OMNI_KIT_ACCEPT_EULA": "1",
    "HOLOSOMA_EVAL_POLICY": "checkpoint_actor",
    "HOLOSOMA_EXPECTED_EVALUATION_CHECKPOINT_SHA256": entry["sha256"],
    "HOLOSOMA_OBJECT_SPAWN_MODE": "mesh",
    "HOLOSOMA_OBJECT_COLLIDER_TYPE": "convex_decomposition",
    "HOLOSOMA_ROBOT_USD_CACHE_DIR": str(cache / "robot_usd"),
    "HOLOSOMA_OBJECT_USD_CACHE_DIR": str(cache / "object_usd"),
    "HOLOSOMA_PERCEPTION_MESH_CACHE_DIR": str(cache / "perception_meshes"),
    "CONTACT_EXPORT_ROOT": str(contacts), "CONTACT_SIDECAR_MODE": "runtime-intervals",
})
# Only the legacy box initializer lacks motion-transition metadata.
os.environ.pop("HOLOSOMA_EVAL_ALLOW_AUTHENTICATED_LEGACY_MOTION_CONTRACT", None)
if checkpoint.name == "box_23000.pt":
    os.environ["HOLOSOMA_EVAL_ALLOW_AUTHENTICATED_LEGACY_MOTION_CONTRACT"] = "1"

import torch
import tyro
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.eval_agent import _bind_training_perception_reference_batch, _validate_eval_policy_contract
from holosoma.utils.eval_utils import CheckpointConfig, load_saved_experiment_config
from holosoma.utils.tyro_utils import TYRO_CONIFG

checkpoint_config = CheckpointConfig(checkpoint=str(checkpoint))
saved, origin = load_saved_experiment_config(checkpoint_config)
prefix = "--command.setup-terms.motion-command.params.motion-config."
cli = [
    f"--training.headless={args.headless}", "--training.multigpu=False", "--training.num-envs=1",
    "--training.export-onnx=False", "--training.name=prism_eval",
    f"--robot.asset.asset-root={assets}", f"--robot.object.object-urdf-path={urdf}",
    prefix + f"motion-file={motion}",
    prefix + f"adaptive-sampling-contact-interval-root={contacts / 'clips'}",
    prefix + "use-adaptive-timesteps-sampler=False",
    prefix + "start-at-timestep-zero-prob=1.0", prefix + "freeze-at-timestep-zero-prob=0.0",
    prefix + "noise-to-initial-pose.overall-noise-scale=0.0",
]
# Visualization needs no training rewards; published assets omit some of their targets.
cli += [f"--reward.terms.{name.replace('_', '-')}.weight=0.0"
        for name in saved.reward.terms]
# ponytail: PhysX buffers sized for one scene; raise these if larger scenes report overflow.
cli += [f"--simulator.config.sim.physx.{name}=1048576" for name in (
    "gpu-max-rigid-contact-count", "gpu-found-lost-pairs-capacity",
    "gpu-found-lost-aggregate-pairs-capacity", "gpu-total-aggregate-pairs-capacity")]
if args.steps is not None:
    cli.append(f"--training.max-eval-steps={args.steps}")
config = tyro.cli(ExperimentConfig, default=saved.get_eval_config(), args=cli, config=TYRO_CONIFG)
_validate_eval_policy_contract(saved, config)
print(f"PRISM actor: {checkpoint.name}; scene: {args.clip}", flush=True)
if args.check:
    from holosoma.agents.modules.module_utils import setup_ppo_actor_module
    torch.set_num_threads(2)
    dims = {"actor_obs_root_contact_aware": 3, "actor_obs_drop_button": 1,
            "actor_obs_proprio_with_actions_no_linvel": 90, "perception_obs": 5046, "robot_action_dim": 29}
    actor = setup_ppo_actor_module(dims, config.algo.config.module_dict.actor, 29, 0.01,
                                   "cpu", {key: 1 for key in dims}).eval()
    actor.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["actor_model_state_dict"], strict=True)
    with torch.inference_mode():
        actions = actor.act_inference({"actor_obs": torch.zeros(1, 94), "perception_obs": torch.zeros(1, 5046)})
    assert actions.shape == (1, 29) and torch.isfinite(actions).all()
    print("PASS: checkpoint SHA256, scene assets, observation contract and CPU actor inference")
else:
    from holosoma.utils.helpers import get_class
    from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
    # The checkpoint already contains its actor config. Reapplying perception
    # presets would rewrite legacy CNN flags and fail upstream's identity check.
    config = _bind_training_perception_reference_batch(saved, config)
    env, device, app = setup_simulation_environment(config)
    try:
        algo = get_class(config.algo._target_)(device=device, env=env, config=config.algo.config,
                                               log_dir=str(output), multi_gpu_cfg=None)
        algo.attach_evaluation_metadata(saved, config, origin)
        if checkpoint.name == "student_28000.pt":
            print("OOD object evaluation: published simplified mesh differs from training geometry", flush=True)
            algo.enable_evaluation_only_ood_object_geometry()
        algo.setup()
        algo.load_evaluation(str(checkpoint))
        audit = algo.evaluation_ood_object_geometry_audit()
        if audit is not None:
            audit_path = output / "student_geometry_audit.json"
            audit_path.parent.mkdir(parents=True, exist_ok=True)
            audit_path.write_text(json.dumps(audit, indent=2))
        if args.demo:
            import sys
            from dataclasses import replace
            from types import SimpleNamespace
            sys.path.insert(0, str(repo / "scripts"))
            algo.config = replace(algo.config, eval_callbacks={"demo": SimpleNamespace(
                _target_="prism_demo.Demo", config={"record": args.record, "box_color": args.box_color})})
        algo.evaluate_policy(max_eval_steps=config.training.max_eval_steps)
        print("PRISM_EVAL_COMPLETE", flush=True)
    except BaseException:
        import sys
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        # Isaac Sim's native close may exit with code 0 even after a Python failure.
        os._exit(1)
    finally:
        if app:
            close_simulation_app(app)
