"""
Real-time Rerun playback of a frozen PPO policy on ManiSkill PickCubeSO100-v1.

Debugging / qualitative tool only: it is NOT the frozen evaluation protocol
(single env, sequential seeds). Use rl_lab/scripts/eval_ppo_maniskill.py for numbers.

The object (cube), TCP and goal shown in 3D are decoded from the 36-d policy
observation, i.e. what the policy sees. Robot links come from the simulator.

Examples:
    # Viewer on this machine (needs a display)
    python scripts/demo_policy_rerun.py --rr spawn
    # Headless server: web viewer; forward ports 9090 and 9876 (VS Code does it automatically)
    python scripts/demo_policy_rerun.py --rr serve
    # Native viewer on your laptop: run `rerun` there, then `ssh -R 9876:localhost:9876 <server>`
    python scripts/demo_policy_rerun.py --rr connect
    # Record to file
    python scripts/demo_policy_rerun.py --rr save --rr_path logs/demo.rrd --speed 0
"""

import argparse
import sys
import time
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch
import gymnasium as gym

ROOT = Path(__file__).resolve().parents[1]
RL_LAB = ROOT / "submodules" / "rl_lab"
sys.path.append(str(RL_LAB))

import rerun as rr  # noqa: E402
import rerun.blueprint as rrb  # noqa: E402
import mani_skill.envs  # noqa: E402,F401  (registers ManiSkill tasks with gym)
from mani_skill.utils.wrappers.flatten import FlattenActionSpaceWrapper  # noqa: E402

from rl.networks import GaussianActor, ValueNet  # noqa: E402
from rl.evaluation import EVAL_SEED_BASE  # noqa: E402
from src.ex3_ppo_config import PPO_MANISKILL_PARAMETERS  # noqa: E402

# Layout of the 36-d "state" observation of PickCubeSO100-v1 (verified against
# env.unwrapped: agent qpos/qvel, then the extras of PickCubeEnv._get_obs_extra).
QPOS = slice(0, 6)
IS_GRASPED = 12
TCP_POSE = slice(13, 20)
GOAL_POS = slice(20, 23)
OBJ_POSE = slice(23, 30)

ACTION_NAMES = ["shoulder_pan", "shoulder_lift", "elbow", "wrist_pitch", "wrist_roll", "gripper"]
LINK_COLORS = {"Fixed_Jaw": (230, 160, 60), "Moving_Jaw": (230, 160, 60)}


def wxyz_to_xyzw(q):
    return np.asarray([q[1], q[2], q[3], q[0]], dtype=np.float32)


def to_np(x):
    return x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)


def init_rerun(args):
    blueprint = rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial3DView(name="scene (policy view)", origin="world"),
                rrb.Horizontal(
                    rrb.Spatial2DView(name="render camera", origin="cameras/render"),
                    rrb.Spatial2DView(name="base camera rgb", origin="world/base_camera/image/rgb"),
                    rrb.Spatial2DView(name="base camera depth", origin="world/base_camera/image/depth"),
                ),
                row_shares=[3, 2],
            ),
            rrb.Vertical(
                rrb.TimeSeriesView(name="reward / value", origin="plots/reward"),
                rrb.TimeSeriesView(name="distances [m]", origin="plots/dist"),
                rrb.TimeSeriesView(name="flags", origin="plots/flags"),
                rrb.TimeSeriesView(name="actions", origin="plots/action"),
                rrb.TextLogView(name="episodes", origin="log"),
            ),
            column_shares=[3, 2],
        ),
        collapse_panels=True,
    )
    rr.init("rl_lab_policy_demo", default_blueprint=blueprint)
    if args.rr == "spawn":
        rr.spawn(default_blueprint=blueprint)
    elif args.rr == "serve":
        uri = rr.serve_grpc(grpc_port=args.grpc_port, default_blueprint=blueprint)
        rr.serve_web_viewer(web_port=args.web_port, open_browser=False, connect_to=uri)
        # '+' in a query string decodes to a space, so the gRPC URL must be percent-encoded.
        url = quote(uri.replace("127.0.0.1", "localhost"), safe="")
        print(f"Web viewer: http://localhost:{args.web_port}/?url={url}")
        print(f"(forward ports {args.web_port} and {args.grpc_port} if running remotely)")
    elif args.rr == "connect":
        rr.connect_grpc(args.rr_url, default_blueprint=blueprint)
    else:
        Path(args.rr_path).parent.mkdir(parents=True, exist_ok=True)
        rr.save(args.rr_path, default_blueprint=blueprint)
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)


def log_static_scene(env):
    u = env.unwrapped
    for link in u.agent.robot.links:
        # Link.get_visual_meshes is broken in mani_skill 3.0.1 (shadows builtin filter).
        meshes = link.generate_mesh(lambda _l, _s: True, "visual")
        mesh = meshes[0] if meshes else None
        if mesh is None or len(mesh.vertices) == 0:
            continue
        rr.log(
            f"world/robot/{link.name}",
            rr.Mesh3D(
                vertex_positions=mesh.vertices,
                triangle_indices=mesh.faces,
                vertex_normals=mesh.vertex_normals,
                albedo_factor=LINK_COLORS.get(link.name, (200, 200, 210)),
            ),
            static=True,
        )

    half = float(u.cube_half_size)
    rr.log("world/cube", rr.Boxes3D(half_sizes=[half] * 3, colors=(220, 50, 50), fill_mode="solid"), static=True)
    rr.log("world/tcp", rr.TransformAxes3D(axis_length=0.03), static=True)

    params = {k: to_np(v[0]) for k, v in u.get_sensor_params()["base_camera"].items()}
    cam = u._sensors["base_camera"].config
    world_from_cam = np.linalg.inv(np.vstack([params["extrinsic_cv"], [0, 0, 0, 1]]))
    rr.log(
        "world/base_camera",
        rr.Transform3D(translation=world_from_cam[:3, 3], mat3x3=world_from_cam[:3, :3]),
        static=True,
    )
    rr.log(
        "world/base_camera/image",
        rr.Pinhole(image_from_camera=params["intrinsic_cv"], resolution=[cam.width, cam.height],
                   camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=0.05),
        static=True,
    )


def log_step(env, obs, action, reward, value, info, log_cameras):
    u = env.unwrapped
    o = to_np(obs[0])

    for link in u.agent.robot.links:
        p = to_np(link.pose.raw_pose[0])
        rr.log(f"world/robot/{link.name}", rr.Transform3D(translation=p[:3], quaternion=wxyz_to_xyzw(p[3:])))

    obj = o[OBJ_POSE]
    tcp = o[TCP_POSE]
    goal = o[GOAL_POS]
    rr.log("world/cube", rr.Transform3D(translation=obj[:3], quaternion=wxyz_to_xyzw(obj[3:])))
    rr.log("world/tcp", rr.Transform3D(translation=tcp[:3], quaternion=wxyz_to_xyzw(tcp[3:])))
    placed = bool(to_np(info["is_obj_placed"])[0])
    rr.log("world/goal", rr.Points3D([goal], radii=float(u.goal_thresh), colors=(40, 200, 80) if placed else (60, 120, 255)))
    rr.log("world/tcp_to_obj", rr.LineStrips3D([[tcp[:3], obj[:3]]], colors=(255, 200, 0), radii=0.001))

    d_tcp_obj = float(np.linalg.norm(obj[:3] - tcp[:3]))
    d_obj_goal = float(np.linalg.norm(goal - obj[:3]))
    rr.log("plots/reward/reward", rr.Scalars(float(reward)))
    rr.log("plots/reward/value", rr.Scalars(float(value)))
    rr.log("plots/dist/tcp_to_obj", rr.Scalars(d_tcp_obj))
    rr.log("plots/dist/obj_to_goal", rr.Scalars(d_obj_goal))
    rr.log("plots/dist/obj_z", rr.Scalars(float(obj[2])))
    rr.log("plots/flags/is_grasped", rr.Scalars(float(o[IS_GRASPED])))
    rr.log("plots/flags/success", rr.Scalars(float(to_np(info["success"])[0])))
    rr.log("plots/flags/robot_static", rr.Scalars(float(to_np(info["is_robot_static"])[0])))
    for name, a in zip(ACTION_NAMES, to_np(action[0])):
        rr.log(f"plots/action/{name}", rr.Scalars(float(a)))

    if log_cameras:
        rr.log("cameras/render", rr.Image(to_np(env.render()[0])).compress(jpeg_quality=85))
        sensor = u._get_obs_sensor_data()["base_camera"]
        rr.log("world/base_camera/image/rgb", rr.Image(to_np(sensor["rgb"][0])))
        # ManiSkill depth is int16 millimetres.
        rr.log("world/base_camera/image/depth", rr.DepthImage(to_np(sensor["depth"][0, ..., 0]), meter=1000.0))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model_path", default=str(ROOT / "checkpoints" / "rl_models" / "iter_2500.pt"))
    parser.add_argument("--num_episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=EVAL_SEED_BASE, help="Episode i uses seed + i.")
    parser.add_argument("--stochastic", action="store_true", help="Sample actions (policy as trained) instead of the mean.")
    parser.add_argument("--speed", type=float, default=1.0, help="Real-time factor; 0 = as fast as possible.")
    parser.add_argument("--sim_backend", default="physx_cpu", choices=["physx_cpu", "physx_cuda"])
    parser.add_argument("--no_cameras", action="store_true", help="Skip render/base camera images.")
    parser.add_argument("--rr", default="serve", choices=["spawn", "serve", "connect", "save"])
    parser.add_argument("--rr_url", default="rerun+http://127.0.0.1:9876/proxy")
    parser.add_argument("--rr_path", default=str(ROOT / "logs" / "demo_policy.rrd"))
    parser.add_argument("--grpc_port", type=int, default=9876)
    parser.add_argument("--web_port", type=int, default=9090)
    args = parser.parse_args()

    cfg = PPO_MANISKILL_PARAMETERS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = gym.make(
        cfg["env_id"],
        num_envs=1,
        obs_mode=cfg["obs_mode"],
        control_mode=cfg["control_mode"],
        reward_mode=cfg["reward_mode"],
        render_mode="rgb_array",
        sim_backend=args.sim_backend,
    )
    if isinstance(env.action_space, gym.spaces.Dict):
        env = FlattenActionSpaceWrapper(env)
    obs_dim = env.observation_space.shape[-1]
    act_dim = env.action_space.shape[-1]

    ckpt = torch.load(args.model_path, map_location=device)
    actor = GaussianActor(obs_dim, act_dim, cfg["hidden_sizes"]).to(device).eval()
    critic = ValueNet(obs_dim, cfg["hidden_sizes"]).to(device).eval()
    actor.load_state_dict(ckpt["actor"])
    critic.load_state_dict(ckpt["critic"])
    print(f"Loaded {args.model_path} (iteration {ckpt.get('iteration', '?')}), "
          f"std={torch.exp(actor.log_std).mean().item():.3f}")

    init_rerun(args)
    dt = env.unwrapped.control_timestep
    gen = torch.Generator(device=device).manual_seed(args.seed)
    frame = 0
    n_success = 0

    try:
        for ep in range(args.num_episodes):
            obs, info = env.reset(seed=args.seed + ep)
            if ep == 0:
                log_static_scene(env)
            obs = obs.to(device)
            ep_return, done, t, first_success = 0.0, False, 0, None
            t_start = time.perf_counter()

            while not done:
                with torch.inference_mode():
                    mu = actor.act_inference(obs)
                    if args.stochastic:
                        noise = torch.randn(mu.shape, generator=gen, device=device)
                        mu = mu + torch.exp(actor.log_std) * noise
                    action = torch.clamp(mu, -1.0, 1.0)
                    value = critic(obs)

                obs, reward, terminated, truncated, info = env.step(action)
                obs = obs.to(device)
                t += 1
                frame += 1
                ep_return += float(reward[0])
                done = bool((terminated | truncated)[0])
                if first_success is None and bool(info["success"][0]):
                    first_success = t

                rr.set_time("frame", sequence=frame)
                rr.set_time("sim_time", duration=frame * dt)
                rr.set_time("episode", sequence=ep)
                rr.set_time("ep_step", sequence=t)
                log_step(env, obs, action, reward[0], value[0], info, not args.no_cameras)

                if args.speed > 0:
                    lag = t_start + t * dt / args.speed - time.perf_counter()
                    if lag > 0:
                        time.sleep(lag)

            success = first_success is not None
            n_success += success
            msg = (f"episode {ep:03d} seed={args.seed + ep} return={ep_return:.2f} len={t} "
                   f"success={success} first_success_step={first_success}")
            rr.log("log", rr.TextLog(msg, level="INFO" if success else "WARN"))
            print(msg)
    except KeyboardInterrupt:
        print("Interrupted.")
    finally:
        env.close()
    print(f"Success: {n_success}/{args.num_episodes}")
    if args.rr == "serve":
        print("Viewer server still running; Ctrl+C to exit.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
