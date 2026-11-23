"""
SuperDec inference on HANDAL object models, rendered as a mosaic image.

Each mesh is sampled uniformly on its surface (dense, complete cloud: the regime
where SuperDec works), normalised as in SuperDec, and decomposed in one batch.
Per object the mosaic shows: input mesh | cloud coloured by primitive | superquadrics.
Panel titles give the number of existing primitives and the Chamfer-L1 (mm)
between the input cloud and points sampled on the superquadric union.

Example:
    python scripts/superdec_handal.py --models_dir ~/data/HANDAL/models
    python scripts/superdec_handal.py --models_dir ~/data/HANDAL/handal_dataset_utensils/models
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from omegaconf import OmegaConf
from scipy.spatial import cKDTree

from superdec.superdec import SuperDec
from superdec.utils.predictions_handler import PredictionHandler

ROOT_DIR = Path(__file__).resolve().parents[1]
# SuperDec was trained y-up (ShapeNet); rotate -90 deg about x to go from z-up.
Z_UP_TO_Y_UP = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
LIGHT = np.array([0.3, -0.5, 0.8]) / np.linalg.norm([0.3, -0.5, 0.8])


def load_model(checkpoint_dir, device):
    cfg = OmegaConf.load(checkpoint_dir / "config.yaml")
    model = SuperDec(cfg.superdec).to(device).eval()
    state = torch.load(checkpoint_dir / "ckpt.pt", map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    return model


@torch.no_grad()
def decompose(model, clouds, device, z_up, lm, batch):
    """clouds (B, N, 3) in the model frame -> PredictionHandler in the same frame."""
    pts = clouds @ Z_UP_TO_Y_UP.T if z_up else clouds
    center = pts.mean(1, keepdims=True)
    norm = 2 * np.abs(pts - center).max(axis=(1, 2), keepdims=True)
    x = torch.as_tensor((pts - center) / norm, dtype=torch.float32, device=device)

    if lm:
        from superdec.lm_optimization.lm_optimizer import LMOptimizer
        lm_opt = LMOptimizer()
    chunks = []
    for xb in x.split(batch):
        ob = model(xb)
        if lm:
            ob = lm_opt(ob, xb)
        chunks.append({k: v.cpu().numpy() for k, v in ob.items() if isinstance(v, torch.Tensor)})
    out = {k: np.concatenate([c[k] for c in chunks]) for k in chunks[0]}

    scale = out["scale"] * norm
    trans = out["trans"] * norm + center
    rotate = out["rotate"]
    if z_up:
        trans = trans @ Z_UP_TO_Y_UP
        rotate = Z_UP_TO_Y_UP.T @ rotate
    return PredictionHandler(dict(
        names=None, pc=clouds, assign_matrix=out["assign_matrix"], scale=scale, rotation=rotate,
        translation=trans, exponents=out["shape"], exist=out["exist"][..., 0],
    ))


def chamfer_l1(a, b, device):
    a = torch.as_tensor(a, dtype=torch.float32, device=device)
    b = torch.as_tensor(b, dtype=torch.float32, device=device)
    d = torch.cdist(a, b)
    return 0.5 * (d.min(1).values.mean() + d.min(0).values.mean()).item()


def shaded(face_rgb, normals):
    return np.clip(face_rgb * (0.35 + 0.65 * np.abs(normals @ LIGHT))[:, None], 0, 1)


def draw_mesh(ax, mesh, face_rgb):
    poly = Poly3DCollection(mesh.vertices[mesh.faces], facecolors=shaded(face_rgb, mesh.face_normals), linewidths=0)
    ax.add_collection3d(poly)


def display_mesh(mesh, max_faces):
    """Decimated copy of mesh for matplotlib, with per-face colours from its texture or vertex colours."""
    if mesh.visual.kind in ("texture", "vertex"):
        vc = mesh.visual.to_color().vertex_colors[:, :3] / 255.0
    else:
        vc = np.full((len(mesh.vertices), 3), 0.7)
    if len(mesh.faces) <= max_faces:
        return mesh, vc[mesh.faces].mean(1)
    small = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False).simplify_quadric_decimation(face_count=max_faces)
    _, idx = cKDTree(mesh.vertices).query(small.triangles_center)
    return small, vc[idx]


def setup_axes(ax, center, half, title):
    for set_lim, c in zip((ax.set_xlim, ax.set_ylim, ax.set_zlim), center):
        set_lim(c - half, c + half)
    ax.set_box_aspect((1, 1, 1), zoom=1.5)
    ax.view_init(elev=35, azim=-50)
    ax.set_axis_off()
    ax.set_title(title, fontsize=9)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models_dir", type=Path, default=Path.home() / "data/HANDAL/models")
    parser.add_argument("--checkpoint", type=Path, default=ROOT_DIR / "submodules/super_dec/checkpoints/normalized")
    parser.add_argument("--out_dir", type=Path, default=None,
                        help="Default: logs/superdec_handal/<parent of models_dir>.")
    parser.add_argument("--n_points", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=16, help="SuperDec batch size (LM needs ~150 MB/cloud).")
    parser.add_argument("--display_faces", type=int, default=5000, help="Decimate input meshes to this for drawing.")
    parser.add_argument("--z_up", action="store_true", help="Treat the models as z-up (rotate to SuperDec's y-up).")
    parser.add_argument("--lm", action="store_true", help="Apply SuperDec's Levenberg-Marquardt refinement.")
    parser.add_argument("--cols", type=int, default=5, help="Objects per mosaic row.")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.models_dir = args.models_dir.expanduser()
    if args.out_dir is None:
        args.out_dir = ROOT_DIR / "logs/superdec_handal" / args.models_dir.parent.name

    device = "cuda" if torch.cuda.is_available() else "cpu"
    paths = sorted(args.models_dir.glob("obj_*.ply"))
    meshes = [trimesh.load(p, force="mesh") for p in paths]
    clouds = np.stack([trimesh.sample.sample_surface(m, args.n_points, seed=args.seed + i)[0]
                       for i, m in enumerate(meshes)])

    model = load_model(args.checkpoint, device)
    pred = decompose(model, clouds, device, args.z_up, args.lm, args.batch)
    palette = plt.get_cmap("tab20")(np.arange(pred.scale.shape[1]))[:, :3]
    pred.colors = palette * 255
    sq_meshes = pred.get_meshes(resolution=24)
    segments = pred.assign_matrix.argmax(-1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pred.names = [p.stem for p in paths]
    tag = ("_zup" if args.z_up else "") + ("_lm" if args.lm else "")
    pred.save_npz(args.out_dir / f"predictions{tag}.npz")

    n = len(paths)
    rows = int(np.ceil(n / args.cols))
    fig = plt.figure(figsize=(2.6 * 3 * args.cols, 2.8 * rows))
    for i, (path, mesh, cloud, sq, seg) in enumerate(zip(paths, meshes, clouds, sq_meshes, segments)):
        n_prim = int((pred.exist[i] > 0.5).sum())
        cd = chamfer_l1(cloud, sq.sample(args.n_points, seed=args.seed), device)
        print(f"{path.stem}: {n_prim} primitives, Chamfer-L1 {cd:.2f} mm")

        lo, hi = mesh.bounds
        center, half = (lo + hi) / 2, 0.55 * (hi - lo).max()
        r, c = divmod(i, args.cols)
        axes = [fig.add_subplot(rows, 3 * args.cols, r * 3 * args.cols + 3 * c + k + 1, projection="3d")
                for k in range(3)]

        draw_mesh(axes[0], *display_mesh(mesh, args.display_faces))
        setup_axes(axes[0], center, half, path.stem)

        axes[1].scatter(*cloud.T, c=palette[seg], s=1.5, linewidths=0, depthshade=False)
        setup_axes(axes[1], center, half, "cloud by primitive")

        draw_mesh(axes[2], sq, sq.visual.face_colors[:, :3] / 255.0)
        setup_axes(axes[2], center, half, f"{n_prim} SQs, CD {cd:.1f} mm")

    fig.suptitle(f"SuperDec ({args.checkpoint.name} ckpt{', z-up' if args.z_up else ''}{', LM' if args.lm else ''}) "
                 f"on {args.models_dir.parent.name}/{args.models_dir.name} ({n} objects)", fontsize=14)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1 - 1.0 / fig.get_figheight(), wspace=0, hspace=0.05)
    out = args.out_dir / f"mosaic{tag}.png"
    fig.savefig(out, dpi=110)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
