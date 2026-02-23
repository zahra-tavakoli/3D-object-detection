import numpy as np
import open3d as o3d
import colorsys
import os
from datetime import datetime

# ========= AUTO COLOR GENERATOR =========
def generate_colors(n, bright=True):
    brightness = 1.0 if bright else 0.7
    hsv = [(i / n, 1, brightness) for i in range(n)]
    colors = [colorsys.hsv_to_rgb(*c) for c in hsv]
    return colors

# ========= LINES for 3D BOX =========
LINES = [
    [0, 1], [1, 2], [2, 3], [3, 0],   # bottom square
    [4, 5], [5, 6], [6, 7], [7, 4],   # top square
    [2, 6], [7, 3], [1, 5], [4, 0]    # vertical edges
]

# ========= POINT CLOUD UTILS =========
def npy2ply(npy):
    ply = o3d.geometry.PointCloud()
    ply.points = o3d.utility.Vector3dVector(npy[:, :3])
    if npy.shape[1] > 3:
        density = npy[:, 3]
        colors = [[item, item, item] for item in density]
        ply.colors = o3d.utility.Vector3dVector(colors)
    return ply

# ========= 3D BOX OBJECT =========
def bbox_obj(points, color=[1, 0, 0]):
    colors = [color for _ in range(len(LINES))]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(LINES)
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def save_boxes_txt(bboxes, labels, scores=None, save_path="boxes.txt"):
    """
    Save boxes to a text file:
    Each line: x y z dx dy dz yaw class_id [score]
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        for i in range(len(bboxes)):
            box = bboxes[i]
            label = labels[i]
            if scores is not None:
                score = scores[i]
                line = " ".join([f"{v:.6f}" for v in box]) + f" {label} {score:.4f}\n"
            else:
                line = " ".join([f"{v:.6f}" for v in box]) + f" {label} 1.0\n"
            f.write(line)
    print(f"[INFO] Saved boxes to {save_path}")

# ========= OFFSCREEN RENDERER (HEADLESS) =========
def vis_core(plys, save_path=None):
    """
    Offscreen visualization using Open3D's headless renderer.
    """
    import open3d.visualization.rendering as rendering
    import open3d.visualization as vis
    import numpy as np
    import os

    # Create an offscreen renderer (no X11 window required)
    width, height = 1280, 720
    renderer = rendering.OffscreenRenderer(width, height)

    scene = renderer.scene
    scene.set_background([0, 0, 0, 1])  # black background
    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"

    # Add all geometries
    for ply in plys:
        scene.add_geometry("obj_" + str(id(ply)), ply, mat)

    # Add lighting
    scene.scene.set_lighting(rendering.Scene.LightingProfile.SOFT_SHADOWS, (0, 0, 0))
    scene.scene.show_axes(True)

    # Render and save
    img = renderer.render_to_image()
    if save_path is not None:
        vis.io.write_image(save_path, img)
        print(f"[INFO] Saved visualization image to {os.path.abspath(save_path)}")
    else:
        print("[INFO] No save_path provided — skipping save.")
        
        
# ========= MAIN VISUALIZER =========
def vis_pc(pc, bboxes=None, labels=None, nclasses=3, save_path=None):
    """
    pc: np.ndarray (N,4) or open3d.geometry.PointCloud
    bboxes: np.ndarray (n,7) or (n,8,3)
    labels: np.ndarray (n,)
    save_path: optional path, else auto in current dir
    """
    if isinstance(pc, np.ndarray):
        pc = npy2ply(pc)

    mesh_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10, origin=[0, 0, 0])

    if bboxes is None:
        vis_core([pc, mesh_frame], save_path=save_path)
        return

    if len(bboxes.shape) == 2:
        from pointpillars.utils import bbox3d2corners
        bboxes = bbox3d2corners(bboxes)

    COLORS = generate_colors(nclasses)
    vis_objs = [pc, mesh_frame]

    for i in range(len(bboxes)):
        bbox = bboxes[i]
        color = [1, 1, 0] if labels is None else COLORS[labels[i] % len(COLORS)]
        vis_objs.append(bbox_obj(bbox, color=color))

    vis_core(vis_objs, save_path=save_path)
