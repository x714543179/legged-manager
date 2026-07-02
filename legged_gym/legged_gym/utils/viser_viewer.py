"""Optional Viser web viewer backed by IsaacGym rigid-body states.

This viewer intentionally does not maintain a separate kinematic model.  It
loads visual meshes from the task URDF once, then updates each mesh from
``env.rigid_body_states`` so the web view follows the simulator state directly.
"""

from __future__ import annotations

import copy
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import trimesh
import trimesh.visual

from legged_gym import LEGGED_GYM_ROOT_DIR

try:
    import viser

    HAS_VISER = True
except ImportError:
    viser = None
    HAS_VISER = False


def _xyzw_to_wxyz(quat: np.ndarray) -> np.ndarray:
    return np.array([quat[3], quat[0], quat[1], quat[2]], dtype=np.float64)


def _rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _parse_vector(value: Optional[str], default: Iterable[float]) -> np.ndarray:
    if value is None:
        return np.asarray(list(default), dtype=np.float64)
    return np.asarray([float(item) for item in value.split()], dtype=np.float64)


@dataclass
class UrdfVisual:
    mesh_path: str
    origin_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3))
    origin_rpy: np.ndarray = field(default_factory=lambda: np.zeros(3))
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    rgba: Optional[np.ndarray] = None


class UrdfVisualMeshLoader:
    """Loads URDF visual meshes keyed by link name."""

    def __init__(self, urdf_path: str) -> None:
        self.urdf_path = os.path.abspath(urdf_path)
        self.urdf_dir = os.path.dirname(self.urdf_path)

    def load_link_meshes(self) -> Dict[str, trimesh.Trimesh]:
        root = ET.parse(self.urdf_path).getroot()
        link_meshes: Dict[str, List[trimesh.Trimesh]] = {}
        for link in root.findall("link"):
            link_name = link.get("name")
            if not link_name:
                continue
            visuals = [self._parse_visual(visual) for visual in link.findall("visual")]
            meshes = [self._load_visual_mesh(visual) for visual in visuals if visual is not None]
            meshes = [mesh for mesh in meshes if mesh is not None]
            if meshes:
                link_meshes[link_name] = trimesh.util.concatenate(meshes)
        return link_meshes

    def _parse_visual(self, visual_elem: ET.Element) -> Optional[UrdfVisual]:
        mesh_elem = visual_elem.find("./geometry/mesh")
        if mesh_elem is None:
            return None
        filename = mesh_elem.get("filename")
        if not filename:
            return None

        origin_elem = visual_elem.find("origin")
        origin_xyz = np.zeros(3)
        origin_rpy = np.zeros(3)
        if origin_elem is not None:
            origin_xyz = _parse_vector(origin_elem.get("xyz"), [0.0, 0.0, 0.0])
            origin_rpy = _parse_vector(origin_elem.get("rpy"), [0.0, 0.0, 0.0])

        rgba = None
        color_elem = visual_elem.find("./material/color")
        if color_elem is not None and color_elem.get("rgba"):
            rgba = _parse_vector(color_elem.get("rgba"), [1.0, 1.0, 1.0, 1.0])

        return UrdfVisual(
            mesh_path=self._resolve_mesh_path(filename),
            origin_xyz=origin_xyz,
            origin_rpy=origin_rpy,
            scale=_parse_vector(mesh_elem.get("scale"), [1.0, 1.0, 1.0]),
            rgba=rgba,
        )

    def _resolve_mesh_path(self, filename: str) -> str:
        if filename.startswith("package://"):
            rel_path = filename[len("package://") :]
            candidate = os.path.join(LEGGED_GYM_ROOT_DIR, "resources", "robots", rel_path)
            if os.path.exists(candidate):
                return candidate
            return os.path.join(LEGGED_GYM_ROOT_DIR, rel_path)
        if os.path.isabs(filename):
            return filename
        return os.path.normpath(os.path.join(self.urdf_dir, filename))

    def _load_visual_mesh(self, visual: UrdfVisual) -> Optional[trimesh.Trimesh]:
        if not os.path.exists(visual.mesh_path):
            print(f"[viser_viewer] Missing mesh: {visual.mesh_path}")
            return None

        try:
            mesh = trimesh.load(visual.mesh_path, force="mesh")
        except Exception as exc:
            print(f"[viser_viewer] Could not load mesh {visual.mesh_path}: {exc}")
            return None

        if isinstance(mesh, trimesh.Scene):
            mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
        if not isinstance(mesh, trimesh.Trimesh):
            return None

        mesh = mesh.copy()
        mesh.apply_scale(visual.scale)
        transform = np.eye(4)
        transform[:3, :3] = _rpy_to_matrix(visual.origin_rpy)
        transform[:3, 3] = visual.origin_xyz
        mesh.apply_transform(transform)

        if visual.rgba is not None:
            color = (np.clip(visual.rgba, 0.0, 1.0) * 255).astype(np.uint8)
            mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=np.tile(color, (len(mesh.vertices), 1)))
        return mesh


class RigidBodyViserViewer:
    """Web viewer that mirrors IsaacGym rigid-body poses."""

    def __init__(
        self,
        env,
        port: int = 8080,
        robot_index: int = 0,
        body_names: Optional[List[str]] = None,
    ) -> None:
        if not HAS_VISER:
            raise ImportError("viser is required for --viewer viser. Install it with: pip install viser")

        self.env = env
        self.robot_index = robot_index
        self.server = viser.ViserServer(port=port)
        self._body_names = body_names or self._get_asset_body_names(env)
        self._body_handles: Dict[int, object] = {}
        self._terrain_handle = None
        self._camera_offset = np.array([2.0, 2.0, 1.5], dtype=np.float64)
        self._camera_look_at_offset = np.array([0.0, 0.0, 0.3], dtype=np.float64)
        self._camera_tracking_enabled = True

        self._build_robot_scene()
        self._build_terrain_scene()
        self._setup_grid()
        self._setup_camera_gui()
        self._setup_command_sliders()

    @staticmethod
    def _get_asset_body_names(env) -> List[str]:
        if hasattr(env, "robot_asset"):
            return list(env.gym.get_asset_rigid_body_names(env.robot_asset))
        if hasattr(env, "asset_body_names"):
            return list(env.asset_body_names)
        raise AttributeError("The environment does not expose robot_asset or asset_body_names.")

    def _build_robot_scene(self) -> None:
        asset_path = self.env.cfg.asset.file.format(LEGGED_GYM_ROOT_DIR=LEGGED_GYM_ROOT_DIR)
        if not asset_path.lower().endswith(".urdf"):
            raise ValueError(f"RigidBodyViserViewer currently expects a URDF asset, got: {asset_path}")

        link_meshes = UrdfVisualMeshLoader(asset_path).load_link_meshes()
        for body_index, body_name in enumerate(self._body_names):
            mesh = self._mesh_for_body(body_name, link_meshes)
            if mesh is None:
                continue
            self._body_handles[body_index] = self.server.scene.add_mesh_trimesh(
                f"/robot/{body_name}",
                mesh,
                cast_shadow=True,
                receive_shadow=True,
            )

    def _mesh_for_body(self, body_name: str, link_meshes: Dict[str, trimesh.Trimesh]) -> Optional[trimesh.Trimesh]:
        candidates = [
            body_name,
            body_name.replace("_link", ""),
            body_name.replace("_body", ""),
        ]
        if body_name == "base_link":
            candidates.append("base")
        for candidate in candidates:
            if candidate in link_meshes:
                return link_meshes[candidate]
        return None

    def _build_terrain_scene(self) -> None:
        terrain = getattr(self.env, "terrain", None)
        if terrain is None:
            return

        mesh = None
        if getattr(terrain, "vertices", None) is not None and getattr(terrain, "triangles", None) is not None:
            mesh = trimesh.Trimesh(
                vertices=np.asarray(terrain.vertices),
                faces=np.asarray(terrain.triangles),
                process=False,
            )
            self._color_height_mesh(mesh)
        elif getattr(terrain, "height_samples", None) is not None:
            mesh = self._mesh_from_heightfield(
                np.asarray(terrain.height_samples),
                horizontal_scale=self.env.cfg.terrain.horizontal_scale,
                vertical_scale=self.env.cfg.terrain.vertical_scale,
            )

        if mesh is None:
            return

        transform = trimesh.transformations.translation_matrix(
            [-self.env.cfg.terrain.border_size, -self.env.cfg.terrain.border_size, 0.0]
        )
        mesh = copy.deepcopy(mesh)
        mesh.apply_transform(transform)
        self._terrain_handle = self.server.scene.add_mesh_trimesh(
            "/terrain",
            mesh,
            cast_shadow=True,
            receive_shadow=True,
        )

    def _mesh_from_heightfield(self, height_samples, horizontal_scale: float, vertical_scale: float):
        heights = height_samples.astype(np.float64) * vertical_scale
        nrow, ncol = heights.shape
        x = np.arange(nrow) * horizontal_scale
        y = np.arange(ncol) * horizontal_scale
        xx, yy = np.meshgrid(x, y, indexing="ij")
        vertices = np.column_stack((xx.ravel(), yy.ravel(), heights.ravel()))
        ri, ci = np.mgrid[: nrow - 1, : ncol - 1]
        i0 = (ri * ncol + ci).ravel()
        faces = np.column_stack(
            [i0, i0 + 1, i0 + ncol + 1, i0, i0 + ncol + 1, i0 + ncol]
        ).reshape(-1, 3)
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
        self._color_height_mesh(mesh)
        return mesh

    @staticmethod
    def _color_height_mesh(mesh: trimesh.Trimesh) -> None:
        z = mesh.vertices[:, 2]
        span = float(z.max() - z.min()) if len(z) else 0.0
        normalized = (z - z.min()) / span if span > 1e-8 else np.zeros_like(z)
        colors = np.zeros((len(z), 4), dtype=np.uint8)
        colors[:, 0] = (70 + 70 * normalized).astype(np.uint8)
        colors[:, 1] = (120 + 100 * (1.0 - normalized)).astype(np.uint8)
        colors[:, 2] = (100 + 50 * normalized).astype(np.uint8)
        colors[:, 3] = 255
        mesh.visual = trimesh.visual.ColorVisuals(vertex_colors=colors)

    def _setup_grid(self) -> None:
        self.server.scene.add_grid(
            "/ground",
            infinite_grid=True,
            fade_distance=50.0,
            shadow_opacity=0.2,
            plane_opacity=0.35,
        )

    def _setup_camera_gui(self) -> None:
        @self.server.on_client_connect
        def _(client) -> None:
            base_pos = self._base_position()
            client.camera.position = base_pos + self._camera_offset
            client.camera.look_at = base_pos + self._camera_look_at_offset
            client.camera.fov = np.radians(60.0)

        with self.server.gui.add_folder("Camera"):
            track = self.server.gui.add_checkbox("Track robot", initial_value=self._camera_tracking_enabled)

            @track.on_update
            def _(_) -> None:
                self._camera_tracking_enabled = bool(track.value)

            fov = self.server.gui.add_slider("FOV", min=30.0, max=120.0, step=1.0, initial_value=60.0)

            @fov.on_update
            def _(_) -> None:
                for client in self.server.get_clients().values():
                    client.camera.fov = np.radians(fov.value)

    def _setup_command_sliders(self) -> None:
        self._command_sliders = {}
        with self.server.gui.add_folder("Velocity Commands", expand_by_default=True):
            self._command_sliders["lin_vel_x"] = self.server.gui.add_slider(
                "Linear X", min=-2.0, max=2.0, step=0.1, initial_value=0.0
            )
            self._command_sliders["lin_vel_y"] = self.server.gui.add_slider(
                "Linear Y", min=-1.0, max=1.0, step=0.1, initial_value=0.0
            )
            self._command_sliders["ang_vel_yaw"] = self.server.gui.add_slider(
                "Yaw Rate", min=-2.0, max=2.0, step=0.1, initial_value=0.0
            )

    def get_command(self) -> np.ndarray:
        return np.array(
            [
                self._command_sliders["lin_vel_x"].value,
                self._command_sliders["lin_vel_y"].value,
                self._command_sliders["ang_vel_yaw"].value,
            ],
            dtype=np.float32,
        )

    def update_from_env(self, env=None, robot_index: Optional[int] = None) -> None:
        env = env or self.env
        robot_index = self.robot_index if robot_index is None else robot_index
        if not hasattr(env, "rigid_body_states"):
            return

        states = env.rigid_body_states[robot_index].detach().cpu().numpy()
        with self.server.atomic():
            for body_index, handle in self._body_handles.items():
                if body_index >= states.shape[0]:
                    continue
                state = states[body_index]
                handle.position = state[:3]
                handle.wxyz = _xyzw_to_wxyz(state[3:7])

            if self._camera_tracking_enabled:
                base_pos = self._base_position(states)
                for client in self.server.get_clients().values():
                    client.camera.position = base_pos + self._camera_offset
                    client.camera.look_at = base_pos + self._camera_look_at_offset
        self.server.flush()

    def _base_position(self, states: Optional[np.ndarray] = None) -> np.ndarray:
        if states is None:
            states = self.env.rigid_body_states[self.robot_index].detach().cpu().numpy()
        base_index = 0
        for idx, name in enumerate(self._body_names):
            if name in ("base", "base_link"):
                base_index = idx
                break
        return states[base_index, :3]

    def stop(self) -> None:
        self.server.stop()


def create_viser_viewer(env, port: int = 8080, robot_index: int = 0) -> RigidBodyViserViewer:
    viewer = RigidBodyViserViewer(env=env, port=port, robot_index=robot_index)
    viewer.update_from_env(env, robot_index)
    return viewer
