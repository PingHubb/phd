"""Parse a URDF and compute link / visual transforms (zero or given joints)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np


def _floats(text: str | None, n: int) -> np.ndarray:
    if not text:
        return np.zeros(n, dtype=np.float64)
    vals = [float(p) for p in text.replace(",", " ").split()]
    if len(vals) < n:
        vals.extend([0.0] * (n - len(vals)))
    return np.asarray(vals[:n], dtype=np.float64)


def rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF RPY: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = (float(rpy[0]), float(rpy[1]), float(rpy[2]))
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def xyz_rpy_matrix(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    t = np.eye(4, dtype=np.float64)
    t[:3, :3] = rpy_matrix(rpy)
    t[:3, 3] = xyz
    return t


def axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    n = float(np.linalg.norm(a))
    if n < 1e-12:
        return np.eye(4, dtype=np.float64)
    x, y, z = a / n
    c, s = np.cos(angle), np.sin(angle)
    t = 1.0 - c
    r = np.array(
        [
            [t * x * x + c, t * x * y - s * z, t * x * z + s * y],
            [t * x * y + s * z, t * y * y + c, t * y * z - s * x],
            [t * x * z - s * y, t * y * z + s * x, t * z * z + c],
        ],
        dtype=np.float64,
    )
    m = np.eye(4, dtype=np.float64)
    m[:3, :3] = r
    return m


def origin_from(elem: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    if elem is None:
        return np.zeros(3), np.zeros(3)
    origin = elem.find("origin")
    if origin is None:
        return np.zeros(3), np.zeros(3)
    return _floats(origin.get("xyz"), 3), _floats(origin.get("rpy"), 3)


@dataclass
class UrdfVisual:
    xyz: np.ndarray
    rpy: np.ndarray
    mesh_path: Path
    rgba: tuple[float, float, float, float]


@dataclass
class UrdfJoint:
    name: str
    joint_type: str
    parent: str
    child: str
    xyz: np.ndarray
    rpy: np.ndarray
    axis: np.ndarray
    lower: float
    upper: float


@dataclass
class UrdfLink:
    name: str
    visuals: list[UrdfVisual] = field(default_factory=list)


@dataclass
class UrdfModel:
    path: Path
    name: str
    root: str
    links: dict[str, UrdfLink]
    joints: list[UrdfJoint]
    parent_joint: dict[str, UrdfJoint]

    @property
    def movable_joints(self) -> list[UrdfJoint]:
        return [j for j in self.joints if j.joint_type in ("revolute", "continuous", "prismatic")]

    def default_q(self) -> dict[str, float]:
        return {j.name: 0.0 for j in self.movable_joints}

    def link_transforms(self, q: dict[str, float] | None = None) -> dict[str, np.ndarray]:
        q = {**self.default_q(), **(q or {})}
        world: dict[str, np.ndarray] = {self.root: np.eye(4, dtype=np.float64)}
        pending = list(self.joints)
        guard = 0
        while pending and guard < 10_000:
            guard += 1
            leftover: list[UrdfJoint] = []
            for j in pending:
                if j.parent not in world:
                    leftover.append(j)
                    continue
                t_origin = xyz_rpy_matrix(j.xyz, j.rpy)
                angle = float(q.get(j.name, 0.0))
                if j.joint_type in ("revolute", "continuous"):
                    t_motion = axis_angle_matrix(j.axis, angle)
                elif j.joint_type == "prismatic":
                    t_motion = np.eye(4, dtype=np.float64)
                    n = float(np.linalg.norm(j.axis))
                    axis = j.axis / n if n > 1e-12 else np.array([1.0, 0.0, 0.0])
                    t_motion[:3, 3] = axis * angle
                else:
                    t_motion = np.eye(4, dtype=np.float64)
                world[j.child] = world[j.parent] @ t_origin @ t_motion
            if len(leftover) == len(pending):
                break
            pending = leftover
        return world

    def visual_poses(
        self, q: dict[str, float] | None = None
    ) -> list[tuple[str, UrdfVisual, np.ndarray]]:
        world = self.link_transforms(q)
        out: list[tuple[str, UrdfVisual, np.ndarray]] = []
        for link_name, link in self.links.items():
            tw = world.get(link_name)
            if tw is None:
                continue
            for i, vis in enumerate(link.visuals):
                tv = tw @ xyz_rpy_matrix(vis.xyz, vis.rpy)
                key = f"{link_name}:{i}"
                out.append((key, vis, tv))
        return out

    def link_visual_pose(self, link_name: str, q: dict[str, float] | None = None) -> np.ndarray | None:
        """World pose of the first visual on ``link_name`` (falls back to link frame)."""
        world = self.link_transforms(q)
        tw = world.get(link_name)
        if tw is None:
            return None
        link = self.links.get(link_name)
        if link is None or not link.visuals:
            return tw
        vis = link.visuals[0]
        return tw @ xyz_rpy_matrix(vis.xyz, vis.rpy)


def _resolve_mesh(filename: str, urdf_dir: Path) -> Path:
    name = filename.replace("\\", "/")
    if name.startswith("package://"):
        name = name.split("/", 3)[-1] if name.count("/") >= 3 else name[len("package://") :]
    return (urdf_dir / name).resolve()


def _visual_color(visual: ET.Element, materials: dict[str, tuple[float, float, float, float]]):
    mat = visual.find("material")
    if mat is None:
        return (0.65, 0.65, 0.65, 1.0)
    color = mat.find("color")
    if color is not None:
        rgba = _floats(color.get("rgba"), 4)
        if rgba[3] == 0:
            rgba[3] = 1.0
        return (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))
    named = materials.get(mat.get("name") or "")
    return named or (0.65, 0.65, 0.65, 1.0)


def load_urdf(path: str | Path) -> UrdfModel:
    path = Path(path)
    tree = ET.parse(path)
    robot = tree.getroot()
    urdf_dir = path.parent
    materials: dict[str, tuple[float, float, float, float]] = {}
    for mat in robot.findall("material"):
        name = mat.get("name")
        color = mat.find("color")
        if name and color is not None:
            rgba = _floats(color.get("rgba"), 4)
            materials[name] = (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3] or 1.0))

    links: dict[str, UrdfLink] = {}
    for link_el in robot.findall("link"):
        name = link_el.get("name")
        if not name:
            continue
        visuals: list[UrdfVisual] = []
        for vis_el in link_el.findall("visual"):
            geom = vis_el.find("geometry")
            if geom is None:
                continue
            mesh_el = geom.find("mesh")
            if mesh_el is None or not mesh_el.get("filename"):
                continue
            xyz, rpy = origin_from(vis_el)
            visuals.append(
                UrdfVisual(
                    xyz=xyz,
                    rpy=rpy,
                    mesh_path=_resolve_mesh(mesh_el.get("filename") or "", urdf_dir),
                    rgba=_visual_color(vis_el, materials),
                )
            )
        links[name] = UrdfLink(name=name, visuals=visuals)

    joints: list[UrdfJoint] = []
    parent_joint: dict[str, UrdfJoint] = {}
    children: set[str] = set()
    for j_el in robot.findall("joint"):
        parent_el = j_el.find("parent")
        child_el = j_el.find("child")
        if parent_el is None or child_el is None:
            continue
        parent = parent_el.get("link") or ""
        child = child_el.get("link") or ""
        xyz, rpy = origin_from(j_el)
        axis_el = j_el.find("axis")
        axis = _floats(axis_el.get("xyz") if axis_el is not None else None, 3)
        if float(np.linalg.norm(axis)) < 1e-12:
            axis = np.array([0.0, 0.0, 1.0])
        limit_el = j_el.find("limit")
        lower = float(limit_el.get("lower", "-3.1416")) if limit_el is not None else -np.pi
        upper = float(limit_el.get("upper", "3.1416")) if limit_el is not None else np.pi
        joint = UrdfJoint(
            name=j_el.get("name") or f"{parent}->{child}",
            joint_type=(j_el.get("type") or "fixed").lower(),
            parent=parent,
            child=child,
            xyz=xyz,
            rpy=rpy,
            axis=axis,
            lower=lower,
            upper=upper,
        )
        joints.append(joint)
        parent_joint[child] = joint
        children.add(child)

    roots = [n for n in links if n not in children]
    root = roots[0] if roots else next(iter(links))
    return UrdfModel(
        path=path,
        name=robot.get("name") or path.stem,
        root=root,
        links=links,
        joints=joints,
        parent_joint=parent_joint,
    )
