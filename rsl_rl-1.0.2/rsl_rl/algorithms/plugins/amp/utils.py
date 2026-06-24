from __future__ import annotations

import torch


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Return the conjugate of a quaternion in wxyz order."""
    return torch.cat((q[..., 0:1], -q[..., 1:]), dim=-1)


def quat_inv(q: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Return the inverse of a quaternion in wxyz order."""
    return quat_conjugate(q) / q.pow(2).sum(dim=-1, keepdim=True).clamp_min(eps)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions in wxyz order."""
    w1, x1, y1, z1 = torch.unbind(q1, dim=-1)
    w2, x2, y2, z2 = torch.unbind(q2, dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by a quaternion in wxyz order."""
    xyz = quat[..., 1:]
    t = torch.cross(xyz, vec, dim=-1) * 2.0
    return vec + quat[..., 0:1] * t + torch.cross(xyz, t, dim=-1)


def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate a vector by the inverse of a quaternion in wxyz order."""
    xyz = quat[..., 1:]
    t = torch.cross(xyz, vec, dim=-1) * 2.0
    return vec - quat[..., 0:1] * t + torch.cross(xyz, t, dim=-1)


def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert quaternions in wxyz order to rotation matrices."""
    r, i, j, k = torch.unbind(quaternions, dim=-1)
    two_s = 2.0 / quaternions.pow(2).sum(dim=-1).clamp_min(1e-9)

    matrix = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        dim=-1,
    )
    return matrix.reshape(quaternions.shape[:-1] + (3, 3))


def subtract_frame_transforms(
    t01: torch.Tensor,
    q01: torch.Tensor,
    t02: torch.Tensor | None = None,
    q02: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute frame-2 pose relative to frame 1 from world-frame poses.

    The transform convention matches Isaac Lab: ``T_12 = inverse(T_01) * T_02``.
    Positions are xyz tensors and quaternions are wxyz tensors.
    """
    q10 = quat_inv(q01)
    q12 = quat_mul(q10, q02) if q02 is not None else q10
    t12 = quat_apply(q10, t02 - t01) if t02 is not None else quat_apply(q10, -t01)
    return t12, q12
