from __future__ import annotations
import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from tqdm import tqdm

from rsl_rl.algorithms.plugins.amp import utils as math_utils

_BODY_NAME_KEYS = ("body_names", "motion_body_names", "robot_body_names")


def _decode_name(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _read_name_list(raw: np.lib.npyio.NpzFile, keys: Sequence[str]) -> list[str] | None:
    for key in keys:
        if key not in raw.files:
            continue
        arr = np.asarray(raw[key])
        if arr.shape == ():
            value = arr.item()
            if isinstance(value, (list, tuple, np.ndarray)):
                arr = np.asarray(value)
            else:
                return [_decode_name(value)]
        return [_decode_name(value) for value in arr.reshape(-1).tolist()]
    return None


def _name_indexes(source_names: Sequence[str], target_names: Sequence[str], path: str) -> list[int]:
    source_to_index = {name: index for index, name in enumerate(source_names)}
    missing = [name for name in target_names if name not in source_to_index]
    if missing:
        raise ValueError(
            f"{path}: npz body_names is missing required AMP body names {missing}. "
            f"Available body_names: {list(source_names)}"
        )
    return [source_to_index[name] for name in target_names]


def _resolve_amp_body_indexes(
    *,
    raw: np.lib.npyio.NpzFile,
    motion_path: str,
    motion_body_count: int,
    body_names: Sequence[str],
    anchor_name: str,
    all_body_names: Sequence[str],
) -> tuple[list[int], int, str]:
    npz_body_names = _read_name_list(raw, _BODY_NAME_KEYS)
    target_body_names = list(body_names)

    if npz_body_names is not None:
        if len(npz_body_names) != motion_body_count:
            raise ValueError(
                f"{motion_path}: body_names length {len(npz_body_names)} does not match "
                f"body array dim {motion_body_count}."
            )
        body_indexes = _name_indexes(npz_body_names, target_body_names, motion_path)
        anchor_index = _name_indexes(npz_body_names, [anchor_name], motion_path)[0]
        return body_indexes, anchor_index, "metadata"

    if motion_body_count == len(all_body_names):
        body_indexes = _name_indexes(all_body_names, target_body_names, motion_path)
        anchor_index = _name_indexes(all_body_names, [anchor_name], motion_path)[0]
        return body_indexes, anchor_index, "legacy_full_body"

    if motion_body_count == len(target_body_names):
        if anchor_name not in target_body_names:
            raise ValueError(
                f"{motion_path}: legacy selected AMP arrays do not include anchor_name '{anchor_name}'. "
                "Add body_names metadata or export full-body arrays so the anchor can be resolved independently."
            )
        return list(range(len(target_body_names))), target_body_names.index(anchor_name), "legacy_selected"

    raise ValueError(
        f"{motion_path}: cannot map npz body dim {motion_body_count} to AMP body dim {len(target_body_names)}. "
        "Add body_names metadata with scripts/attach_npz_names.py or regenerate the npz."
    )


class AMPLoader:
    def __init__(self, motion_file: str,
                 body_names: Sequence[str],
                 anchor_name: str,
                 all_body_names: Sequence[str],
                 device: str = "cuda:0"):
        """Load AMP motion data.

        Args:
            motion_file: Path to a single .npz file or a directory of .npz files.
            body_names: Names of the target bodies to track.
            anchor_name: Name of the anchor (root) body.
            all_body_names: Ordered list of *all* body names in the model.
                The index of each name in this list must match the body
                dimension in the .npz arrays.
            device: Torch device.
        """
        assert os.path.exists(motion_file), f"Invalid path: {motion_file}"

        all_names_list = list(all_body_names)
        body_names_list = list(body_names)
        self._body_names = body_names_list
        self._anchor_name = anchor_name
        self._body_indexes = list(range(len(body_names_list)))
        self._anchor_indexes = None
        self._anchor_motion_indexes = []
        self._num_bodies = len(body_names_list)

        # 检查是文件还是文件夹
        if os.path.isfile(motion_file):
            # 单个文件的情况（保持向后兼容）
            motion_files = [motion_file]
            motion_names = [os.path.splitext(os.path.basename(motion_file))[0]]
        elif os.path.isdir(motion_file):
            # 文件夹的情况：递归查找所有子目录下的 .npz 文件
            motion_names = []
            motion_files = []
            for root, _dirs, files in os.walk(motion_file):
                for filename in sorted(files):
                    if filename.endswith('.npz'):
                        motion_names.append(os.path.splitext(filename)[0])
                        motion_files.append(os.path.join(root, filename))
            motion_files, motion_names = zip(*sorted(zip(motion_files, motion_names))) if motion_files else ([], [])
            motion_files, motion_names = list(motion_files), list(motion_names)
            assert len(motion_files) > 0, f"No npz files found in directory: {motion_file}"
        else:
            raise ValueError(f"Path is neither a file nor a directory: {motion_file}")
        
        # 存储所有motion的数据列表
        self.motion_names = motion_names
        self._body_pos_b_list = []
        self._body_quat_b_list = []
        self._body_ori_b_list = []
        self._body_lin_vel_b_list = []
        self._body_ang_vel_b_list = []
        
        # 处理每个motion文件
        for motion_idx, (motion_name, motion_path) in enumerate(zip(motion_names, motion_files)):
            print(f"Processing motion {motion_idx+1}/{len(motion_files)}: {motion_name}")
            data = np.load(motion_path, allow_pickle=True)
            
            if motion_idx == 0:
                self.fps = data["fps"]
            
            _dof_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
            _dof_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
            _body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
            _body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
            _body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
            _body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
            body_indexes, anchor_index, body_order_source = _resolve_amp_body_indexes(
                raw=data,
                motion_path=motion_path,
                motion_body_count=int(_body_pos_w.shape[1]),
                body_names=body_names_list,
                anchor_name=anchor_name,
                all_body_names=all_names_list,
            )
            self._anchor_motion_indexes.append(anchor_index)
            if self._anchor_indexes is None:
                self._anchor_indexes = anchor_index

            # >>> AMP BODY ID DEBUG START
            if motion_idx == 0:
                print("\n========== AMP BODY ID DEBUG: NPZ ==========")
                print("[AMPDBG] motion_path:", motion_path)
                print("[AMPDBG] npz body count:", _body_pos_w.shape[1])
                print("[AMPDBG] npz body order source:", body_order_source)
                print("[AMPDBG] npz has body_names:", _read_name_list(data, _BODY_NAME_KEYS) is not None)
                print("[AMPDBG] runtime all_body_names count:", len(all_names_list))
                print("[AMPDBG] amp body indexes in npz:", body_indexes)
                print("[AMPDBG] amp body names:", body_names_list)
                print("[AMPDBG] amp anchor index in source:", anchor_index)
                print("[AMPDBG] amp anchor name:", anchor_name)
                print("============================================\n")
            # <<< AMP BODY ID DEBUG END
            
            time_step_total = _dof_pos.shape[0]
            
            # 为当前motion初始化存储
            _body_pos_b = torch.zeros((time_step_total, self._num_bodies, 3), dtype=torch.float32, device=device)
            _body_quat_b = torch.zeros((time_step_total, self._num_bodies, 4), dtype=torch.float32, device=device)
            _body_ori_b = torch.zeros((time_step_total, self._num_bodies, 6), dtype=torch.float32, device=device)
            _body_lin_vel_b = torch.zeros((time_step_total, self._num_bodies, 3), dtype=torch.float32, device=device)
            _body_ang_vel_b = torch.zeros((time_step_total, self._num_bodies, 3), dtype=torch.float32, device=device)
            
            # 处理所有帧
            for frame_idx in tqdm(range(time_step_total), desc=f"Preloading AMP data for {motion_name}"):
                # 获取当前帧的anchor和body数据
                tgt_anchor_pos_w = (
                    _body_pos_w[frame_idx, anchor_index, :].squeeze().unsqueeze(0).repeat(self._num_bodies, 1)
                )
                tgt_anchor_quat_w = (
                    _body_quat_w[frame_idx, anchor_index, :].squeeze().unsqueeze(0).repeat(self._num_bodies, 1)
                )
                tgt_body_pos_w = _body_pos_w[frame_idx, body_indexes, :]
                tgt_body_quat_w = _body_quat_w[frame_idx, body_indexes, :]
                tgt_body_lin_vel_w = _body_lin_vel_w[frame_idx, body_indexes, :]
                tgt_body_ang_vel_w = _body_ang_vel_w[frame_idx, body_indexes, :]

                # 计算body相对于anchor的位置和姿态 (局部坐标系)
                tgt_robot_body_pos_b, tgt_robot_body_quat_b = (
                    math_utils.subtract_frame_transforms(
                        tgt_anchor_pos_w,
                        tgt_anchor_quat_w,
                        tgt_body_pos_w,
                        tgt_body_quat_w,
                    )
                )

                # 将姿态四元数转换为旋转矩阵的前两列
                mat = math_utils.matrix_from_quat(tgt_robot_body_quat_b)
                tgt_robot_body_ori_b = mat[..., :, :2].reshape(self._num_bodies, 6)

                # 将速度转换到每个body自己的局部坐标系
                tgt_body_lin_vel_b = math_utils.quat_apply_inverse(
                    tgt_body_quat_w,
                    tgt_body_lin_vel_w,
                )

                tgt_body_ang_vel_b = math_utils.quat_apply_inverse(
                    tgt_body_quat_w,
                    tgt_body_ang_vel_w,
                )

                # 存储当前帧的局部坐标系数据
                _body_pos_b[frame_idx] = tgt_robot_body_pos_b
                _body_quat_b[frame_idx] = tgt_robot_body_quat_b
                _body_ori_b[frame_idx] = tgt_robot_body_ori_b
                _body_lin_vel_b[frame_idx] = tgt_body_lin_vel_b
                _body_ang_vel_b[frame_idx] = tgt_body_ang_vel_b
            
            # 将当前motion的数据添加到列表
            self._body_pos_b_list.append(_body_pos_b)
            self._body_quat_b_list.append(_body_quat_b)
            self._body_ori_b_list.append(_body_ori_b)
            self._body_lin_vel_b_list.append(_body_lin_vel_b)
            self._body_ang_vel_b_list.append(_body_ang_vel_b)
        
        # 为了向后兼容，使用第一个motion的数据作为默认值
        self.time_step_total = self._body_pos_b_list[0].shape[0]
        self.motion_total_time = self.time_step_total / self.fps
        self._body_pos_b = self._body_pos_b_list[0]
        self._body_quat_b = self._body_quat_b_list[0]
        self._body_ori_b = self._body_ori_b_list[0]
        self._body_lin_vel_b = self._body_lin_vel_b_list[0]
        self._body_ang_vel_b = self._body_ang_vel_b_list[0]

    @property
    def observation_dim(self) -> int:
        num_bodies = len(self._body_indexes)
        obs_dim = (3 + 6 + 3 + 3) * num_bodies  # pos, mat[:,:2], lin_vel, ang_vel
        return obs_dim

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        num_motions = len(self._body_pos_b_list)
        
        for batch_idx in range(num_mini_batch):
            # 按顺序循环选择motion文件
            motion_idx = batch_idx % num_motions
            
            # 获取当前motion的数据
            current_body_pos_b = self._body_pos_b_list[motion_idx]
            current_body_ori_b = self._body_ori_b_list[motion_idx]
            current_body_lin_vel_b = self._body_lin_vel_b_list[motion_idx]
            current_body_ang_vel_b = self._body_ang_vel_b_list[motion_idx]
            current_time_step_total = current_body_pos_b.shape[0]
            
            # 从当前motion中随机采样
            idxs = torch.randint(0, current_time_step_total, (mini_batch_size,), device=current_body_pos_b.device)
            idxs = torch.clamp(idxs, max=current_time_step_total - 1)
            
            batch_body_pos_b = current_body_pos_b[idxs]  # (mini_batch_size, num_bodies, 3)
            batch_body_ori_b = current_body_ori_b[idxs]  # (mini_batch_size, num_bodies, 6)
            batch_body_lin_vel_b = current_body_lin_vel_b[idxs]  # (mini_batch_size, num_bodies, 3)
            batch_body_ang_vel_b = current_body_ang_vel_b[idxs]  # (mini_batch_size, num_bodies, 3)
            s = torch.cat(
                [
                    batch_body_pos_b.reshape(mini_batch_size, -1),
                    batch_body_ori_b.reshape(mini_batch_size, -1),
                    batch_body_lin_vel_b.reshape(mini_batch_size, -1),
                    batch_body_ang_vel_b.reshape(mini_batch_size, -1),
                ],
                dim=-1,
            )  # (mini_batch_size, obs_dim)

            next_idxs = (idxs + 1)
            next_idxs = torch.clamp(next_idxs, max=current_time_step_total - 1)
            batch_next_body_pos_b = current_body_pos_b[next_idxs]  # (mini_batch_size, num_bodies, 3)
            batch_next_body_ori_b = current_body_ori_b[next_idxs]  # (mini_batch_size, num_bodies, 6)
            batch_next_body_lin_vel_b = current_body_lin_vel_b[next_idxs]  # (mini_batch_size, num_bodies, 3)
            batch_next_body_ang_vel_b = current_body_ang_vel_b[next_idxs]  # (mini_batch_size, num_bodies, 3)
            s_next = torch.cat(
                [
                    batch_next_body_pos_b.reshape(mini_batch_size, -1),
                    batch_next_body_ori_b.reshape(mini_batch_size, -1),
                    batch_next_body_lin_vel_b.reshape(mini_batch_size, -1),
                    batch_next_body_ang_vel_b.reshape(mini_batch_size, -1),
                ],
                dim=-1,
            )  # (mini_batch_size, obs_dim)
            yield s, s_next
