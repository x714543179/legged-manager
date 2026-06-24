from __future__ import annotations

from .model_base import Model_Base

import torch
import torch.nn as nn
from tensordict import TensorDict
from rsl_rl.modules import  HiddenState
from rsl_rl.modules import MLP
from rsl_rl.utils import resolve_callable
import torch.nn.functional as F
try:
    from vector_quantize_pytorch import FSQ
except ImportError:
    FSQ = None  # type: ignore[assignment,misc]


class MyModel(Model_Base):
    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        **backbone_cfg
    ) -> None:
        super().__init__(obs, obs_groups, obs_set, output_dim, **backbone_cfg)

        # [FSQ] 从 fsq config 推导 latent_dim = num_fsq_levels × max_num_tokens；无 fsq 时沿用 backbone_cfg["latent_dim"]
        fsq_cfg = backbone_cfg.get("fsq", None)
        self.loss_cfg = backbone_cfg.get("loss", None)
        if fsq_cfg is not None:
            self.num_fsq_levels = fsq_cfg["num_fsq_levels"]
            self.max_num_tokens = fsq_cfg["max_num_tokens"]
            fsq_level_list = fsq_cfg["fsq_level_list"]
            self.fsq = FSQ(levels=[fsq_level_list] * self.num_fsq_levels)
            latent_dim = self.num_fsq_levels * self.max_num_tokens
        else:
            self.fsq = None
            latent_dim = backbone_cfg["latent_dim"]

        # encoder
        self.encoders = nn.ModuleDict()
        self.encoder_cfg = backbone_cfg.get("encoder", {})
        self.main_encoder_name = backbone_cfg.get("main_encoder", None)
        self.encoder_names = list(self.encoder_cfg.keys())
        self._encoder_mode = "split"
        self._fsq_sample_mode = "encode"

        for k, v in self.encoder_cfg.items():
            encoder_groups = v.get("encoder_groups", [])
            encoder_hidden_dims = v.get("hidden_dims", [])
            activation = v.get("activation", "relu")
            mlp = MLP(
                input_dim=sum(self.obs_dim[g] for g in encoder_groups),
                hidden_dims=encoder_hidden_dims,
                output_dim=latent_dim,
                activation=activation,
            )
            self.encoders[k] = mlp

        self.num_encoders = len(self.encoder_names)

        # decoder
        self.decoders = nn.ModuleDict()
        self.decoder_cfg = backbone_cfg.get("decoder", {})

        for k, v in self.decoder_cfg.items():
            decoder_groups = v.get("decoder_groups", [])
            decoder_hidden_dims = v.get("hidden_dims", [])
            activation = v.get("activation", "relu")
            output = v.get("outputs", [])

            if "actions" in output:
                output_dim = self.output_dim
            else:
                output_dim = sum(self.obs_dim[g] for g in output)

            input_dim = latent_dim + sum(self.obs_dim[g] for g in decoder_groups)
            mlp = MLP(
                input_dim=input_dim,
                hidden_dims=decoder_hidden_dims,
                output_dim=output_dim,
                activation=activation,
            )
            self.decoders[k] = mlp

    def set_encoder_mode(self, mode: str) -> None:
        aliases = {"g1": "robot", "encoder_g1": "robot", "smpl": "latent", "encoder_smpl": "latent"}
        mode = aliases.get(mode, mode)
        if mode not in ("split", "robot", "latent"):
            raise ValueError(f"Unknown encoder mode: {mode}")
        if mode == "robot" and "encoder_g1" not in self.encoder_names:
            raise ValueError("encoder_mode='robot' requires an encoder named 'encoder_g1'.")
        if mode == "latent" and "encoder_smpl" not in self.encoder_names:
            raise ValueError("encoder_mode='latent' requires an encoder named 'encoder_smpl'.")
        if mode == "split" and self.num_encoders <= 1:
            mode = "robot" if "encoder_g1" in self.encoder_names else self._encoder_mode
        self._encoder_mode = mode

    def get_encoder_mode(self) -> str:
        return self._encoder_mode

    def set_fsq_sample_mode(self, mode: str) -> None:
        aliases = {"normal": "encode", "none": "encode"}
        mode = aliases.get(mode, mode)
        if mode not in ("encode", "random", "shuffle"):
            raise ValueError(f"Unknown FSQ sample mode: {mode}")
        self._fsq_sample_mode = mode

    def get_fsq_sample_mode(self) -> str:
        return self._fsq_sample_mode

    def _build_encoder_masks(self, batch_size: int, device: torch.device) -> torch.Tensor | None:
        if self.num_encoders <= 1:
            return None

        masks = torch.zeros(self.num_encoders, batch_size, 1, device=device)
        if self._encoder_mode == "robot":
            encoder_indices = torch.full(
                (batch_size,), self.encoder_names.index("encoder_g1"), dtype=torch.long, device=device
            )
        elif self._encoder_mode == "latent":
            encoder_indices = torch.full(
                (batch_size,), self.encoder_names.index("encoder_smpl"), dtype=torch.long, device=device
            )
        else:
            batch_indices = torch.arange(batch_size, device=device)
            encoder_indices = batch_indices * self.num_encoders // batch_size
        batch_indices = torch.arange(batch_size, device=device)
        masks[encoder_indices, batch_indices, 0] = 1.0
        return masks

    def _apply_fsq_sample_mode(
        self,
        main_latent: torch.Tensor,
        train_mode: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if self.fsq is None:
            return main_latent, None, None

        batch_size = main_latent.shape[0]
        z = main_latent.view(batch_size, self.max_num_tokens, self.num_fsq_levels)
        if self._fsq_sample_mode == "random":
            codebook_size = int(getattr(self.fsq, "implicit_codebook").shape[0])
            indices = torch.randint(codebook_size, (batch_size, self.max_num_tokens), device=z.device)
            z_q = self.fsq.indices_to_codes(indices)
        else:
            z_q, indices = self.fsq(z)
            if self._fsq_sample_mode == "shuffle" and batch_size > 1:
                z_q = z_q[torch.randperm(batch_size, device=z_q.device)]
                indices = self.fsq.codes_to_indices(z_q)

        level_indices = self.fsq.indices_to_level_indices(indices) if train_mode else None
        return z_q.view(batch_size, -1), z_q if train_mode else None, level_indices

    def forward(
            self,
            obs: TensorDict,
            masks: torch.Tensor | None = None,
            hidden_state: HiddenState = None,
            train_mode: bool = False,
        ) -> dict[str, torch.Tensor]:
            obs = super().forward(obs, masks, hidden_state, train_mode)

            latents = {}
            for k, encoder in self.encoders.items():
                encoder_groups = self.encoder_cfg[k].get("encoder_groups", [])
                encoder_input = torch.cat([obs[g] for g in encoder_groups], dim=-1)
                latents[k] = encoder(encoder_input)

            if self.main_encoder_name is not None: # 选择主 encoder 输出作为 latent，或对多个 encoder 输出加权求和（权重由 self.encoder_masks 指定）
                encoder_masks = self._build_encoder_masks(
                    batch_size=next(iter(latents.values())).shape[0],
                    device=next(iter(latents.values())).device,
                )

                if encoder_masks is None:
                    main_latent = latents[self.main_encoder_name]
                else:
                    # latent_stack: (n, B, D); masks: (n, B, 1) → sum over encoders → (B, D)
                    latent_stack = torch.stack([latents[name] for name in self.encoder_names], dim=0)
                    main_latent = (latent_stack * encoder_masks).sum(0)

            # [FSQ] (batch, latent_dim) -> (batch, max_num_tokens, num_fsq_levels) -> quantized latent.
            main_latent, fsq_z_q, fsq_level_indices = self._apply_fsq_sample_mode(main_latent, train_mode)

            backbone_output = {}
            actions = self.decoders["action_decoder"](
                torch.cat([main_latent] + [obs[g] for g in self.decoder_cfg["action_decoder"].get("decoder_groups", [])], dim=-1)
            )
            backbone_output["actions"] = actions
            if train_mode and fsq_z_q is not None and fsq_level_indices is not None:
                backbone_output["fsq_z_q"] = fsq_z_q
                backbone_output["fsq_level_indices"] = fsq_level_indices

            if train_mode:
                recon = {}
                losses = {}
                for k, decoder in self.decoders.items():
                    if k == "action_decoder":
                        continue
                    cfg = self.decoder_cfg[k]
                    decoder_groups = cfg.get("decoder_groups", [])
                    decoder_input = torch.cat([main_latent] + [obs[g] for g in decoder_groups], dim=-1)
                    recon[k] = decoder(decoder_input)
                    target = torch.cat([obs[g] for g in cfg.get("outputs", [])], dim=-1)

                    if self.loss_cfg is not None:
                        if self.loss_cfg.get("token", 0.0) > 0.0:
                            losses[f"{k}_token"] = F.mse_loss(latents["encoder_g1"], latents["encoder_smpl"]) * self.loss_cfg["token"]
                        
                        
                        if self.loss_cfg.get("recon", 0.0) > 0.0:
                            losses[f"{k}_recon"] = F.mse_loss(recon["g1_kin_decoder"], target) * self.loss_cfg["recon"]
                        
                        if self.loss_cfg.get("re_encode", 0.0) > 0.0:
                            recon_kin_smpl = self.decoders["g1_kin_decoder"](latents["encoder_smpl"])
                            latents_matched = self.encoders["encoder_g1"](recon_kin_smpl)
                            losses[f"{k}_re_encode"] = F.mse_loss(
                                latents_matched, latents["encoder_g1"].detach()
                            ) * self.loss_cfg["re_encode"]




                backbone_output["aux_losses"] = losses

            return backbone_output
