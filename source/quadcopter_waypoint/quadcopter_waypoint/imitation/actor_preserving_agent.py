"""Project-local RL-Games agent for actor-preserving PPO."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Iterable
from typing import Any

import torch
from rl_games.algos_torch import a2c_continuous, players, torch_ext
from rl_games.common import common_losses

from .actor_preserving_checkpoint import (
    ACTOR_KEYS,
    CRITIC_KEYS,
    OBS_RMS_KEYS,
    SCHEMA_VERSION,
    FrozenReferenceActor,
    actor_preserving_metadata,
    actor_weights_sha256,
    critic_weights_sha256,
    load_reference_actor,
    obs_rms_sha256,
    reference_state_from_model,
)


def warmup_active(epoch_num: int, warmup_epochs: int) -> bool:
    """RL-Games increments epoch before optimization; epochs 1..K are warm-up."""
    return 1 <= int(epoch_num) <= int(warmup_epochs)


def warmup_scheduler_frozen(epoch_num: int, warmup_epochs: int, enabled: bool) -> bool:
    """Return whether adaptive KL scheduling must be bypassed for this epoch."""
    return bool(enabled) and warmup_active(epoch_num, warmup_epochs)


def parameter_grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = torch.zeros((), dtype=torch.float64)
    for parameter in parameters:
        if parameter.grad is not None:
            total += torch.sum(torch.square(parameter.grad.detach().double().cpu()))
    return float(torch.sqrt(total).item())


def parameter_delta_l2(before: list[torch.Tensor], parameters: list[torch.nn.Parameter]) -> float:
    if len(before) != len(parameters):
        raise ValueError("parameter snapshot length mismatch")
    total = torch.zeros((), dtype=torch.float64)
    for old, current in zip(before, parameters, strict=True):
        total += torch.sum(torch.square(current.detach().double().cpu() - old.double()))
    return float(torch.sqrt(total).item())


def bc_anchor_loss(current_mu: torch.Tensor, reference_mu: torch.Tensor) -> torch.Tensor:
    """Mean over batch and action dimensions; reject invalid values before optimization."""
    if current_mu.shape != reference_mu.shape or current_mu.ndim != 2:
        raise ValueError("current and reference means must have identical [batch, action] shapes")
    if not torch.isfinite(current_mu).all() or not torch.isfinite(reference_mu).all():
        raise FloatingPointError("BC anchor received NaN or Inf")
    loss = torch.square(current_mu - reference_mu.detach()).mean()
    if not torch.isfinite(loss):
        raise FloatingPointError("BC anchor loss is NaN or Inf")
    return loss


class ActorPreservingA2CAgent(a2c_continuous.A2CAgent):
    """Separate actor/critic PPO with critic warm-up, frozen observation RMS, and a BC anchor."""

    def __init__(self, base_name: str, params: dict[str, Any]):
        actor_preserving_config = copy.deepcopy(params["config"].get("actor_preserving", {}))
        if actor_preserving_config.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(f"config actor_preserving.schema_version must be {SCHEMA_VERSION}")
        super().__init__(base_name, params)
        if not self.model.a2c_network.is_separate_critic():
            raise ValueError("actor-preserving PPO requires network.separate=true")

        self.actor_preserving_config = actor_preserving_config
        self.warmup_epochs = int(actor_preserving_config["warmup_epochs"])
        self.freeze_lr_scheduler_during_warmup = bool(
            actor_preserving_config["freeze_lr_scheduler_during_warmup"]
        )
        self.base_learning_rate = float(params["config"]["learning_rate"])
        self.freeze_observation_rms = bool(actor_preserving_config["freeze_observation_rms"])
        self.bc_anchor_type = str(actor_preserving_config["bc_anchor_type"])
        self.bc_anchor_coefficient = float(actor_preserving_config["bc_anchor_coefficient"])
        if self.bc_anchor_type != "mse_mean_action" or self.bc_anchor_coefficient < 0.0:
            raise ValueError("actor-preserving PPO supports only a non-negative mse_mean_action anchor")

        self.reference_actor = FrozenReferenceActor().to(self.ppo_device)
        self.reference_actor.load_state_dict(reference_state_from_model(self.model.state_dict()), strict=True)
        self.reference_actor.requires_grad_(False)
        self.reference_actor.eval()
        self._migration_metadata: dict[str, Any] = {}
        self._epoch_actor_before: list[torch.Tensor] = []
        self._epoch_critic_before: list[torch.Tensor] = []
        self._last_anchor_loss = 0.0
        self._last_actor_grad_norm = 0.0
        self._last_critic_grad_norm = 0.0
        self._last_actor_parameter_delta = 0.0
        self._last_critic_parameter_delta = 0.0
        self._set_policy_trainable(not warmup_active(self.epoch_num, self.warmup_epochs))
        self._enforce_frozen_observation_rms()
        self._assert_parameter_isolation()

    def _actor_parameters(self) -> list[torch.nn.Parameter]:
        network = self.model.a2c_network
        return [*network.actor_mlp.parameters(), *network.mu.parameters(), network.sigma]

    def _actor_weight_parameters(self) -> list[torch.nn.Parameter]:
        network = self.model.a2c_network
        return [*network.actor_mlp.parameters(), *network.mu.parameters()]

    def _critic_parameters(self) -> list[torch.nn.Parameter]:
        network = self.model.a2c_network
        return [*network.critic_mlp.parameters(), *network.value.parameters()]

    def _assert_parameter_isolation(self) -> None:
        actor_storage = {parameter.untyped_storage().data_ptr() for parameter in self._actor_weight_parameters()}
        critic_storage = {parameter.untyped_storage().data_ptr() for parameter in self._critic_parameters()}
        if actor_storage & critic_storage:
            raise RuntimeError("actor and critic parameters share storage")

    def _set_policy_trainable(self, trainable: bool) -> None:
        for parameter in self._actor_parameters():
            parameter.requires_grad_(trainable)
        for parameter in self._critic_parameters():
            parameter.requires_grad_(True)
        self.reference_actor.requires_grad_(False)

    def _enforce_frozen_observation_rms(self) -> None:
        if self.freeze_observation_rms and self.normalize_input:
            self.model.running_mean_std.eval()

    def set_train(self) -> None:
        super().set_train()
        self._set_policy_trainable(not warmup_active(self.epoch_num, self.warmup_epochs))
        self._enforce_frozen_observation_rms()
        self.reference_actor.eval()

    def set_eval(self) -> None:
        super().set_eval()
        self._enforce_frozen_observation_rms()
        self.reference_actor.eval()

    def update_epoch(self) -> int:
        previous_epoch = int(self.epoch_num)
        epoch_num = super().update_epoch()
        self._set_policy_trainable(not warmup_active(epoch_num, self.warmup_epochs))
        if self.freeze_lr_scheduler_during_warmup and (
            warmup_active(epoch_num, self.warmup_epochs)
            or previous_epoch <= self.warmup_epochs < epoch_num
        ):
            self.last_lr = self.base_learning_rate
            self.update_lr(self.base_learning_rate)
        self._enforce_frozen_observation_rms()
        self._epoch_actor_before = [parameter.detach().cpu().clone() for parameter in self._actor_parameters()]
        self._epoch_critic_before = [parameter.detach().cpu().clone() for parameter in self._critic_parameters()]
        return epoch_num

    def train_epoch(self):
        """Disable adaptive KL scheduling while the policy is intentionally frozen."""
        if not warmup_scheduler_frozen(
            self.epoch_num,
            self.warmup_epochs,
            self.freeze_lr_scheduler_during_warmup,
        ):
            return super().train_epoch()
        original_schedule_type = self.schedule_type
        self.schedule_type = "actor_preserving_frozen_warmup"
        self.last_lr = self.base_learning_rate
        self.update_lr(self.base_learning_rate)
        try:
            return super().train_epoch()
        finally:
            self.schedule_type = original_schedule_type
            self.last_lr = self.base_learning_rate
            self.update_lr(self.base_learning_rate)

    def _normalized_observation(self, observation: torch.Tensor) -> torch.Tensor:
        self._enforce_frozen_observation_rms()
        with torch.no_grad():
            return self.model.running_mean_std(observation) if self.normalize_input else observation

    def _reference_mean(self, normalized_observation: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.reference_actor(normalized_observation)

    def calc_gradients(self, input_dict: dict[str, torch.Tensor]) -> None:
        value_preds_batch = input_dict["old_values"]
        old_action_log_probs_batch = input_dict["old_logp_actions"]
        advantage = input_dict["advantages"]
        old_mu_batch = input_dict["mu"]
        old_sigma_batch = input_dict["sigma"]
        return_batch = input_dict["returns"]
        actions_batch = input_dict["actions"]
        raw_obs_batch = self._preproc_obs(input_dict["obs"])

        curr_e_clip = self.e_clip
        batch_dict: dict[str, Any] = {"is_train": True, "prev_actions": actions_batch, "obs": raw_obs_batch}
        rnn_masks = None
        if self.is_rnn:
            rnn_masks = input_dict["rnn_masks"]
            batch_dict["rnn_states"] = input_dict["rnn_states"]
            batch_dict["seq_length"] = self.seq_length
            if self.zero_rnn_on_done:
                batch_dict["dones"] = input_dict["dones"]

        self._enforce_frozen_observation_rms()
        with torch.cuda.amp.autocast(enabled=self.mixed_precision):
            result = self.model(batch_dict)
            action_log_probs = result["prev_neglogp"]
            values = result["values"]
            entropy = result["entropy"]
            mu = result["mus"]
            sigma = result["sigmas"]
            actor_loss = self.actor_loss_func(
                old_action_log_probs_batch, action_log_probs, advantage, self.ppo, curr_e_clip
            )
            if self.has_value_loss:
                critic_loss = common_losses.critic_loss(
                    self.model, value_preds_batch, values, curr_e_clip, return_batch, self.clip_value
                )
            else:
                critic_loss = torch.zeros(1, device=self.ppo_device)
            if self.bound_loss_type == "regularisation":
                bounds_loss = self.reg_loss(mu)
            elif self.bound_loss_type == "bound":
                bounds_loss = self.bound_loss(mu)
            else:
                bounds_loss = torch.zeros(1, device=self.ppo_device)

            losses, _ = torch_ext.apply_masks(
                [actor_loss.unsqueeze(1), critic_loss, entropy.unsqueeze(1), bounds_loss.unsqueeze(1)], rnn_masks
            )
            actor_loss, critic_loss, entropy, bounds_loss = losses
            normalized_obs = self._normalized_observation(raw_obs_batch)
            reference_mu = self._reference_mean(normalized_obs)
            anchor = bc_anchor_loss(mu, reference_mu)
            effective_anchor = 0.0 if warmup_active(self.epoch_num, self.warmup_epochs) else self.bc_anchor_coefficient
            loss = (
                actor_loss
                + 0.5 * critic_loss * self.critic_coef
                - entropy * self.entropy_coef
                + bounds_loss * self.bounds_loss_coef
                + effective_anchor * anchor
            )
            auxiliary = self.model.get_aux_loss()
            self.aux_loss_dict = {}
            if auxiliary is not None:
                for key, value in auxiliary.items():
                    loss += value
                    self.aux_loss_dict.setdefault(key, []).append(value.detach())
            if not torch.isfinite(loss):
                raise FloatingPointError("actor-preserving PPO total loss is NaN or Inf")
            if self.multi_gpu:
                self.optimizer.zero_grad()
            else:
                for parameter in self.model.parameters():
                    parameter.grad = None

        self.scaler.scale(loss).backward()
        self._last_anchor_loss = float(anchor.detach().item())
        self._last_actor_grad_norm = parameter_grad_norm(self._actor_parameters())
        self._last_critic_grad_norm = parameter_grad_norm(self._critic_parameters())
        self.trancate_gradients_and_step()
        self._enforce_frozen_observation_rms()

        with torch.no_grad():
            reduce_kl = rnn_masks is None
            kl_dist = torch_ext.policy_kl(mu.detach(), sigma.detach(), old_mu_batch, old_sigma_batch, reduce_kl)
            if rnn_masks is not None:
                kl_dist = (kl_dist * rnn_masks).sum() / rnn_masks.numel()

        self.diagnostics.mini_batch(
            self,
            {
                "values": value_preds_batch,
                "returns": return_batch,
                "new_neglogp": action_log_probs,
                "old_neglogp": old_action_log_probs_batch,
                "masks": rnn_masks,
            },
            curr_e_clip,
            0,
        )
        self.train_result = (
            actor_loss,
            critic_loss,
            entropy,
            kl_dist,
            self.last_lr,
            1.0,
            mu.detach(),
            sigma.detach(),
            bounds_loss,
        )

    def get_full_state_weights(self) -> dict[str, Any]:
        state = super().get_full_state_weights()
        model_state = state["model"]
        metadata = copy.deepcopy(self._migration_metadata)
        metadata.update(
            {
                "schema_version": SCHEMA_VERSION,
                "warmup_epochs": self.warmup_epochs,
                "freeze_lr_scheduler_during_warmup": self.freeze_lr_scheduler_during_warmup,
                "base_learning_rate": self.base_learning_rate,
                "freeze_observation_rms": self.freeze_observation_rms,
                "bc_anchor_type": self.bc_anchor_type,
                "bc_anchor_coefficient": self.bc_anchor_coefficient,
                "warmup_active": warmup_active(self.epoch_num, self.warmup_epochs),
                "actor_weights_sha256": actor_weights_sha256(model_state),
                "critic_weights_sha256": critic_weights_sha256(model_state),
                "observation_rms_sha256": obs_rms_sha256(model_state),
                "reference_actor_state": {
                    key: value.detach().cpu().clone() for key, value in self.reference_actor.state_dict().items()
                },
            }
        )
        state["actor_preserving_ppo"] = metadata
        return state

    def set_full_state_weights(self, weights: dict[str, Any], set_epoch: bool = True) -> None:
        metadata = actor_preserving_metadata(weights)
        super().set_full_state_weights(weights, set_epoch=set_epoch)
        reference_state = metadata.get("reference_actor_state")
        if not isinstance(reference_state, dict):
            raise ValueError("actor-preserving PPO checkpoint is missing embedded reference_actor_state")
        self.reference_actor = load_reference_actor(reference_state, self.ppo_device)
        self._migration_metadata = {
            key: copy.deepcopy(value)
            for key, value in metadata.items()
            if key != "reference_actor_state"
        }
        self._assert_parameter_isolation()
        self._set_policy_trainable(not warmup_active(self.epoch_num, self.warmup_epochs))
        self._enforce_frozen_observation_rms()

    def write_stats(
        self,
        total_time: float,
        epoch_num: int,
        step_time: float,
        play_time: float,
        update_time: float,
        actor_losses: list[torch.Tensor],
        critic_losses: list[torch.Tensor],
        entropies: list[torch.Tensor],
        kls: list[torch.Tensor],
        last_lr: float,
        lr_mul: float,
        frame: int,
        scaled_time: float,
        scaled_play_time: float,
        curr_frames: int,
    ) -> None:
        super().write_stats(
            total_time,
            epoch_num,
            step_time,
            play_time,
            update_time,
            actor_losses,
            critic_losses,
            entropies,
            kls,
            last_lr,
            lr_mul,
            frame,
            scaled_time,
            scaled_play_time,
            curr_frames,
        )
        self._last_actor_parameter_delta = parameter_delta_l2(
            self._epoch_actor_before, self._actor_parameters()
        )
        self._last_critic_parameter_delta = parameter_delta_l2(
            self._epoch_critic_before, self._critic_parameters()
        )
        model_state = self.model.state_dict()
        diagnostics = {
            "epoch": int(epoch_num),
            "warmup_active": warmup_active(epoch_num, self.warmup_epochs),
            "actor_trainable": all(parameter.requires_grad for parameter in self._actor_parameters()),
            "critic_trainable": all(parameter.requires_grad for parameter in self._critic_parameters()),
            "actor_hash": actor_weights_sha256(model_state),
            "critic_hash": critic_weights_sha256(model_state),
            "obs_rms_hash": obs_rms_sha256(model_state),
            "actor_grad_norm": self._last_actor_grad_norm,
            "critic_grad_norm": self._last_critic_grad_norm,
            "actor_parameter_delta": self._last_actor_parameter_delta,
            "critic_parameter_delta": self._last_critic_parameter_delta,
            "anchor_loss": self._last_anchor_loss,
            "learning_rate": float(self.last_lr),
        }
        for key in (
            "actor_grad_norm",
            "critic_grad_norm",
            "actor_parameter_delta",
            "critic_parameter_delta",
            "anchor_loss",
            "learning_rate",
        ):
            value = float(diagnostics[key])
            if not math.isfinite(value):
                raise FloatingPointError(f"actor-preserving PPO diagnostic {key} is NaN or Inf")
            self.writer.add_scalar(f"actor_preserving/{key}", value, frame)
        self.writer.add_scalar("actor_preserving/warmup_active", float(diagnostics["warmup_active"]), frame)
        self.writer.add_scalar("actor_preserving/actor_trainable", float(diagnostics["actor_trainable"]), frame)
        print("[ACTOR_PRESERVING_DIAGNOSTIC] " + json.dumps(diagnostics, sort_keys=True))


def register_actor_preserving_runner(runner: Any) -> None:
    """Register the project-local agent/player without modifying installed RL-Games source."""
    runner.algo_factory.register_builder(
        "actor_preserving_ppo", lambda **kwargs: ActorPreservingA2CAgent(**kwargs)
    )
    runner.player_factory.register_builder(
        "actor_preserving_ppo", lambda **kwargs: players.PpoPlayerContinuous(**kwargs)
    )
