from typing import Any, Optional, Union

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer
from tianshou.policy import DiscreteBCQPolicy
from tianshou.utils.rec_mask import get_recommended_ids, removed_recommended_id_from_embedding

class SQNPolicy(DiscreteBCQPolicy):
    def __init__(
            self,
            model_final_layer: torch.nn.Module,
            imitation_final_layer: torch.nn.Module,
            optim: torch.optim.Optimizer,
            discount_factor: float = 0.99,
            estimation_step: int = 1,
            target_update_freq: int = 8000,
            eval_eps: float = 1e-3,
            unlikely_action_threshold: float = 0.3,
            imitation_logits_penalty: float = 1e-2,
            reward_normalization: bool = False,
            state_tracker=None,
            buffer=None,
            which_head="shead",
            **kwargs: Any,
    ) -> None:
        super().__init__(
            model_final_layer, imitation_final_layer, optim, discount_factor, estimation_step, target_update_freq,
            eval_eps, unlikely_action_threshold, imitation_logits_penalty, reward_normalization, state_tracker, buffer, **kwargs
        )
        assert which_head in {"shead", "qhead", "bcq"}
        self.which_head = which_head

    def forward(  # type: ignore
        self,
        batch: Batch,
        buffer: ReplayBuffer,
        indices: np.ndarray = None,
        remove_recommended_ids=False,
        is_train = True,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        input: str = "obs",
        use_batch_in_statetracker=False,
        **kwargs: Any,
    ) -> Batch:
        # obs = batch[input]
        is_obs = True if input == "obs" else False
        obs_emb = self.state_tracker(buffer=buffer, indices=indices, is_obs=is_obs, batch=batch,  is_train=is_train, use_batch_in_statetracker=use_batch_in_statetracker)
        q_value, state = self.model(obs_emb, state=state, info=batch.info)
        if not hasattr(self, "max_action_num"):
            self.max_action_num = q_value.shape[1]
        imitation_logits, _ = self.imitator(obs_emb, state=state, info=batch.info)

        recommended_ids = get_recommended_ids(buffer) if remove_recommended_ids else None
        q_value_masked, indices_masked = removed_recommended_id_from_embedding(q_value, recommended_ids)
        imitation_logits_masked, indices_masked = removed_recommended_id_from_embedding(imitation_logits, recommended_ids)

        if self.which_head == "bcq":
            # BCQ way
            ratio = imitation_logits_masked - imitation_logits_masked.max(dim=-1, keepdim=True).values
            mask = (ratio < self._log_tau).float()
            act_masked = (q_value_masked - np.inf * mask).argmax(dim=-1)
        elif self.which_head == "shead":
            # Supervised head
            act_masked = imitation_logits_masked.argmax(dim=-1)
        elif self.which_head == "qhead":
            act_masked = q_value_masked.argmax(dim=-1)

        act_unsqueezed = act_masked.unsqueeze(-1)
        act = indices_masked.gather(dim=1, index=act_unsqueezed).squeeze(1)

        return Batch(
            act=act, state=state, q_value=q_value, imitation_logits=imitation_logits
        )
