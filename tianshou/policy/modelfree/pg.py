from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer, to_torch, to_torch_as
from tianshou.policy import BasePolicy
from tianshou.utils import RunningMeanStd
from tianshou.utils.rec_mask import get_recommended_ids, removed_recommended_id_from_embedding


class PGPolicy(BasePolicy):
    """Implementation of REINFORCE algorithm.

    :param torch.nn.Module model: a model following the rules in
        :class:`~tianshou.policy.BasePolicy`. (s -> logits)
    :param torch.optim.Optimizer optim: a torch.optim for optimizing the model.
    :param dist_fn: distribution class for computing the action.
    :type dist_fn: Type[torch.distributions.Distribution]
    :param float discount_factor: in [0, 1]. Default to 0.99.
    :param bool action_scaling: whether to map actions from range [-1, 1] to range
        [action_spaces.low, action_spaces.high]. Default to True.
    :param str action_bound_method: method to bound action to range [-1, 1], can be
        either "clip" (for simply clipping the action), "tanh" (for applying tanh
        squashing) for now, or empty string for no bounding. Default to "clip".
    :param Optional[gym.Space] action_space: env's action space, mandatory if you want
        to use option "action_scaling" or "action_bound_method". Default to None.
    :param lr_scheduler: a learning rate scheduler that adjusts the learning rate in
        optimizer in each policy.update(). Default to None (no lr_scheduler).
    :param bool deterministic_eval: whether to use deterministic action instead of
        stochastic action sampled by the policy. Default to False.

    .. seealso::

        Please refer to :class:`~tianshou.policy.BasePolicy` for more detailed
        explanation.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optim: torch.optim.Optimizer,
        dist_fn: Type[torch.distributions.Distribution],
        discount_factor: float = 0.99,
        reward_normalization: bool = False,
        state_tracker = None,
        action_scaling: bool = True,
        action_bound_method: str = "clip",
        deterministic_eval: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            action_scaling=action_scaling,
            action_bound_method=action_bound_method,
            **kwargs
        )
        self.actor = model
        try:
            if action_scaling and not np.isclose(model.max_action, 1.):  # type: ignore
                import warnings
                warnings.warn(
                    "action_scaling and action_bound_method are only intended"
                    "to deal with unbounded model action space, but find actor model"
                    f"bound action space with max_action={model.max_action}."
                    "Consider using unbounded=True option of the actor model,"
                    "or set action_scaling to False and action_bound_method to \"\"."
                )
        except Exception:
            pass
        self.optim = optim
        self.dist_fn = dist_fn
        assert 0.0 <= discount_factor <= 1.0, "discount factor should be in [0, 1]"
        self._gamma = discount_factor
        self._rew_norm = reward_normalization
        self.ret_rms = RunningMeanStd()
        self._eps = 1e-8
        self._deterministic_eval = deterministic_eval
        self.state_tracker = state_tracker
    
    def set_collector(self, train_collector):
        self.train_collector = train_collector

    def set_eps(self, eps: float) -> None:
        """Set the eps for epsilon-greedy exploration."""
        self.eps = eps

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        r"""Compute the discounted returns for each transition.

        .. math::
            G_t = \sum_{i=t}^T \gamma^{i-t}r_i

        where :math:`T` is the terminal time step, :math:`\gamma` is the
        discount factor, :math:`\gamma \in [0, 1]`.
        """
        batch.indices = indices
        v_s_ = np.full(indices.shape, self.ret_rms.mean)
        unnormalized_returns, _ = self.compute_episodic_return(
            batch, buffer, indices, v_s_=v_s_, gamma=self._gamma, gae_lambda=1.0
        )
        if self._rew_norm:
            batch.returns = (unnormalized_returns - self.ret_rms.mean) / \
                np.sqrt(self.ret_rms.var + self._eps)
            self.ret_rms.update(unnormalized_returns)
        else:
            batch.returns = unnormalized_returns
        return batch

    def forward(
        self,
        batch: Batch,
        buffer: Optional[ReplayBuffer],
        indices: np.ndarray = None,
        is_obs = None,
        remove_recommended_ids = False,
        is_train = True, 
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:
        """Compute action over the given batch data.

        :return: A :class:`~tianshou.data.Batch` which has 4 keys:

            * ``act`` the action.
            * ``logits`` the network's raw output.
            * ``dist`` the action distribution.
            * ``state`` the hidden state.

        .. seealso::

            Please refer to :meth:`~tianshou.policy.BasePolicy.forward` for
            more detailed explanation.
        """
        obs_emb = self.state_tracker(buffer=buffer, indices=indices, obs=batch.obs, is_obs=is_obs, is_train=is_train)
        logits, hidden = self.actor(obs_emb, state=state, info=batch.info)

        recommended_ids = get_recommended_ids(buffer) if remove_recommended_ids else None
        logits_masked, indices_masked = removed_recommended_id_from_embedding(logits, recommended_ids)

        if isinstance(logits_masked, tuple):
            dist = self.dist_fn(*logits_masked)
        else:
            dist = self.dist_fn(logits_masked)
        if self._deterministic_eval and not self.training:
            if self.action_type == "discrete":
                act_masked = logits_masked.argmax(-1)
            elif self.action_type == "continuous":
                act_masked = logits_masked[0]
        else:
            act_masked = dist.sample()
        
        act_unsqueezed = act_masked.unsqueeze(-1)
        act = indices_masked.gather(dim=1, index=act_unsqueezed).squeeze(1)
        return Batch(logits=logits, act=act, state=hidden, dist=dist)

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        losses = []
        optim_RL, optim_state = self.optim
        for _ in range(repeat):
            for minibatch in batch.split(batch_size, merge_last=True):
                optim_RL.zero_grad()
                optim_state.zero_grad()
                result = self(minibatch, self.train_collector.buffer, minibatch.indices, is_obs=True) # TODO is_obs=True/False
                dist = result.dist
                act = to_torch_as(minibatch.act, result.act)
                ret = to_torch(minibatch.returns, torch.float, result.act.device)
                log_prob = dist.log_prob(act).reshape(len(ret), -1).transpose(0, 1)
                loss = -(log_prob * ret).mean()
                loss.backward()
                optim_RL.step()
                optim_state.step()
                losses.append(loss.item())

        return {"loss": losses}

    def exploration_noise(
            self,
            act: Union[np.ndarray, Batch],
            batch: Batch,
    ) -> Union[np.ndarray, Batch]:
        if isinstance(act, np.ndarray) and not np.isclose(self.eps, 0.0):
            bsz = len(act)
            rand_mask = np.random.rand(bsz) < self.eps
            q = np.random.rand(bsz, self.actor.output_dim)  # [0, 1]
            if hasattr(batch.obs, "mask"):
                q += batch.obs.mask
            rand_act = q.argmax(axis=1)
            act[rand_mask] = rand_act[rand_mask]
        return act