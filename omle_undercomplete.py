# omle_undercomplete.py
# Reference implementation of Algorithm 1 (OMLE) for undercomplete, tabular POMDPs.
# Dependencies: numpy only.

from __future__ import annotations
import itertools
import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable, Optional
import numpy as np

# --------------------------
# Basic tabular POMDP model
# --------------------------

@dataclass(frozen=True)
class POMDPParams:
    """
    Tabular POMDP parameters (undercomplete setting).
    S: # latent states
    A: # actions
    O: # observations
    H: horizon
    mu1: (S,) initial state distribution
    T: list length H-1; each entry is a list of A transition matrices of shape (S,S)
       T[h][a][s, s'] = P(s_{h+1}=s' | s_h=s, a_h=a)
    Oh: list length H; each entry is an emission matrix of shape (O,S)
       Oh[h][o, s] = P(o_h=o | s_h=s)
    r: list length H; each entry is shape (O,), reward r_h(o) in [0,1]
       (as in the paper, rewards are functions of the observation)
    """
    S: int
    A: int
    O: int
    H: int
    mu1: np.ndarray
    T: List[List[np.ndarray]]
    Oh: List[np.ndarray]
    r: List[np.ndarray]

    def check_shapes(self) -> None:
        S, A, O, H = self.S, self.A, self.O, self.H
        assert self.mu1.shape == (S,)
        assert len(self.T) == H-1
        for h in range(H-1):
            assert len(self.T[h]) == A
            for a in range(A):
                assert self.T[h][a].shape == (S, S)
                # Row-stochastic over s' given current s:
                np.testing.assert_allclose(self.T[h][a].sum(axis=1), np.ones(S), rtol=1e-6, atol=1e-8)
        assert len(self.Oh) == H
        for h in range(H):
            assert self.Oh[h].shape == (O, S)
            # Column-stochastic over o given s (or row-stochastic over o if columns are states):
            np.testing.assert_allclose(self.Oh[h].sum(axis=0), np.ones(S), rtol=1e-6, atol=1e-8)
        assert len(self.r) == H
        for h in range(H):
            assert self.r[h].shape == (O,)

# --------------------------
# Reactive (stage-dependent) policy class
# --------------------------

@dataclass(frozen=True)
class ReactivePolicy:
    """
    Stage-dependent reactive policy: pi_h(o) -> action in {0,...,A-1}
    Internally stored as a list of arrays of shape (O,) with integer actions.
    """
    actions_by_stage: List[np.ndarray]  # length H, each is shape (O,), dtype=int

    def action_prob(self, h: int, o: int, history: Tuple[int, ...]) -> float:
        # Deterministic reactive policy: prob=1 for chosen action, 0 otherwise
        return 1.0

    def act(self, h: int, o: int, history: Tuple[int, ...]) -> int:
        return int(self.actions_by_stage[h][o])

# --------------------------
# Likelihood and value under a model
# --------------------------

def forward_likelihood_of_trajectory(theta: POMDPParams,
                                     policy: ReactivePolicy,
                                     trajectory: List[Tuple[int, int]]) -> float:
    """
    Compute P^pi_theta(τ) for τ = [(o1,a1),...,(oH,aH)] using the forward/filtering recursion.
    Policy is deterministic reactive; so pi(a_h|history,o_h) is 1 if a_h equals prescribed action, 0 otherwise.
    """
    S, H = theta.S, theta.H
    mu = theta.mu1.copy()  # belief over states before seeing o1
    total = 1.0

    for h in range(H):
        o_h, a_h = trajectory[h]
        # policy probability factor: must match deterministic action
        if policy.actions_by_stage[h][o_h] != a_h:
            return 0.0  # impossible under this policy
        # emission likelihood given current belief
        # p(o_h | b_h) = sum_s Oh[h][o_h, s] * b_h[s]
        p_oh = float(np.dot(theta.Oh[h][o_h, :], mu))
        total *= p_oh
        if p_oh == 0.0:
            return 0.0
        # posterior over states after observation (Bayes)
        # b^+(s) ∝ Oh[h][o_h, s] * mu[s]
        mu_post = theta.Oh[h][o_h, :] * mu
        mu_post /= mu_post.sum()
        # propagate through transition (unless final step)
        if h < H - 1:
            mu = theta.T[h][a_h].T @ mu_post
    return float(total)

def expected_return_under_model(theta: POMDPParams, policy: ReactivePolicy) -> float:
    """
    Compute V^pi(theta) = E[sum_h r_h(o_h)] under the model, with reactive policy, via belief dynamics.
    """
    S, O, H = theta.S, theta.O, theta.H
    mu = theta.mu1.copy()
    total = 0.0

    for h in range(H):
        # Predict distribution over o_h: p(o) = sum_s Oh[h][o,s] * mu[s]
        p_o = theta.Oh[h] @ mu  # shape (O,)
        # Expected reward at step h:
        total += float(np.dot(theta.r[h], p_o))
        # Choose actions per observation, then update belief as mixture over o
        mu_next = np.zeros(S)
        for o in range(O):
            a = int(policy.actions_by_stage[h][o])
            # posterior after seeing o:
            numer = theta.Oh[h][o, :] * mu
            if numer.sum() == 0.0:
                continue
            b_post = numer / numer.sum()
            if h < H - 1:
                mu_next += p_o[o] * (theta.T[h][a].T @ b_post)
        if h < H - 1:
            mu = mu_next
    return total

# --------------------------
# Trajectory simulation in the (unknown) true environment
# --------------------------

def simulate_episode(env: POMDPParams, policy: ReactivePolicy, rng: np.random.Generator) -> List[Tuple[int, int]]:
    """
    Roll out one episode in the true environment (latent states sampled),
    returning trajectory τ = [(o1,a1),...,(oH,aH)] (we omit rewards; reward is function of o).
    """
    S, O, H = env.S, env.O, env.H
    # sample initial state
    s = int(rng.choice(S, p=env.mu1))
    traj: List[Tuple[int, int]] = []

    for h in range(H):
        # sample observation
        o = int(rng.choice(O, p=env.Oh[h][:, s]))
        # choose action from policy (deterministic reactive)
        a = policy.act(h, o, history=tuple(x for pair in traj for x in pair))
        traj.append((o, a))
        if h < H - 1:
            s = int(rng.choice(S, p=env.T[h][a][s, :]))
    return traj

# --------------------------
# Confidence set maintenance
# --------------------------

def initial_confidence_filter_alpha(models, alpha: float) -> List[int]:
    keep = []
    tol = 1e-9  # numerical slack
    for idx, m in enumerate(models):
        ok = True
        for h in range(m.H):
            svals = np.linalg.svd(m.Oh[h], compute_uv=False)
            if len(svals) < m.S or svals[m.S - 1] + tol < alpha:
                ok = False
                break
        if ok:
            keep.append(idx)
    return keep


def select_confidence_set_by_ll(models: List[POMDPParams],
                                D: List[Tuple[ReactivePolicy, List[Tuple[int,int]]]],
                                base_indices: List[int],
                                beta: float) -> List[int]:
    """
    B_{k+1} = { theta in base_indices : sum_{(pi,tau) in D} log P^pi_theta(tau) >= max_{theta'} sum log P^pi_{theta'}(tau) - beta }
    """
    if len(base_indices) == 0:
        return []
    ll = []
    for idx in base_indices:
        theta = models[idx]
        s = 0.0
        for (pi, tau) in D:
            p = forward_likelihood_of_trajectory(theta, pi, tau)
            if p <= 0.0:
                return float('-inf')
            s += math.log(p)
        ll.append(s)
    ll = np.array(ll, dtype=float)
    max_ll = float(np.max(ll))
    new_indices = [base_indices[i] for i in range(len(base_indices)) if ll[i] >= max_ll - beta]
    return new_indices

# --------------------------
# Policy enumeration (reactive, stage-dependent)
# --------------------------

def enumerate_reactive_policies(H: int, O: int, A: int) -> List[ReactivePolicy]:
    """
    Enumerate all reactive stage-dependent policies: for each h and o, pick an action a.
    Count: A^(O*H). Use with care.
    """
    policies: List[ReactivePolicy] = []
    action_choices = list(range(A))
    # For each stage, a vector of length O over actions:
    per_stage = list(itertools.product(action_choices, repeat=O))
    for combo in itertools.product(per_stage, repeat=H):
        acts = [np.array(list(stage), dtype=int) for stage in combo]
        policies.append(ReactivePolicy(actions_by_stage=acts))
    return policies

# --------------------------
# Optimistic selection
# --------------------------

def argmax_value_over_models_and_policies(models: List[POMDPParams],
                                          candidate_indices: List[int],
                                          policies: List[ReactivePolicy]) -> Tuple[int, ReactivePolicy, float]:
    """
    Return (model_index, best_policy, best_value) maximizing V^pi(theta) over theta in candidate_indices and pi in policies.
    """
    best_val = -1e100
    best_idx = None
    best_pi = None
    for idx in candidate_indices:
        theta = models[idx]
        for pi in policies:
            val = expected_return_under_model(theta, pi)
            if val > best_val:
                best_val = val
                best_idx = idx
                best_pi = pi
    assert best_idx is not None
    return best_idx, best_pi, float(best_val)

# --------------------------
# OMLE driver
# --------------------------

class OMLE:
    """
    Algorithm 1: Optimistic Maximum Likelihood Estimation (undercomplete, reactive policy class).
    - models: finite candidate grid Θ
    - alpha: weakly-revealing threshold for initial filter B1
    - beta: log-likelihood tolerance for the confidence set
    - policy_space: list of ReactivePolicy objects (e.g., full enumeration for small O,H; or a user-specified subset)
    """
    def __init__(self,
                 models: List[POMDPParams],
                 alpha: float,
                 beta: float,
                 policy_space: List[ReactivePolicy]):
        self.models = models
        self.alpha = alpha
        self.beta = beta
        self.policy_space = policy_space
        # B1 filter
        self.B = initial_confidence_filter_alpha(models, alpha)
        self.D: List[Tuple[ReactivePolicy, List[Tuple[int,int]]]] = []

    def step(self,
             true_env: POMDPParams,
             rng: np.random.Generator) -> Dict[str, object]:
        """
        One OMLE episode:
        1) optimistic planning over current confidence set B_k
        2) execute policy in true env to get τ_k
        3) add (π_k, τ_k) to D
        4) update B_{k+1} by log-likelihood band β intersect B1
        """
        if len(self.B) == 0:
            raise RuntimeError("Confidence set is empty. Adjust model grid or alpha.")
        # optimistic selection
        opt_idx, opt_pi, opt_val = argmax_value_over_models_and_policies(self.models, self.B, self.policy_space)
        # rollout
        tau_k = simulate_episode(true_env, opt_pi, rng)
        # update D and B
        self.D.append((opt_pi, tau_k))
        # keep within the initial B1 (undercomplete α condition)
        B1 = initial_confidence_filter_alpha(self.models, self.alpha)
        B_next = select_confidence_set_by_ll(self.models, self.D, B1, self.beta)
        self.B = B_next
        return {
            "optimistic_model_index": opt_idx,
            "optimistic_value": opt_val,
            "policy": opt_pi,
            "trajectory": tau_k,
            "confidence_set_size": len(self.B)
        }

    def run(self,
            true_env: POMDPParams,
            K: int,
            seed: Optional[int] = None) -> List[Dict[str, object]]:
        rng = np.random.default_rng(seed)
        out = []
        for _ in range(K):
            out.append(self.step(true_env, rng))
        return out
