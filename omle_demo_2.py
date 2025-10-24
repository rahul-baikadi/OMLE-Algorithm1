import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any, Optional


# -----------------------------
# Utilities for valid matrices
# -----------------------------
def random_stochastic_matrix(rows: int, cols: int) -> np.ndarray:
    """Each column sums to 1 (we use column-stochastic for state transitions)."""
    x = np.random.gamma(shape=1.0, scale=1.0, size=(rows, cols))
    x = x / x.sum(axis=0, keepdims=True)
    return x

def random_emission_matrix(O: int, S: int, min_sigma: float = 0.15) -> np.ndarray:
    """
    Generate an O x S matrix with columns summing to 1 and with smallest singular value >= min_sigma.
    We enforce weakly-revealing-ish condition (undercomplete; O=S).
    """
    assert O == S, "This simple demo keeps O=S (undercomplete) to enforce full rank easily."
    for _ in range(500):
        M = random_stochastic_matrix(O, S)
        # tweak: add small diagonal bump then renormalize columns
        M = M + 0.05 * np.eye(O, S)
        M = M / M.sum(axis=0, keepdims=True)
        svals = np.linalg.svd(M, compute_uv=False)
        if svals.min() >= min_sigma:
            return M
    # Fall back (rare): just return last try; not guaranteed, but usually fine with above loop.
    return M


# -----------------------------
# POMDP Model Containers
# -----------------------------
@dataclass
class POMDPModel:
    S: int
    A: int
    O: int
    H: int
    mu1: np.ndarray                       # shape (S,)
    T: List[List[np.ndarray]]             # T[h][a]: shape (S,S), column-stochastic (next_state | prev_state, action a at step h)
    Oh: List[np.ndarray]                  # Oh[h]: shape (O,S), column-stochastic (obs | state at step h)
    r_map: List[np.ndarray]               # r_map[h]: shape (O,), reward depends on observation only (per paper)

    def simulate_episode(self, policy: np.ndarray, rng: np.random.Generator) -> Dict[str, Any]:
        """
        Simulate one episode under the TRUE environment.
        policy: array of shape (O, A) with row-stochastic rows π(a|o).
        Returns dict containing states, obs, actions, rewards, total_return.
        """
        S, A, O, H = self.S, self.A, self.O, self.H
        # sample initial state s1 ~ mu1
        s = rng.choice(S, p=self.mu1)
        states, obs, acts, rews = [], [], [], []
        for h in range(H):
            # sample observation from Oh[h][:, s]
            o = rng.choice(O, p=self.Oh[h][:, s])
            # sample action from policy row for observation o
            a = rng.choice(A, p=policy[o])
            # reward depends ONLY on observation (paper's setup)
            r = self.r_map[h][o]
            states.append(s)
            obs.append(o)
            acts.append(a)
            rews.append(r)
            if h < H - 1:
                # sample next state from T[h][a][:, s]
                s = rng.choice(S, p=self.T[h][a][:, s])
        return {
            "states": np.array(states, dtype=int),
            "obs": np.array(obs, dtype=int),
            "acts": np.array(acts, dtype=int),
            "rews": np.array(rews, dtype=float),
            "G": float(np.sum(rews))
        }

    def log_likelihood_of_trajectory(self, trajectory: Dict[str, Any]) -> float:
        """
        Compute log P(o_1..o_H | a_1..a_H ; θ) via forward beliefs.
        Note: the policy probabilities cancel when comparing θ's; we only need the POMDP part.
        """
        S, H = self.S, self.H
        obs = trajectory["obs"]
        acts = trajectory["acts"]
        # belief before emission at step 1
        b = self.mu1.copy()  # shape (S,)
        ll = 0.0
        for h in range(H):
            o = obs[h]
            a = acts[h]
            # p(o_h | b) = sum_s O_h[o,s]*b[s]
            p_oh = float(self.Oh[h][o] @ b)  # Oh[h][o,:] dot b
            # numerical guard
            p_oh = max(p_oh, 1e-12)
            ll += np.log(p_oh)
            if h < H - 1:
                # posterior (unnormalized) ~ O_h[o,:] .* b
                post = self.Oh[h][o] * b
                if post.sum() <= 0:
                    post = post + 1e-12
                post = post / post.sum()
                # predictive next belief under action a: b' = T[h][a]^T @ post
                b = self.T[h][a].T @ post
                # normalize to combat drift (should already be normalized)
                ssum = b.sum()
                if ssum <= 0:
                    b = np.ones(S) / S
                else:
                    b = b / ssum
        return ll

    def value_of_policy_by_sim(self, policy: np.ndarray, n_rollouts: int = 200, rng: Optional[np.random.Generator] = None) -> float:
        if rng is None:
            rng = np.random.default_rng()
        returns = []
        for _ in range(n_rollouts):
            ep = self.simulate_episode(policy, rng)
            returns.append(ep["G"])
        return float(np.mean(returns))


def random_pomdp(S: int, A: int, O: int, H: int, min_sigma: float = 0.15) -> POMDPModel:
    mu1 = np.random.dirichlet(np.ones(S))
    # transitions: for each step h and action a, a column-stochastic SxS
    T = []
    for h in range(H - 1):
        TA = []
        for a in range(A):
            TA.append(random_stochastic_matrix(S, S))
        T.append(TA)
    # emissions: for each step h, O x S, with minimum singular value
    Oh = [random_emission_matrix(O, S, min_sigma=min_sigma) for _ in range(H)]
    # rewards: per step, per observation (in [0,1])
    r_map = [np.random.rand(O) for _ in range(H)]
    return POMDPModel(S=S, A=A, O=O, H=H, mu1=mu1, T=T, Oh=Oh, r_map=r_map)


def random_policy(O: int, A: int) -> np.ndarray:
    """Row-stochastic π(a|o)."""
    M = np.random.gamma(1.0, 1.0, size=(O, A))
    M = M / M.sum(axis=1, keepdims=True)
    return M


# ------------------------------------
# OMLE-like Training Loop (simplified)
# ------------------------------------
@dataclass
class EpisodeRecord:
    policy: np.ndarray      # (O, A)
    trajectory: Dict[str, Any]
    optimistic_model_id: int
    optimistic_value_estimate: float
    realized_return: float


@dataclass
class OMLEResult:
    true_env: POMDPModel
    candidate_models: List[POMDPModel]
    episodes: List[EpisodeRecord]
    Vstar_est: float
    Vstar_policy: np.ndarray
    cumulative_regret: np.ndarray


def run_omle_simple(
    S: int = 2,
    A: int = 2,
    O: int = None,
    H: int = 5,
    K: int = 30,
    pool_models: int = 60,
    per_round_policy_samples: int = 40,
    beta: float = 3.0,
    min_sigma: float = 0.15,
    value_mc_rollouts: int = 200,
    seed: Optional[int] = None
) -> OMLEResult:
    """
    Simplified OMLE:
      - Generate true environment θ*
      - Pre-generate a pool of candidate models satisfying the weakly-revealing-ish check
      - Each round:
          * sample a batch of random policies, evaluate value under each candidate model by sim
          * pick (θ_k, π_k) with highest estimated value
          * execute π_k in TRUE env, get trajectory τ_k
          * update confidence set B_{k+1} via log-likelihood threshold β
      - Empirical regret uses a Monte Carlo estimate of V* over many random policies on θ*
    """
    if O is None:
        O = S  # keep undercomplete O=S for this simple demo

    rng = np.random.default_rng(seed)  # seed None => random each run

    # true env (random each run)
    true_env = random_pomdp(S, A, O, H, min_sigma=min_sigma)

    # candidate model pool (B1)
    candidate_models = [random_pomdp(S, A, O, H, min_sigma=min_sigma) for _ in range(pool_models)]

    # Estimate V* by sampling a big set of policies on TRUE env
    best_val = -1e9
    best_pi = None
    for _ in range(300):
        pi = random_policy(O, A)
        v = true_env.value_of_policy_by_sim(pi, n_rollouts=value_mc_rollouts, rng=rng)
        if v > best_val:
            best_val = v
            best_pi = pi
    Vstar_est = float(best_val)
    Vstar_policy = best_pi

    # The dataset D = list of (policy, trajectory)
    D: List[Tuple[np.ndarray, Dict[str, Any]]] = []

    # Initial confidence set B1 = all candidates (we already constructed to be "weakly-revealing-ish")
    active_ids = list(range(len(candidate_models)))

    episodes: List[EpisodeRecord] = []

    for k in range(1, K + 1):
        # --- Optimistic planning (approx) over active models and sampled policies
        best_pair_val = -1e9
        best_model_id = None
        best_policy = None

        # Sample per_round_policy_samples random memoryless policies
        policies = [random_policy(O, A) for _ in range(per_round_policy_samples)]

        for mid in active_ids:
            model = candidate_models[mid]
            # For each sampled policy, estimate its value via simulation under THAT MODEL
            # (This keeps things simple; exact planning in POMDP is PSPACE-hard)
            for pi in policies:
                vhat = model.value_of_policy_by_sim(pi, n_rollouts=max(30, H * 10), rng=rng)
                if vhat > best_pair_val:
                    best_pair_val = vhat
                    best_model_id = mid
                    best_policy = pi

        # --- Execute π_k in the TRUE environment, collect a trajectory τ_k
        traj = true_env.simulate_episode(best_policy, rng)
        D.append((best_policy, traj))

        # record realized return and the model/policy chosen
        episodes.append(EpisodeRecord(
            policy=best_policy,
            trajectory=traj,
            optimistic_model_id=best_model_id,
            optimistic_value_estimate=float(best_pair_val),
            realized_return=float(traj["G"])
        ))

        # --- Confidence set update: keep models within β of the max total log-likelihood
        # Compute total log-likelihood over D for each candidate model
        total_ll = np.zeros(len(candidate_models))
        for mid, model in enumerate(candidate_models):
            s = 0.0
            for (pi_used, tr) in D:
                # When comparing models, the policy factor cancels; we use P(o|a; θ) by forward pass
                s += model.log_likelihood_of_trajectory(tr)
            total_ll[mid] = s
        ll_max = float(total_ll.max())
        # retain if within β
        new_active = [mid for mid in active_ids if total_ll[mid] >= ll_max - beta]
        # Always intersect with original B1 (which is just the pool); here new_active already subset
        active_ids = new_active if len(new_active) > 0 else active_ids  # guard: never go empty

    # Empirical regret (per-episode; cumulative)
    ep_returns = np.array([rec.realized_return for rec in episodes], dtype=float)
    regrets = Vstar_est - ep_returns
    cumulative_regret = np.cumsum(regrets)

    return OMLEResult(
        true_env=true_env,
        candidate_models=candidate_models,
        episodes=episodes,
        Vstar_est=Vstar_est,
        Vstar_policy=Vstar_policy,
        cumulative_regret=cumulative_regret
    )


# ------------------------------------
# Reporting Helpers for your questions
# ------------------------------------
def explain_generation(env: POMDPModel) -> str:
    return (
        "How reward/action/state transition are generated:\n"
        f"- Initial state s1 ~ mu1 = {np.round(env.mu1,3)}\n"
        f"- At step h, observation o ~ O_h[:, s]: emission matrix Oh[h] (O x S) column-stochastic.\n"
        f"- Action a ~ π(a|o): memoryless (row-stochastic) policy over actions given the current observation.\n"
        f"- Next state s' ~ T_h^a[:, s]: transition matrix T[h][a] (S x S) column-stochastic.\n"
        f"- Reward r_h = r_map[h][o]: depends only on observation (per paper's setup).\n"
    )

def explain_observation(env: POMDPModel) -> str:
    return (
        "How observations are modeled:\n"
        "- For each step h, Oh[h] maps latent state s to a distribution over observations: P(o|s,h).\n"
        "- In code, we sample o using rng.choice(O, p=Oh[h][:, s]).\n"
        "- We enforce a minimum singular value on Oh[h] (O=S here) to keep the model weakly-revealing-ish.\n"
    )

def show_policy(policy: np.ndarray) -> str:
    return "Policy π(a|o) rows (one row per observation):\n" + str(np.round(policy, 3))

def summarize_trajectories(episodes: List[EpisodeRecord], first_n: int = 5) -> str:
    lines = []
    n = min(first_n, len(episodes))
    for i in range(n):
        ep = episodes[i]
        lines.append(
            f"Episode {i+1}: "
            f"o={ep.trajectory['obs'].tolist()}, "
            f"a={ep.trajectory['acts'].tolist()}, "
            f"s={ep.trajectory['states'].tolist()}, "
            f"r={np.round(ep.trajectory['rews'],3).tolist()}, "
            f"G={np.round(ep.trajectory['G'],3)}"
        )
    return "\n".join(lines)

def rewards_over_time(episodes: List[EpisodeRecord]) -> Tuple[np.ndarray, np.ndarray]:
    ep_rewards = np.array([ep.trajectory["G"] for ep in episodes], dtype=float)
    return ep_rewards, np.cumsum(ep_rewards)

def compare_configurations(
    configs: List[Dict[str,int]],
    K: int = 30,
    seed: Optional[int] = None
) -> List[OMLEResult]:
    """Run multiple env settings to see reward/regret differences."""
    results = []
    for cfg in configs:
        res = run_omle_simple(
            S=cfg.get("S",2),
            A=cfg.get("A",2),
            O=cfg.get("O",cfg.get("S",2)),
            H=cfg.get("H",5),
            K=K,
            seed=seed
        )
        results.append(res)
    return results


# -----------------------------
# Demo / Usage
# -----------------------------
if __name__ == "__main__":
    # Single run (random each time). Change seed=None to a fixed int to reproduce.
    result = run_omle_simple(S=3, A=3, O=3, H=5, K=25, seed=None)

    print("=== EXPLANATIONS ===")
    print(explain_generation(result.true_env))
    print(explain_observation(result.true_env))

    print("=== TRUE ENV REWARD MAP (per step, per observation) ===")
    for h, rvec in enumerate(result.true_env.r_map):
        print(f"Step {h+1} r(o):", np.round(rvec, 3))

    print("\n=== SAMPLE POLICY (Episode 1) ===")
    print(show_policy(result.episodes[0].policy))

    print("\n=== FIRST FEW TRAJECTORIES (states, obs, actions, rewards, return) ===")
    print(summarize_trajectories(result.episodes, first_n=5))

    ep_rewards, cum_rewards = rewards_over_time(result.episodes)
    print("\n=== REWARDS OVER EPISODES ===")
    print("Per-episode returns:", np.round(ep_rewards, 3).tolist())
    print("Cumulative returns:", np.round(cum_rewards, 3).tolist())
    plt(EpisodeRecord,ep_rewards)

    print("\n=== EMPIRICAL REGRET ===")
    print("Estimated V* (by Monte Carlo over many random policies):", np.round(result.Vstar_est, 4))
    print("Per-episode regret:", np.round(result.Vstar_est - ep_rewards, 3).tolist())
    print("Cumulative regret:", np.round(result.cumulative_regret, 3).tolist())


