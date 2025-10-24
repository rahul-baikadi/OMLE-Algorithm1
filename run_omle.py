from omle_undercomplete import *

import numpy as np

S, A, O, H = 2, 2, 2, 3

def make_params(mu1, T, Oh, r):
    p = POMDPParams(S, A, O, H, mu1, T, Oh, r)
    p.check_shapes()
    return p

mu1_true = np.array([0.6, 0.4])
T_true = [
    [np.array([[0.9,0.1],[0.3,0.7]]), np.array([[0.7,0.3],[0.4,0.6]])],
    [np.array([[0.85,0.15],[0.2,0.8]]), np.array([[0.6,0.4],[0.5,0.5]])]
]
Oh_true = [
    np.array([[0.8,0.2],[0.2,0.8]]),
    np.array([[0.7,0.3],[0.3,0.7]]),
    np.array([[0.6,0.4],[0.4,0.6]])
]
r = [np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([0.0, 1.0])]

true_env = make_params(mu1_true, T_true, Oh_true, r)

models = [true_env]
for eps in [0.05, 0.1]:
    T_alt = [
        [np.clip(T_true[0][0] + np.array([[eps,-eps],[-eps,eps]]), 1e-6, 1.0),
         T_true[0][1]],
        T_true[1]
    ]
    for h in range(H-1):
        for a in range(A):
            T_alt[h][a] = T_alt[h][a] / T_alt[h][a].sum(axis=1, keepdims=True)
    models.append(make_params(mu1_true, T_alt, Oh_true, r))

policy_space = enumerate_reactive_policies(H, O, A)
alpha = 0.15
beta  = 5.0

omle = OMLE(models=models, alpha=alpha, beta=beta, policy_space=policy_space)
history = omle.run(true_env, K=5, seed=0)

for k, info in enumerate(history, 1):
    print(f"Ep {k}: |B|={info['confidence_set_size']}, "
          f"opt_model={info['optimistic_model_index']}, "
          f"V={info['optimistic_value']:.3f}, "
          f"traj={info['trajectory']}")
