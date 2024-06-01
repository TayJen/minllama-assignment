from typing import Callable, Iterable, Tuple

import math
import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-6,
            weight_decay: float = 0.0,
            correct_bias: bool = True,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # State should be stored in this dictionary
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m_t'] = torch.zeros_like(p.data)
                    state['v_t'] = torch.zeros_like(p.data)

                state['step'] += 1

                # Access hyperparameters from the `group` dictionary
                alpha = group["lr"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]
                lmbda = group["weight_decay"]
                correct_bias = group["correct_bias"]

                # Update first and second moments of the gradients
                m_t, v_t = state['m_t'], state['v_t']
                m_t1 = beta1 * m_t + (1 - beta1) * grad
                v_t1 = beta2 * v_t + (1 - beta2) * torch.square(grad)

                state['m_t'], state['v_t'] = m_t1, v_t1

                # Bias correction
                # Please note that we are using the "efficient version" given in
                # https://arxiv.org/abs/1412.6980
                # Decay the first and second moment running average coefficient
                if correct_bias:
                    bias_correction_1 = 1 - beta1 ** state['step']
                    m_t1 = m_t1 / bias_correction_1

                    bias_correction_2 = 1 - beta2 ** state['step']
                    v_t1 = v_t1 / bias_correction_2

                # Update parameters
                new_weights = p.data - (alpha * m_t1) / (eps + torch.sqrt(v_t1))

                # Add weight decay after the main gradient-based updates.
                # Please note that the learning rate should be incorporated into this update.
                p.data = new_weights - alpha * lmbda * p.data

        return loss
