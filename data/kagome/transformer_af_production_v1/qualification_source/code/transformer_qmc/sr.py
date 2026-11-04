"""Shared sample-space stochastic reconfiguration for normalized real guides.

The metric is the equal-context covariance of d log(psi), divided by N.
The force is the gradient of the equal-context energy per spin. Centering is
performed separately in each Hamiltonian, never between different contexts.
"""

import math
import torch
from torch.func import functional_call, grad, vmap
try:
    from torch.nn.attention import SDPBackend, sdpa_kernel

    def math_attention():
        return sdpa_kernel(SDPBackend.MATH)
except ImportError:  # PyTorch 2.2 on the local analysis machine.
    def math_attention():
        return torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True,
                                             enable_mem_efficient=False)


def score_matrix(model, spins, chunk=16):
    """Per-configuration log-amplitude derivatives, without higher derivatives."""
    parameters = dict(model.named_parameters())

    # functional_call invokes forward; these guides expose log_prob instead.
    class Probability(torch.nn.Module):
        def __init__(self, guide):
            super().__init__()
            self.guide = guide

        def forward(self, states):
            return self.guide.log_prob(states)

    wrapper = Probability(model)
    parameters = {'guide.' + name: value for name, value in parameters.items()}

    def amplitude(parameters, state):
        return .5 * functional_call(wrapper, parameters, (state[None],))[0]

    result = []
    # The math backend supports vmap over parameter gradients on CPU and CUDA.
    with math_attention():
        for part in spins.split(chunk):
            derivatives = vmap(grad(amplitude), in_dims=(None, 0))(parameters, part)
            result.append(torch.cat([value.detach().reshape(len(part), -1)
                                     for value in derivatives.values()], dim=1))
    return torch.cat(result)


def context_rows(scores, energy, n_sites, contexts):
    """Unbiased context covariance and physical per-spin energy gradient."""
    scale = math.sqrt(contexts * (len(energy)-1) * n_sites)
    return ((scores-scores.mean(0))/scale,
            2*(energy-energy.mean()).to(scores.dtype)/scale)


def solve_direction(rows, force, damping, precision='refined'):
    """Woodbury solve; algebraically identical to damped parameter-space SR."""
    if damping <= 0:
        raise ValueError('SR damping must be positive')
    if precision not in ('refined', 'double'):
        raise ValueError('unknown solve precision')
    # A float32 factor is a preconditioner. Compute residuals and reconstruct
    # the direction in float64, where the dual null-space cancellation is safe.
    # Refinement uses two matrix-vector products, avoiding an expensive double
    # precision Gram product on GPUs with limited float64 throughput.
    output_dtype = rows.dtype
    work = rows.double() if precision == 'double' else rows
    kernel = work @ work.T
    kernel.diagonal().add_(damping)
    factor, info = torch.linalg.cholesky_ex(kernel)
    if int(info) != 0:
        if precision == 'refined':
            return solve_direction(rows, force, damping, 'double')
        raise FloatingPointError('SR covariance Cholesky failed')
    dual = torch.cholesky_solve(force.to(work.dtype)[:, None], factor)[:, 0].double()
    rows, force = rows.double(), force.double()
    for _ in range(5):
        residual = force-rows @ (rows.T @ dual)-damping*dual
        relative = float(residual.norm()/force.norm().clamp_min(1e-20))
        if relative < 1e-9:
            break
        correction = torch.cholesky_solve(residual.to(factor.dtype)[:, None], factor)[:, 0]
        dual += correction.double()
    if relative > 1e-7 and precision == 'refined':
        return solve_direction(rows.to(output_dtype), force.to(output_dtype), damping, 'double')
    direction = rows.T @ dual
    residual = rows @ direction + damping*dual-force
    relative = float(residual.norm()/force.norm().clamp_min(1e-20))
    if not math.isfinite(relative) or relative > 1e-7:
        raise FloatingPointError(f'SR solve residual too large: {relative}')
    return direction.to(output_dtype), relative


@torch.no_grad()
def apply_direction(model, direction, score_blocks, learning_rate=.05, trust=.05):
    """Limit the predicted RMS change of log(psi) in every sampled context."""
    largest = max(float(((scores-scores.mean(0)) @ direction).square().mean().sqrt())
                  for scores in score_blocks)
    rate = min(learning_rate, trust/max(largest, 1e-20))
    if not torch.isfinite(direction).all():
        raise FloatingPointError('nonfinite SR direction')
    offset = 0
    for parameter in model.parameters():
        count = parameter.numel()
        parameter.add_(direction[offset:offset+count].view_as(parameter), alpha=-rate)
        offset += count
    if offset != len(direction):
        raise ValueError('parameter dimension mismatch')
    return dict(rate=rate, predicted_max_logpsi_rms=rate*largest,
                direction_norm=float(direction.norm()))
