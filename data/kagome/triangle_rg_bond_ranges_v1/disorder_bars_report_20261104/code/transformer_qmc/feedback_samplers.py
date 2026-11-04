"""Fast two-copy RBM tempering and empirical guide-step safeguards.

The hot marginal is exp(2*beta*a.s) prod_j[2cosh(beta*(sW+b)_j)]^2,
NOT psi(s)^(2*beta). At beta=1 it is exactly our squared-amplitude density.
Each temperature admits blocked Gibbs; marginal-density ratios give exact
replica exchanges. No ground-state information enters these kernels.
"""
import torch


def tempered_log_density(model, spins, beta):
    beta = torch.as_tensor(beta, dtype=spins.dtype, device=spins.device)
    field = beta[..., None]*(spins@model.w+model.b)
    return 2*beta*(spins@model.a)+2*torch.logaddexp(field, -field).sum(-1)


def tempered_exchange(model, a, b, beta_a, beta_b):
    return (tempered_log_density(model, b, beta_a)
            + tempered_log_density(model, a, beta_b)
            - tempered_log_density(model, a, beta_a)
            - tempered_log_density(model, b, beta_b))


class TwoCopyTempering:
    """Persistent marginal replica exchange, including full checkpoint state."""
    def __init__(self, model, chains, temperatures, generator, state=None):
        self.model = model; self.generator = generator
        device = next(model.parameters()).device
        self.betas = torch.linspace(0, 1, temperatures, dtype=torch.float64, device=device)
        if temperatures < 2 or chains < 2:
            raise ValueError('need at least two temperatures and chains')
        if state is None:
            self.states = (2*torch.randint(2, (temperatures, chains, model.n_sites),
                device=device, generator=generator)-1).double()
            self.labels = torch.arange(temperatures, device=device)[:, None].expand(-1, chains).clone()
            self.phase = torch.zeros_like(self.labels); self.trips = torch.zeros_like(self.labels)
            self.accepted = torch.zeros(temperatures-1, dtype=torch.long, device=device)
            self.attempted = torch.zeros_like(self.accepted); self.sweeps = 0
        else:
            for name, value in state.items():
                setattr(self, name, value.to(device) if isinstance(value, torch.Tensor) else value)

    def state_dict(self):
        return {name: getattr(self, name).detach().cpu().clone() if isinstance(getattr(self, name), torch.Tensor)
                else getattr(self, name) for name in
                ['states', 'labels', 'phase', 'trips', 'accepted', 'attempted', 'sweeps']}

    @torch.no_grad()
    def sweep(self):
        model = self.model; beta = self.betas[:, None]; x = self.states
        probability = torch.sigmoid(2*beta[..., None]*(x@model.w+model.b))
        hidden = [2*(torch.rand(probability.shape, device=x.device, generator=self.generator)<probability).double()-1
                  for _ in range(2)]
        field = beta[..., None]*(2*model.a+(hidden[0]+hidden[1])@model.w.T)
        x = 2*(torch.rand(x.shape, device=x.device, generator=self.generator)<torch.sigmoid(2*field)).double()-1
        ratio = tempered_log_density(model, -x, beta)-tempered_log_density(model, x, beta)
        take = torch.log(torch.rand(ratio.shape, device=x.device, generator=self.generator)) < ratio
        x *= torch.where(take, -1., 1.)[..., None]
        self.states = x
        # Batched disjoint swaps. Recompute densities from current model parameters.
        a = torch.arange(self.sweeps % 2, len(self.betas)-1, 2, device=x.device)
        if len(a):
            lr = tempered_exchange(model, x[a], x[a+1], self.betas[a, None], self.betas[a+1, None])
            take = torch.log(torch.rand(lr.shape, device=x.device, generator=self.generator)) < lr
            old = x[a].clone(); other = x[a+1].clone()
            x[a] = torch.where(take[..., None], other, old)
            x[a+1] = torch.where(take[..., None], old, other)
            old = self.labels[a].clone(); other = self.labels[a+1].clone()
            self.labels[a] = torch.where(take, other, old)
            self.labels[a+1] = torch.where(take, old, other)
            self.accepted[a] += take.sum(1); self.attempted[a] += x.shape[1]
        col = torch.arange(x.shape[1], device=x.device)
        hot = self.labels[0]; cold = self.labels[-1]
        self.trips[hot, col] += (self.phase[hot, col] == 2).long()
        self.phase[hot, col] = 1
        self.phase[cold, col] = torch.where(self.phase[cold, col] == 1, 2, self.phase[cold, col])
        self.sweeps += 1
        return x[-1]

    def diagnostics(self):
        return dict(sweeps=self.sweeps, round_trips=int(self.trips.sum()),
            exchange_fraction=(self.accepted/self.attempted.clamp_min(1)).cpu().tolist(),
            warning='Adaptive transport is diagnostic, not proof of equilibrium or valley mixing.')


@torch.no_grad()
def bounded_step(model, old_parameters, reference, old_logs, rms_limit=.025, max_limit=.15):
    """Backtrack an Adam direction on empirical centered log-density changes.

The reference is the current positive/negative minibatch. This is NOT a KL
bound or a global guarantee. Adam moments retain all observed gradients;
only its proposed parameter displacement is damped, including on rejection.
Returns scale, achieved RMS and max; rejected parameters are exactly restored.
"""
    proposed = [p.detach().clone() for p in model.parameters()]
    for power in range(9):
        scale = 2.**(-power)
        for parameter, old, new in zip(model.parameters(), old_parameters, proposed):
            parameter.copy_(old+scale*(new-old))
        delta = 2*(model(reference)-old_logs); delta -= delta.mean()
        rms = float(delta.square().mean().sqrt()); maximum = float(delta.abs().max())
        if torch.isfinite(delta).all() and rms <= rms_limit and maximum <= max_limit:
            return [scale, rms, maximum]
    for parameter, old in zip(model.parameters(), old_parameters):
        parameter.copy_(old)
    return [0., 0., 0.]
