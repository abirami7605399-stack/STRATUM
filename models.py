"""Network definitions for STRATUM.

The encoder is deliberately small: the deployment target is a wearable node,
and the scientific claims concern the structure of the objective and of the
aggregation rule rather than raw capacity. Parameters are partitioned into
three disjoint groups whose names drive the tier at which each group is
aggregated:

    phi.*   shared representation and task heads   -> aggregated at the cloud
    psi.*   context adapter                        -> aggregated inside a cell
    omega.* device calibration and normalisation   -> never leaves the device
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C


class GradientReversal(torch.autograd.Function):
    """Identity forward, sign-flipped and scaled backward."""

    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def grad_reverse(x, lam=1.0):
    return GradientReversal.apply(x, lam)


class ECGEncoder(nn.Module):
    def __init__(self, dim=C.EMB_DIM):
        super().__init__()
        self.c1 = nn.Conv1d(1, 32, 7, padding=3)
        self.c2 = nn.Conv1d(32, 48, 5, padding=2)
        self.c3 = nn.Conv1d(48, 48, 3, padding=1)
        self.n1, self.n2, self.n3 = (nn.BatchNorm1d(32), nn.BatchNorm1d(48),
                                     nn.BatchNorm1d(48))
        self.n_bins = 8
        self.proj = nn.Linear(48 * self.n_bins, dim)

    def forward(self, e):
        h = F.max_pool1d(F.relu(self.n1(self.c1(e))), 2)
        h = F.max_pool1d(F.relu(self.n2(self.c2(h))), 2)
        h = F.relu(self.n3(self.c3(h)))
        # Coarse temporal binning retains the position of the neighbouring
        # depolarisations, which carries the interval information the rate
        # head needs and which a global average would discard.
        h = F.adaptive_avg_pool1d(h, self.n_bins).flatten(1)
        return self.proj(h)


class ContextEncoder(nn.Module):
    def __init__(self, dim=48):
        super().__init__()
        self.c1 = nn.Conv1d(C.N_IMU_CH, 32, 7, stride=2, padding=3)
        self.c2 = nn.Conv1d(32, dim, 5, stride=2, padding=2)
        self.n1, self.n2 = nn.BatchNorm1d(32), nn.BatchNorm1d(dim)
        self.null = nn.Parameter(torch.zeros(dim))

    def forward(self, m, has_ctx):
        h = F.relu(self.n1(self.c1(m)))
        h = F.relu(self.n2(self.c2(h))).mean(dim=-1)
        mask = has_ctx.unsqueeze(-1)
        return mask * h + (1.0 - mask) * self.null.unsqueeze(0)


class ContextAdapter(nn.Module):
    """Edge-resident feature-wise modulation of the rhythm embedding."""

    def __init__(self, dim=C.EMB_DIM, ctx_dim=48):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(ctx_dim, 64), nn.ReLU(),
                               nn.Linear(64, 2 * dim))

    def forward(self, z_r, z_c):
        gb = self.f(z_c)
        gamma, beta = gb.chunk(2, dim=-1)
        return z_r * (1.0 + torch.tanh(gamma)) + beta


class DeviceCalibration(nn.Module):
    """Per-node affine correction retained locally."""

    def __init__(self, dim=C.EMB_DIM):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, z):
        return z * self.scale + self.shift


class STRATUM(nn.Module):
    """Joint rhythm, rate and activity model with a mediated rhythm head."""

    def __init__(self, dim=C.EMB_DIM, ctx_dim=48, use_adapter=True,
                 mediated=True):
        super().__init__()
        self.mediated = mediated
        self.use_adapter = use_adapter

        self.phi_ecg = ECGEncoder(dim)
        self.phi_ctx = ContextEncoder(ctx_dim)
        self.psi_adapter = ContextAdapter(dim, ctx_dim) if use_adapter else None
        self.omega_cal = DeviceCalibration(dim)

        self.phi_head_a = nn.Sequential(nn.Linear(ctx_dim, 64), nn.ReLU(),
                                        nn.Linear(64, C.N_ACTIVITY))
        self.phi_head_v = nn.Sequential(nn.Linear(dim + ctx_dim, 64), nn.ReLU(),
                                        nn.Linear(64, 1))
        rhythm_in = dim + 1 if mediated else dim + 1 + ctx_dim
        self.phi_head_y = nn.Sequential(nn.Linear(rhythm_in, 64), nn.ReLU(),
                                        nn.Linear(64, C.N_RHYTHM))
        self.phi_adv = nn.Sequential(nn.Linear(dim + 1, 64), nn.ReLU(),
                                     nn.Linear(64, C.N_ACTIVITY))

    # ------------------------------------------------------------------
    def encode(self, e, m, has_ctx):
        """Pre-modulation rhythm embedding and the inertial context embedding."""
        return self.phi_ecg(e), self.phi_ctx(m, has_ctx)

    def modulate(self, z_base, z_ctx):
        z = self.psi_adapter(z_base, z_ctx) if self.psi_adapter is not None else z_base
        return self.omega_cal(z)

    def rhythm(self, z_r, v_hat, z_c):
        if self.mediated:
            inp = torch.cat([z_r, v_hat.unsqueeze(-1)], dim=-1)
        else:
            inp = torch.cat([z_r, v_hat.unsqueeze(-1), z_c], dim=-1)
        return self.phi_head_y(inp)

    def forward(self, e, m, has_ctx, ctx_override=None):
        z_base, z_c = self.encode(e, m, has_ctx)
        z_r = self.modulate(z_base, z_c if ctx_override is None else ctx_override)
        logit_a = self.phi_head_a(z_c)
        v_hat = self.phi_head_v(torch.cat([z_r, z_c], dim=-1)).squeeze(-1)
        logit_y = self.rhythm(z_r, v_hat, z_c)
        return {"z_base": z_base, "z_r": z_r, "z_c": z_c, "logit_a": logit_a,
                "v_hat": v_hat, "logit_y": logit_y}

    def counterfactual(self, out, perm):
        """Rhythm logits under a swapped context with the mediator held fixed.

        Only the inertial context is exchanged; the estimated rate that the
        rhythm head consumes is pinned to the value obtained from the true
        context. Any change in the resulting decision is therefore attributable
        to the direct context path alone.
        """
        z_r = self.modulate(out["z_base"], out["z_c"][perm].detach())
        return self.rhythm(z_r, out["v_hat"].detach(), out["z_c"][perm].detach())

    def adversary(self, z_r, v_hat, detach_features=False):
        """Conditional probe for residual activity information.

        The probe is asked to recover the activity from the rhythm embedding
        once the estimated rate has been supplied. Whatever it can still recover
        is information that reaches the rhythm head without passing through the
        physiological mediator, which is exactly the direct path the framework
        is meant to close.
        """
        z = z_r.detach() if detach_features else z_r
        v = v_hat.unsqueeze(-1).detach()
        return self.phi_adv(torch.cat([z, v], dim=-1))

    # ------------------------------------------------------------------
    def group(self, prefix):
        return {n: p for n, p in self.named_parameters() if n.startswith(prefix)}

    def buffers_local(self):
        return {n: b for n, b in self.named_buffers()}
