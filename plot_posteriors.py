"""Plot the true posterior p(z|x) against the approximate posteriors q(z|x).

Only for models with a 2-D latent space (train with `config_2d.yaml`), where the
true posterior can be computed exactly on a grid:

    p(z|x) = p(x|z) p(z) / p(x),   p(x) = integral of p(x|z) p(z) over the grid

For each checkpoint (one row) and each test image (one column) the panel shows
  * the true posterior (the target), filled grey contours
  * the amortized q(z|x) from the encoder, blue solid contours
  * a Gaussian optimized for that image alone (local FFG), orange dashed contours
and the numbers behind the paper's gaps: log p(x), and the ELBO under each q.
Pass `init` as a checkpoint to include the untrained (randomly initialized) model.

Example:
    python plot_posteriors.py --config config_2d.yaml \
        --checkpoints init checkpoints/mnist/warmup/ffg_z2/1_model.pth \
                           checkpoints/mnist/warmup/ffg_z2/1000_model.pth
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import math
import os
import re

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import grad as torchgrad

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from utils.hparams import HParams
from utils.config import parse_args
from utils.helper import get_model, get_loaders
from utils.math_ops import log_bernoulli, log_normal


parser = argparse.ArgumentParser(description='plot_posteriors')
parser.add_argument('--checkpoints', '-ck', nargs='+', required=True,
                    help='checkpoint paths, one row each; `init` = untrained model')
parser.add_argument('--n-images', '-ni', type=int, default=5,
                    help='number of test images, one column each (default: 5)')
parser.add_argument('--no-local', action='store_true',
                    help='skip the locally optimized Gaussian')
parser.add_argument('--grid-size', type=int, default=200,
                    help='grid points per axis for the true posterior (default: 200)')
parser.add_argument('--n-samples', type=int, default=20000,
                    help='samples from q for the ELBO and for flow densities (default: 20000)')
parser.add_argument('--out', '-o', type=str, default=None,
                    help='output image (default: plots/posteriors_<dataset>.png)')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--no-cuda', '-nc', action='store_true')

# model configuration flags (same as the other scripts, usually from --config)
parser.add_argument('--dataset', '-d', type=str, default='mnist', choices=['mnist', 'fashion'])
parser.add_argument('--z-size', '-zs', type=int, default=2)
parser.add_argument('--batch-size', '-bs', type=int, default=100)
parser.add_argument('--wide-encoder', '-we', action='store_true')
parser.add_argument('--has-flow', '-hf', action='store_true')
parser.add_argument('--hamiltonian-flow', '-hamil-f', action='store_true')
parser.add_argument('--n-flows', '-nf', type=int, default=2)

args = parse_args(parser)
args.cuda = not args.no_cuda and torch.cuda.is_available()
if args.z_size != 2:
    parser.error('the true posterior can only be plotted for a 2-D latent space '
                 '(z_size: 2, e.g. --config config_2d.yaml); got z_size=%d' % args.z_size)

LOG_2PI = math.log(2 * math.pi)
CHUNK = 20000  # latent points per decoder call

# colors: reference palette slots 1-2 for the two approximations, neutral ramp for the target
C_AMORT, C_LOCAL = '#2a78d6', '#eb6834'
TEXT, MUTED = '#0b0b0b', '#52514e'


def build_model():
    hps = HParams(z_size=args.z_size, act_func=F.elu, has_flow=args.has_flow,
                  hamiltonian_flow=args.hamiltonian_flow, n_flows=args.n_flows,
                  wide_encoder=args.wide_encoder, cuda=args.cuda)
    return get_model(args.dataset, hps)


def load(model, path):
    state = torch.load(path, map_location='cuda' if args.cuda else 'cpu')['state_dict']
    z = state['fc3.weight'].shape[0] // 2
    if z != args.z_size:
        raise SystemExit('%s has a %d-D latent space; this script needs z_size 2' % (path, z))
    model.load_state_dict(state)


def label_for(path):
    if path == 'init':
        return 'untrained'
    m = re.match(r'(\d+)_', os.path.basename(path))
    return 'epoch %s' % m.group(1) if m else os.path.basename(path)


# --- densities ----------------------------------------------------------------

def log_joint(model, x, z):
    """log p(x, z) = log p(x|z) + log N(z; 0, I), with all constants. x: [784], z: [N, 2]."""
    out = []
    for zc in torch.split(z, CHUNK):
        logits = model.decode(zc)
        logpx = log_bernoulli(logits, x.expand_as(logits))
        logpz = -0.5 * (zc ** 2).sum(1) - LOG_2PI
        out.append(logpx + logpz)
    return torch.cat(out)


def log_gauss(z, mean, logvar):
    """log N(z; mean, diag(exp(logvar))) in 2-D, with constants."""
    return (-0.5 * ((z - mean) ** 2 / logvar.exp() + logvar).sum(-1)) - LOG_2PI


def grid(center, half, n):
    ax0 = torch.linspace(center[0] - half, center[0] + half, n)
    ax1 = torch.linspace(center[1] - half, center[1] + half, n)
    g0, g1 = torch.meshgrid(ax0, ax1, indexing='xy')
    cell = (ax0[1] - ax0[0]).item() * (ax1[1] - ax1[0]).item()
    return g0, g1, torch.stack([g0.reshape(-1), g1.reshape(-1)], 1), cell


def moments(z, logw):
    w = torch.softmax(logw, 0).unsqueeze(1)
    mean = (w * z).sum(0)
    std = (w * (z - mean) ** 2).sum(0).sqrt()
    return mean, std


def true_posterior(model, x, dtype):
    """Locate the true posterior and integrate log p(x) on a grid fitted to it.

    Starts from a coarse grid over the prior and zooms in until the grid spans
    +-6 std of the posterior at fine resolution, so log p(x) is accurate however
    far away the approximate posteriors are.
    """
    center, half, n = torch.zeros(2), 6., 150
    for _ in range(4):
        _, _, z, cell = grid(center, half, n)
        logj = log_joint(model, x, z.type(dtype)).cpu()
        mean, std = moments(z, logj)
        spacing = 2 * half / (n - 1)
        # keep a margin of a few old grid cells in case the posterior was under-resolved
        center, half = mean, max(6 * std.max().item(), 3 * spacing)
    log_px = (torch.logsumexp(logj, 0) + math.log(cell)).item()
    return mean, std, log_px


# --- approximate posteriors ----------------------------------------------------

def amortized(model, x):
    """Samples and ELBO from the encoder's q(z|x) (Gaussian, or flow on top)."""
    with torch.set_grad_enabled(model.hamiltonian_flow):
        xs = x.unsqueeze(0).repeat(args.n_samples, 1)
        mu, logvar, x_info = model.encode(xs)

        def grad_U(z):  # as in VAE.forward, only used by the Hamiltonian flow
            U = -log_bernoulli(model.decode(z), xs) - log_normal(z)
            g = torchgrad(U, z, grad_outputs=torch.ones_like(U), create_graph=True)[0]
            return (g / torch.sqrt(torch.norm(g, p=2, dim=1)).view(-1, 1)).detach()

        z, logpz, logqz = model.sample(mu, logvar, x_info=x_info,
                                       **({'grad_fn': grad_U} if model.hamiltonian_flow else {}))
        # log_normal drops the same constant from log p(z) and log q(z), so it cancels
        elbo = (log_bernoulli(model.decode(z), xs) + logpz - logqz).mean().item()
    return z.detach(), mu[0].detach(), logvar[0].detach(), elbo


def local_gaussian(model, x, dtype, init, steps=3000, k=64):
    """Fit a diagonal Gaussian to this one image by maximizing its ELBO.

    Starts from the encoder's Gaussian (`init` = (mean, logvar)): from N(0, I) the
    variance takes far longer to shrink to the posterior's scale."""
    mean = init[0].clone().type(dtype).requires_grad_()
    logvar = init[1].clone().type(dtype).requires_grad_()
    opt = torch.optim.Adam([mean, logvar], lr=1e-2)
    for step in range(steps):
        if step == 2 * steps // 3:
            for g in opt.param_groups:
                g['lr'] = 1e-3
        z = mean + torch.randn(k, 2).type(dtype) * (0.5 * logvar).exp()
        loss = -(log_joint(model, x, z) - log_gauss(z, mean, logvar)).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        z = mean + torch.randn(args.n_samples, 2).type(dtype) * (0.5 * logvar).exp()
        elbo = (log_joint(model, x, z) - log_gauss(z, mean, logvar)).mean().item()
    return mean.detach(), logvar.detach(), elbo


def density_from_samples(z, center, half, n):
    """Histogram density of 2-D samples on the plotting grid (for flows)."""
    edges0 = np.linspace(center[0] - half, center[0] + half, n + 1)
    edges1 = np.linspace(center[1] - half, center[1] + half, n + 1)
    h, _, _ = np.histogram2d(z[:, 0], z[:, 1], bins=[edges0, edges1], density=True)
    k = np.outer(*[np.exp(-0.5 * (np.arange(-3, 4) / 1.2) ** 2)] * 2)  # light smoothing
    k /= k.sum()
    from scipy.signal import convolve2d
    return convolve2d(h.T, k, mode='same')


# every density is drawn at the same fractions of its peak, so the shapes compare directly
FRACS = (0.1, 0.4, 0.8)
FILL = ['#e4e3df', '#b9b8b3', '#8a8985', '#5c5b58']  # neutral ramp, lightest still visible


def levels(dens):
    top = dens.max()
    return [f * top for f in FRACS] if top > 0 else None


# --- one panel ----------------------------------------------------------------

def panel(ax, model, x, dtype):
    with torch.no_grad():
        p_mean, p_std, log_px = true_posterior(model, x, dtype)
    z_amort, q_mean, q_logvar, elbo_amort = amortized(model, x)
    local = None if args.no_local else local_gaussian(model, x, dtype, (q_mean, q_logvar))

    # window: the true posterior plus wherever the approximations put their mass
    center = p_mean
    reach = [4 * p_std.max().item()]
    if model.has_flow:  # the flow moves mass away from its base Gaussian: use the samples
        lo, hi = np.percentile(z_amort.cpu().numpy(), [0.5, 99.5], axis=0)
        reach.append(max(np.abs(lo - center.numpy()).max(), np.abs(hi - center.numpy()).max()))
    else:
        reach.append(((q_mean.cpu() - center).abs() + 3 * (0.5 * q_logvar.cpu()).exp()).max().item())
    if local:
        reach.append(((local[0].cpu() - center).abs() + 3 * (0.5 * local[1].cpu()).exp()).max().item())
    half = max(reach)

    n = args.grid_size
    g0, g1, z, _ = grid(center, half, n)
    with torch.no_grad():
        post = (log_joint(model, x, z.type(dtype)).cpu() - log_px).exp().view(n, n).numpy()

    g0, g1 = g0.numpy(), g1.numpy()
    ax.contourf(g0, g1, post, levels=levels(post) + [post.max() * 1.001], colors=FILL[:3])

    if model.has_flow:
        q = density_from_samples(z_amort.cpu().numpy(), center.numpy(), half, n)
    else:
        q = log_gauss(z.type(dtype), q_mean, q_logvar).exp().view(n, n).cpu().numpy()
    if levels(q):
        ax.contour(g0, g1, q, levels=levels(q), colors=C_AMORT, linewidths=1.5)

    lines = ['log p(x)  %.1f' % log_px, 'ELBO amortized  %.1f' % elbo_amort]
    if local:
        l_mean, l_logvar, elbo_local = local
        ql = log_gauss(z.type(dtype), l_mean, l_logvar).exp().view(n, n).cpu().numpy()
        if levels(ql):
            ax.contour(g0, g1, ql, levels=levels(ql), colors=C_LOCAL,
                       linewidths=1.5, linestyles='--')
        lines.append('ELBO local  %.1f' % elbo_local)

    ax.set_title('\n'.join(lines), fontsize=7, color=TEXT, family='monospace', loc='left')
    ax.set_xlim(center[0] - half, center[0] + half)
    ax.set_ylim(center[1] - half, center[1] + half)
    ax.set_aspect('equal')
    ax.tick_params(labelsize=7, colors=MUTED, length=2)
    for s in ax.spines.values():
        s.set_color('#d6d5d0')
    return log_px, elbo_amort, local[2] if local else None


def main():
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    _, test_loader = get_loaders(dataset=args.dataset, evaluate=True, batch_size=args.batch_size)
    images = next(iter(test_loader))[0][:args.n_images]
    images = images.view(images.size(0), -1)

    model = build_model()
    dtype = model.dtype
    model.eval()

    rows, cols = len(args.checkpoints), images.size(0)
    fig = plt.figure(figsize=(2.6 * cols, 3.0 * rows + 1.4))
    gs = fig.add_gridspec(rows + 1, cols, height_ratios=[0.45] + [1] * rows,
                          hspace=0.6, wspace=0.3)

    for j in range(cols):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(images[j].view(28, 28).numpy(), cmap='Greys')
        ax.set_axis_off()
        ax.set_title('test image %d' % j, fontsize=8, color=MUTED)

    for i, path in enumerate(args.checkpoints):
        torch.manual_seed(args.seed)  # same random init for `init` and same sampling noise
        model = build_model()
        if path != 'init':
            load(model, path)
        model.eval()
        print('%s (%s)' % (label_for(path), path))
        for j in range(cols):
            ax = fig.add_subplot(gs[i + 1, j])
            x = images[j].type(dtype)
            log_px, e_a, e_l = panel(ax, model, x, dtype)
            print('  image %d: log p(x) %.2f | amortized ELBO %.2f (gap %.2f)%s' % (
                j, log_px, e_a, log_px - e_a,
                '' if e_l is None else ' | local ELBO %.2f (approximation gap %.2f, '
                'amortization gap %.2f)' % (e_l, log_px - e_l, e_l - e_a)))
            if j == 0:
                ax.set_ylabel(label_for(path), fontsize=9, color=TEXT)

    handles = [Patch(facecolor=FILL[1], label='true posterior p(z|x) (target)'),
               Line2D([], [], color=C_AMORT, lw=1.5,
                      label='amortized q(z|x)' + (' (flow)' if args.has_flow else ' (FFG)'))]
    if not args.no_local:
        handles.append(Line2D([], [], color=C_LOCAL, lw=1.5, ls='--',
                              label='locally optimized q(z|x) (FFG)'))
    fig.legend(handles=handles, loc='lower center', ncol=len(handles), frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle('Posteriors in the 2-D latent space (%s); contours at %s of each peak'
                 % (args.dataset, ', '.join('%d%%' % (100 * f) for f in FRACS)),
                 fontsize=11, color=TEXT)

    out = args.out or 'plots/posteriors_%s.png' % args.dataset
    if os.path.dirname(out):
        os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print('Saved', out)


if __name__ == '__main__':
    main()
