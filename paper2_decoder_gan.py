"""
paper2_decoder_gan.py  --  stronger decoder + adversarial losses for the f=8 spatial VAE (Paper 2)

Drop-in pieces:
  StrongDecoder        z [B, C, 8, 8] -> x [B, 3, 64, 64]   (~10-15 M params, SD-VAE style)
  PatchDiscriminator   70x70-style PatchGAN shrunk for 64x64 inputs
  AutoencoderGANLoss   rec (Gaussian NLL, learnable sigma) + LPIPS + KL + GAN
                       critic in {"hinge", "wgan_gp", "sobolev"}

Critic variants (the ablation):
  hinge    : standard SD/taming baseline. Uses spectral norm.
  wgan_gp  : Kantorovich-Rubinstein dual (W1). 1-Lipschitz enforced by gradient penalty
             on real/fake interpolates (Gulrajani et al. 2017). No norm layers that mix samples.
  sobolev  : Sobolev-GAN constraint E_mu ||grad_x f||^2 = 1 with mu = (real + fake)/2,
             enforced by an augmented Lagrangian (Mroueh et al. 2018). Relaxes the hard
             Lipschitz constraint of WGAN-GP.

Usage: see `train_step` at the bottom. Two optimisers: one for encoder+decoder, one for the critic.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


# ----------------------------------------------------------------------------- decoder blocks
def Norm(c):
    return nn.GroupNorm(num_groups=32 if c % 32 == 0 else 8, num_channels=c, eps=1e-6)


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.n1, self.c1 = Norm(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.n2, self.c2 = Norm(cout), nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x):
        h = self.c1(F.silu(self.n1(x)))
        h = self.c2(F.silu(self.n2(h)))
        return self.skip(x) + h


class AttnBlock(nn.Module):
    """Single-head self-attention over spatial positions (used only at 8x8 = 64 tokens)."""
    def __init__(self, c):
        super().__init__()
        self.norm = Norm(c)
        self.qkv = nn.Conv2d(c, 3 * c, 1)
        self.proj = nn.Conv2d(c, c, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, C, H * W).unbind(1)
        h = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
        return x + self.proj(h.transpose(1, 2).reshape(B, C, H, W))


class Upsample(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class StrongDecoder(nn.Module):
    """
    Asymmetric decoder: keep your existing (small) encoder, swap in this decoder.
    latent_ch=5 matches the 5x8x8 f=8 model. ch=(256, 256, 128, 128) for levels 8,16,32,64.
    Output is the decoder MEAN in [-1, 1]; pair with a learnable log_sigma in the loss.
    """
    def __init__(self, latent_ch=5, out_ch=3, ch=(256, 256, 128, 128), n_res=3, attn_8x8=True):
        super().__init__()
        self.conv_in = nn.Conv2d(latent_ch, ch[0], 3, padding=1)
        mid = [ResBlock(ch[0], ch[0])]
        if attn_8x8:
            mid.append(AttnBlock(ch[0]))
        mid.append(ResBlock(ch[0], ch[0]))
        self.mid = nn.Sequential(*mid)

        levels, cin = [], ch[0]
        for i, cout in enumerate(ch):
            blocks = []
            for _ in range(n_res):
                blocks.append(ResBlock(cin, cout))
                cin = cout
            if i < len(ch) - 1:
                blocks.append(Upsample(cout))
            levels.append(nn.Sequential(*blocks))
        self.up = nn.Sequential(*levels)
        self.norm_out = Norm(cin)
        self.conv_out = nn.Conv2d(cin, out_ch, 3, padding=1)

    def forward(self, z):
        h = self.mid(self.conv_in(z))
        h = self.up(h)
        return self.conv_out(F.silu(self.norm_out(h)))

    @property
    def last_layer(self):
        return self.conv_out.weight


# ----------------------------------------------------------------------------- critic / discriminator
class PatchDiscriminator(nn.Module):
    """
    PatchGAN for 64x64: 3 stride-2 convs -> 8x8 patch logits (receptive field ~ 46 px).
    critic="hinge"            -> spectral norm, no extra norm layers
    critic in {wgan_gp,sobolev} -> no spectral norm (the penalty does the constraining),
                                 GroupNorm-free LayerNorm-free plain convs (BatchNorm would
                                 break per-sample gradient penalties)
    """
    def __init__(self, in_ch=3, nf=64, n_layers=3, critic="hinge"):
        super().__init__()
        sn = spectral_norm if critic == "hinge" else (lambda m: m)
        layers = [sn(nn.Conv2d(in_ch, nf, 4, 2, 1)), nn.LeakyReLU(0.2, True)]
        c = nf
        for i in range(1, n_layers):
            layers += [sn(nn.Conv2d(c, c * 2, 4, 2, 1)), nn.LeakyReLU(0.2, True)]
            c *= 2
        layers += [sn(nn.Conv2d(c, c, 3, 1, 1)), nn.LeakyReLU(0.2, True),
                   sn(nn.Conv2d(c, 1, 3, 1, 1))]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _grad_sqnorm(D, x):
    """Per-sample ||grad_x D(x)||^2, where D(x) is summed over patch logits."""
    x = x.detach().requires_grad_(True)
    out = D(x).flatten(1).mean(1)              # one scalar per sample
    g, = torch.autograd.grad(out.sum(), x, create_graph=True)
    return g.flatten(1).pow(2).sum(1)          # [B]


# ----------------------------------------------------------------------------- loss
class AutoencoderGANLoss(nn.Module):
    """
    Generator side:  NLL(x | x_hat, sigma) + w_lpips*LPIPS + beta*KL + lambda_adapt*w_gan*G_adv
    Critic side:     hinge / WGAN-GP / Sobolev-GAN

    lambda_adapt is the taming-transformers adaptive weight: ratio of the gradient norms of the
    reconstruction and adversarial terms at the decoder's last layer. It stops the GAN term from
    overpowering reconstruction early on.
    """
    # disc_start: train reconstruction-only first. With an untrained critic the adaptive weight
    # saturates at its 1e4 clamp (seen in the smoke test), so the GAN must not switch on at step 0.
    def __init__(self, critic="hinge", disc_start=5000, w_lpips=1.0, w_gan=0.5, beta_kl=1.0,
                 gp_weight=10.0, sobolev_rho=1e-5, use_lpips=True, n_pixels=3 * 64 * 64):
        super().__init__()
        assert critic in ("hinge", "wgan_gp", "sobolev")
        self.critic, self.disc_start = critic, disc_start
        self.w_lpips, self.w_gan, self.beta_kl = w_lpips, w_gan, beta_kl
        self.gp_weight, self.rho = gp_weight, sobolev_rho
        self.log_sigma = nn.Parameter(torch.zeros(()))   # learnable decoder sigma (keep from rebuild)
        self.n_pixels = n_pixels
        self.register_buffer("sob_lambda", torch.zeros(()))  # Lagrange multiplier (Sobolev)
        self.lpips = None
        if use_lpips:
            import lpips                                    # pip install lpips
            self.lpips = lpips.LPIPS(net="vgg").eval()
            for p in self.lpips.parameters():
                p.requires_grad_(False)

    # ---- generator / autoencoder step
    def nll(self, x, x_hat):
        """Gaussian NLL per image, summed over pixels, averaged over batch (nats)."""
        s2 = torch.exp(2 * self.log_sigma)
        per_px = 0.5 * (x - x_hat).pow(2) / s2 + self.log_sigma + 0.5 * math.log(2 * math.pi)
        return per_px.flatten(1).sum(1).mean()

    def adaptive_weight(self, rec_loss, g_loss, last_layer):
        rg = torch.autograd.grad(rec_loss, last_layer, retain_graph=True)[0]
        gg = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        return (rg.norm() / (gg.norm() + 1e-4)).clamp(0.0, 1e4).detach()

    def generator_loss(self, x, x_hat, kl, D, last_layer, step):
        rec = self.nll(x, x_hat)
        perc = x.new_zeros(())
        if self.lpips is not None:
            # LPIPS is per-image ~O(0.1); scale to the per-image NLL scale so w_lpips is meaningful
            perc = self.lpips(x_hat, x).mean() * self.n_pixels
        rec_total = rec + self.w_lpips * perc
        loss = rec_total + self.beta_kl * kl
        logs = {"nll": rec.item(), "lpips": (perc / self.n_pixels).item(), "kl": kl.item(),
                "sigma": self.log_sigma.exp().item()}

        if step >= self.disc_start:
            g_adv = -D(x_hat).mean()          # same form for hinge, WGAN-GP and Sobolev
            lam = self.adaptive_weight(rec_total, g_adv, last_layer)
            loss = loss + lam * self.w_gan * g_adv
            logs.update(g_adv=g_adv.item(), lam=lam.item())
        return loss, logs

    # ---- critic step
    def critic_loss(self, x, x_hat, D):
        x_hat = x_hat.detach()
        dr, df = D(x).mean(), D(x_hat).mean()
        logs = {}
        if self.critic == "hinge":
            loss = F.relu(1 - D(x)).mean() + F.relu(1 + D(x_hat)).mean()
        elif self.critic == "wgan_gp":
            eps = torch.rand(x.size(0), 1, 1, 1, device=x.device)
            xi = eps * x + (1 - eps) * x_hat
            gp = (_grad_sqnorm(D, xi).sqrt() - 1).pow(2).mean()
            loss = (df - dr) + self.gp_weight * gp
            logs["gp"] = gp.item()
        else:  # sobolev: E_{mu}||grad f||^2 = 1, mu = equal mixture of real and fake
            omega = torch.cat([_grad_sqnorm(D, x), _grad_sqnorm(D, x_hat)]).mean()
            c = 1.0 - omega
            # critic maximises  (dr - df) + lambda*c - rho/2 * c^2   -> minimise the negative
            lam = self.sob_lambda.clone()       # snapshot: the in-place update below must not
            loss = -(dr - df) - lam * c + 0.5 * self.rho * c.pow(2)   # touch the autograd graph
            self.sob_lambda.sub_(self.rho * c.detach())               # dual ascent on the multiplier
            logs.update(omega=omega.item(), sob_lambda=self.sob_lambda.item())
        logs.update(d_real=dr.item(), d_fake=df.item(), w1_est=(dr - df).item())
        return loss, logs


# ----------------------------------------------------------------------------- training step
def train_step(encoder, decoder, D, loss_fn, opt_ae, opt_d, x, step, n_critic=1):
    """
    encoder(x) must return (mu, logvar) shaped [B, C, 8, 8].
    opt_ae must include encoder, decoder AND loss_fn.log_sigma.
    For wgan_gp / sobolev, n_critic=2-5 is typical; for hinge, 1.
    """
    mu, logvar = encoder(x)
    z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
    x_hat = decoder(z)
    kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).flatten(1).sum(1).mean()

    loss, logs = loss_fn.generator_loss(x, x_hat, kl, D, decoder.last_layer, step)
    opt_ae.zero_grad(set_to_none=True)
    loss.backward()
    opt_ae.step()

    if step >= loss_fn.disc_start:
        for _ in range(n_critic):
            d_loss, d_logs = loss_fn.critic_loss(x, x_hat, D)
            opt_d.zero_grad(set_to_none=True)
            d_loss.backward()
            opt_d.step()
        logs.update(d_logs)
    return logs


if __name__ == "__main__":
    dec = StrongDecoder()
    n = sum(p.numel() for p in dec.parameters())
    print(f"decoder params: {n/1e6:.2f} M")
    z = torch.randn(4, 5, 8, 8)
    print("out:", tuple(dec(z).shape))
    for crit in ("hinge", "wgan_gp", "sobolev"):
        D = PatchDiscriminator(critic=crit)
        print(crit, "D params:", f"{sum(p.numel() for p in D.parameters())/1e6:.2f} M",
              "logits:", tuple(D(torch.randn(4, 3, 64, 64)).shape))
