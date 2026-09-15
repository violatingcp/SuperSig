import torch
import torch.nn as nn
import torch.nn.functional as F


class DualSuperVisReg(nn.Module):
    """
    Every class is an isotropic gaussian 
    Every class centroid lives in a global gaussian
    
    global loss = L_center + L_scale + L_cov
    local loss = L_local_center + L_local_cov + L_local_shape
    total loss = global loss + local loss
    """

    def __init__(
        self,
        num_classes=9,
        embed_dim=4,
        num_projections=512,
        global_scale=10.0,
        local_target_scale=1.0,
        w_center=1,
        w_scale=3,  
        w_shape=2, 
        w_global=0.2,
        w_local_center=1.0,
        w_local_shape=10.0,
        w_local_cov=1.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.num_projections = num_projections
        self.global_scale = global_scale
        self.local_target_scale = local_target_scale

        self.w_center = w_center
        self.w_scale = w_scale
        self.w_shape = w_shape
        self.w_global = w_global
        self.w_local_center = w_local_center
        self.w_local_shape = w_local_shape
        self.w_local_cov = w_local_cov

        # Fixed projection matrices to monitor loss curve
        W_init = torch.randn(self.embed_dim, self.num_projections)
        W_fixed = F.normalize(W_init, p=2, dim=0)
        self.register_buffer("W_fixed", W_fixed)

    def _global_macro_visreg(self, points):
        N, D = points.shape
        device = points.device
        if N < 4:
            zero = torch.zeros((), device=device)
            return zero, {"global_center": zero, "global_scale": zero, "global_shape": zero}

        mu = points.mean(dim=0)
        L_center = mu.pow(2).mean()

        p_cent = points - mu
        std = p_cent.std(dim=0, unbiased=False) + 1e-6
        L_scale = (self.global_scale - std).pow(2).mean()

        p_norm = p_cent / std.detach()

        proj = p_norm @ self.W_fixed
        proj_sorted = torch.sort(proj, dim=0).values

        u = torch.arange(1, N + 1, device=device).float() / (N + 1)
        target = (self.global_scale * torch.distributions.Normal(0, 1).icdf(u)).unsqueeze(-1)
        L_shape = (proj_sorted - target).pow(2).mean()

        weighted = self.w_center * L_center + self.w_scale * L_scale + self.w_shape * L_shape
        raw = {
            "global_center": L_center.detach(),
            "global_scale": L_scale.detach(),
            "global_shape": L_shape.detach(),
        }
        return weighted, raw

    def forward(self, z, y):
        device = z.device
        unique_classes = torch.unique(y)

        loss_local = torch.zeros((), device=device)
        raw_local_center_sum = torch.zeros((), device=device)
        raw_local_cov_sum = torch.zeros((), device=device)
        raw_local_shape_sum = torch.zeros((), device=device)
        centroids = []
        count = 0

        for c in unique_classes:
            if c.item() < 0:
                continue

            mask = y == c
            zc = z[mask]
            if zc.shape[0] < 4:
                continue

            N, D = zc.shape

            mu_c = zc.mean(dim=0, keepdim=True)
            centroids.append(mu_c.squeeze(0))

            L_local_center = mu_c.pow(2).mean()

            zc_cent = zc - mu_c
            std_c = zc_cent.std(dim=0, unbiased=False) + 1e-6
            zc_norm = zc_cent / std_c.detach()

            if N > 1:
                cov_c = (zc_cent.t() @ zc_cent) / (N - 1)
                target_cov = (self.local_target_scale ** 2) * torch.eye(D, device=device)
                L_cov = F.mse_loss(cov_c, target_cov)
            else:
                L_cov = torch.zeros((), device=device)

            # Fixed Projection Slice Step
            p = zc_norm @ self.W_fixed
            p_sorted = torch.sort(p, dim=0).values

            u = torch.arange(1, N + 1, device=device).float() / (N + 1)
            target = torch.distributions.Normal(0, 1).icdf(u).unsqueeze(-1)
            L_local_shape = (p_sorted - target).pow(2).mean()

            loss_local = loss_local + (
                self.w_local_center * L_local_center
                + self.w_local_shape * L_local_shape
                + self.w_local_cov * L_cov
            )

            raw_local_center_sum = raw_local_center_sum + L_local_center.detach()
            raw_local_cov_sum = raw_local_cov_sum + L_cov.detach()
            raw_local_shape_sum = raw_local_shape_sum + L_local_shape.detach()

            count += 1

        if count >= 4:
            centroid_points = torch.stack(centroids, dim=0)
            loss_global, raw_global = self._global_macro_visreg(centroid_points)
        else:
            zero = torch.zeros((), device=device)
            loss_global = zero
            raw_global = {"global_center": zero, "global_scale": zero, "global_shape": zero}

        if count == 0:
            return self.w_global * loss_global, {
                "global_loss": loss_global.detach(),
                "local_loss": torch.tensor(0.0, device=device),
                **raw_global,
                "raw_local_center": torch.tensor(0.0, device=device),
                "raw_local_cov": torch.tensor(0.0, device=device),
                "raw_local_shape": torch.tensor(0.0, device=device),
            }

        loss_local = loss_local / count
        total_loss = self.w_global * loss_global + loss_local

        return total_loss, {
            "global_loss": loss_global.detach(),
            "local_loss": loss_local.detach(),
            **raw_global,
            "raw_local_center": (raw_local_center_sum / count).detach(),
            "raw_local_cov": (raw_local_cov_sum / count).detach(),
            "raw_local_shape": (raw_local_shape_sum / count).detach(),
        }
