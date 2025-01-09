import torch
import torch.nn as nn
import numpy as np

class GMVAE(nn.Module):
    def __init__(self, k, n_z, n_x, qy_dims, qz_dims, pz_dims, px_dims, r_nent, use_batch_norm=False):
        super(GMVAE, self).__init__()

        self.k = k
        self.n_z = n_z
        self.n_x = n_x
        self.r_nent = r_nent

        # y transformation layer
        self.y_transform = nn.Linear(self.k, self.k)

        # Qy network
        qy_layers = []
        previous_dim = self.n_x
        for hidden_dim in qy_dims:
            qy_layers.append(nn.Linear(previous_dim, hidden_dim, bias=not use_batch_norm))
            qy_layers.append(nn.ReLU())
            if use_batch_norm:
                qy_layers.append(nn.BatchNorm1d(hidden_dim, affine=True))
            previous_dim = hidden_dim
        qy_layers.append(nn.Linear(previous_dim, k))
        qy_layers.append(nn.ReLU())
        self.qy_nn = nn.Sequential(*qy_layers)

        # Qz network
        qz_layers = []
        previous_dim = self.k + self.n_x
        for hidden_dim in qz_dims:
            qz_layers.append(nn.Linear(previous_dim, hidden_dim))
            qz_layers.append(nn.ReLU())
            if use_batch_norm:
                qz_layers.append(nn.BatchNorm1d(hidden_dim, affine=True))
            previous_dim = hidden_dim
        self.qz_nn = nn.Sequential(*qz_layers)
        self.zm_layer = nn.Linear(previous_dim, n_z)
        self.zv_layer = nn.Linear(previous_dim, n_z)

        # Pz network
        pz_layers = []
        previous_dim = self.k
        for hidden_dim in pz_dims:
            pz_layers.append(nn.Linear(previous_dim, hidden_dim))
            pz_layers.append(nn.ReLU())
            if use_batch_norm:
                pz_layers.append(nn.BatchNorm1d(hidden_dim, affine=True))
            previous_dim = hidden_dim
        self.pz_nn = nn.Sequential(*pz_layers)
        self.zm_prior_layer = nn.Linear(previous_dim, n_z)
        self.zv_prior_layer = nn.Linear(previous_dim, n_z)

        # Px network
        px_layers = []
        previous_dim = self.n_z
        for hidden_dim in px_dims:
            px_layers.append(nn.Linear(previous_dim, hidden_dim, bias=not use_batch_norm))
            px_layers.append(nn.ReLU())
            if use_batch_norm:
                px_layers.append(nn.BatchNorm1d(hidden_dim, affine=True))
            previous_dim = hidden_dim
        self.px_nn = nn.Sequential(*px_layers)
        self.xm_layer = nn.Linear(previous_dim, n_x)
        self.xv_layer = nn.Linear(previous_dim, n_x)

    @staticmethod
    def log_normal(x, mu, var, eps=1e-10):
        return -0.5 * torch.sum(torch.log(torch.tensor(2.0) * torch.pi) + (x - mu).pow(2) / var + var.log(), dim=-1)  # log probability of a normal (Gaussian) distribution

    def loss_function(self, x, xm, xv, z, zm, zv, zm_prior, zv_prior):
        return (
            -self.log_normal(x, xm, xv)                                                             # Reconstruction Loss
            + self.log_normal(z, zm, zv) - self.log_normal(z, zm_prior, zv_prior)                   # Regularization Loss (KL Divergence)
            - torch.log(torch.tensor(1/self.k, device=x.device))                                    # Entropy Regularization
        )
    
    def sum_aggregation(self, xz_list, qy):
        a = np.zeros(xz_list[0].shape)
        for a_i in range(xz_list[0].shape[1]):
            for y_i in range(qy.shape[1]):
                a[:, a_i] += xz_list[y_i][:,a_i]*qy[:,y_i]
        return a
    
    def forward(self, data):

        qy_logit = self.qy_nn(data)
        qy = torch.softmax(qy_logit, dim=1)

        y_ = torch.zeros([data.shape[0], self.k]).to(data.device)

        zm_list, zv_list, z_list = [], [], []
        xm_list, xv_list, x_list = [], [], []
        zm_prior_list, zv_prior_list = [], []

        for i in range(self.k):
            # One-hot y
            y = y_ + torch.eye(self.k).to(data.device)[i]

            # Qz
            h0 = self.y_transform(y)
            xy = torch.cat([data, h0], dim=1)
            qz_logit = self.qz_nn(xy)
            zm = self.zm_layer(qz_logit)
            zv = torch.nn.functional.softplus(self.zv_layer(qz_logit))
            noise = torch.randn_like(torch.sqrt(zv))
            z_sample = zm + noise * zv

            zm_list.append(zm)
            zv_list.append(zv)
            z_list.append(z_sample)

            # Pz (prior)
            pz_logit = self.pz_nn(y)
            zm_prior = self.zm_prior_layer(pz_logit)
            zv_prior = torch.nn.functional.softplus(self.zv_prior_layer(pz_logit))
            noise = torch.randn_like(torch.sqrt(zv_prior))
            z_prior_sample = zm_prior + noise * zv_prior

            zm_prior_list.append(zm_prior)
            zv_prior_list.append(zv_prior)

            # Px
            px_logit = self.px_nn(z_prior_sample)
            xm = self.xm_layer(px_logit)
            xv = torch.nn.functional.softplus(self.xv_layer(px_logit))
            noise = torch.randn_like(torch.sqrt(xv))
            x_sample = xm + noise * xv

            xm_list.append(xm)
            xv_list.append(xv)
            x_list.append(x_sample)

        # Cross Entropy
        nent = -torch.sum(qy * torch.log_softmax(qy_logit, dim=1), dim=1).mean()

        # Reconstruction (log probability) + Regularization Loss (KL Divergence)
        losses = [None] * self.k
        for i in range(self.k):
            losses[i] = self.loss_function(data, xm_list[i], xv_list[i],
                                           z_list[i], zm_list[i], zv_list[i],
                                           zm_prior_list[i], zv_prior_list[i])

        # Class-wise Weighted total loss function: Cross_entropy + w · [reconstruction + regularization]
        total_loss = sum([self.r_nent*nent]+[qy[:,i]*losses[i] for i in range(self.k)])

        return qy, qy_logit, z_list, x_list, total_loss, nent