import torch
from itertools import chain
import torch.optim as optim

from .base import BaseModel
from utils import norm


class FeatDecompNetDensityMatch(BaseModel):

    def __init__(self, config, save_dir_path):
        super().__init__(config, save_dir_path)

    def configure_optimizers(self):
        e_optim = optim.Adam(filter(lambda p: p.requires_grad,
                             chain(self.encoder.parameters(),
                                   self.vq1.parameters(),
                                   self.vq2.parameters(),
                                   self.seg_decoder.parameters(),
                                   self.img_decoder.parameters())),
                             self.config.optimizer.e_lr, [0.9, 0.9999],
                             weight_decay=self.config.optimizer.weight_decay)

        l_optim = optim.Adam(filter(lambda p: p.requires_grad,
                             chain(self.discriminator.parameters())),
                             self.config.optimizer.l_lr, [0.9, 0.9999],
                             weight_decay=self.config.optimizer.weight_decay)

        return [e_optim, l_optim], []

    def l_gradient_penalty(self,
                           n_qlat: torch.Tensor,
                           a_qlat: torch.Tensor,
                           ) -> torch.Tensor:

        eta = n_qlat.new_empty(n_qlat.size(0), 1, 1, 1).uniform_(0, 1)
        interpolated = eta * n_qlat + (1 - eta) * a_qlat
        interpolated.requires_grad_()

        prob_interpolated = self.discriminator(interpolated)

        grad = torch.autograd.grad(
            outputs=prob_interpolated,
            inputs=interpolated,
            grad_outputs=torch.ones_like(prob_interpolated),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        return (grad.norm(2, dim=1) - 1).pow(2).mean()

    def training_step(self, batch, batch_idx, optimizer_idx):
        (e_optim, l_optim) = self.optimizers()

        image = batch['image']
        s_label = batch['seg_label']
        c_label = batch['class_label']
        n_label = torch.zeros_like(c_label)

        n_mask = (s_label == 0).unsqueeze(1).expand_as(image)
        a_mask = (s_label > 0).unsqueeze(1).expand_as(image)

        if self.config.model.apply_input_norm:
            image = norm(image)

        recon1, recon2, seg_logit, \
        lat_loss_1, lat_loss_2, qlat_1, qlat_2 = self.forward(image)

        latent_loss = self.l_latent(lat_loss_1, lat_loss_2)
        recon_loss_1 = self.l_recon(recon1, image, a_mask, mode='emphasize')
        recon_loss_2 = self.l_recon(recon2, image, n_mask, mode='crop')
        recon_loss = recon_loss_1 + recon_loss_2
        seg_loss = self.l_seg(seg_logit, s_label)

        total_loss = latent_loss + recon_loss + seg_loss

        a_indices = (c_label == 1).nonzero(as_tuple=True)[0]
        n_indices = (c_label == 0).nonzero(as_tuple=True)[0]

        length = min(len(a_indices), len(n_indices))
        w_wgan_dis = self.config.loss_weight.w_wgan_discriminator
        w_wgan_gen = self.config.loss_weight.w_wgan_generator

        if length > 0:
            a_indices = a_indices[:length]
            n_indices = n_indices[:length]

            a_qlat_1 = qlat_1[a_indices, ...]
            n_qlat_1 = qlat_1[n_indices, ...]

            for _ in range(self.config.run.discriminator_iter):
                self.discriminator.requires_grad_(True)
                d_abnormal_loss = w_wgan_dis * self.discriminator(a_qlat_1.detach()).mean()
                d_normal_loss = - w_wgan_dis * self.discriminator(n_qlat_1.detach()).mean()
                gp_loss = w_wgan_dis * self.l_gradient_penalty(n_qlat_1.detach(), a_qlat_1.detach())

                total_discriminator_loss = d_abnormal_loss + d_normal_loss + gp_loss

                self.manual_backward(total_discriminator_loss.sum(), l_optim)
                l_optim.step()
                l_optim.zero_grad()

            self.discriminator.requires_grad_(False)

            g_abnormal_loss = - w_wgan_gen * self.discriminator(a_qlat_1).mean()

            total_loss += g_abnormal_loss

            self.manual_backward(total_loss.sum(), e_optim)
            e_optim.step()
            e_optim.zero_grad()

        else:
            g_abnormal_loss = d_abnormal_loss = \
            d_normal_loss = gp_loss = image.new_tensor(0.0).sum()

            self.manual_backward(total_loss.sum(), e_optim)
            e_optim.step()
            e_optim.zero_grad()

        self.log('epoch', self.current_epoch)
        self.log('iteration', self.global_step)
        self.log('total_loss', total_loss.sum(), prog_bar=True)
        self.log('latent_loss', latent_loss.sum(), prog_bar=True)
        self.log('latent_loss_1', lat_loss_1.sum(), prog_bar=True)
        self.log('latent_loss_2', lat_loss_2.sum(), prog_bar=True)
        self.log('recon_loss', recon_loss.sum(), prog_bar=True)
        self.log('recon_loss_1', recon_loss_1.sum(), prog_bar=True)
        self.log('recon_loss_2', recon_loss_2.sum(), prog_bar=True)
        self.log('seg_loss', seg_loss.sum(), prog_bar=True)
        self.log('g_abnormal_loss', g_abnormal_loss.sum(), prog_bar=True)
        self.log('d_abnormal_loss', d_abnormal_loss.sum(), prog_bar=True)
        self.log('d_normal_loss', d_normal_loss.sum(), prog_bar=True)
        self.log('gp_loss', gp_loss.sum(), prog_bar=True)

        return total_loss.sum()
