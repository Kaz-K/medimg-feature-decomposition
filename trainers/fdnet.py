import torch
from itertools import chain
import torch.optim as optim

from .base import BaseModel
from utils import norm


class FeatDecompNet(BaseModel):

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
        return [e_optim], []

    def training_step(self, batch, batch_idx):
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
        similarity_loss = self.l_similarity(recon1, recon2, n_mask)
        seg_loss = self.l_seg(seg_logit, s_label)

        total_loss = (latent_loss + recon_loss + similarity_loss + seg_loss).sum()

        self.log('epoch', self.current_epoch)
        self.log('iteration', self.global_step)
        self.log('total_loss', total_loss.sum(), prog_bar=True)
        self.log('latent_loss', latent_loss.sum(), prog_bar=True)
        self.log('latent_loss_1', self.config.loss_weight.w_latent * lat_loss_1.sum(), prog_bar=True)
        self.log('latent_loss_2', self.config.loss_weight.w_latent * lat_loss_2.sum(), prog_bar=True)
        self.log('recon_loss', recon_loss.sum(), prog_bar=True)
        self.log('recon_loss_1', recon_loss_1.sum(), prog_bar=True)
        self.log('recon_loss_2', recon_loss_2.sum(), prog_bar=True)
        self.log('similarity_loss', similarity_loss.sum(), prog_bar=True)
        self.log('seg_loss', seg_loss.sum(), prog_bar=True)

        return total_loss
