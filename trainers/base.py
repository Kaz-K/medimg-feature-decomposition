import os
import numpy as np

import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from dataio import get_data_loader
from networks import init_models
from functions import GeneralizedDiceLoss
from functions import SoftDiceLoss
from functions import FocalLoss
from functions import DiceCoefficient
from functions import OneHotEncoder
import functions.pytorch_ssim as pytorch_ssim
from utils import norm
from utils import denorm
from utils import minmax_norm


class BaseModel(pl.LightningModule):

    def __init__(self, config, save_dir_path):
        super().__init__()

        self.config = config
        self.save_dir_path = save_dir_path

        self.class_name_to_index = self.config.metric.class_name_to_index._asdict()
        self.index_to_class_name = {v: k for k, v in self.class_name_to_index.items()}

        self.encoder, self.vq1, self.vq2, \
        self.seg_decoder, self.img_decoder, self.discriminator, self.img_dis_1, self.img_dis_2 = init_models(
            input_dim=self.config.model.input_dim,
            img_output_dim=self.config.model.img_output_dim,
            seg_output_dim=self.config.model.seg_output_dim,
            img_output_act=self.config.model.img_output_act,
            emb_dim=self.config.model.emb_dim,
            dict_size=self.config.model.dict_size,
            enc_filters=self.config.model.enc_filters,
            dec_filters=self.config.model.dec_filters,
            latent_size=self.config.model.latent_size,
            init_type=self.config.model.init_type,
            init_latent_discriminator=self.config.run.use_latent_discrimination,
            latent_discriminator_type=self.config.run.latent_discriminator_type,
            init_image_discriminator=self.config.run.use_image_discrimination,
            img_dis_filters=self.config.model.img_dis_filters,
            faiss_backend=self.config.model.faiss_backend,
            apply_spectral_norm_to_discriminator=config.run.apply_spectral_norm,
        )

        if hasattr(self.config, 'dice_loss'):
            if self.config.dice_loss.type == 'generalized':
                self.l_dice = GeneralizedDiceLoss()
            elif self.config.dice_loss.type == 'soft':
                self.l_dice = SoftDiceLoss(ignore_index=self.config.dice_loss.ignore_index)
        else:
            self.l_dice = GeneralizedDiceLoss()

        self.l_focal = FocalLoss(gamma=self.config.focal_loss.gamma,
                                 alpha=self.config.focal_loss.alpha)

        self.one_hot_encoder = OneHotEncoder(n_classes=self.config.metric.n_classes).forward

        self.dice_metric = DiceCoefficient(n_classes=self.config.metric.n_classes,
                                           index_to_class_name=self.index_to_class_name)

    def export_models(self, path):
        os.makedirs(path, exist_ok=True)

        torch.save(self.encoder.state_dict(), os.path.join(path, 'encoder.pth'))
        torch.save(self.vq1.state_dict(), os.path.join(path, 'vq1.pth'))
        torch.save(self.vq2.state_dict(), os.path.join(path, 'vq2.pth'))
        torch.save(self.seg_decoder.state_dict(), os.path.join(path, 'seg_decoder.pth'))
        torch.save(self.img_decoder.state_dict(), os.path.join(path, 'img_decoder.pth'))

        if self.discriminator is not None:
            torch.save(self.discriminator.state_dict(), os.path.join(path, 'discriminator.pth'))

    def train_dataloader(self):
        data_loader = get_data_loader(
            dataset_name=self.config.dataset.name,
            modalities=self.config.dataset.modalities,
            root_dir_paths=self.config.dataset.root_dir_paths,
            augmentation_type=self.config.dataset.augmentation_type,
            use_shuffle=self.config.dataset.use_shuffle,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            use_image_discrimination=self.config.run.use_image_discrimination,
        )
        self.train_dataset = data_loader.dataset
        return data_loader

    def val_dataloader(self):
        return get_data_loader(
            dataset_name=self.config.dataset.name,
            modalities=self.config.dataset.modalities,
            root_dir_paths=self.config.dataset.root_dir_paths,
            augmentation_type='none',
            use_shuffle=False,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
        )

    def test_dataloader(self):
        return get_data_loader(
            dataset_name=self.config.dataset.name,
            modalities=self.config.dataset.modalities,
            root_dir_paths=self.config.dataset.root_dir_paths,
            augmentation_type='none',
            use_shuffle=False,
            batch_size=self.config.dataset.batch_size,
            num_workers=self.config.dataset.num_workers,
            initial_randomize=False,
        )

    def forward(self, x, with_ids=False):
        lat_1, lat_2 = self.encoder(x)
        qlat_1, lat_loss_1, id_1 = self.vq1(lat_1)
        qlat_2, lat_loss_2, id_2 = self.vq2(lat_2)

        seg_logit = self.seg_decoder(qlat_2)

        null_feat = torch.zeros_like(seg_logit)
        seg_composite = seg_logit.clone().detach()
        seg_mask = torch.argmax(seg_composite, dim=1, keepdim=True)
        seg_feat = torch.where(seg_mask==0, null_feat, seg_composite)

        recon1 = self.img_decoder(qlat_1, seg_feat)
        recon2 = self.img_decoder(qlat_1, null_feat)

        if with_ids:
            return recon1, recon2, seg_logit, lat_loss_1, lat_loss_2, qlat_1, qlat_2, id_1, id_2
        else:
            return recon1, recon2, seg_logit, lat_loss_1, lat_loss_2, qlat_1, qlat_2

    def l_latent(self, latent1, latent2):
        return self.config.loss_weight.w_latent * (latent1 + latent2)

    def l_recon(self, recon, target, mask=None, mode='none'):
        if self.config.recon_loss.apply_ssim:
            return self.l_mse_ssim(recon, target, mask, mode)
        else:
            return self.l_mse(recon, target, mask, mode)

    def l_mse(self, recon, target, mask=None, mode='none'):
        assert mode in ['none', 'crop', 'emphasize']
        if mode == 'none':
            loss = F.mse_loss(recon, target, reduction='mean')
        elif mode == 'crop':
            loss = F.mse_loss(recon[mask], target[mask], reduction='mean')
        elif mode == 'emphasize':
            loss = F.mse_loss(recon, target, reduction='mean')
            if mask.byte().any().item() == 1:
                loss += F.mse_loss(recon[mask], target[mask], reduction='mean')
        return self.config.loss_weight.w_recon * loss

    def l_mse_ssim(self, recon, target, mask=None, mode='none'):
        assert mode in ['none', 'crop', 'emphasize']
        ssim_loss = pytorch_ssim.SSIM(window_size=11)
        if mode == 'none':
            loss = F.mse_loss(recon, target, reduction='sum') \
                 + (1.0 - ssim_loss(recon, target)) * torch.numel(recon)
        elif mode == 'crop':
            loss = F.mse_loss(recon[mask], target[mask], reduction='sum') \
                 + (1.0 - ssim_loss(recon, target, mask)) * torch.numel(recon[mask])
        elif mode == 'emphasize':
            loss = F.mse_loss(recon, target, reduction='sum') \
                 + (1.0 - ssim_loss(recon, target)) * torch.numel(recon)
            if mask.byte().any().item() == 1:
                loss += F.mse_loss(recon[mask], target[mask], reduction='sum') \
                     + (1.0 - ssim_loss(recon, target, mask)) * torch.numel(recon[mask])
        return self.config.loss_weight.w_recon * loss / recon.numel()

    def l_similarity(self, recon1, recon2, mask):
        loss = F.mse_loss(recon1[mask], recon2[mask], reduction='mean')
        return self.config.loss_weight.w_similarity * loss

    def l_seg(self, seg_logit, seg_label):
        target = self.one_hot_encoder(seg_label)
        dice_loss = self.l_dice(seg_logit, target)
        focal_loss = self.l_focal(seg_logit, target)
        return self.config.loss_weight.w_seg * (dice_loss + focal_loss)

    def validation_step(self, batch, batch_idx):
        image = batch['image']
        s_label = batch['seg_label']

        if self.config.model.apply_input_norm:
            image = norm(image)

        with torch.no_grad():
            recon1, recon2, seg_logit, \
            lat_loss_1, lat_loss_2, qlat_1, qlat_2 = self.forward(image)

        dice = self.dice_metric(seg_logit, s_label)

        if batch_idx == 0:
            image = image.detach().cpu()
            recon1 = recon1.detach().cpu()
            recon2 = recon2.detach().cpu()

            if self.config.model.apply_input_norm:
                image = denorm(image)
                recon1 = denorm(recon1)
                recon2 = denorm(recon2)

            else:
                vmin = image.min()
                vmax = image.max()
                image = minmax_norm(image, vmin=vmin, vmax=vmax)
                recon1 = minmax_norm(recon1, vmin=vmin, vmax=vmax)
                recon2 = minmax_norm(recon2, vmin=vmin, vmax=vmax)

            seg_label = s_label.detach().cpu()
            seg_output = seg_logit.argmax(dim=1).detach().cpu()

            n_images = min(self.config.save.n_save_images, image.size(0))

            save_modalities = ['t1ce']
            if 'flair' in self.config.dataset.modalities:
                save_modalities.append('flair')

            for save_modality in save_modalities:
                idx = self.config.dataset.modalities.index(save_modality)

                save_image = image[:n_images, ...][:, idx, ...][:, np.newaxis, ...]
                save_recon1 = recon1[:n_images, ...][:, idx, ...][:, np.newaxis, ...]
                save_recon2 = recon2[:n_images, ...][:, idx, ...][:, np.newaxis, ...]

                image_grid = torch.cat([save_image, save_recon1, save_recon2])
                self.logger.log_images(save_modality, image_grid, self.current_epoch, self.global_step, nrow=n_images)

            seg_label = seg_label[:n_images, ...].float()[:, np.newaxis, ...]
            seg_output = seg_output[:n_images, ...].float()[:, np.newaxis, ...]

            max_label_val = 3
            seg_label /= max_label_val
            seg_output /= max_label_val

            label_grid = torch.cat([seg_label, seg_output])
            self.logger.log_images('labels', label_grid, self.current_epoch, self.global_step, nrow=n_images)

        return dice

    def validation_epoch_end(self, outputs):
        metrics = {
            'epoch': self.current_epoch,
            'iteration': self.global_step,
        }

        for key in self.class_name_to_index.keys():
            avg_value = np.stack([x[key] for x in outputs]).mean()
            metrics.update({
                key: avg_value
            })

        self.logger.log_val_metrics(metrics)

    def test_step(self, batch, batch_idx):
        patient_ids = batch['patient_id']
        n_slice = batch['n_slice']
        image = batch['image']
        s_label = batch['seg_label']

        if self.config.model.apply_input_norm:
            image = norm(image)

        with torch.no_grad():
            _, _, _, _, _, _, _, id_1, id_2 = self.forward(image, with_ids=True)

        id_1 = id_1.detach().cpu().numpy()
        id_2 = id_2.detach().cpu().numpy()

        for i in range(len(patient_ids)):
            patient_id = patient_ids[i]
            slice_num = n_slice[i].item()

            save_path = os.path.join(self.save_dir_path, patient_id)
            os.makedirs(save_path, exist_ok=True)

            np.save(os.path.join(save_path, 'id_1_{}'.format(str(slice_num).zfill(4))), id_1[i, ...])
            np.save(os.path.join(save_path, 'id_2_{}'.format(str(slice_num).zfill(4))), id_2[i, ...])
