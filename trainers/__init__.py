import os
import pytorch_lightning as pl

from utils import Logger
from utils import ModelSaver
from .base import BaseModel
from .fdnet import FeatDecompNet
from .fdnetdm import FeatDecompNetDensityMatch


def build_monitoring_metrics(config):

    monitoring_metrics = ['epoch', 'iteration', 'total_loss', 'latent_loss',
                          'latent_loss_1', 'latent_loss_2', 'recon_loss',
                          'recon_loss_1', 'recon_loss_2', 'similarity_loss',
                          'seg_loss']

    if config.run.use_latent_discrimination:
        monitoring_metrics += ['g_abnormal_loss', 'd_abnormal_loss',
                               'd_normal_loss', 'gp_loss']

    monitoring_metrics += ['NET', 'ED', 'ET']

    return monitoring_metrics


def const_trainer_train(config, seed, args):

    monitoring_metrics = build_monitoring_metrics(config)

    logger = Logger(save_dir=config.save.save_dir,
                    config=config,
                    seed=seed,
                    name=config.save.study_name,
                    monitoring_metrics=monitoring_metrics)

    save_dir_path = logger.log_dir

    checkpoint_callback = ModelSaver(limit_num=10,
                                     monitor=None,
                                     filepath=os.path.join(
                                         save_dir_path, 'ckpt-{epoch:04d}-{total_loss:.2f}'),
                                     save_top_k=-1)

    if config.run.use_latent_discrimination:
        if config.run.resume_checkpoint:
            print('Training will resume from: {}'.format(
                config.run.resume_checkpoint))
            encoding_model = FeatDecompNetDensityMatch.load_from_checkpoint(
                config.run.resume_checkpoint,
                config=config,
                save_dir_path=save_dir_path,
            )
        else:
            encoding_model = FeatDecompNetDensityMatch(config, save_dir_path)

        automatic_optimization = False

    else:
        if config.run.resume_checkpoint:
            print('Training will resume from: {}'.format(
                config.run.resume_checkpoint))
            encoding_model = FeatDecompNet.load_from_checkpoint(
                config.run.resume_checkpoint,
                config=config,
                save_dir_path=save_dir_path,
            )
        else:
            encoding_model = FeatDecompNet(config, save_dir_path)

        automatic_optimization = True

    trainer = pl.Trainer(gpus=[0],
                         num_nodes=1,
                         max_epochs=config.run.n_epochs,
                         progress_bar_refresh_rate=1,
                         automatic_optimization=automatic_optimization,
                         distributed_backend=config.run.distributed_backend,
                         deterministic=True,
                         logger=logger,
                         sync_batchnorm=True,
                         checkpoint_callback=checkpoint_callback,
                         resume_from_checkpoint=config.run.resume_checkpoint,
                         limit_val_batches=10)

    return trainer, encoding_model


def const_trainer_test(config, seed, args):

    save_dir_path = config.save.save_dir

    if config.run.use_latent_discrimination:
        print('Training will resume from: {}'.format(
            config.run.resume_checkpoint))
        encoding_model = FeatDecompNetDensityMatch.load_from_checkpoint(
            config.run.resume_checkpoint,
            config=config,
            save_dir_path=save_dir_path,
        )

    else:
        print('Training will resume from: {}'.format(
            config.run.resume_checkpoint))
        encoding_model = FeatDecompNet.load_from_checkpoint(
            config.run.resume_checkpoint,
            config=config,
            save_dir_path=save_dir_path,
        )

    trainer = pl.Trainer(gpus=[0],
                         num_nodes=1,
                         max_epochs=config.run.n_epochs,
                         progress_bar_refresh_rate=20,
                         automatic_optimization=True,
                         distributed_backend=config.run.distributed_backend,
                         deterministic=True,
                         logger=None,
                         sync_batchnorm=True,
                         checkpoint_callback=None,
                         resume_from_checkpoint=config.run.resume_checkpoint,
                         limit_val_batches=10)

    return trainer, encoding_model
