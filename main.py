import warnings
warnings.filterwarnings("ignore")

import argparse, os, sys, datetime, glob, importlib
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
import torch
import torchvision
from torch.utils.data import random_split, DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, Callback, LearningRateMonitor
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from pytorch_lightning.loggers import TensorBoardLogger

from ldm.data.utils import custom_collate


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
        "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument("-p", "--project", help="name of new or path to existing project")
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=2023,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "--basedir",
        type=str,
        default="checkpoints",
        help="the base directory",
    )
    parser.add_argument(
        "--test_first",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="test before training",
    )

    return parser


def nondefault_trainer_args(opt):
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


def instantiate_from_config(config):
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""
    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class DataModuleFromConfig(pl.LightningDataModule):
    def __init__(self, batch_size, val_scale=4, train=None, validation=None, test=None,
                 wrap=False, num_workers=None):
        super().__init__()
        self.batch_size = batch_size
        self.val_scale = val_scale  # use larger batch size during validation and testing.
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size*2
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = self._val_dataloader
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = self._test_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    def _train_dataloader(self):
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=True, collate_fn=custom_collate,
                          pin_memory=False, drop_last=False)

    def _val_dataloader(self):
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size * self.val_scale,
                          num_workers=self.num_workers, collate_fn=custom_collate,
                          pin_memory=False, drop_last=False)

    def _test_dataloader(self):
        return DataLoader(self.datasets["test"], batch_size=self.batch_size * self.val_scale,
                          num_workers=self.num_workers, collate_fn=custom_collate,
                          pin_memory=False, drop_last=False)


class SetupCallback(Callback):
    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_pretrain_routine_start(self, trainer, pl_module):
        if not getattr(trainer, 'get_ready', False):
            setattr(trainer, 'get_ready', True)
            if trainer.global_rank == 0:

                # Create logdirs and save configs
                os.makedirs(self.logdir, exist_ok=True)
                os.makedirs(self.ckptdir, exist_ok=True)
                os.makedirs(self.cfgdir, exist_ok=True)

                print("Save project config")
                OmegaConf.save(self.config,
                               os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

                print("Save lightning config")
                OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                               os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

                OmegaConf.save(OmegaConf.merge(self.config, OmegaConf.create({"lightning": self.lightning_config})),
                               os.path.join(self.cfgdir, "{}-all.yaml".format(self.now)))

            else:
                # ModelCheckpoint callback created log directory --- remove it
                if not self.resume and os.path.exists(self.logdir):
                    dst, name = os.path.split(self.logdir)
                    dst = os.path.join(dst, "child_runs", name)
                    os.makedirs(os.path.split(dst)[0], exist_ok=True)
                    try:
                        os.rename(self.logdir, dst)
                    except FileNotFoundError:
                        pass


class DummyImageLogger(Callback):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def check_frequency(self, epoch_idx, batch_idx):
        return False


class ImageLoggerBase(Callback):
    def __init__(self, batch_frequency, frequency_base=2, nrow=8, max_images=8, clamp=True, increase_log_steps=True):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.nrow = nrow
        self.logger_log_images = {
            pl.loggers.tensorboard.TensorBoardLogger: self._tensorboard,
            pl.loggers.WandbLogger: self._wandb,
        }
        self.log_steps = [frequency_base ** n for n in range(int(np.log(self.batch_freq) / np.log(frequency_base)) + 1)]
        if not increase_log_steps:
            self.log_steps = [self.batch_freq]
        self.clamp = clamp

    @rank_zero_only
    def _tensorboard(self, pl_module, images, split, batch_idx=None):
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=self.nrow)
            grid = (grid + 1.0) / 2.0
            label = f'{split}/{k}' if batch_idx is None else f'{split}/{k}_{batch_idx}'
            pl_module.logger.experiment.add_image(label, grid.detach().cpu(), pl_module.global_step)

    @rank_zero_only
    def _wandb(self, pl_module, images, batch_idx, split):
        raise ValueError("No way wandb")
        grids = dict()
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grids[f"{split}/{k}"] = wandb.Image(grid)
        pl_module.logger.experiment.log(grids)

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid+1.0)/2.0 # -1,1 -> 0,1; c,h,w

            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(
                tag, grid,
                global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images,
                  global_step, current_epoch, batch_idx):
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=self.nrow)

            grid = grid.transpose(0,1).transpose(1,2).squeeze(-1)
            grid = grid.numpy().astype(np.uint8)
            filename = "{}_gs-{:06}_e-{:06}_b-{:06}.png".format(
                k,
                global_step,
                current_epoch,
                batch_idx)
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    def log_img(self, pl_module, batch, batch_idx, split="train"):
        pass

    @rank_zero_only
    def log_histogram(self, pl_module, histogram, batch_idx, split, factor_name):
        pl_module.logger.experiment.add_histogram(f'{split}/{factor_name}', histogram.detach().cpu(), pl_module.global_step)

    def check_frequency(self, iter_idx):
        if (iter_idx % self.batch_freq) == 0 or iter_idx in self.log_steps:
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if batch_idx % trainer.accumulate_grad_batches == 0:
            self.log_img(pl_module, batch, trainer.global_step, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="val")

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        self.log_img(pl_module, batch, batch_idx, split="test")


class ImageLogger(ImageLoggerBase):
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        if not hasattr(pl_module, 'log_images') or not callable(pl_module.log_images) or self.max_images <= 0:
            return
        if self.check_log(split, pl_module.global_step, batch_idx):
            logger = type(pl_module.logger)

            is_train = pl_module.training
            if is_train:
                pl_module.eval()

            images = pl_module.log_images(batch)

            for k in images:
                if len(images[k]) < self.nrow:
                    N = len(images[k])
                else:
                    N = self.nrow * (len(images[k]) // self.nrow)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    # if self.clamp:
                    #     images[k] = torch.clamp(images[k], -1., 1.)
                elif isinstance(images[k], list):
                    images[k] = torch.stack(images[k], dim=0)

            self.log_local(pl_module.logger.save_dir, split, images,
                           pl_module.global_step, pl_module.current_epoch, batch_idx)

            logger_log_images = self.logger_log_images.get(logger, lambda *args, **kwargs: None)
            for k in images:
                images[k] /= 255.
                images[k] = images[k] * 2. - 1.
            logger_log_images(pl_module, images, split, batch_idx if split.startswith('val') else None)

            if is_train:
                pl_module.train()

    def check_log(self, split, global_step, batch_idx):
        return (split == 'train' and self.check_frequency(global_step)) or (split == 'val' and batch_idx == 0)


if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")

    # add cwd for convenience and to make classes in this file available when
    # running as `python main.py`
    # (in particular `main.DataModuleFromConfig`)
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)

    opt, unknown = parser.parse_known_args()
    if opt.name and opt.resume:
        raise ValueError(
            "-n/--name and -r/--resume cannot be specified both."
            "If you want to resume training in a new log folder, "
            "use -n/--name in combination with --resume_from_checkpoint"
        )
    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError("Cannot find {}".format(opt.resume))
        if os.path.isfile(opt.resume):
            paths = opt.resume.split("/")
            idx = -2
            logdir = "/".join(paths[:idx])
            ckpt = opt.resume
        else:
            assert os.path.isdir(opt.resume), opt.resume
            logdir = opt.resume
            ckpt = os.path.join(logdir, "models", "last.ckpt")

        opt.resume_from_checkpoint = ckpt
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.base = base_configs + opt.base
        _tmp = logdir.split("/")
        nowname = _tmp[-1]
    else:
        if opt.name:
            name = "_" + opt.name
        elif opt.base:
            cfg_fname = os.path.split(opt.base[0])[-1]
            cfg_name = os.path.splitext(cfg_fname)[0]
            name = "_" + cfg_name
        else:
            name = ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.basedir, nowname)

    ckptdir = os.path.join(logdir, "models")
    cfgdir = os.path.join(logdir, "configs")
    tensorboard_dir = os.path.join(logdir, 'tensorboard')
    os.makedirs(ckptdir, exist_ok=True)
    os.makedirs(cfgdir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    seed_everything(opt.seed)

    try:
        # init and save configs
        configs = [OmegaConf.load(cfg) for cfg in opt.base]
        cli = OmegaConf.from_dotlist(unknown)
        config = OmegaConf.merge(*configs, cli)
        lightning_config = config.pop("lightning", OmegaConf.create())
        # merge trainer cli with config
        trainer_config = lightning_config.get("trainer", OmegaConf.create())
        # default to ddp
        trainer_config["distributed_backend"] = "ddp"
        for k in nondefault_trainer_args(opt):
            trainer_config[k] = getattr(opt, k)
        if not "gpus" in trainer_config:
            del trainer_config["distributed_backend"]
            cpu = True
        else:
            gpuinfo = trainer_config["gpus"]
            print(f"Running on GPUs {gpuinfo}")
            cpu = False
        trainer_opt = argparse.Namespace(**trainer_config)
        lightning_config.trainer = trainer_config

        # model
        model = instantiate_from_config(config.model)

        # trainer and callbacks
        trainer_kwargs = dict()

        trainer_kwargs['logger'] = \
            TensorBoardLogger(save_dir=tensorboard_dir, name='', version='', default_hp_metric=False)

        # modelcheckpoint - use TrainResult/EvalResult(checkpoint_on=metric) to
        # specify which metric is used to determine best models
        default_modelckpt_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "epoch={epoch}_step={step}_loss={train/loss_epoch:.4f}",
                "auto_insert_metric_name": False,
                "every_n_epochs": 1,
                "monitor": "train/loss_epoch",
                "save_top_k": 3,
                "mode": "min",
                "save_last": True,
                "verbose": False,
            },
        }
        modelckpt_cfg = lightning_config.get('modelcheckpoint', OmegaConf.create())
        modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, modelckpt_cfg)

        default_modelckpt_epoch_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{epoch:03d}_{step}",
                "save_top_k": -1,
                "every_n_epochs": 25,
                "every_n_train_steps": None,
                "save_last": False,
                "verbose": False,
                "save_on_train_epoch_end": True,
            },
        }
        modelckpt_epoch_cfg = lightning_config.get('modelcheckpoint_epoch', OmegaConf.create())
        modelckpt_epoch_cfg = OmegaConf.merge(default_modelckpt_epoch_cfg, modelckpt_epoch_cfg)

        default_modelckpt_step_cfg = {
            "target": "pytorch_lightning.callbacks.ModelCheckpoint",
            "params": {
                "dirpath": ckptdir,
                "filename": "{step}",
                "save_top_k": -1,
                "every_n_epochs": None,
                "every_n_train_steps": None,
                "save_last": False,
                "verbose": False,
                "save_on_train_epoch_end": True,
            },
        }
        modelckpt_step_cfg = lightning_config.get('modelcheckpoint_step', OmegaConf.create())
        modelckpt_step_cfg = OmegaConf.merge(default_modelckpt_step_cfg, modelckpt_step_cfg)

        # add callback which sets up log directory
        default_callbacks_cfg = {
            "setup_callback": {
                "target": "main.SetupCallback",
                "params": {
                    "resume": opt.resume,
                    "now": now,
                    "logdir": logdir,
                    "ckptdir": ckptdir,
                    "cfgdir": cfgdir,
                    "config": config,
                    "lightning_config": lightning_config,
                }
            },
            "image_logger": {
                "target": "main.ImageLogger",
                "params": {
                    "batch_frequency": 1024,
                    "max_images": 64,
                    "clamp": True
                }
            },
            "learning_rate_logger": {
                "target": "main.LearningRateMonitor",
                "params": {
                    "logging_interval": "step",
                    #"log_momentum": True
                }
            },
            'checkpoints': modelckpt_cfg,
            'checkopints_epoch': modelckpt_epoch_cfg,
            'checkopints_step': modelckpt_step_cfg,
        }
        callbacks_cfg = lightning_config.get('callbacks', OmegaConf.create())
        callbacks_cfg = OmegaConf.merge(default_callbacks_cfg, callbacks_cfg)
        trainer_kwargs["callbacks"] = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

        trainer = Trainer.from_argparse_args(trainer_opt, **trainer_kwargs)

        # configure learning rate
        model.embedding_lr = config.model.embedding_learning_rate
        model.target_output_lr = config.model.target_output_learning_rate

        # accumulate grad batches
        accumulate_grad_batches = lightning_config.trainer.get('accumulate_grad_batches', 1)
        print(f"accumulate_grad_batches = {accumulate_grad_batches}")
        lightning_config.trainer.accumulate_grad_batches = accumulate_grad_batches

        # data
        data = instantiate_from_config(config.data)
        data.prepare_data()
        data.setup()

        # allow checkpointing via USR1
        def melk(*args, **kwargs):
            if trainer.global_rank == 0:
                print("Summoning checkpoint.")
                ckpt_path = os.path.join(ckptdir, "last.ckpt")
                trainer.save_checkpoint(ckpt_path)
                print(f"Save last in {ckpt_path}")

        def divein(*args, **kwargs):
            if trainer.global_rank == 0:
                import pudb; pudb.set_trace()

        import signal
        signal.signal(signal.SIGUSR1, melk)
        signal.signal(signal.SIGUSR2, divein)

        if opt.test_first and 'test' in data.datasets:
            trainer.test(model, data)

        # run
        if opt.train:
            try:
                setattr(model, 'data', data)
                trainer.fit(model, data)
            except Exception:
                print('exception!!!!!')
                melk()
                raise
        if not opt.no_test and not trainer.interrupted and 'test' in data.datasets:
            trainer.test(model, data)
    except Exception:
        if opt.debug and trainer.global_rank==0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        # move newly created debug project to debug_runs
        if opt.debug and not opt.resume and trainer.global_rank==0:
            dst, name = os.path.split(logdir)
            dst = os.path.join(dst, "debug_runs", name)
            os.makedirs(os.path.split(dst)[0], exist_ok=True)
            os.rename(logdir, dst)
