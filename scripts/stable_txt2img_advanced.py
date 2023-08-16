import argparse, os, sys, glob
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid, save_image
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config
from scripts.helpers import chunk, load_model_from_config, sample


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a photo of a {}",
        help="the prompt to render"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--skip_save",
        action='store_true',
        help="do not save individual samples. For speed measurements.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of sampling steps",
    )
    parser.add_argument(
        "--laion400m",
        action='store_true',
        help="uses the LAION400M model",
    )
    parser.add_argument(
        "--fixed_code",
        action='store_true',
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=2,
        help="sample this often",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=3,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.7,
        help="bias used in gated rank-1 update",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=0.15,
        help="temperature used in gated rank-1 update",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/perfusion_inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--sampler_config",
        type=str,
        default="configs/sampler/sampler.yaml",
        help="path to config which constructs sampler",
    )
    parser.add_argument(
        "--denoiser_config",
        type=str,
        default="configs/denoiser/denoiser.yaml",
        help="path to config which constructs sampler",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="./ckpt/v1-5-pruned-emaonly.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--personalized_ckpt",
        type=str,
        required=True,
        help="Path to a pre-trained personalized checkpoint"
    )
    parser.add_argument(
        "--global_locking",
        action="store_true",
        help="the superclass word for global locking. None for disable."
    )

    opt = parser.parse_args()

    if opt.laion400m:
        print("Falling back to LAION 400M model...")
        opt.config = "configs/latent-diffusion/txt2img-1p4B-eval.yaml"
        opt.ckpt = "models/ldm/text2img-large/model.ckpt"
        opt.outdir = "outputs/txt2img-samples-laion400m"

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    config.model.params.beta = opt.beta
    config.model.params.tau = opt.tau
    model = load_model_from_config(config, opt.ckpt, opt.personalized_ckpt)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler_config = OmegaConf.load(f"{opt.sampler_config}")
    sampler_config.params.num_steps = opt.steps
    sampler_config.params.guider_config.scale = opt.scale
    sampler = instantiate_from_config(sampler_config)

    denoiser_config = OmegaConf.load(f"{opt.denoiser_config}")
    denoiser = instantiate_from_config(denoiser_config).to(device)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples
    n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    # prompts with superclass word for global locking
    c_superclass_list = None
    if opt.global_locking:
        superclass = model.embedding_manager.initializer_words[0]
        data_superclass = [[p.format(superclass) for p in data[i]] for i in range(len(data))]
        c_superclass_list = [model.get_learned_conditioning(batch_superclass) for batch_superclass in data_superclass]

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if opt.fixed_code:
        start_code = torch.randn([opt.n_samples, opt.C, opt.H // opt.f, opt.W // opt.f], device=device)

    placeholder = list(model.embedding_manager.string_to_token_dict.keys())[0]
    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                for n in trange(opt.n_iter, desc="Sampling"):
                    for data_i, prompts in tqdm(enumerate(data), desc="data"):
                        uc = None
                        c_superclass = c_superclass_list[data_i] if c_superclass_list is not None else None
                        if opt.scale != 1.0:
                            encoding_uc = model.get_learned_conditioning(batch_size * [""])
                            uc = dict(c_crossattn=encoding_uc,
                                      c_super=encoding_uc if c_superclass is not None else None)
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        prompts = [p.format(placeholder) for p in prompts]
                        encoding = model.cond_stage_model.encode(prompts, embedding_manager=model.embedding_manager)
                        c = dict(c_crossattn=encoding, c_super=c_superclass)
                        shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

                        z_samples = sample(model.apply_model, denoiser, sampler, c, uc, batch_size, shape, model.device)
                        x_samples = model.decode_first_stage(z_samples)
                        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                        if not opt.skip_save:
                            for x_sample in x_samples:
                                x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                                Image.fromarray(x_sample.astype(np.uint8)).save(
                                    os.path.join(sample_path, f"{base_count:05}.jpg"))
                                base_count += 1

                        if not opt.skip_grid:
                            all_samples.append(x_samples)

                if not opt.skip_grid:
                    # additionally, save as grid
                    grid = torch.stack(all_samples, 0)
                    grid = rearrange(grid, 'n b c h w -> (n b) c h w')

                    for i in range(grid.size(0)):
                        save_image(grid[i, :, :, :], os.path.join(outpath, opt.prompt + '_{}.png'.format(i)))
                    grid = make_grid(grid, nrow=n_rows)

                    # to image
                    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                    Image.fromarray(grid.astype(np.uint8)).save(
                        os.path.join(outpath, f'{prompt.replace(" ", "-")}-{grid_count:04}.jpg'))
                    grid_count += 1

                toc = time.time()

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    main()
