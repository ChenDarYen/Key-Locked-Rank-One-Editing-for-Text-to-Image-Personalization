from itertools import islice
import torch
from ldm.util import instantiate_from_config


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, personalized_ckpt, verbose=False):
    model = instantiate_from_config(config.model)

    print(f"Loading model from {ckpt} and {personalized_ckpt}")

    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.init_from_personalized_ckpt(personalized_ckpt)

    model.cuda()
    model.eval()
    return model


@torch.no_grad()
def sample(
        model,
        denoiser,
        sampler,
        cond,
        uc=None,
        batch_size=16,
        shape=None,
        device='cuda',
        **kwargs,
):
    randn = torch.randn(batch_size, *shape).to(device)
    samples = sampler(
        lambda input, sigma, c: denoiser(model, input, sigma, c, **kwargs),
        randn, cond, uc=uc,
    )
    return samples