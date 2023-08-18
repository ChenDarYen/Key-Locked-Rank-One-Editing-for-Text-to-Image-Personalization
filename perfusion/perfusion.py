import copy
import os

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.util import log_txt_as_img, instantiate_from_config, default
from perfusion.roe import ROELinear, roe_to_mc_roe
from perfusion.dataset import PROMPT_TEMPLATE


def roe_state_dict(model: torch.nn.Module):
    sd = model.state_dict()
    to_return = {}
    for k in sd:
        if 'target_output' in k or 'target_input' in k:
            to_return[k] = sd[k]
    return to_return


def set_submodule(module, submodule_name, new_submodule):
    submodule_names = submodule_name.split('.')
    current_module = module
    for name in submodule_names[:-1]:
        current_module = getattr(current_module, name)
    setattr(current_module, submodule_names[-1], new_submodule)


class Perfusion(LatentDiffusion):
    def __init__(
            self,
            personalization_config,
            C_inv_path='./ckpt/C_inv.npy',
            personalization_ckpt=None,
            ema_p=0.99,
            beta=0.75,
            tau=0.1,
            concept_token_idx_key='concept_token_idx',
            mask_key='mask',
            *args, **kwargs):
        """
        Args:
            C_inv_path: path to the inverse of the uncentered covariance metric.
            ema_p: p for calculating exponential moving average on target input.
            beta: bias used in gated rank-1 editing.
            tau: temperature used in gated rank-1 editing.
        """
        ckpt_path = kwargs.pop('ckpt_path', None)
        ignore_keys = kwargs.pop('ignore_keys', list())
        load_only_unet = kwargs.pop('load_only_unet', False)
        super().__init__(*args, **kwargs)
        self.ema_p = ema_p
        self.beta = torch.tensor(beta)
        self.tau = torch.tensor(tau)
        self.concept_token_idx_key = concept_token_idx_key
        self.mask_key = mask_key

        self.register_buffer('C_inv', torch.from_numpy(np.load(C_inv_path)).float())
        self.register_buffer('target_input',
                             torch.empty(self.model.diffusion_model.context_dim, dtype=torch.float32))

        self.embedding_manager = instantiate_from_config(personalization_config, embedder=self.cond_stage_model)
        assert len(self.embedding_manager.string_to_param_dict) == 1

        # Set requires_grad for different components.
        self.cond_stage_model.requires_grad_(False)
        for param in self.embedding_manager.embedding_parameters():
            param.requires_grad = True
        for name, param in self.model.diffusion_model.named_parameters():
            if 'target_output' not in name:
                param.requires_grad = False

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        if personalization_ckpt is not None:
            self.init_from_personalization_ckpt(personalization_ckpt)

    def init_from_personalized_ckpt(self, path):
        sd = torch.load(path, map_location="cpu")
        self.C_inv.data = sd['C_inv']
        self.target_input.data = sd['target_input']
        self.embedding_manager.load_state_dict(sd['embedding'])
        self.model.diffusion_model.load_state_dict(sd['target_output'], strict=False)

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        super().on_train_batch_start(batch, batch_idx, dataloader_idx)
        if self.global_step == 0 and batch_idx == 0:
            prompts = batch[self.cond_stage_key]
            concept_indices = batch[self.concept_token_idx_key]

            prompts_splits = [p.split(' ') for p in prompts]
            for i, c_i in enumerate(concept_indices):
                prompts_splits[i][c_i] = self.embedding_manager.initializer_words[0]
            prompts_superclass = [' '.join(s) for s in prompts_splits]
            input_superclass = self.get_learned_conditioning(prompts_superclass)

            concept_encoding = (
                input_superclass[torch.arange(len(input_superclass)), [c_i + 1 for c_i in concept_indices]].mean(dim=0))
            self.target_input.data = concept_encoding

            for _, mod in self.model.diffusion_model.named_modules():
                if isinstance(mod, ROELinear):
                    mod.initialize_target_output(input_superclass)

    @torch.no_grad()
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, *args, **kwargs)

        concept_token_idx = batch[self.concept_token_idx_key]
        mask = batch[self.mask_key]
        if bs is not None:
            concept_token_idx = concept_token_idx[:bs]
            mask = mask[:bs]

        return x, dict(c_crossattn=c, concept_token_idx=concept_token_idx), mask

    def shared_step(self, batch, **kwargs):
        x, c, mask = self.get_input(batch, self.first_stage_key)
        loss = self(x, c, mask=mask)
        return loss

    def forward(self, x, c, mask=None, *args, **kwargs):
        encoding = self.cond_stage_model.encode(c['c_crossattn'], embedding_manager=self.embedding_manager)
        c['c_crossattn'] = encoding

        if self.training:
            concept_encoding = encoding[torch.arange(len(encoding)), c['concept_token_idx'] + 1].mean(dim=0)
            self.target_input.data = self.target_input.data * self.ema_p + concept_encoding * (1 - self.ema_p)
        del c['concept_token_idx']

        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, c, t, mask=mask, *args, **kwargs)

    def p_losses(self, x_start, cond, t, noise=None, mask=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        if mask is not None:
            mask = F.interpolate(mask, model_output.size()[-2:], mode='bilinear')
            model_output = model_output * mask
            target = target * mask

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        eps = diffusion_model(
            x=x_noisy, timesteps=t, context=cond['c_crossattn'], target_input=self.target_input, C_inv=self.C_inv,
            beta=self.beta, tau=self.tau,
            context_super=cond['c_super'] if 'c_super' in cond else None,  # for global locking
        )
        return eps

    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)

    @torch.autocast('cuda')
    @torch.no_grad()
    def log_images(self, batch, N=4, sample=False, ddim_steps=50, ddim_eta=0.0,
                   unconditional_guidance_scale=6.0, **kwargs):
        use_ddim = ddim_steps is not None

        log = dict()
        z, c, _ = self.get_input(batch, self.first_stage_key, bs=N)
        encoding = self.cond_stage_model.encode(c['c_crossattn'], embedding_manager=self.embedding_manager)
        c['c_crossattn'] = encoding
        c['target_input'] = self.target_input
        del c['concept_token_idx']

        N = min(z.shape[0], N)
        log["reconstruction"] = self.tensor_to_rgb(self.decode_first_stage(z))
        log["conditioning"] = self.tensor_to_rgb(log_txt_as_img((512, 512), batch[self.cond_stage_key], size=16))
        log['mask'] = batch[self.mask_key] * 255

        if sample:
            samples, z_denoise_row = self.sample_log(
                cond=c,
                C_inv=self.C_inv, beta=self.beta, tau=self.tau,
                batch_size=N, ddim=use_ddim,
                ddim_steps=ddim_steps, eta=ddim_eta)
            x_samples = self.decode_first_stage(samples)
            x_samples = self.tensor_to_rgb(x_samples)
            log["samples"] = x_samples

        if unconditional_guidance_scale > 1.0:
            uc = copy.deepcopy(c)
            uc['c_crossattn'] = self.get_unconditional_conditioning(N)
            samples_cfg, _ = self.sample_log(
                cond=c,
                C_inv=self.C_inv, beta=self.beta, tau=self.tau,
                batch_size=N, ddim=use_ddim,
                ddim_steps=ddim_steps, eta=ddim_eta,
                unconditional_guidance_scale=unconditional_guidance_scale,
                unconditional_conditioning=uc,
                )
            x_samples_cfg = self.decode_first_stage(samples_cfg)
            x_samples_cfg = self.tensor_to_rgb(x_samples_cfg)
            log[f"samples_cfg_scale_{unconditional_guidance_scale:.2f}"] = x_samples_cfg

        return log

    def configure_optimizers(self):
        embedding_params = list(self.embedding_manager.embedding_parameters())
        target_output_params = list()
        for name, param in self.model.diffusion_model.named_parameters():
            if 'target_output' in name and param.requires_grad:
                target_output_params.append(param)

        opt = torch.optim.AdamW(
            [{"params": embedding_params, "lr": self.embedding_lr}, {"params": target_output_params}],
            lr=self.target_output_lr,
        )
        return opt

    def on_save_checkpoint(self, checkpoint):
        global_step = checkpoint.pop('global_step')
        checkpoint.clear()
        checkpoint.update({
            'global_step': global_step,
            'C_inv': self.C_inv.data,
            'target_input': self.target_input.data,
            'embedding': self.embedding_manager.state_dict(),
            'target_output': roe_state_dict(self.model.diffusion_model),
        })

    @staticmethod
    def tensor_to_rgb(x):
        return torch.clip((x + 1.) * 127.5, 0., 255.)


# inference only
class MultiConceptsPerfusion(LatentDiffusion):
    def __init__(
            self,
            personalization_config,
            C_inv_path='./ckpt/C_inv.npy',
            personalization_ckpt_list=None,
            beta=0.6,
            tau=0.15,
            n_concepts=1,
            *args, **kwargs):
        """
        Args:
            C_inv_path: path to the inverse of the uncentered covariance metric.
            ema_p: p for calculating exponential moving average on target input.
            beta: bias used in gated rank-1 editing.
            tau: temperature used in gated rank-1 editing.
        """
        ckpt_path = kwargs.pop('ckpt_path', None)
        ignore_keys = kwargs.pop('ignore_keys', list())
        load_only_unet = kwargs.pop('load_only_unet', False)
        super().__init__(*args, **kwargs)
        self.n_concepts = n_concepts

        if isinstance(beta, float):
            beta = [beta] * self.n_concepts
        if isinstance(tau, float):
            tau = [tau] * self.n_concepts
        self.beta = beta
        self.tau = tau

        for name, mod in self.model.diffusion_model.named_modules():
            if isinstance(mod, ROELinear):
                set_submodule(self.model.diffusion_model, name, roe_to_mc_roe(mod, self.n_concepts))

        self.register_buffer('C_inv', torch.from_numpy(np.load(C_inv_path)).float())
        self.register_buffer('target_inputs',
                             torch.empty(self.n_concepts, self.model.diffusion_model.context_dim, dtype=torch.float32))
        self.register_buffer('target_inputs_basis',
                             torch.empty_like(self.target_inputs))

        self.embedding_manager = instantiate_from_config(personalization_config, embedder=self.cond_stage_model)
        assert len(self.embedding_manager.string_to_param_dict) == 1

        # Set requires_grad for different components.
        self.cond_stage_model.requires_grad_(False)
        for param in self.embedding_manager.embedding_parameters():
            param.requires_grad = True
        for name, param in self.model.diffusion_model.named_parameters():
            if 'target_output' not in name:
                param.requires_grad = False

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)
        if personalization_ckpt_list is not None:
            self.init_from_personalization_ckpt_list(personalization_ckpt_list)

    def prepare(self):
        U = torch.linalg.cholesky(self.C_inv, upper=True)
        U_inv = torch.linalg.inv(U)
        tilde_target_inputs = F.linear(self.target_inputs, U)
        Q = torch.linalg.qr(tilde_target_inputs.transpose(0, 1))[0].transpose(0, 1)  # orthogonal basis
        Q = F.normalize(Q, dim=1)  # orthonormal basis
        self.target_inputs_basis.data = F.linear(Q, U_inv)

    def init_from_personalized_ckpt(self, path_list):
        assert len(path_list) == self.n_concepts
        for i, path in enumerate(path_list):
            sd = torch.load(path, map_location="cpu")
            sd_C_inv = sd['C_inv']
            sd_embedding = sd['embedding']
            sd_target_input = sd['target_input']
            sd_target_output = sd['target_output']

            # C_inv and embeddings
            if i == 0:
                self.C_inv.data = sd_C_inv
                self.embedding_manager.load_state_dict(sd_embedding)
            else:
                assert torch.equal(self.C_inv, sd_C_inv), 'all personalization concepts must share the same C_inv.'
                placeholder_str = list(sd_embedding['string_to_token'].keys())[0]
                if placeholder_str in self.embedding_manager.string_to_token_dict:
                    new_placeholder_str = placeholder_str * (i + 1)
                    new_token = self.embedding_manager.get_token_for_string(new_placeholder_str)
                    sd_embedding = {
                        'initializer_words': sd_embedding['initializer_words'],
                        'string_to_token': {new_placeholder_str: new_token},
                        'string_to_param': {new_placeholder_str: sd_embedding['string_to_param'][placeholder_str]}
                    }
                self.embedding_manager.update_state_dict(sd_embedding)

            # target inputs and outputs
            self.target_inputs.data[i] = sd_target_input
            for k in list(sd_target_output.keys()):
                if 'target_output' in k:
                    v = sd_target_output.pop(k)
                    new_k = k.replace('target_output', f'target_outputs.{i}')
                    sd_target_output[new_k] = v
            self.model.diffusion_model.load_state_dict(sd_target_output, strict=False)

        self.prepare()

    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model
        eps = diffusion_model(
            x=x_noisy, timesteps=t, context=cond['c_crossattn'],
            target_inputs=self.target_inputs, target_inputs_basis=self.target_inputs_basis,
            C_inv=self.C_inv, beta=self.beta, tau=self.tau,
            context_super=cond['c_super'] if 'c_super' in cond else None,  # for global locking
        )
        return eps
