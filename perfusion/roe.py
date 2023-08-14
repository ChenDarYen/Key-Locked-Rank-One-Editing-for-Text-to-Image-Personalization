import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ROELinear(nn.Linear):
    def __init__(self, length=77, lock=False, *args, **kwargs):
        """
        Args:
            length: length of text encoding.
            lock: apply locking or not.
        """
        super().__init__(*args, **kwargs)
        assert self.bias is None
        self.length = length
        self.lock = lock

        self.weight.requires_grad = False

        self.target_output = nn.Parameter(
            torch.empty((self.length, self.weight.data.shape[0]), dtype=torch.float32), requires_grad=not lock)

    @torch.no_grad()
    def initialize_target_output(self, init_input):
        """
        Args:
            init_input: the encoding of prompt using the super-class word. shape: B x N x D
        """
        self.target_output.data = F.linear(init_input, self.weight).mean(dim=0)

    def forward(self, input, target_input, C_inv, betta=0.75, tau=0.1, **kwargs):
        """
        Args:
            input: the encoding of prompt using the concept word. shape: B x N x D
            target_input: the target input.
            C_inv: the inverse of the uncentered covariance metric.
            betta: bias used in gated rank-1 update.
            tau: temperature used in gated rank-1 update.
        """
        tmp = (C_inv @ target_input)
        target_input_energy = (tmp[None, :] @ target_input).squeeze()
        sim = (input @ tmp)[..., None]
        sigmoid_term = torch.sigmoid((sim / target_input_energy - betta) / tau)
        em_orthogonal_term = input - sim * target_input / target_input_energy
        W_em_orthogonal_term = F.linear(em_orthogonal_term, self.weight)
        h = W_em_orthogonal_term + sigmoid_term * self.target_output[None, :]

        return h.to(W_em_orthogonal_term.dtype)


# Inference only
class MultiConceptsROELinear(nn.Linear):
    def __init__(self, length=77, n_concepts=1, *args, **kwargs):
        """
        Args:
            length: length of text encoding.
            n_concepts: number of concepts.
        """
        super().__init__(*args, **kwargs)
        self.length = length
        self.target_outputs = nn.ParameterList([
            nn.Parameter(torch.empty((self.length, self.weight.data.shape[0]), dtype=torch.float32))
            for _ in range(n_concepts)
        ])
        self.requires_grad_(False)

    @torch.no_grad()
    def forward(self, input, target_inputs, target_inputs_basis, C_inv, betta=0.75, tau=0.1, **kwargs):
        """
        Args:
            input: the encoding of prompt using the concept word. shape: B x N x D
            target_inputs: the target inputs.
            target_inputs_basis: a basis of the spce spanned by target_inputs.
            C_inv: the inverse of the uncentered covariance metric.
            betta: bias used in gated rank-1 update.
            tau: temperature used in gated rank-1 update.
        """
        assert len(target_inputs) == len(self.target_outputs)

        parallel_term = 0
        for i, target_input in enumerate(target_inputs):
            tmp = (C_inv @ target_input)
            target_input_energy = (tmp[None, :] @ target_input).squeeze()
            sim = (input @ tmp)[..., None]
            sigmoid_term = torch.sigmoid((sim / target_input_energy - betta) / tau)
            parallel_term += sigmoid_term * self.target_outputs[i][None, ...]

        em_proj_term = 0
        for u in target_inputs_basis:
            em_proj_term += u * (input @ C_inv @ u)[..., None]
        em_orthogonal_term = input - em_proj_term
        W_em_orthogonal_term = F.linear(em_orthogonal_term, self.weight)

        h = W_em_orthogonal_term + parallel_term

        if self.bias is not None:
            h += self.bias
        return h


# Only use when initialization, no weight copy
def roe_to_mc_roe(roe: ROELinear, n_concepts):
    mc_roe = MultiConceptsROELinear(
        n_concepts=n_concepts, bias=False,
        in_features=roe.in_features, out_features=roe.out_features, length=roe.length,
    )
    return mc_roe
