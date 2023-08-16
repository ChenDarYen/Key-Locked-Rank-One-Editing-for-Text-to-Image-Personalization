import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
import math

import numpy as np
import torch
from tqdm import tqdm

from ldm.modules.encoders.modules import FrozenCLIPEmbedder, FrozenOpenCLIPEmbedder


def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()  # Read all lines from the file
        lines = [line.strip() for line in lines]  # Remove leading/trailing whitespace

    return lines


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--caption', type=str, required=True)
    parser.add_argument('--save_path', type=str, default='./ckpt/C_inv.npy')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_captions', type=int, default=100_000)
    parser.add_argument('--type', type=str, default='clip', choices=['clip', 'openclip'])
    args = parser.parse_args()

    device = 'cuda'
    caption_path = args.caption
    bs = args.batch_size
    save_path = args.save_path
    n_captions = args.n_captions
    encoder_type = args.type

    captions = read_txt_file(caption_path)[:n_captions]
    print(f'Total {len(captions)} captions.')

    if encoder_type == 'clip':
        clip_encoder = FrozenCLIPEmbedder()
        enc_dim = 768
    else:
        clip_encoder = FrozenOpenCLIPEmbedder()
        enc_dim = 1024
    clip_encoder = clip_encoder.to(device)

    num_batches = math.ceil(len(captions) / bs)
    cov_accumulator = np.zeros((enc_dim, enc_dim))
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            emb = clip_encoder(captions[i * bs:(i + 1) * bs])
            cov_batch = torch.einsum('ijk, ijl -> kl', emb, emb).cpu().numpy()
            cov_accumulator += cov_batch

    # Normalize by the number of data points
    uncentered_cov_matrix = (1 / len(captions)) * cov_accumulator
    uncentered_cov_matrix_inv = np.linalg.inv(uncentered_cov_matrix)

    np.save(save_path, uncentered_cov_matrix_inv)
