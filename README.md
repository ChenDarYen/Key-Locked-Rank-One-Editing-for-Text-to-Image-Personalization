# Perfusion <br> <sub>Key-Locked Rank One Editing for Text-to-Image Personalization</sub>

A Pytorch implementation of the paper [Key-Locked Rank One Editing for Text-to-Image Personalization](https://arxiv.org/abs/2305.01644) ([project page](https://research.nvidia.com/labs/par/Perfusion/)).


<p align="center">
<img src=assets/paper_samples.png />
<img src=assets/paper_diagram.png />
</p>

### ⏳ To do
- [x] Multiple concepts inference
- [ ] Support SDXL-1.0
- [ ] CLIP metrics
- [ ] Evaluation


### News
- Support advanced sampler from [Stable Diffusion](https://github.com/Stability-AI/generative-models), like EulerEDMSampler.
- Now we support training and generating using SD V2.1 as the basement!

## Samples
### Paper
<p align="center">
<img src=assets/paper_samples_teddy.png />
<img src=assets/paper_samples_cat.png />
</p>

### Our
Using CLIP similarity to automatically select a balanced weight is necessary. 
We'll implement it in the near future.
<p align="center">
<img src=assets/our_samples_teddy.png />
<img src=assets/our_samples_cat.png />
<img src=assets/our_samples_Hepburn.png />
<img src=assets/our_samples_Hepburn_cat.png />
</p>

## Environment
Create and activate the conda environment:

```
conda env create -f environment.yaml
conda activate perfusion
```

## Training
Download the [SD V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt) or [SD V2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt) to `./ckpt/`.

Then run the commands:

```
python main.py \
    --name teddy \
    --base ./configs/perfusion_teddy.yaml \
    --basedir ./ckpt \
    -t True \
    --gpus 0,
```

or:

```
python main.py \
    --name teddy \
    --base ./configs/perfusion_teddy_sd_v2.yaml \
    --basedir ./ckpt \
    -t True \
    --gpus 0,
```

To prepare your own training data, please ensure that they are placed in a folder `/path/to/your/images/`.
You need to download pretrained weight of [clipseg](https://github.com/timojl/clipseg):
```
wget https://owncloud.gwdg.de/index.php/s/ioHbRzFx6th32hn/download -O ./clipseg/weights.zip
unzip -d ./clipseg/weights -j ./clipseg/weights.zip
```
Then run:
```
python ./data/soft_segment.py --image_dir /path/to/your/images/ --super_class your_own_super_class
```
Modify the `initializer_words`, `data_root`, `flip_p` in `./configs/perfusion_custom.yaml` or `./configs/perfusion_custom_sd_v2.yaml`.

Finally, run:
```
python main.py \
    --name experiment_name \
    --base ./configs/perfusion_custom.yaml \
    --basedir ./ckpt \
    -t True \
    --gpus 0,
```
You can find weights along with tensorboard in `./ckpt`.


## Pretrained Weight
Pretrained concepts' weights can be found in `./ckpt`.

## Generating
Personalized samples can be obtained by running the command
```
python scripts/perfusion_txt2img.py --ddim_eta 0.0 \
                                    --steps 50  \
                                    --scale 6.0 \
                                    --beta 0.7 \
                                    --tau 0.15 \
                                    --n_samples 4 \
                                    --n_iter 1 \
                                    --personalized_ckpt ./ckpt/teddy.ckpt \
                                    --prompt "photo of a {}"
```

### Global Locking
Global locking will be applied with the label `--global_locking`.

### Stable Diffusion V2
Set `--config configs/perfusion_inference_sd_v2.yaml` and `--ckpt ./ckpt/v2-1_512-ema-pruned.ckpt` when using SD v2.1.

### Multiple Concepts Generating
When generating with multiple concepts, use commas to separate checkpoints like  `--personalized_ckpt /path/to/personalized/ckpt1,/path/to/personalized/ckpt2`, 
and use `{1}`,`{2}`, ..., `{n}` to distinguish different concepts in the prompt as `--prompt "photo of a {1} and {2}"`.

If you want to apply different biases and temperatures to each concept, set `--beta b1,b2` and `--tau t1,t2`.

### Advanced Sampler
If you want to use advanced samplers other than DDIM, use the label `--advanced_sampler`.
The default advanced sampler is the EulerEDMSampler. 
You can modify `./configs/sampler/sampler.yaml` and `./configs/denoiser/denoiser.yaml` depend on you preference.

Following are results by utilizing SD v2.1 and EulerEDMSampler.
<p align="center">
<img src=assets/our_samples_Hepburn_sd_v2-1_edm.png />
</p>

## BibTeX
If you find this repository useful, please cite origin papers using the following.

```bibtex
@article{Tewel2023KeyLockedRO,
    title   = {Key-Locked Rank One Editing for Text-to-Image Personalization},
    author  = {Yoad Tewel and Rinon Gal and Gal Chechik and Yuval Atzmon},
    journal = {ACM SIGGRAPH 2023 Conference Proceedings},
    year    = {2023},
    url     = {https://api.semanticscholar.org/CorpusID:258436985}
}
```

```bibtex
@inproceedings{Meng2022LocatingAE,
    title   = {Locating and Editing Factual Associations in GPT},
    author  = {Kevin Meng and David Bau and Alex Andonian and Yonatan Belinkov},
    booktitle = {Neural Information Processing Systems},
    year    = {2022},
    url     = {https://api.semanticscholar.org/CorpusID:255825985}
}
```

