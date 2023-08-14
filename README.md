# Perfusion <br> <sub>Key-Locked Rank One Editing for Text-to-Image Personalization</sub>

A Pytorch implementation of the paper [Key-Locked Rank One Editing for Text-to-Image Personalization](https://arxiv.org/abs/2305.01644) ([project page](https://research.nvidia.com/labs/par/Perfusion/)).

<p align="center">
<img src=assets/paper_samples.png />
<img src=assets/paper_diagram.png />
</p>

### ⏳ To do
- [ ] Multiple concepts inference
- [ ] CLIP metrics
- [ ] Evaluation

## Samples
### Paper
<p align="center">
<img src=assets/paper_samples_teddy.png />
</p>

### Our
Using CLIP similarity to automatically select a balanced weight is necessary. 
We'll implement it in the near future.
<p align="center">
<img src=assets/our_samples_teddy.png />
</p>

## Environment
Create and activate the conda environment:

```
conda env create -f environment.yaml
conda activate perfusion
```

## Training
Download the [SD V1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5/resolve/main/v1-5-pruned-emaonly.ckpt) to `./ckpt/`.

Then run the commands:

```
python main.py \
    --name teddy \
    --base ./configs/perfusion_teddy.yaml \
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
Modify the `initializer_words`, `data_root` and `flip_p` in `./configs/perfusion_custom.yaml`.

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
Pretrained weight of teddy* is in `./ckpt`.

## Generation
Personalized samples can be obtained by running the command
```
python scripts/stable_txt2img.py --ddim_eta 0.0 \
                                 --ddim_steps 50  \
                                 --scale 6.0 \
                                 --beta 0.7 \
                                 --tau 0.15 \
                                 --n_samples 4 \
                                 --n_iter 1 \
                                 --personalized_ckpt ./ckpt/teddy.ckpt \
                                 --prompt "photo of a {}"
```

## Generation with multiple concepts
Use {1}, {2}, ..., {n} to distinguish different concepts.
```
python scripts/stable_txt2img_multi_concepts.py --ddim_eta 0.0 \
                                                --ddim_steps 50  \
                                                --scale 6.0 \
                                                --beta 0.7 \
                                                --tau 0.15 \
                                                --n_samples 4 \
                                                --n_iter 1 \
                                                --personalized_ckpts ./ckpt/teddy.ckpt,./ckpt/cat.ckpt \
                                                --prompt "photo of a {1} and a {2}"
```

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

