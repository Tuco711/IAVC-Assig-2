# Image Encoding Assignment

This project trains visual encoders from local image descriptors and extracts
one global embedding per image. The supported encoders are:

- **BoVW**: a histogram of visual-word assignments.
- **VLAD**: aggregated residuals to visual-word centers.
- **Fisher vector**: first- and second-order statistics from a GMM.

Run the commands below from the project root, the directory containing
`README.md`.

## Install Anaconda Environment

Create the project environment from the included `env.yml` file:

On Linux, run:
```bash
conda env create -f env.yml
```

On Windows, download the `wheel` folder from the OneDrive materials and copy it
to the project root. It should contain:

```text
wheel/cyvlfeat-0.7.1-cp311-cp311-win_amd64.whl
```

Then run:
```bash
conda env create -f env-win.yml
```

Then
```
conda activate cvc-a2
```

Install the Python packages listed in `requirements.txt`:

```bash
python -m pip install -r requirements.txt
```

The requirements file is configured for CUDA 12.6 PyTorch by default. If CUDA
installation or execution does not work on your machine, edit
`requirements.txt`: comment out the CUDA lines and uncomment the CPU-only
`torch` and `torchvision` lines, then rerun the installation command.


## Project Structure

```text
.
├── data/
│   ├── images/                 # Dataset images.
│   ├── splits/                 # Train/validation split text files.
│   ├── encoders/               # Trained encoder configurations (*.npz).
│   └── encodings/              # Per-image embeddings grouped by encoder.
├── src/
│   ├── train.py                # Train BoVW, VLAD, or Fisher encoders.
│   ├── extract.py              # Extract embeddings using a trained encoder.
│   ├── encoders.py             # Encoder implementations and image aggregation.
│   ├── deep_models.py          # Pretrained models and intermediate features.
│   ├── features.py             # Image preparation and dense SIFT/PHOW features.
│   ├── utils.py                # Dataset tables, splits, and SVM helpers.
│   └── visualization.py        # Ranking, precision-recall, and embedding plots.
├── 1_handcraft.ipynb           # Hand-crafted feature experiments.
├── 2_deep_models.ipynb         # Deep-model features and CAM visualization.
```

Encoder files are named `<ENCODER>_K<K>_C<C>.npz`, where `K<K>` is the number of
visual words or GMM components and `C<C>` is the PCA dimension. `C0` means PCA
is disabled. Extracted files use the same encoder name as a directory, for
example:

```text
data/encodings/bovw_K512_C0/
├── aeroplane_train.npz
├── aeroplane_val.npz
├── background_train.npz
└── ...
```

Each encoding archive contains `file_names` and `histograms`. The latter is a
matrix with one row per image.

## Train Encoders

Train a BoVW encoder with 512 visual words and no PCA:

```bash
python -m src.train -e bovw -k 512 -c 0
```

Train VLAD and Fisher encoders with 32 components and an 80-dimensional PCA
projection:

```bash
python -m src.train -e vlad -k 32 -c 80
python -m src.train -e fv -k 32 -c 80
```

The resulting configurations are written to `data/encoders/` using names such
as `vlad_K32_C80.npz`. Useful options include:

```text
--data-dir       Dataset split directory (default: data/splits)
--img-dir        Image directory (default: data/images)
--output-dir     Encoder output directory (default: data/encoders)
-i, --per-class  Images sampled per class; -1 uses all (default: 150)
-n, --per-image  Local descriptors sampled per image (default: 400)
-j, --jobs       Reserved parallel-job option (default: 4)
--cache-dir      Optional feature-cache directory (default: disabled)
```

## Extract Embeddings

Extract embeddings for selected categories with a trained encoder:

```bash
python -m src.extract \
  -e data/encoders/bovw_K512_C0.npz \
  --catg background motorbike aeroplane person

python -m src.extract \
  -e data/encoders/vlad_K32_C80.npz \
  --catg background motorbike aeroplane person

python -m src.extract \
  -e data/encoders/fv_K32_C80.npz \
  --catg background motorbike aeroplane person
```

Extraction writes files under a directory named after the encoder, for
example `data/encodings/vlad_K32_C80/`. By default, existing files are kept;
add `-f` to overwrite them. Use `-j` to control the number of worker threads
and `--cache-dir cache/` to reuse cached local descriptors.

```bash
python -m src.extract \
  -e data/encoders/vlad_K32_C80.npz \
  --catg background motorbike aeroplane person \
  --cache-dir cache/ \
  -j 4 \
  -f
```

## Deep Models and References

The deep-model notebook uses pretrained backbones and utilities from these
projects:

- [TorchVision models documentation](https://pytorch.org/vision/stable/models.html)
  and [TorchVision GitHub repository](https://github.com/pytorch/vision) for
  ResNet-18 and ViT-B/32 weights and preprocessing.
- [DINOv2 GitHub repository](https://github.com/facebookresearch/dinov2) for
  the self-supervised `dinov2_vits14` model loaded through Torch Hub.
- [OpenAI CLIP GitHub repository](https://github.com/openai/CLIP) for the
  `ViT-B/32` image-text model and preprocessing pipeline.

These models may download pretrained weights the first time they are loaded.
The notebook automatically selects CUDA when available and otherwise uses CPU.

## Reproducing the report

Run the notebooks from `code/` (they locate `../data` automatically), in this order:

| Notebook | Covers | Needs `cyvlfeat`? |
|---|---|---|
| `1_handcraft.ipynb` | Part 1 (BoVW + SVM, normalisation, Hellinger, learning curves), Part 2 (VLAD / FV), Part 3 (PCA) | No, except the *relevant visual words* cell and re-encoding with other `K` |
| `2_deep_models.ipynb` | Part 4 (ResNet-18, ViT-B/32, DINOv2, CLIP; few-shot, C sweep, zero-shot, failure cases) + final comparison table | No |
| `3_retrieval.ipynb` | Part 5 (retrieval challenge, motorbike) | No (needs the feature cache written by notebook 2) |

All tables/plots data are also written to `code/results/*.csv`. Shared helpers: `src/eval.py` (normalisations,
dev split, SVM fit/eval with timings, P@K/R@K, OvR scores); `src/deep_models.py::extractEmbeddings` (frozen features).

**Protocol.** The provided `val` split is used as the held-out **test** set (reported once). All choices (C, normalisation,
spatial vs global, prompt set) use a stratified `dev` split from `train` (or the train split for zero-shot prompts). One-vs-rest
tasks = category vs `background`; 5-way confusion/F1 use val images with exactly one category label. Seeds: 0-4.

**Deviations / not run.** `K=256` and `K=1024` vocabularies (Part 3, Part 5 vocabulary question) require
`python -m src.train -e bovw -k 256 -c 0` (and `-k 1024`) followed by `python -m src.extract -e data/encodings/...` on a machine with
`cyvlfeat`; notebooks 1 and 3 pick up any `data/encodings/bovw_K*_C0` automatically. Extraction time of the handcrafted features is
not re-measured (provided pre-computed descriptors).

**Environment.** `requirements.txt` (versions pinned); `cyvlfeat` is optional (imports are guarded). To execute notebooks headless:
`jupyter nbconvert --to notebook --execute --inplace <notebook>` (needs `nbconvert`). Hardware/device and timings are printed in the notebooks.
