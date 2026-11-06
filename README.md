# inference-suboptimality
Code regarding evaluation for paper
*Inference Suboptimality in Variational Autoencoders.*
[[arxiv](https://arxiv.org/abs/1801.03558)]

## Dependencies
* `python3` (3.10–3.12 recommended)
* `pytorch>=2.0` and `torchvision`
* `numpy`, `scipy`, `tqdm`, `pyyaml`
* `visdom` (optional, only for `--visdom`)

The code was originally written for `pytorch==0.2.0` and has been updated to run on current PyTorch,
on both CPU and GPU.

### Local setup
```bash
python3 -m venv .venv && source .venv/bin/activate
# CPU-only PyTorch; for a CUDA build see https://pytorch.org
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

### Running on Kaggle
Open [`kaggle/inference_suboptimality.ipynb`](kaggle/inference_suboptimality.ipynb) on Kaggle
(**Create → New Notebook → File → Import Notebook**), set **Accelerator** to `GPU T4` and
turn **Internet** on, then run the cells. The notebook clones this repository, trains a model,
runs every evaluation, and packages checkpoints and logs as the notebook's output.
For long runs use **Save Version → Save & Run All**.

To run your own changes: commit, `git push`, then rerun the notebook's first cell (it pulls).
To train without keeping the browser open, use **Save Version → Save & Run All (Commit)**:
it runs in the background (up to 12 h) from the pushed branch, and saves `/kaggle/working`
as the version's output.

## Configuration
[`config.yaml`](config.yaml) holds the experiment settings (dataset, model, training and
evaluation sample counts). Every script reads it with `--config`, and flags given on the
command line override the file:

```bash
python run.py --train --config config.yaml
python run.py --train --config config.yaml --epochs 2 --display-epoch 1   # quick check
python run.py --eval-ais --config config.yaml --eval-path <ckpt path>
```

Each script ignores keys it does not use, so one file serves all of them. The action flags
(`--train`, `--eval-iwae`, `--eval-ais`) and `--eval-path` stay on the command line.
Boolean settings set to `true` in the file cannot be switched off from the command line; edit
the file (or use a copy) instead.

## Training
MNIST is extracted from `datasets/mnist.pkl.tar.gz` automatically on first use; Fashion is read
directly from the `.gz` files in `datasets/fashion/`.

For a quick check that everything works, train for a couple of epochs first:
`python run.py --train --dataset mnist --epochs 2 --display-epoch 1`

`python run.py --train --dataset <dataset> (--lr-schedule --warmup --early-stopping)`

Besides the `<epoch>_model.pth` checkpoints, each evaluation during training keeps the model
with the best validation ELBO as `best_model.pth`; use that one for evaluation. Validation data
is held out from training for every dataset (for MNIST, 10,000 of the 60,000 images).

To resume an interrupted run, add `--load-path <ckpt> --load-epoch <epoch>`. `--epochs` is the
total, so training continues up to it, and warmup and the learning-rate schedule carry on from
`<epoch>`. Adam's internal state and the early-stopping patience count start afresh.

To train on CIFAR, set the argument for the dataset flag to `cifar`. The dataset should be
downloaded automatically, if not already downloaded.

## Evaluation
* IWAE: `python run.py --eval-iwae --dataset <dataset> --eval-path <ckpt path>`
* AIS: `python run.py --eval-ais --dataset <dataset> --eval-path <ckpt path>`
* Local FFG: `python local_ffg.py --dataset <dataset> --eval-path <ckpt path>`
* Local Flow: `python local_flow.py --dataset <dataset> --eval-path <ckpt path>`
* BDMC: `python bdmc.py --eval-path <ckpt path> --n-ais-iwae <num samples> --n-ais-dist <num dist>`

## Plotting the posteriors
With a 2-D latent space the true posterior p(z|x) can be computed exactly on a grid, so it can
be plotted against the approximate posteriors. Train with `z_size: 2` in
[`config.yaml`](config.yaml) (checkpoints go to `.../ffg_z2/`), then:

```bash
python run.py --train --config config.yaml
python plot_posteriors.py --config config.yaml \
    --checkpoints init checkpoints/mnist/warmup/ffg_z2/1_model.pth \
                       checkpoints/mnist/warmup/ffg_z2/best_model.pth
```

Each checkpoint is a row (`init` is the untrained model) and each test image a column. A panel
shows the true posterior (the target, grey), the encoder's amortized q(z|x) (blue) and a Gaussian
optimized for that image alone (orange, dashed), with log p(x) and the ELBO under each q.
The script also prints the approximation gap (log p(x) - local ELBO) and the amortization gap
(local ELBO - amortized ELBO) for each image. The figure is saved to `plots/posteriors_<dataset>.png`.

## Other Experiments
For decoder size, flow affect amortization, test set gap and other experiments, refer to [this](https://github.com/chriscremer/Inference-Suboptimality).

## Citation
If you use our code, please consider cite the following:
Chris Cremer, Xuechen Li, David Duvenaud.
Inference Suboptimality in Variational Autoencoders.

```
@article{cremer2018inference,
  title={Inference Suboptimality in Variational Autoencoders},
  author={Cremer, Chris and Li, Xuechen and Duvenaud, David},
  journal={ICML},
  year={2018}
}
```
