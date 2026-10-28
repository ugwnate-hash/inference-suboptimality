# inference-suboptimality
Code regarding evaluation for paper
*Inference Suboptimality in Variational Autoencoders.*
[[arxiv](https://arxiv.org/abs/1801.03558)]

## Dependencies
* `python3` (3.10–3.12 recommended)
* `pytorch>=2.0` and `torchvision`
* `numpy`, `scipy`, `tqdm`
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

## Training
MNIST is extracted from `datasets/mnist.pkl.tar.gz` automatically on first use; Fashion is read
directly from the `.gz` files in `datasets/fashion/`.

For a quick check that everything works, train for a couple of epochs first:
`python run.py --train --dataset mnist --epochs 2 --display-epoch 1`

`python run.py --train --dataset <dataset> (--lr-schedule --warmup --early-stopping)`

To train on CIFAR, set the argument for the dataset flag to `cifar`. The dataset should be
downloaded automatically, if not already downloaded.

## Evaluation
* IWAE: `python run.py --eval-iwae --dataset <dataset> --eval-path <ckpt path>`
* AIS: `python run.py --eval-ais --dataset <dataset> --eval-path <ckpt path>`
* Local FFG: `python local_ffg.py --dataset <dataset> --eval-path <ckpt path>`
* Local Flow: `python local_flow.py --dataset <dataset> --eval-path <ckpt path>`
* BDMC: `python bdmc.py --eval-path <ckpt path> --n-ais-iwae <num samples> --n-ais-dist <num dist>`

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
