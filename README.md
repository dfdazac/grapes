
  <img src="grapes-logo.png" alt="drawing" style="width:400px;"/>
  <h3>Sampling graphs with GFlowNets</h3>


## Instructions

### 1. Install dependencies

Create a conda environment with the provided file, then activate it:

```sh
conda env create -f environment.yml
conda activate gflow-sampling
```

### 2. Train a model

Run the following to train a GCN classifier on the Cora dataset:

```sh
python main.py
```

Available datasets are `cora`, `citeseer`, and `pubmed` via the `--dataset` flag. For example,

```sh
python main.py --dataset=citeseer
```

When using a GPU, training on Cora should take less than 10 seconds.

### 3. Inspect results on W&B

Logging on Weights & Biases is disabled by default. To enable it, add the flag `--log_wandb`.
This will log results to a project with name `gflow-sampling`.
