<div align="center">
<img src="grapes-logo.png" alt="drawing" style="width:400px;"/>
</div>
<br>

This repository contains the official implementation of the graph sampling method presented in ["GRAPES: Learning to Sample Graphs for Scalable Graph Neural Networks"]
## Instructions

### 1. Install dependencies

Create a conda environment with the provided file, then activate it:

```sh
conda env create -f environment.yml
conda activate grapes
```

### 2. Train a model

Run the following to train a GCN classifier on the Cora dataset:

```sh
python main.py
```

We provide configuration files to reproduce the results in our experiments with all datasets.
To use them, run:

```sh
python main.py --config_file=configs/<dataset>.txt
```

Replacing `<dataset>` with the name of the dataset.

### 3. Inspect results on W&B

Logging on Weights & Biases is enabled by default. Results will be logged to a project with name `gflow-sampling`.
To disable this, add the flag `--log_wandb=False`.

---

### Baselines and Data Analysis

For the baseline implementation anad data analysis, please check out the following repos:

* GraphSAINT: https://github.com/dfdazac/grapes/blob/main/graphsaint.py
* LADIES & FastGCN: https://anonymous.4open.science/r/LADIES-9589
* GAS: https://anonymous.4open.science/r/pyg_autoscale-2A4C
* AS-GCN: https://anonymous.4open.science/r/as-gcn-B0FA
* Data Analysis: https://anonymous.4open.science/r/GRAPES-plots-and-analyses-F4B6
