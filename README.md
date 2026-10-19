# biotorch

PyTorch toolkit for **biological sequences**, **biological networks** and **biological graphs**, with an
automated training / dataset / hyper-parameter-search loop. Only `torch` and `numpy` are required.

```bash
pip install .            # or: pip install -e ".[test]"
pip install ".[chem]"    # optional: RDKit for SMILES -> graph
biotorch selftest
```

## Modules

| module | contents |
|---|---|
| `biotorch.sequences` | DNA/RNA/protein alphabets, k-mers, reverse complement, translation, FASTA/FASTQ, `SequenceDataset` (multiclass, binary, multilabel, regression, per-residue, MLM) |
| `biotorch.networks` | `BioNetwork` (PPI, regulatory, metabolic; weighted / directed / typed), random walks, k-hop, components, de Bruijn graphs |
| `biotorch.graphs` | `BioGraph`, `GraphBatch`, collation, negative sampling, edge splits, protein-structure and molecular graphs |
| `biotorch.layers` | scatter ops + GCN, SAGE, GAT, GIN, relational GCN, `GNNEncoder`, pooling |
| `biotorch.models` | `SeqModel` (CNN/Transformer/GRU/LSTM), `SeqMLM`, `GraphModel`, `NodeModel`, `LinkModel`, `Node2Vec`, `SeqGraphNodeModel` |
| `biotorch.training` | `Trainer`: AMP, grad accumulation, clipping, warmup+cosine, early stopping, checkpoints, resume |
| `biotorch.auto` | `auto_split`, `auto_fit` (random search -> retrain -> test report), `load_dataset` |
| `biotorch.synthetic`, `biotorch.demos` | synthetic data, demos, self-test |

## Usage

```python
import biotorch as bt

ds = bt.SequenceDataset.from_csv("data.csv", "sequence", "label", task="binary", alphabet=bt.DNA, augment=True)
res = bt.auto_fit(lambda dim=64, num_layers=3: bt.SeqModel(bt.DNA, 1, "binary", dim=dim, num_layers=num_layers),
                  ds, "binary", search_space={"lr": (1e-3, 1e-2, "log"), "dim": [32, 64, 128]},
                  n_trials=6, final_epochs=30, stratify_labels=ds.labels, out_dir="runs/exp1")
print(res.test_metrics)
```

Graph / network tasks:

```python
net = bt.BioNetwork.from_tsv("ppi.tsv")                      # name \t name [\t weight]
g = net.to_graph(feature="random")                           # or node features via node_features=
train, val, test = bt.link_split_loaders(g)                  # link prediction
res = bt.auto_fit(lambda hidden=64: bt.LinkModel(g.x.size(1), hidden), None, "binary",
                  {"lr": (1e-3, 2e-2, "log"), "hidden": [32, 64]}, loaders=(train, val, test),
                  monitor="val_auroc", mode="max")
```

Every model implements `step(batch) -> StepOutput(loss, pred, target)`, so custom models only need that method
to work with `Trainer` and `auto_fit`.

## CLI

```bash
biotorch demo all
biotorch train-seq --csv data.csv --alphabet dna --task binary --arch cnn --epochs 30 --trials 4
biotorch train-link --edges ppi.tsv --layer sage
python -m biotorch selftest
```

## Tests

```bash
pip install -e ".[test]" && pytest
```

Gaurav Sablok \
gsablok@proton.me
