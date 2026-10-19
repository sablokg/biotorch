"""Minimal end-to-end example: sequence classification with automatic hyper-parameter search."""
import biotorch as bt

seqs, labels = bt.synthetic_sequences(n=400)                 # replace with your own data
ds = bt.SequenceDataset(seqs, labels, bt.DNA, task="binary", augment=True)

result = bt.auto_fit(
    lambda dim=64, num_layers=3, arch="cnn": bt.SeqModel(bt.DNA, 1, "binary", arch=arch, dim=dim, num_layers=num_layers),
    ds, "binary",
    search_space={"lr": (1e-3, 1e-2, "log"), "dim": [32, 64], "arch": ["cnn", "gru"], "num_layers": (2, 4, "int")},
    n_trials=3, trial_epochs=3, final_epochs=10, stratify_labels=labels, out_dir="runs/quickstart",
)
print(result.test_metrics)
