import math
import torch
import biotorch as bt


def test_sequence_ops():
    assert bt.reverse_complement("ATGC") == "GCAT"
    assert bt.reverse_complement("AUGC".replace("U", "T"), rna=True) == "GCAU"
    assert bt.translate("ATGGCCTAA", to_stop=True) == "MA"
    assert abs(bt.gc_content("GGCC") - 1.0) < 1e-9
    assert bt.kmer_counts("ACGT", 1).sum() == 1.0
    assert bt.DNA.decode(bt.DNA.encode("ACGT", add_special=True)) == "ACGT"


def test_fasta_roundtrip(tmp_path):
    p = tmp_path / "x.fa"
    bt.write_fasta([("a", "ACGT" * 30), ("b", "GGCC")], str(p))
    assert bt.read_fasta(str(p)) == [("a", "ACGT" * 30), ("b", "GGCC")]


def test_dataset_collate_and_token_labels():
    seqs, labels = bt.synthetic_sequences(n=8, task="token_multiclass")
    ds = bt.SequenceDataset(seqs, labels, bt.DNA, task="token_multiclass", add_special=True)
    b = ds.collate([ds[i] for i in range(4)])
    assert b["input_ids"].shape == b["labels"].shape and b["mask"].dtype == torch.bool


def test_network_and_graph():
    net = bt.BioNetwork.from_edge_list([("A", "B"), ("B", "C"), ("C", "A"), ("D", "E")])
    assert net.num_nodes == 5 and len(net.connected_components()) == 2
    assert net.k_hop("A", 1) == ["A", "B", "C"]
    g = net.to_graph(feature="random")
    batch = bt.collate_graphs([g, g])
    assert batch.num_graphs == 2 and batch.x.size(0) == 10
    assert bt.debruijn_graph(["ACGTACGT"], 3).num_nodes > 0


def test_negative_sampling_excludes_edges():
    ei = torch.tensor([[0, 1, 2], [1, 2, 3]])
    neg = bt.negative_sampling(bt.to_undirected(ei), 6, 20)
    pos = {(int(a), int(b)) for a, b in torch.cat([ei, ei.flip(0)], 1).T}
    assert all((int(a), int(b)) not in pos for a, b in neg.T)


def test_gnn_layers_forward():
    g = bt.synthetic_graphs(n=2)[0]
    for layer in ["gcn", "sage", "gat", "gin"]:
        enc = bt.GNNEncoder(2, 16, 2, layer, heads=4)
        assert enc(g.x, g.edge_index).shape == (g.num_nodes, 16)
    enc = bt.GNNEncoder(2, 16, 2, "rgcn", num_relations=2)
    et = torch.randint(0, 2, (g.num_edges,))
    assert enc(g.x, g.edge_index, edge_type=et).shape[1] == 16


def test_metrics():
    assert bt.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0
    m = bt.compute_metrics("regression", *(lambda a: (a, a))(__import__("numpy").arange(10.0)))
    assert m["r2"] == 1.0


def test_sequence_training_and_predict():
    seqs, labels = bt.synthetic_sequences(n=600)
    ds = bt.SequenceDataset(seqs, labels, bt.DNA, task="binary")
    tr, va, te = bt.auto_split(ds, stratify=labels)
    model = bt.SeqModel(bt.DNA, 1, "binary", dim=64, num_layers=3)
    t = bt.Trainer(model, "binary", bt.TrainConfig(epochs=12, lr=2e-3, patience=0, verbose=False))
    t.fit(bt.make_loader(tr, 32, True), bt.make_loader(va, 64))
    m = t.evaluate(bt.make_loader(te, 64), "test")
    assert m["test_acc"] > 0.7
    p, y = t.predict(bt.make_loader(te, 64))
    assert p.shape == y.shape


def test_checkpoint_resume(tmp_path):
    seqs, labels = bt.synthetic_sequences(n=64)
    ds = bt.SequenceDataset(seqs, labels, bt.DNA, task="binary")
    dl = bt.make_loader(ds, 32, True)
    mk = lambda e, r: bt.Trainer(bt.SeqModel(bt.DNA, 1, "binary", dim=16, num_layers=1), "binary",
                                 bt.TrainConfig(epochs=e, patience=0, verbose=False, out_dir=str(tmp_path), resume=r))
    mk(2, False).fit(dl, dl)
    t = mk(4, True)
    t.fit(dl, dl)
    assert len(t.history) == 4


def test_graph_node_link_pipelines():
    for name in ["graph", "node", "link"]:
        m = bt.DEMOS[name](False)
        assert all(not math.isnan(v) for k, v in m.items() if k.startswith("test_acc") or k == "test_loss")


def test_auto_fit_runs():
    assert "test_loss" in bt.demo_auto(False)
