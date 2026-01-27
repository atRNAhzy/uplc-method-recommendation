import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader as GeoDataLoader
from torch_geometric.utils import from_smiles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from rdkit import Chem
from transformers import AutoTokenizer


def _worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)


def _split_indices(n, train_ratio, shuffle, random_state):
    idx = np.arange(n)
    if train_ratio is None:
        return idx, None
    return train_test_split(
        idx,
        train_size=train_ratio,
        shuffle=shuffle,
        random_state=random_state)


class GNNDataset(Dataset):
    def __init__(self, smiles, y, phys=None):
        self.smiles = smiles
        self.y_scaler = StandardScaler()
        self.y = torch.tensor(
            self.y_scaler.fit_transform(np.array(y).reshape(-1, 1)).squeeze(),
            dtype=torch.float32
        )

        self.phys = None
        if phys is not None:
            self.phys_scaler = StandardScaler()
            self.phys = self.phys_scaler.fit_transform(phys)

        self.graphs = self._build_graphs()

    def _build_graphs(self):
        graphs = []
        for i, smi in enumerate(self.smiles):
            try:
                g = from_smiles(smi)
                if g is None or g.x is None:
                    continue
                g.x = g.x.float()
                if self.phys is not None:
                    extra = torch.tensor(self.phys[i], dtype=torch.float32)
                    g.x = torch.cat(
                        [g.x, extra.expand(g.num_nodes, -1)], dim=1
                    )
                g.y = self.y[i].unsqueeze(0)
                g.smiles = smi
                graphs.append(g)
            except Exception:
                continue
        return graphs

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]

    @property
    def rt_scaler(self):
        return self.y_scaler


# =========================
# BERT Dataset
# =========================

class BERTDataset(Dataset):
    def __init__(
        self,
        smiles,
        y,
        tokenizer,
        max_length=128,
        phys=None,
        augment=False
    ):
        self.smiles = smiles
        self.augment = augment
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._phys_dim = 0

        self.y_scaler = StandardScaler()
        self.y = torch.tensor(
            self.y_scaler.fit_transform(np.array(y).reshape(-1, 1)).squeeze(),
            dtype=torch.float32
        )

        self.phys = None
        if phys is not None:
            self.phys_scaler = StandardScaler()
            self.phys = self.phys_scaler.fit_transform(phys)
            self._phys_dim = self.phys.shape[1]

    def _rand_smiles(self, s):
        mol = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(mol, doRandom=True) if mol else s

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smi = self.smiles[idx]
        if self.augment:
            smi = self._rand_smiles(smi)

        enc = self.tokenizer(
            smi,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = self.y[idx].unsqueeze(0)

        if self.phys is not None:
            item["physicochemical"] = torch.tensor(
                self.phys[idx], dtype=torch.float32
            )

        return item

    @property
    def phys_dim(self):
        return self._phys_dim

    @property
    def rt_scaler(self):
        return self.y_scaler


# =========================
# DataLoader Factory
# =========================

def create_data_loaders(
    smiles,
    y,
    phys=None,
    model_type="gnn",
    train_ratio=0.8,
    batch_size=32,
    shuffle=True,
    random_state=42,
    bert_model_name=None,
    max_length=128,
    augment=False
):
    gen = torch.Generator().manual_seed(random_state)

    train_idx, test_idx = _split_indices(
        len(smiles), train_ratio, shuffle, random_state
    )

    def _subset(arr, idx):
        return [arr[i] for i in idx] if idx is not None else arr

    if model_type == "bert":
        tokenizer = AutoTokenizer.from_pretrained(bert_model_name)

        train_ds = BERTDataset(
            _subset(smiles, train_idx),
            _subset(y, train_idx),
            tokenizer,
            max_length=max_length,
            phys=_subset(phys, train_idx) if phys is not None else None,
            augment=augment
        )

        test_ds = BERTDataset(
            _subset(smiles, test_idx),
            _subset(y, test_idx),
            tokenizer,
            max_length=max_length,
            phys=_subset(phys, test_idx) if phys is not None else None,
            augment=False
        )

        train_loader = torch.utils.data.DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=gen
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=batch_size,
            shuffle=False
        )

        return train_loader, test_loader, train_ds, test_ds

    # ---- GNN ----
    train_ds = GNNDataset(
        _subset(smiles, train_idx),
        _subset(y, train_idx),
        phys=_subset(phys, train_idx) if phys is not None else None
    )
    test_ds = GNNDataset(
        _subset(smiles, test_idx),
        _subset(y, test_idx),
        phys=_subset(phys, test_idx) if phys is not None else None
    )

    train_loader = GeoDataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        worker_init_fn=_worker_init_fn,
        generator=gen
    )
    test_loader = GeoDataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader, train_ds, test_ds
