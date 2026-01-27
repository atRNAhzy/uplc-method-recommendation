# models.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import (
    GCNConv, GATConv, GINConv, SAGEConv,
    global_mean_pool, BatchNorm
)

from transformers import AutoModel




class GNNModel(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim=128,
        output_dim=1,
        num_layers=3,
        dropout=0.2,
        gnn_type="GIN",
        use_batch_norm=False,
    ):
        super().__init__()

        self.use_batch_norm = use_batch_norm
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() if use_batch_norm else None

        def build_conv(in_dim, out_dim):
            if gnn_type == "GCN":
                return GCNConv(in_dim, out_dim)
            if gnn_type == "GAT":
                return GATConv(in_dim, out_dim, heads=4, concat=False)
            if gnn_type == "GraphSAGE":
                return SAGEConv(in_dim, out_dim)
            if gnn_type == "GIN":
                mlp = nn.Sequential(
                    nn.Linear(in_dim, out_dim),
                    nn.ReLU(),
                    nn.Linear(out_dim, out_dim),
                )
                return GINConv(mlp)
            raise ValueError(f"Unsupported gnn_type: {gnn_type}")

        # first layer
        self.convs.append(build_conv(input_dim, hidden_dim))
        if use_batch_norm:
            self.bns.append(BatchNorm(hidden_dim))

        # hidden layers
        for _ in range(num_layers - 1):
            self.convs.append(build_conv(hidden_dim, hidden_dim))
            if use_batch_norm:
                self.bns.append(BatchNorm(hidden_dim))

        # head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if self.use_batch_norm:
                x = self.bns[i](x)
            x = F.relu(x)

        x = global_mean_pool(x, batch)
        return self.head(x)


class BERTModel(nn.Module):
    """
    BERT encoder + optional physicochemical features -> regression head
    """

    def __init__(
        self,
        bert_dir,
        phys_dim=0,
        hidden_dim=256,
        dropout=0.5,
        freeze_bert=False,
    ):
        super().__init__()

        self.bert = AutoModel.from_pretrained(bert_dir)
        bert_dim = self.bert.config.hidden_size

        if freeze_bert:
            for p in self.bert.parameters():
                p.requires_grad = False

        self.use_phys = phys_dim > 0
        if self.use_phys:
            self.phys_proj = nn.Sequential(
                nn.Linear(phys_dim, phys_dim),
                nn.ReLU(),
                nn.LayerNorm(phys_dim),
                nn.Dropout(dropout * 0.5),
            )

        fusion_dim = bert_dim + phys_dim if self.use_phys else bert_dim

        self.regressor = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, input_ids, attention_mask, physicochemical=None, **kwargs):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_feat = out.last_hidden_state[:, 0]

        if self.use_phys and physicochemical is not None:
            phys_feat = self.phys_proj(physicochemical)
            feat = torch.cat([cls_feat, phys_feat], dim=1)
        elif self.use_phys:
            raise ValueError("physicochemical features are required but missing")
        else:
            feat = cls_feat

        return self.regressor(feat)



def get_model(
    model_type="gnn",
    pretrained_model=None,
    freeze_backbone=False,
    **kwargs):

    if model_type == "bert":
        model = BERTModel(
            bert_dir=kwargs["bert_model_name"],
            phys_dim=kwargs.get("phys_dim", 0),
            hidden_dim=kwargs.get("hidden_dim", 256),
            dropout=kwargs.get("dropout", 0.5),
            freeze_bert=freeze_backbone,
        )

    elif model_type == "gnn":
        model = GNNModel(
            input_dim=kwargs["input_dim"],
            hidden_dim=kwargs.get("hidden_dim", 128),
            output_dim=kwargs.get("output_dim", 1),
            num_layers=kwargs.get("num_layers", 3),
            dropout=kwargs.get("dropout", 0.2),
            gnn_type=kwargs.get("gnn_type", "GIN"),
            use_batch_norm=kwargs.get("use_batch_norm", False),
        )

    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # -------- load checkpoint --------
    if pretrained_model and os.path.exists(pretrained_model):
        ckpt = torch.load(pretrained_model, map_location="cpu")
        state_dict = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state_dict, strict=False)
        print(f"[INFO] Loaded pretrained model from {pretrained_model}")

    return model
