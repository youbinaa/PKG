import argparse
import os.path as osp

import torch
import torch.nn.functional as F

import torch_geometric.transforms as T
from torch_geometric.logging import init_wandb, log
from torch_geometric.nn import GATConv, GATv2Conv, RGATConv
'''
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='Cora')
parser.add_argument('--hidden_channels', type=int, default=8)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--wandb', action='store_true', help='Track experiment')
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
init_wandb(name=f'GAT-{args.dataset}', heads=args.heads, epochs=args.epochs,
           hidden_channels=args.hidden_channels, lr=args.lr, device=device)

path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data', 'Planetoid')
dataset = Planetoid(path, args.dataset, transform=T.NormalizeFeatures())
data = dataset[0].to(device)

'''
class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads):
        super().__init__()
        #default = GATConv, but GATv2Conv outperforms
        self.conv1 = GATv2Conv(in_channels, hidden_channels, heads, dropout=0.6)
        self.conv2 = GATv2Conv(hidden_channels * heads, out_channels, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, x, edge_index, edge_type=None):
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        node_emb = x.clone()

        return node_emb[:, 0, :]
        #return x, F.log_softmax(x, dim=1)

class RGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations):
        super().__init__()
        self.conv1 = RGATConv(in_channels, hidden_channels, num_relations)
        self.conv2 = RGATConv(hidden_channels, hidden_channels, num_relations)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, edge_type):
        x = self.conv1(x, edge_index, edge_type).relu()
        x = self.conv2(x, edge_index, edge_type).relu()
        x = self.lin(x)
        node_emb = x.clone()
        return node_emb[:, 0, :]
        #return F.log_softmax(x, dim=-1)


'''
#train.py로 넘겨야하는 부분
#num_featuers = 768
#hidden_channels = 256 (on average)
#num_classes = 768
#heads = 8 (on average)
model = GAT(dataset.num_features, args.hidden_channels, dataset.num_classes,
            args.heads).to(device)
model = RGAT(16, 16, dataset.num_classes, dataset.num_relations).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4) #얘는 필요없음
#train.py로 넘겨야하는 부분
'''
