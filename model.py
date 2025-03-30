import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GNN(torch.nn.Module):
    def __init__(self, num_features, num_classes):
        super(GNN, self).__init__()
        self.conv1 = GATConv(num_features, 32, heads=8, dropout=0.6)  # Updated to match saved model
        self.conv2 = GATConv(32 * 8, num_classes, heads=1, concat=True, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
