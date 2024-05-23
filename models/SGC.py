import torch
import torch.nn.functional as F
from torch_geometric.nn import SGConv


class SGC(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, dropout):
        super(SGC, self).__init__()
        self.dropout = dropout
        self.conv1 = SGConv(input_channels, hidden_channels, K=2, cached=False)
        self.conv2 = SGConv(hidden_channels, output_channels, K=2, cached=False)
        self.reset_parameters()
        
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)        
        
    