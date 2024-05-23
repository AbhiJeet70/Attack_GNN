import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, dropout):
        super(GAT, self).__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 1
        self.dropout = dropout
        
        self.conv1 = GATConv(input_channels, self.hid, heads=self.in_head, dropout=self.dropout)
        self.conv2 = GATConv(self.hid*self.in_head, output_channels, concat=False,
                             heads=self.out_head, dropout=self.dropout)
        self.reset_parameters()
        
        
    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
    
    def forward(self, data):
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
   