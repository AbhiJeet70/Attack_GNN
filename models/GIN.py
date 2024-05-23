import torch
import torch.nn.functional as F
from torch_geometric.nn import GINConv

from torch.nn import Linear, Sequential, BatchNorm1d, ReLU

class GIN(torch.nn.Module):
    def __init__(self, input_channels, hidden_channels, output_channels, dropout):
        super(GIN, self).__init__()
        self.dropout = dropout
        self.conv1 = GINConv(
            Sequential(Linear(input_channels, hidden_channels),
                        BatchNorm1d(hidden_channels), 
                        ReLU(),
                        Linear(hidden_channels, hidden_channels), 
                        ReLU()))
        self.conv2 = GINConv(
            Sequential(Linear(hidden_channels, hidden_channels), 
                        BatchNorm1d(hidden_channels), 
                        ReLU(),
                        Linear(hidden_channels, hidden_channels), 
                        ReLU()))
        
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, output_channels)
        self.reset_parameters()  

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)

        # Classifier
        h = F.relu(self.lin1(x))
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.lin2(h)

        return F.log_softmax(h, dim=1)  
        