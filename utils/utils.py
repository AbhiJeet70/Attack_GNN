
import statistics
from dataset.dataset import load_data
from models.GCN import GCN
from training.training import train_model, evaluate_model
import torch.optim as optim
import torch

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Main experiment function
def run_experiment(ptb_rate=None):
    pyg_data = load_data(ptb_rate=ptb_rate)
    if ptb_rate:
        pyg_data.update_edge_index(pyg_data.adj)
    data = pyg_data[0].to(device)
    model = GCN(pyg_data.num_node_features, 512, pyg_data.num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0)
    
    average = []
    for _ in range(50):
        train_model(data, model, optimizer)
        acc = evaluate_model(data, model)
        average.append(acc)
    
    return statistics.mean(average)
