
import torch
import statistics
import torch.optim as optim


from dataset.dataset import load_data
from models.GCN import GCN
from models.GAT import GAT
from models.SGC import SGC
from models.GIN import GIN
from training.training import train_model, evaluate_model



# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Main experiment function
def run_experiment(model_type = "GCN", ptb_rate=None):
    pyg_data = load_data(ptb_rate=ptb_rate)
    if ptb_rate:
        pyg_data.update_edge_index(pyg_data.adj)
    data = pyg_data[0].to(device)

    ## PARAMETERS
    input_dim = pyg_data.num_node_features
    hidden_dim = 512
    output_dim = pyg_data.num_classes

    lr=0.01
    weight_decay=0

    if model_type == "GCN":
        model = GCN(input_dim, hidden_dim, output_dim).to(device)
    elif model_type == "SGC":
        model = SGC(input_dim, hidden_dim, output_dim).to(device)
    elif model_type == "GAT":
        model = GAT(input_dim, hidden_dim, output_dim).to(device)
    elif model_type == "GIN":
        model = GIN(input_dim, hidden_dim, output_dim).to(device)
    else:
        print("Error: No valid model selected")
        model = GCN(input_dim, hidden_dim, output_dim).to(device)
    
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    average = []
    for _ in range(50):
        train_model(data, model, optimizer)
        acc = evaluate_model(data, model)
        average.append(acc)
    
    return statistics.mean(average)
