
import torch
import copy
import os
import torch.nn.functional as F

from models.gcn import GCN
from models.sgc import SGC
from models.gat import GAT
from models.gin import GIN


class GNN():
    def __init__(self, model_type, input_dim, hidden_dim, output_dim, epochs, lr, weight_decay, dropout, patience):

        self.n_epochs = epochs
        self.lr = lr
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.patience = patience

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.model_type = model_type
        if model_type == "GCN":
            self.model = GCN(self.input_dim, self.hidden_dim, self.output_dim, self.dropout)
        elif model_type == "SGC":
            self.model = SGC(self.input_dim, self.hidden_dim, self.output_dim, self.dropout)
        elif model_type == "GAT":
            self.model = GAT(self.input_dim, self.hidden_dim, self.output_dim, self.dropout)
        elif model_type == "GIN":
            self.model = GIN(self.input_dim, self.hidden_dim, self.output_dim, self.dropout)
        else:
            self.model = GCN(self.input_dim, self.hidden_dim, self.output_dim, self.dropout)
            print("Error: No valid model selected")
       
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr= self.lr, weight_decay=self.weight_decay)
        self.criterion = F.nll_loss

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
    
    # Train the model
    def train(self, loader):
        best_val_acc = test_acc = 0
        
        best_model_wts = copy.deepcopy(self.model.state_dict())
        counter_improvement = 0
        for epoch in range(0, self.n_epochs):
            train_loss = self._train(loader)
            train_acc, val_acc, tmp_test_acc = self.test()
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc = tmp_test_acc
                best_model_wts = copy.deepcopy(self.model.state_dict())
                counter_improvement = 0
            else:
                counter_improvement += 1
                if counter_improvement >= self.patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

            if epoch % 100 == 0:
                print(f'Epoch: {epoch:03d}, Train Loss: {train_loss:.4f}, '
                f'Train Acc: {train_acc:.4f}, Val Acc: {best_val_acc:.4f}, Test Acc: {test_acc:.4f}')

       
        # Restore the best model state dict
        self.model.load_state_dict(best_model_wts)
        
        train_acc, val_acc, tmp_test_acc = self.test()

        print(f'Best Model: {self.gnn_type}, Val Acc: {best_val_acc:.4f}, Test Acc: {test_acc:.4f}')
        print('----------------------------------------------------')


    def _train(self, loader):
        self.model.train()
        total_train_loss = total_examples = 0
        for batch in loader:
            batch = batch.to(self.device)
            self.optimizer.zero_grad()
            
            out = self.model(batch)
            loss = F.nll_loss(out[batch.train_mask], batch.y[batch.train_mask])
            loss.backward()
            self.optimizer.step()
            
            total_train_loss += loss.item() 

        return total_train_loss/len(loader)

    # Evaluate the model
    def test(self, loader):
        self.model.eval()
        accs = [0, 0, 0]
        for batch in loader:
            batch = batch.to(self.device)
            out = self.model(batch).argmax(dim=-1)
            i = 0
            for _, mask in batch('train_mask', 'val_mask', 'test_mask'):
                acc = 0
                if (mask.sum().item() > 0):
                    acc = out[mask].eq(batch.y[mask]).sum().item() / mask.sum().item()
                accs[i] += acc
                i += 1
                    
        return accs[0]/len(loader), accs[1]/len(loader), accs[2]/len(loader)
            
        