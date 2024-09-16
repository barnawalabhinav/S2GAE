import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split

# Load the PROTEINS dataset
dataset = TUDataset(root=f'./dataset/graph-class', name='PROTEINS')

# Split dataset into training and testing
train_idx, test_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
train_dataset = dataset[train_idx]
test_dataset = dataset[test_idx]

# Create DataLoader for mini-batch processing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Define the GCN model
class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.fc = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, batch):
        # 1st Graph Convolution + ReLU
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        
        # 2nd Graph Convolution
        x = self.conv2(x, edge_index)
        
        # Global mean pooling (graph-level representation)
        x = global_mean_pool(x, batch)  # Shape: [num_graphs, hidden_channels]
        
        # Output layer (graph classification)
        x = self.fc(x)  # Shape: [num_graphs, out_channels]
        
        return x

# Initialize model, optimizer, and loss function
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GCN(in_channels=dataset.num_features, hidden_channels=64, out_channels=dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

# Training function
def train():
    model.train()
    total_loss = 0
    correct = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x, data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        correct += (out.argmax(dim=1) == data.y).sum().item()
    return total_loss / len(train_dataset), correct / len(train_dataset)

# Testing function
def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        correct += (out.argmax(dim=1) == data.y).sum().item()
    return correct / len(loader.dataset)

# Training loop
for epoch in range(1, 101):
    loss, train_acc = train()
    test_acc = test(test_loader)
    print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
