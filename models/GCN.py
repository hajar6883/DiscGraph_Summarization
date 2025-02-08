import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from sklearn.metrics import f1_score
import dgl
from collections import Counter

# Sample GCN model
class GCNModel(nn.Module):
    def __init__(self, in_feats, hidden_feats, num_classes):
        super(GCNModel, self).__init__()
        self.conv1 = dglnn.GraphConv(in_feats, hidden_feats)
        self.conv2 = dglnn.GraphConv(hidden_feats, num_classes)

    def forward(self, g, node_embeddings, edge_encodings):
        # Apply the first GCN layer
        h = self.conv1(g, node_embeddings)
        h = F.relu(h)
        # Apply the second GCN layer
        h = self.conv2(g, h)
        return h

# Load  graphs
labeled_graphs = torch.load('processed_train_graphs2.pth')
print(labeled_graphs[0])
for i in range(len(labeled_graphs)):
    labeled_graphs[i] = dgl.add_self_loop(labeled_graphs[i])

random.shuffle(labeled_graphs) 
split_index = int(len(labeled_graphs) * 0.8)
train_graphs = labeled_graphs[:split_index]
validation_graphs = labeled_graphs[split_index:]




labels = [label.item() for g in labeled_graphs for label in g.ndata['label']]

class_counts = Counter(labels)
print(class_counts)


model = GCNModel(in_feats=772, hidden_feats=128, num_classes=1)  

# class_weights = torch.tensor([1., (59331/13292)]).to(device)  # Adjust weights
# loss_func = nn.BCEWithLogitsLoss(pos_weight=class_weights[1])
loss_func = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for g in labeled_graphs:
        node_embeddings = g.ndata['features'].float()
        edge_encodings = g.edata['rel_type'].float()
        labels = g.ndata['label'].float()

        optimizer.zero_grad()
        logits = model(g, node_embeddings, edge_encodings)
        loss = loss_func(logits.view(-1), labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(labeled_graphs)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for g in validation_graphs:
        node_embeddings = g.ndata['features'].float()
        edge_encodings = g.edata['rel_type'].float()
        labels = g.ndata['label'].long() 
        logits = model(g, node_embeddings, edge_encodings)
        predicted_labels = (logits >= 0.5).long() 
        y_true.extend(labels.tolist())
        y_pred.extend(predicted_labels.tolist())

f1 = f1_score(y_true, y_pred, average='weighted')
print(f"F1 Score on Validation Set: {f1:.4f}")


model_path = 'GCN.pth' 
torch.save(model.state_dict(), model_path)

print(f"Model saved to {model_path}")
