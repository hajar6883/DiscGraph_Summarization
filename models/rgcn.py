import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn import GraphConv
import random
from sklearn.metrics import f1_score
import dgl
import dgl.nn as dglnn
from someData import num_rel_types

# Load  graphs
labeled_graphs = torch.load('processed_dialogue_graphs2.pth')

for i in range(len(labeled_graphs)):
    labeled_graphs[i] = dgl.add_self_loop(labeled_graphs[i])

random.shuffle(labeled_graphs) 
split_index = int(len(labeled_graphs) * 0.8)
train_graphs = labeled_graphs[:split_index]
validation_graphs = labeled_graphs[split_index:]


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels):
        super(RGCNLayer, self).__init__()
        self.conv = dglnn.HeteroGraphConv({
            rel: dglnn.GraphConv(in_feat, out_feat) for rel in range(num_rels)
        })

    def forward(self, g, features):
        return self.conv(g, features)

class RGCNModel(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, num_rels):
        super(RGCNModel, self).__init__()
        self.layer1 = RGCNLayer(in_feat, hidden_feat, num_rels)
        self.layer2 = RGCNLayer(hidden_feat, out_feat, num_rels)

    def forward(self, adj_matrix, features):
        g = dgl.graph(adj_matrix, 'user', 'movie')
        h = self.layer1(g, features)
        h = F.relu(h)
        h = self.layer2(g, h)
        return h
    


model = RGCNModel(in_feat=768, hidden_feat=128, out_feat=2, num_rels=num_rel_types)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for g in train_graphs:
        node_features = g.ndata['features'].float()
        labels = g.ndata['label'].long()

        # Extract adjacency matrix from the graph
        adj_matrix = g.adjacency_matrix().toarray()

        optimizer.zero_grad()
        logits = model(adj_matrix, node_features)
        loss = loss_func(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(train_graphs)
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}")

#save trainned model 


model.eval()
y_true = []
y_pred = []

with torch.no_grad():
    for g in validation_graphs:
        features = g.ndata['features']
        labels = g.ndata['label']
        logits = model(g, features)
        _, predicted = torch.max(logits, 1)

        y_true.extend(labels.tolist())
        y_pred.extend(predicted.tolist())

f1 = f1_score(y_true, y_pred, average='weighted')
print(f"F1 Score on Validation Set: {f1:.4f}")

model_path = 'RGCN.pth' 

torch.save(model.state_dict(), model_path)

print(f"Model saved to {model_path}")
