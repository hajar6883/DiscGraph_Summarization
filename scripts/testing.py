from GCN import GCNModel
import torch
import dgl 
import json
    
model = GCNModel(in_feats=772, hidden_feats=128, num_classes=1)

model_path = 'GCN.pth' 

model.load_state_dict(torch.load(model_path))


unlabeled_graphs = torch.load('processed_test_graphs2.pth') 
for file_id, graph in unlabeled_graphs.items():
    unlabeled_graphs[file_id] = dgl.add_self_loop(graph)

predicted_labels_dict = {}

model.eval()

with torch.no_grad():
    for file_id, graph in unlabeled_graphs.items():
        node_embeddings = graph.ndata['features'].float()
        edge_encodings = graph.edata['rel_type'].float()  

        logits = model(graph, node_embeddings, edge_encodings)

        predicted_labels = (logits >= 0.5).long().tolist()
        

        predicted_labels_dict[file_id] =  [label for sublist in predicted_labels for label in sublist]
       
        

json_filename = "predicted_labels.json"

with open(json_filename, 'w') as json_file:
    json.dump(predicted_labels_dict, json_file)

print(f"Predicted labels have been exported to {json_filename}.")