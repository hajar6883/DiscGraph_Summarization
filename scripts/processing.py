from preprocessing import *
from baseline import flatten
from transformers import BertTokenizer, BertModel
import torch
import dgl
from someData import rel_types

training_set = ['ES2002', 'ES2005', 'ES2006', 'ES2007', 'ES2008', 'ES2009', 'ES2010', 'ES2012', 'ES2013', 'ES2015', 'ES2016', 'IS1000', 'IS1001', 'IS1002', 'IS1003', 'IS1004', 'IS1005', 'IS1006', 'IS1007', 'TS3005', 'TS3008', 'TS3009', 'TS3010', 'TS3011', 'TS3012']
training_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in training_set])
training_set.remove('IS1002a')
training_set.remove('IS1005d')
training_set.remove('TS3012c')

test_set = ['ES2003', 'ES2004', 'ES2011', 'ES2014', 'IS1008', 'IS1009', 'TS3003', 'TS3004', 'TS3006', 'TS3007']
test_set = flatten([[m_id+s_id for s_id in 'abcd'] for m_id in test_set])


#encoding relations 
mapping =  {rel_type: i for i, rel_type in enumerate(rel_types)}
num_rel_types = len(rel_types)
one_hot_encodings = torch.eye(num_rel_types)




def ProcessDialogue(file_id, tokenizer, model):
    
    df = parse_jsons(file_id , train=False)
    graph_dict, _= parse_discourse_graph(file_id, train=False) 
    edge_list, edge_types = extract_edges(graph_dict)
    
    df['encoded'] = df['text'].apply(lambda x: tokenize(x, tokenizer))
    df['input_ids'] = df['encoded'].apply(lambda x: x['input_ids'])
    df['attention_mask'] = df['encoded'].apply(lambda x: x['attention_mask']) 
    df['token_type_ids'] = df['encoded'].apply(lambda x: x['token_type_ids'])
     
    embeddings_tensor= transform(df, model)
    # print( mbeddings_tensor.shape[1 ])#input features param in the GCN model
   
    # Creating DGL graph
    g = dgl.graph(edge_list)
    # if not g: 
    #     print(f"Failed to create graph for {file_id}")
    #     return None

    # Adding features
    edge_features = torch.stack([one_hot_encodings[mapping[rel_type]] for rel_type in edge_types])
    g.edata['rel_type'] = edge_features
    g.ndata['features'] = embeddings_tensor
    
    one_hot_encoded_speakers = pd.get_dummies(df['speaker'])
    speaker_features = torch.tensor(one_hot_encoded_speakers.values, dtype=torch.float32)
    combined_features = torch.cat([g.ndata['features'], speaker_features], dim=1)
    
    g.ndata['features'] = combined_features
    
    # labels = get_labels(file_id)
    # labels_tensor = torch.tensor(labels, dtype=torch.long)
    # g.ndata['label'] = labels_tensor

    
    return g



# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# model = BertModel.from_pretrained('bert-base-uncased')

# all_labeled_graphs = []
# for file_id in training_set:
#     graph = ProcessDialogue(file_id, tokenizer, model)
#     if graph is not None:
#         print(f"graph for {file_id} constructed")
#         all_labeled_graphs.append(graph)
#     else:
#         print(f"No graph returned for {file_id}")
  
# # torch.save(all_labeled_graphs, 'processed_train_graphs2.pth')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

unlabeled_graphs = {}
for file_id in test_set:
    
    graph = ProcessDialogue(file_id, tokenizer, model)
    if graph is not None:
        print(f"graph for {file_id} constructed")
        unlabeled_graphs[file_id]=graph
    else:
        print(f"No graph returned for {file_id}")
  

# Save the processed graphs
torch.save(unlabeled_graphs, 'processed_test_graphs2.pth')