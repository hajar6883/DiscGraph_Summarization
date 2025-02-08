import numpy as np
import json
import pandas as pd
import torch
from typing import Dict, List, Tuple, Set
import os



def parse_jsons(file_id: str, train= True) : #parsing dialogues 
    path = os.path.join('training' if train else 'test', f'{file_id}.json')
    
    with open(path) as f:
        utterances = json.load(f)
        
    df = pd.json_normalize(utterances, 
                        meta=['speaker', 'text', 'index'])
    df.columns=['speaker', 'text', 'index']
    df.drop(columns=['index'], inplace=True)
    return df


def get_labels(file_id): #return utterances labels of the given dialogue 
    with open("training_labels.json") as f:
        data = json.load(f)
        labels = data.get(file_id, "Key not found")
    return labels
    

def parse_discourse_graph(file_id: str, train=True ) :
    path = os.path.join('training' if train else 'test', f'{file_id}.txt')
    
    graph_dict = {}
    relationship_Set=set()
    threshold_distance = 1  
    
    with open(path) as file:
        for line in file:
            parel_types = line.strip().split()
            if len(parel_types) == 3:
                src, relation, dest = parel_types
                src, dest = int(src), int(dest)
                
                relationship_Set.add(relation)
                
                # Determine if the link is long-distance
                is_long_distance = abs(dest - src) > threshold_distance

                if src not in graph_dict:
                    graph_dict[src] = []
                graph_dict[src].append((dest, relation, is_long_distance))  

    return graph_dict, relationship_Set


def extract_edges(graph_dict):
    edge_list = []
    edge_types = []

    for source_node, connections in graph_dict.items():
        for target_node, relation, _ in connections:
            edge_list.append((source_node, target_node))
            edge_types.append(relation)

    return edge_list, edge_types


def tokenize(text, tokenizer):
    max_length =64 #max 512
    return tokenizer.encode_plus(
        text, 
        add_special_tokens=True, 
        max_length=max_length, 
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        # return_token_type_ids=True,
        return_tensors='pt'
        
    )
    
def transform(df, model): #transforming tokens embedding to contextually aware embeddings.. 
    embeddings = []

    for index, row in df.iterrows():
        input_ids = row['input_ids']
        attention_mask = row['attention_mask']
        
        with torch.no_grad():
            # Forward pass, get model output
            output = model(input_ids, attention_mask=attention_mask) 

        # Get the embeddings from the last hidden state
        last_hidden_states = output.last_hidden_state # embeddings for all tokens in the current input seq
        # print((last_hidden_states).size())
        # Optionally, you can take the embedding of the `[CLS]` token (index 0) as the utterance representation
        utterance_embedding = last_hidden_states[:, 0, :].squeeze()

        embeddings.append(utterance_embedding)

    # Convert list of embeddings into a tensor
    embeddings_tensor = torch.stack(embeddings)
    return embeddings_tensor


  
# def augmentNodes(graph): # augmente node feat w\ edge feat(since GCN doesn't directly uses edge features))
#     num_nodes = graph.number_of_nodes()
#     edge_feat_dim = graph.edata['rel_type'].shape[1]
#     aggregated_edge_features = torch.zeros((num_nodes, edge_feat_dim))
#     src, dst = graph.edges()

#     for i in range(src.size(0)):  # src and dst have the same size
#         source_node = src[i].item()
#         destination_node = dst[i].item()
#         edge_feature = graph.edata['rel_type'][i]

        
#         aggregated_edge_features[destination_node] += edge_feature # Aggregate features to destination node
#         aggregated_edge_features[source_node] += edge_feature # Aggregate features to source node
        
        
#         original_node_features = graph.ndata['features']
#         # Concatenate
#         augmented_node_features = torch.cat([original_node_features, aggregated_edge_features], dim=1)

#         graph.ndata['features'] = augmented_node_features
#     return graph
        


  

    
            
  


