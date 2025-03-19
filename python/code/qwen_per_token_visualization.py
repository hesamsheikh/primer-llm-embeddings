import torch.nn as nn
import torch
from transformers import AutoTokenizer
import networkx as nx
import plotly.graph_objects as go

def find_similar_embeddings(target_embedding, n=10):
    """
    Find the n most similar embeddings to the target embedding using cosine similarity
    
    Args:
        target_embedding: The embedding vector to compare against
        n: Number of similar embeddings to return (default 3)
    
    Returns:
        List of tuples containing (word, similarity_score) sorted by similarity
    """
    # Convert target to tensor if not already
    if not isinstance(target_embedding, torch.Tensor):
        target_embedding = torch.tensor(target_embedding)
        
    # Get all embeddings from the model
    all_embeddings = model.embedding.weight
    
    # Compute cosine similarity between target and all embeddings
    similarities = torch.nn.functional.cosine_similarity(
        target_embedding.unsqueeze(0), 
        all_embeddings
    )
    
    # Get top n similar embeddings
    top_n_similarities, top_n_indices = torch.topk(similarities, n)
    
    # Convert to word-similarity pairs
    results = []
    for idx, score in zip(top_n_indices, top_n_similarities):
        word = tokenizer.decode(idx)
        results.append((word, score.item()))
        
    return results

def prompt_to_embeddings(prompt:str):
    # tokenize the input text
    tokens = tokenizer(prompt, return_tensors="pt")
    input_ids = tokens['input_ids']

    # make a forward pass
    outputs = model(input_ids)

    # directly use the embeddings layer to get embeddings for the input_ids
    embeddings = outputs

    # print each token
    token_id_list = tokenizer.encode(prompt, add_special_tokens=True)
    token_str = [tokenizer.decode(t_id, skip_special_tokens=True) for t_id in token_id_list]

    return token_id_list, embeddings, token_str

class EmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(EmbeddingModel, self).__init__()
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

    def forward(self, input_ids):
        return self.embedding(input_ids)
    

vocab_size = 151936
dimensions = 1536
embeddings_filename = r"python\code\files\embeddings_qwen.pth"
tokenizer_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

# Initialize the custom embedding model
model = EmbeddingModel(vocab_size, dimensions)

# Load the saved embeddings from the file
saved_embeddings = torch.load(embeddings_filename)

# Ensure the 'weight' key exists in the saved embeddings dictionary
if 'weight' not in saved_embeddings:
    raise KeyError("The saved embeddings file does not contain 'weight' key.")

embeddings_tensor = saved_embeddings['weight']

# Check if the dimensions match
if embeddings_tensor.size() != (vocab_size, dimensions):
    raise ValueError(f"The dimensions of the loaded embeddings do not match the model's expected dimensions ({vocab_size}, {dimensions}).")

# Assign the extracted embeddings tensor to the model's embedding layer
model.embedding.weight.data = embeddings_tensor

# put the model in eval mode
model.eval()

token_id_list, prompt_embeddings, prompt_token_str = prompt_to_embeddings("""We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely""")

tokens_and_neighbors = {}
for i in range(1, len(prompt_embeddings[0])):
    token_results = find_similar_embeddings(prompt_embeddings[0][i], n=20)
    similar_embs = []
    for word, score in token_results:
        if word.strip().lower() != prompt_token_str[i].strip().lower():
            similar_embs.append(word)
    tokens_and_neighbors[prompt_token_str[i]] = similar_embs

all_token_embeddings = {}

# Process each token and its neighbors
for token, neighbors in tokens_and_neighbors.items():
    # Get embedding for the original token
    token_id, token_emb, _ = prompt_to_embeddings(token)
    all_token_embeddings[token] = token_emb[0][1]
    
    # Get embeddings for each neighbor token
    for neighbor in neighbors:
        # Get embedding
        neighbor_id, neighbor_emb, _ = prompt_to_embeddings(neighbor)
        all_token_embeddings[neighbor] = neighbor_emb[0][1]

# Create the graph
G = nx.Graph()

# Add edges from tokens to their neighbors
for token, neighbors in tokens_and_neighbors.items():
    for neighbor in neighbors:
        G.add_edge(token, neighbor)

# Generate positions using force-directed layout
pos = nx.spring_layout(G, k=0.3, iterations=100)

# pos = nx.circular_layout(G)
# pos = nx.kamada_kawai_layout(G)

# Define visualization dimensions
viz_width = 300
viz_height = 100

# Extract edge coordinates and scale them
edge_x, edge_y = [], []
for edge in G.edges():
    x0, y0 = pos[edge[0]]
    x1, y1 = pos[edge[1]]
    # Scale coordinates to fill the width/height
    x0, x1 = x0 * viz_width, x1 * viz_width  # Scale x coordinates to width
    y0, y1 = y0 * viz_height, y1 * viz_height    # Scale y coordinates to height
    edge_x.extend([x0, x1, None])
    edge_y.extend([y0, y1, None])


# Node coordinates and data - scale the positions
node_x = [pos[node][0] * viz_width for node in G.nodes()]  # Scale x to width
node_y = [pos[node][1] * viz_height for node in G.nodes()]   # Scale y to height
node_degrees = dict(G.degree())
node_sizes = [(degree + 1) * 1 for degree in node_degrees.values()]

# Node trace with custom colorscale
node_trace = go.Scatter(
    x=node_x, y=node_y,
    mode='markers+text',
    text=[node for node in G.nodes()],
    textposition="top center",
    marker=dict(
        size=node_sizes,
        color=list(node_degrees.values()),
        colorscale='Viridis',
        opacity=0.8,
        line_width=0.5
    ),
    customdata=[[f"Connections: {deg}"] for deg in node_degrees.values()],
    hovertemplate="<b>%{text}</b><br>%{customdata[0]}<extra></extra>",
    hoverlabel=dict(namelength=0)
)

# Edge trace (thin lines)
edge_trace = go.Scatter(
    x=edge_x, y=edge_y,
    line=dict(width=0.3, color='rgba(150,150,150,0.6)'),
    hoverinfo='none',
    mode='lines'
)

# Set up Plotly figure
fig = go.Figure(data=[edge_trace, node_trace],
                layout=go.Layout(
                    width=1200,
                    height=400,
                    paper_bgcolor='white',
                    plot_bgcolor='white',
                    showlegend=False,
                    margin=dict(l=0, r=0, t=0, b=0),
                    xaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False,
                    ),
                    yaxis=dict(
                        showgrid=False,
                        zeroline=False,
                        showticklabels=False,
                        scaleanchor="x",
                        scaleratio=1
                    )
                ))

fig.show()

fig.write_html(r"src\fragments\token_visualization.html",
               include_plotlyjs=False,
               full_html=False,
               config={
                   'displayModeBar': False,
                   'responsive': True,
                   'scrollZoom': False,
               })

...