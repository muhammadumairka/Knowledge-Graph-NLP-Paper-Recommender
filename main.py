import pandas as pd
import networkx as nx
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("data.csv")

# Load NLP model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(df['abstract'].tolist())

# Create graph
G = nx.Graph()

for i, title in enumerate(df['title']):
    G.add_node(i, title=title)

# Add edges based on similarity
for i in range(len(embeddings)):
    for j in range(i+1, len(embeddings)):
        sim = cosine_similarity([embeddings[i]], [embeddings[j]])[0][0]
        if sim > 0.5:
            G.add_edge(i, j, weight=sim)

# Compute PageRank
pagerank = nx.pagerank(G)

# Recommendation function
def recommend(query, top_k=3):
    query_emb = model.encode([query])
    sims = cosine_similarity(query_emb, embeddings)[0]
    
    scores = []
    for i, sim in enumerate(sims):
        score = 0.7 * sim + 0.3 * pagerank.get(i, 0)
        scores.append((i, score))
    
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    
    print("\nTop Recommendations:\n")
    for idx, score in scores[:top_k]:
        print(f"{df['title'][idx]} (Score: {score:.3f})")

# Run
if __name__ == "__main__":
    query = input("Enter research topic: ")
    recommend(query)
