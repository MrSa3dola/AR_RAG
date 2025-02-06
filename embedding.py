from sentence_transformers import SentenceTransformer, util

sentences = [
    "Find a bed that matches a room with a coffee color",
    "a bed with a green upholstered headboard and foot board, two pillows, and a blanket",
    "a yellow bed with wooden legs and a wooden headboard",
    "a round dining table with a black marble top and a metal base",
]

model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embeddings = model.encode(sentences)
print(embeddings.shape)
similarity_matrix = util.cos_sim(embeddings, embeddings)

# Find most similar pairs
most_similar_indices = []
for i in range(len(sentences)):
    similarities = similarity_matrix[i].clone()  # Work on a copy
    similarities[i] = -1  # Ignore self-similarity
    most_similar_idx = similarities.argmax().item()  # Convert to Python int
    most_similar_score = similarities[
        most_similar_idx
    ].item()  # Convert score to Python float
    most_similar_indices.append((i, most_similar_idx, most_similar_score))

print("Most similar pairs:")
for pair in most_similar_indices:
    i, j, score = pair
    print(f"Index {i} -> Index {j} (Score: {score:.3f})")

# For the query (index 0)
query_similarities = similarity_matrix[0].clone()
query_similarities[0] = -1  # Ignore query itself
sorted_indices = query_similarities.argsort(descending=True)

print("\nTop matches for the query (Index 0):")
for idx in sorted_indices:
    if idx == 0:  # Skip query itself
        continue
    score = query_similarities[idx].item()
    print(f"- Index {idx}: {sentences[idx]} (Score: {score:.3f})")
