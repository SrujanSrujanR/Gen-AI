import gensim.downloader as api
import numpy as np
import matplotlib.pyplot as plt

# Load pre-trained Word2Vec model
# (This is a large model, first run will take time)
model = api.load("word2vec-google-news-300")

# Example words
words = ["king", "man", "woman", "queen"]

# Get vectors for visualization
vectors = [model[word] for word in words]

# Word vector arithmetic: king - man + woman ≈ queen
king = model["king"]
man = model["man"]
woman = model["woman"]

result_vector = king - man + woman

# Find most similar words
similar_words = model.most_similar(positive=[result_vector], topn=5)

print("Most similar words to 'king - man + woman':")
for word, similarity in similar_words:
    print(f"{word}: {similarity}")

# -----------------------------
# Visualization using PCA (NumPy SVD)
# -----------------------------
word_matrix = np.array(vectors)
word_matrix -= word_matrix.mean(axis=0)
_, _, vt = np.linalg.svd(word_matrix, full_matrices=False)
reduced_vectors = word_matrix @ vt[:2].T

plt.figure(figsize=(6, 6))
plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1])

# Add labels
for i, word in enumerate(words):
    plt.text(
        reduced_vectors[i, 0] + 0.02,
        reduced_vectors[i, 1] + 0.02,
        word,
        fontsize=12
    )

plt.title("2D Visualization of Word Relationships")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()
