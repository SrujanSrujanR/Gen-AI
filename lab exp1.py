import gensim.downloader as api
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load pre-trained Word2Vec model
model = api.load('word2vec-google-news-300')

# Example word relationships using vector arithmetic
words = ['king', 'man', 'woman', 'queen']
vectors = [model[word] for word in words]

# Performing word vector arithmetic: king - man + woman = queen
king = model['king']
man = model['man']
woman = model['woman']
queen = king - man + woman

# Find the most similar words to the resulting vector (queen)
similar_words = model.similar_by_vector(queen, topn=5)
print("Most similar words to 'queen' (result of king - man + woman):")
for word, similarity in similar_words:
    print(f"{word}: {similarity}")

# Visualizing the word vectors
# Reduce the dimensionality to 2D using PCA
pca = PCA(n_components=2)
result = pca.fit_transform(vectors)

# Plotting the result
plt.figure(figsize=(6, 6))
plt.scatter(result[:, 0], result[:, 1])

# Adding text annotations for each word
for i, word in enumerate(words):
    plt.text(result[i, 0] + 0.05, result[i, 1] + 0.05, word, fontsize=12)

plt.title("2D Visualization of Word Relationships (king, man, woman, queen)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()
