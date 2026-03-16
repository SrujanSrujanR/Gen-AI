import gensim
from gensim.models import Word2Vec
from gensim.parsing.preprocessing import STOPWORDS
import string
import re
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Sample corpus: A small example legal text corpus (replace this with your own data)
corpus = [
    "The plaintiff has filed a lawsuit against the defendant for breach of contract.",
    "The court will hear arguments on the motion for summary judgment.",
    "The lawyer presented evidence that supported the claim of negligence.",
    "The defendant's counsel argued that the case should be dismissed on procedural grounds.",
    "The judge ruled in favor of the plaintiff, awarding damages."
]

# Preprocess the corpus: tokenization, removing stop words and punctuation
def preprocess_text(text):
    stop_words = set(STOPWORDS)
    # Tokenize the text into lowercase alphabetic words.
    tokens = re.findall(r"[a-zA-Z]+", text.lower())
    # Remove stop words and punctuation
    tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
    return tokens

# Preprocess the entire corpus
processed_corpus = [preprocess_text(doc) for doc in corpus]

# Train Word2Vec model
print("Training Word2Vec model...")
model = Word2Vec(sentences=processed_corpus, 
vector_size=100, window=5, min_count=1, sg=0, workers=4)

# Save the trained model (optional)
model.save("legal_word2vec.model")

# Explore the trained embeddings: Find the most similar words to a given word
def find_similar_words(word, top_n=5):
    try:
        similar_words = model.wv.most_similar(word, topn=top_n)
        print(f"Most similar words to '{word}':")
        for similar_word, similarity in similar_words:
            print(f"{similar_word}: {similarity}")
    except KeyError:
        print(f"Word '{word}' not found in the vocabulary.")

# Example: Find similar words to 'plaintiff'
find_similar_words('plaintiff')

# Example: Find similar words to 'defendant'
find_similar_words('defendant')
# Extract vectors and corresponding words
words = list(model.wv.index_to_key)
vectors = [model.wv[word] for word in words]

# Reduce dimensionality to 2D using PCA
pca = PCA(n_components=2)
pca_result = pca.fit_transform(vectors)

# Plot the embeddings in 2D
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1])

# Annotate the words
for i, word in enumerate(words):
    plt.text(pca_result[i, 0] + 0.01, pca_result[i, 1] + 0.01, word, fontsize=12)

plt.title("2D Visualization of Word Embeddings (Legal Corpus)")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.grid(True)
plt.show()
