import gensim.downloader as api
import random

# Load pre-trained word vectors (GloVe 50D)
word_vectors = api.load("glove-wiki-gigaword-50")

def get_similar_words(seed_word, top_n=5):
    """Retrieve similar words using word embeddings."""
    try:
        similar_words = word_vectors.most_similar(seed_word, topn=top_n)
        return [word[0] for word in similar_words]
    except KeyError:
        return []  # Word not found in embeddings

def create_story(seed_word):
    """Generate a creative paragraph using the seed word and its similar words."""
    similar_words = get_similar_words(seed_word)

    if not similar_words:
        return f"Couldn't find similar words for '{seed_word}'. Try a different word."

    # Define a simple story template
    story_templates = [
        f"In a world of {similar_words[0]} and {similar_words[1]}, {seed_word} stood as a beacon of hope. "
        f"With a heart full of {similar_words[2]}, they embarked on a journey to discover the true essence of {similar_words[3]}. "
        f"The path was filled with {similar_words[4]}, but courage and resilience led them forward."
    ]

    return random.choice(story_templates)

# Example: Get user input and generate a story
seed_word = input("Enter a seed word: ").strip().lower()
story = create_story(seed_word)

print("\nGenerated Story:")
print(story)

