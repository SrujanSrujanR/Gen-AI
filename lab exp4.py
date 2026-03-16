import gensim.downloader as api
import random

# Load pre-trained word vectors
word_vectors = api.load("glove-wiki-gigaword-50")  # 50D GloVe vectors

def get_similar_words(word, top_n=3):
    """Retrieve similar words using word embeddings"""
    try:
        similar_words = word_vectors.most_similar(word, topn=top_n)
        return [w[0] for w in similar_words]
    except KeyError:
        return []  # Word not found in embeddings

def enrich_prompt(prompt):
    """Replace key words in the prompt with similar words to enhance richness"""
    words = prompt.split()
    enriched_words = []

    for word in words:
        similar_words = get_similar_words(word)
        if similar_words:
            enriched_words.append(random.choice(similar_words))  # Randomly pick a synonym
        else:
            enriched_words.append(word)

    return " ".join(enriched_words)

def generate_response(prompt):
    """Generate a short local story response without external APIs."""
    words = [w.strip(".,!?\"'").lower() for w in prompt.split()]
    nouns = [w for w in words if len(w) > 3]

    hero = random.choice(["warrior", "guardian", "knight", "hunter"])
    place = random.choice(["ancient valley", "misty mountain", "forgotten citadel", "burning plain"])
    conflict = random.choice(["shadow dragon", "fire serpent", "storm beast", "iron wyrm"])

    if nouns:
        focus = random.choice(nouns)
    else:
        focus = "courage"

    return (
        f"At dawn in the {place}, a {hero} followed the trail of {focus}. "
        f"By nightfall, the {hero} confronted a {conflict} whose roar shook the sky. "
        f"With patience and nerve, the {hero} turned fear into strategy, "
        f"struck at the right moment, and restored peace to the realm."
    )

# Example prompt
original_prompt = "Write a short story about a brave warrior fighting a dragon."

# Enrich the prompt
enriched_prompt = enrich_prompt(original_prompt)

# Generate responses
original_response = generate_response(original_prompt)
enriched_response = generate_response(enriched_prompt)

# Print results
print("Original Prompt: ", original_prompt)
print("Enriched Prompt: ", enriched_prompt)
print("\nGenerated Response for Original Prompt:\n", original_response)
print("\nGenerated Response for Enriched Prompt:\n", enriched_response)
