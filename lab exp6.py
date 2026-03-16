import importlib

pipeline = None
transformers_module = importlib.util.find_spec("transformers")
if transformers_module is not None:
    transformers = importlib.import_module("transformers")
    pipeline = getattr(transformers, "pipeline", None)

# Load the sentiment analysis pipeline from Hugging Face when available.
sentiment_pipeline = pipeline("sentiment-analysis") if pipeline else None

def analyze_sentiment(text):
    """Analyze sentiment of the given text using a pre-trained model."""
    if sentiment_pipeline is not None:
        result = sentiment_pipeline(text)[0]  # Get the first result
        return result  # Returns a dictionary with label and score

    # Lightweight offline fallback if transformers is not installed.
    positive_words = {
        "amazing", "love", "fantastic", "perfect", "great", "good", "excellent", "happy"
    }
    negative_words = {
        "disappointed", "slow", "bad", "worst", "awful", "hate", "poor", "terrible"
    }
    words = [w.strip(".,!?\"'").lower() for w in text.split()]
    pos_score = sum(1 for w in words if w in positive_words)
    neg_score = sum(1 for w in words if w in negative_words)

    if pos_score >= neg_score:
        confidence = 0.5 + (pos_score / max(len(words), 1))
        return {"label": "POSITIVE", "score": min(confidence, 0.99)}

    confidence = 0.5 + (neg_score / max(len(words), 1))
    return {"label": "NEGATIVE", "score": min(confidence, 0.99)}

# Example sentences for real-world application
feedbacks = [
    "The product is amazing! I love it.",
    "It was an average experience, nothing special.",
    "I'm extremely disappointed with the service.",
    "The food was okay, but the service was slow.",
    "I had a fantastic time at the hotel. Everything was perfect!"
]
for feedback in feedbacks:
    sentiment = analyze_sentiment(feedback)
    print(f"Text: {feedback}\nSentiment: {sentiment['label']}, Confidence: {sentiment['score']:.2f}\n")
