from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def summarize_text(text, max_length=150, min_length=50):
    # Use direct generation to avoid pipeline task-name differences across versions.
    model_name = "sshleifer/distilbart-cnn-12-6"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    inputs = tokenizer(text.strip(), return_tensors="pt", truncation=True, max_length=1024)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=max_length,
        min_length=min_length,
        num_beams=4,
        early_stopping=True,
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Example long passage.
    passage = """
    Artificial Intelligence (AI) is transforming industries by enabling machines to learn from data,
    adapt to new inputs, and perform human-like tasks. AI applications include natural language processing,
    computer vision, and robotics. Companies are increasingly using AI to improve efficiency, reduce costs,
    and create innovative products. However, concerns around ethics, bias, and job displacement continue
    to be topics of discussion as AI technology advances.
    """

    summary = summarize_text(passage)
    print("Summarized Text:")
    print(summary)
