import os
from langchain_cohere import Cohere
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import GoogleDriveLoader


def load_text_from_google_drive(file_id):
    """
    Loads text from a Google Drive file.
    """
    loader = GoogleDriveLoader(file_id=file_id)
    documents = loader.load()
    return documents[0].page_content if documents else ""


def summarize_text(text):
    """
    Uses Cohere API to summarize the given text.
    """
    cohere_api_key = os.getenv("COHERE_API_KEY")
    if not cohere_api_key:
        raise ValueError("Cohere API key not found. Please set the  COHERE_API_KEY environment variable.")

    llm = Cohere(cohere_api_key=cohere_api_key)
    prompt_template = PromptTemplate(
        input_variables=["text"],
        template="Summarize the following text:\n{text}\n\nSummary:"
    )

    summary = llm.invoke(prompt_template.format(text=text))
    return summary


if __name__ == "__main__":
    file_id = "YOUR_GOOGLE_DRIVE_FILE_ID"
    try:
        text = load_text_from_google_drive(file_id)
        if text:
            summary = summarize_text(text)
            print("Summarized Text:")
            print(summary)
        else:
            print("Failed to load text from Google Drive.")
    except Exception as e:
        print(f"An error occurred: {e}")
