# Standard library imports
import os
from openai import AzureOpenAI

from common.ApiConfiguration import ApiConfiguration

config = ApiConfiguration()

def ensure_directory_exists(directory):
    """
    Checks if the directory at the given destination exists.
    If it does not exist, creates the directory.

    Parameters:
    directory (str): The path to the directory.
    """
    # Use os.path.join() to handle path construction across different platforms
    if not os.path.exists(directory):
        os.makedirs(directory)
        #print(f"Directory '{directory}' created.")  #  can remove or comment out this print statement for production
    else:
        # print(f"Directory '{directory}' already exists.")
        pass

# Construct the path using os.path.join() for cross-platform compatibility
HTML_DESTINATION_DIR = os.path.join("data", "web")
ensure_directory_exists(HTML_DESTINATION_DIR)

def get_embedding(text: str, embedding_client: AzureOpenAI, config: ApiConfiguration, model: str = "text-embedding-3-large"):
    # Replace newlines with spaces 
    text = text.replace("\n", " ")

    # Use the provided model parameter if given, otherwise fall back to config's deployment name
    chosen_model = model if model else config.embedModelName

    # Generate embedding using the chosen model and configuration
    response = embedding_client.embeddings.create(
        input=[text],
        model=chosen_model,
        timeout=config.openAiRequestTimeout
    )
    
    return response.data[0].embedding
