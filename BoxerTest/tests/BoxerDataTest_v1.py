# Copyright (c) 2024 Braid Technologies Ltd
# Standard Library Imports
import logging
import os
import json
import sys
from logging import Logger
from typing import List, Dict, Any
import numpy as np
from numpy.linalg import norm
import datetime

# Third-Party Packages
from openai import AzureOpenAI, OpenAIError, BadRequestError, APIConnectionError
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_not_exception_type

# Add the project root and scripts directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

# Local Modules
from common.ApiConfiguration import ApiConfiguration
from common.common_functions import get_embedding

# Constants
SIMILARITY_THRESHOLD = 0.8
MAX_RETRIES = 15

OPENAI_PERSONA_PROMPT =  "You are an AI assistant helping an application developer understand generative AI. You explain complex concepts in simple language, using Python examples if it helps. You limit replies to 50 words or less. If you don't know the answer, say 'I don't know'. If the question is not related to building AI applications, Python, or Large Language Models (LLMs), say 'That doesn't seem to be about AI'."
ENRICHMENT_PROMPT = "You will be provided with a question about building applications that use generative AI technology. Write a 50 word summary of an article that would be a great answer to the question. Consider enriching the question with additional topics that the question asker might want to understand. Write the summary in the present tense, as though the article exists. If the question is not related to building AI applications, Python, or Large Language Models (LLMs), say 'That doesn't seem to be about AI'.\n"
FOLLOW_UP_PROMPT =  "You will be provided with a summary of an article about building applications that use generative AI technology. Write a question of no more than 10 words that a reader might ask as a follow up to reading the article."

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to configure the Azure OpenAI API client
def configure_openai_for_azure(config: ApiConfiguration) -> AzureOpenAI:
    """
    Configures OpenAI for Azure using the provided ApiConfiguration.

    Args:
        config (ApiConfiguration): The ApiConfiguration object containing the necessary settings.

    Returns:
        AzureOpenAI: An instance of AzureOpenAI configured with the provided settings.
    """
    return AzureOpenAI(
        azure_endpoint=config.resourceEndpoint, 
        api_key=config.apiKey.strip(),
        api_version=config.apiVersion
    )

# Class to hold test results
class TestResult:
    def __init__(self) -> None:
        """
        Initializes a new instance of the TestResult class.
        
        Sets the initial state of the test result, including the question, enriched question, 
        hit status, hit relevance, hit summary, follow-up question, and follow-up topic.
        
        Args:
            None
        
        Returns:
            None
        """
        self.question: str = ""
        self.enriched_question_summary: str = ""
        self.hit: bool = False
        self.hit_relevance: float = 0.0
        self.follow_up: str = ""
        self.follow_up_on_topic: str = ""

# Function to call the OpenAI API with retry logic
@retry(wait=wait_random_exponential(min=5, max=15), stop=stop_after_attempt(MAX_RETRIES), retry=retry_if_not_exception_type(BadRequestError))
def call_openai_chat(client: AzureOpenAI, messages: List[Dict[str, str]], config: ApiConfiguration, logger: logging.Logger) -> str:
    """
    Retries the OpenAI chat API call with exponential backoff and retry logic.

    :param client: An instance of the AzureOpenAI class.
    :type client: AzureOpenAI
    :param messages: A list of dictionaries representing the messages to be sent to the API.
    :type messages: List[Dict[str, str]]
    :param config: An instance of the ApiConfiguration class.
    :type config: ApiConfiguration
    :param logger: An instance of the logging.Logger class.
    :type logger: logging.Logger
    :return: The content of the first choice in the API response.
    :rtype: str
    :raises RuntimeError: If the finish reason in the API response is not 'stop', 'length', or an empty string.
    :raises OpenAIError: If there is an error with the OpenAI API.
    :raises APIConnectionError: If there is an error with the API connection.
    """
    try:
        response = client.chat.completions.create(
            model=config.azureDeploymentName,
            messages=messages,
            temperature=0.7,
            max_tokens=config.maxTokens,
            top_p=0.0,
            frequency_penalty=0,
            presence_penalty=0,
            timeout=config.openAiRequestTimeout,
        )
        content = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason

        if finish_reason not in {"stop", "length", ""}:
            logger.warning("Unexpected stop reason: %s", finish_reason)
            logger.warning("Content: %s", content)
            logger.warning("Consider increasing max tokens and retrying.")
            raise RuntimeError("Unexpected finish reason in API response.")

        return content

    except (OpenAIError, APIConnectionError) as e:
        logger.error(f"Error: {e}")
        raise

# Function to retrieve text embeddings using OpenAI API with retry logic
@retry(wait=wait_random_exponential(min=5, max=15), stop=stop_after_attempt(MAX_RETRIES), retry=retry_if_not_exception_type(BadRequestError))
def get_text_embedding(client: AzureOpenAI, config: ApiConfiguration, text: str, logger: Logger) -> np.ndarray:
    """
    Retrieves the text embedding for a given text using the OpenAI API.

    Args:
        client (AzureOpenAI): The OpenAI client instance.
        config (ApiConfiguration): The API configuration instance.
        text (str): The text for which to retrieve the embedding.
        logger (Logger): The logger instance.

    Returns:
        np.ndarray: The text embedding as a numpy array.

    Raises:
        OpenAIError: If an error occurs while retrieving the text embedding.
    """
    try:
        embedding = get_embedding(text, client, config)
        return np.array(embedding)
    except OpenAIError as e:
        logger.error(f"Error getting text embedding: {e}")
        raise

# Function to calculate cosine similarity between two vectors
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Calculates the cosine similarity between two vectors.

    Args:
        a (np.ndarray): The first vector.
        b (np.ndarray): The second vector.

    Returns:
        float: The cosine similarity between the two vectors.

    Raises:
        ValueError: If the input vectors are not numpy arrays or convertible to numpy arrays.
        ValueError: If the input vectors do not have the same shape.
        ValueError: If either of the input vectors is a zero vector.
    """
    try:
        a, b = np.array(a), np.array(b)
    except Exception:
        raise ValueError("Input vectors must be numpy arrays or convertible to numpy arrays")

    if a.shape != b.shape:
        raise ValueError("Input vectors must have the same shape")

    dot_product = np.dot(a, b)
    a_norm, b_norm = norm(a), norm(b)

    if a_norm == 0 or b_norm == 0:
        raise ValueError("Input vectors must not be zero vectors")

    return dot_product / (a_norm * b_norm)

# Function to generate enriched questions using OpenAI API
@retry(wait=wait_random_exponential(min=5, max=15), stop=stop_after_attempt(MAX_RETRIES), retry=retry_if_not_exception_type(BadRequestError))
def generate_enriched_question(client: AzureOpenAI, config: ApiConfiguration, question: str, logger: logging.Logger) -> str:
    """
    Generates an enriched question using the OpenAI API.

    Args:
        client (AzureOpenAI): The OpenAI client instance.
        config (ApiConfiguration): The API configuration instance.
        question (str): The question to be enriched.
        logger (logging.Logger): The logger instance.

    Returns:
        str: The enriched question.

    Raises:
        BadRequestError: If the API request fails.
    """
    messages = [
        {"role": "system", "content": OPENAI_PERSONA_PROMPT},
        {"role": "user", "content": ENRICHMENT_PROMPT + "Question: " + question},
    ]
    logger.info("Making API request to OpenAI...")
    logger.info("Request payload: %s", messages)

    response = call_openai_chat(client, messages, config, logger)
    logger.info("API response received: %s", response)

    return response

def process_questions(client: AzureOpenAI, config: ApiConfiguration, questions: List[str], processed_question_chunks: List[Dict[str, Any]], logger: logging.Logger) -> List[TestResult]:
    """
    Processes a list of test questions and evaluates their relevance based on their similarity to pre-processed question chunks.

    Args:
        client (AzureOpenAI): The OpenAI client instance.
        config (ApiConfiguration): The API configuration instance.
        questions (List[str]): The list of test questions to be processed.
        processed_question_chunks (List[Dict[str, Any]]): The list of pre-processed question chunks.
        logger (logging.Logger): The logger instance.

    Returns:
        List[TestResult]: A list of test results, each containing the original question, its enriched version, and its relevance to the pre-processed chunks.
    """
    question_results: List[TestResult] = []
    
    for question in questions:
        question_result = TestResult()
        question_result.question = question
        question_result.enriched_question_summary = generate_enriched_question(client, config, question, logger)
        embedding = get_text_embedding(client, config, question_result.enriched_question_summary, logger)

        best_hit_relevance = 0  # To track the highest similarity score
        best_hit_summary = None  # To track the summary corresponding to the highest similarity

        for chunk in processed_question_chunks:
            if chunk and isinstance(chunk, dict) :  
                ada_embedding = chunk.get("ada_v2")
                similarity = cosine_similarity(ada_embedding, embedding)

                if similarity > SIMILARITY_THRESHOLD:
                    question_result.hit = True

                # Check if this is the best match so far
                if similarity > best_hit_relevance:
                    best_hit_relevance = similarity
                    best_hit_summary = chunk.get("summary")

        # Set the best hit relevance and summary for the question result
        question_result.hit_relevance = best_hit_relevance
        question_result.hit_summary = best_hit_summary

        question_results.append(question_result)

    logger.debug("Total tests processed: %s", len(question_results))
    return question_results


# Function to read processed chunks from the source directory
def read_processed_chunks(source_dir: str) -> List[Dict[str, Any]]:
    """
    Reads and processes JSON files from a specified source directory.

    Args:
        source_dir (str): The path to the source directory containing JSON files.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries containing the processed JSON data.

    Raises:
        FileNotFoundError: If the source directory or a JSON file is not found.
        IOError: If an I/O error occurs while reading a JSON file.
    """
    processed_question_chunks: List[Dict[str, Any]] = []
    try:
        for filename in os.listdir(source_dir):
            if filename.endswith(".json"):
                file_path = os.path.join(source_dir, filename)
                with open(file_path, "r", encoding="utf-8") as f:
                    chunk = json.load(f)
                    processed_question_chunks = chunk
    except (FileNotFoundError, IOError) as e:
        logger.error(f"Error reading files: {e}")
        raise
    return processed_question_chunks

def save_results(test_destination_dir: str, question_results: List[TestResult]) -> None:
    """
    Saves the test results to a JSON file in the specified destination directory.

    Args:
        test_destination_dir (str): The path to the directory where the test results will be saved.
        question_results (List[TestResult]): A list of TestResult objects containing the test results.

    Returns:
        None
    """
    # Define the output structure with the specified columns
    output_results = [
        {
            "question": result.question,
            "enriched_question": result.enriched_question_summary, 
            "hit": result.hit,
            "summary": result.hit_summary, 
            "hitRelevance": result.hit_relevance,  
        }
        for result in question_results
    ]

    # Create a timestamp for the output file name
    current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_file = os.path.join(test_destination_dir, f"test_output_v1_{current_datetime}.json")

    try:
        # Open the output file in write mode, using utf-8 encoding and create it if it doesn't exist
        with open(output_file, "w", encoding="utf-8") as f:
            # Write the output results to the file in JSON format 
            json.dump(output_results, f, indent=4)  # Save the results to the file 
        logger.info(f"Test results saved to: {output_file}")
    except IOError as e:
        logger.error(f"Error saving results: {e}")
        raise


# Main function to run tests
def run_tests(config: ApiConfiguration, test_destination_dir: str, source_dir: str, questions: List[str]) -> None:
    """
    Runs tests using the provided configuration, test destination directory, source directory, and questions.

    Args:
        config (ApiConfiguration): The configuration for the API.
        test_destination_dir (str): The directory where the test results will be saved.
        source_dir (str): The directory containing the source files.
        questions (List[str]): A list of questions to be processed.

    Returns:
        None
    """
    client = configure_openai_for_azure(config)

    if not test_destination_dir:
        logger.error("Test data folder not provided")
        raise ValueError("Test destination directory not provided")

    processed_question_chunks = read_processed_chunks(source_dir)
    question_results = process_questions(client, config, questions, processed_question_chunks, logger)
    save_results(test_destination_dir, question_results)
