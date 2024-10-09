# Copyright (c) 2024 Braid Technologies Ltd

# Standard library imports
import os

azure = True                  

if azure:
   API_TYPE = "Azure" #AZURE VERSION WAS "Azure"
   # API_KEY = os.environ["AZURE_OPENAI_API_KEY"] #AZURE VERSION WAS os.environ["AZURE_OPENAI_API_KEY"]           #uncomment if code breaks - changes for script to exectable on both Windows and Unix machines.  
   API_KEY = os.getenv("AZURE_OPENAI_API_KEY")  # Use os.getenv() to safely retrieve environment variables       #comment if code breaks  - changes for script to exectable on both Windows and Unix machines. 
   API_VERSION = "2024-02-01" #AZURE VERSION WAS "2023-07-01-preview"
   RESOURCE_ENDPOINT = "https://braidlms.openai.azure.com/" #AZURE VERSION WAS os.environ["AZURE_OPENAI_ENDPOINT"] 
else:
   API_TYPE = "open_ai" #AZURE VERSION WAS "Azure"
   API_KEY = os.environ["OPENAI_API_KEY"] #AZURE VERSION WAS os.environ["AZURE_OPENAI_API_KEY"] 
   API_VERSION = "2020-11-07" #AZURE VERSION WAS "2023-07-01-preview"
   RESOURCE_ENDPOINT = "https://api.openai.com/v1" #AZURE VERSION WAS os.environ["AZURE_OPENAI_ENDPOINT"] 


# fetch Gemini API key from environment variables
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_SERVICE_ENDPOINT = "https://generativelanguage.googleapis.com"
# API_VERSION = "v1"  # Gemini API version


class ApiConfiguration:
    def __init__(self) -> None:
        """
        Initialises an instance of the ApiConfiguration class, setting default values for Azure OpenAI and Gemini API parameters.

        :param None: No parameters are required to create an instance of this class.

        :return: Nothing is returned by this method.
        """
        self.apiKey = API_KEY
        self.apiVersion = API_VERSION
        self.resourceChatCompletionEndpoint = "https://studiomodels.openai.azure.com/openai/deployments/StudioLarge/chat/completions?api-version=2024-06-01"  # Chat completions endpoint
        self.resourceEmbeddingEndpoint = "https://studiomodels.openai.azure.com/openai/deployments/StudioEmbeddingLarge/embeddings?api-version=2024-06-01"  # Embeddings endpoint
        self.azureDeploymentName = "StudioLarge"
        self.azureEmbedDeploymentName = "StudioEmbeddingLarge"
        self.modelName = "gpt-4"
        self.embedModelName = "text-embedding-3-large"
        self.processingThreads = 4
        self.openAiRequestTimeout = 60
        self.summaryWordCount = 50      # 50 word summary
        self.chunkDurationMins = 10     # 10 minute long video clips
        self.maxTokens = 4096           # Upper limit on total tokens in an API call. 10 minutes of video = 600 words = 2400 tokens, plus approx 2x headroom
        self.discardIfBelow = 100       # Dont index if less than 100 tokens in an article
        self.GeminiApiKey = GEMINI_API_KEY
        self.GeminiServiceEndpoint = GEMINI_SERVICE_ENDPOINT

    apiType: str
    apiKey: str
    apiVersion: str
    resourceEndpoint: str
    azureDeploymentName: str
    azureEmbedDeploymentName: str
    modelName: str
    embedModelName: str
    processingThreads: int
    openAiRequestTimeout: int
    summaryWordCount: int
    chunkDurationMins: int
    maxTokens: int
    discardIfBelow: int 
    GeminiApiKey: str
    GeminiServiceEndpoint: str