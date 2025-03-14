# --- Imports and Dependencies ---
import configparser
import logging
import os
import prompts  # Custom prompts module containing predefined prompts

from alive_progress import alive_bar
from azure.storage.blob import BlobServiceClient
from openai import AzureOpenAI


# --- Logging Configuration ---
# Create a named logger for tracking script execution
logger = logging.getLogger(__name__)

# Set up logging configuration with timestamp, level, and message format
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# --- Configuration Setup ---
# Load configuration settings from config.prop file
config = configparser.ConfigParser()
config.read("config.prop")
azure_llm_config = config["azure_openai_gpt4o-mini"]
azure_blob_config = config["azure_blob"]
azure_assistant_config = config["azure_assistant"]

# Set required Azure OpenAI environment variables
os.environ["AZURE_OPENAI_ENDPOINT"] = azure_llm_config["endpoint"]
os.environ["AZURE_OPENAI_API_KEY"] = azure_llm_config["api_key"]

# --- Configuration Variables ---
# Store Azure-specific configuration values
CONNECTION_STRING = azure_blob_config["connection_string"]
ASSISTANT1_NAME = azure_assistant_config["assistant1_name"]
ASSISTANT2_NAME = azure_assistant_config["assistant2_name"]
ASSISTANT1_PROMPT = prompts.ASSISTANT1_PROMPT
ASSISTANT2_PROMPT = prompts.ASSISTANT2_PROMPT
API_VERSION = azure_llm_config["api_version"]
MODEL_NAME = azure_llm_config["deployment"]
TEMPERATURE = float(azure_llm_config["temperature"])
SCORE_THRESHOLD = float(azure_assistant_config["score_threshold"]) # for ranking options
RANKER = azure_assistant_config["ranker"]

# --- Initialize Azure OpenAI Client ---
client = AzureOpenAI(
    azure_endpoint=azure_llm_config["endpoint"],
    api_key=azure_llm_config["api_key"],
    api_version=API_VERSION
)

# --- Create and Configure Assistant ---
def make_assistant(ASSISTANT_NAME, ASSISTANT_PROMPT, file_search=True):
    # Initialize the Azure OpenAI assistant with specified parameters
    if file_search:
        assistant = client.beta.assistants.create(
            name=ASSISTANT_NAME,
            instructions=ASSISTANT_PROMPT,
            model=MODEL_NAME,
            tools=[{"type":"file_search","file_search":{"ranking_options":{"ranker":RANKER,"score_threshold":SCORE_THRESHOLD}}}],
            tool_resources={"file_search":{"vector_store_ids":[]}},
            temperature=TEMPERATURE,
            top_p=1
        )
    else:
        assistant = client.beta.assistants.create(
            name=ASSISTANT_NAME,
            instructions=ASSISTANT_PROMPT,
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            top_p=1
        )

    # Log assistant details
    logger.info(f"Configured Assistant Details:")
    logger.info(f"ID: {assistant.id}")
    logger.info(f"Name: {assistant.name}")
    logger.info(f"Model: {assistant.model}")


if __name__ == "__main__":
    # Create assistant 1
    logger.info(f"Creating Assistant 1: {ASSISTANT1_NAME}")
    make_assistant(ASSISTANT1_NAME, ASSISTANT1_PROMPT, file_search=True)

    # Create assistant 2
    logger.info(f"Creating Assistant 2: {ASSISTANT2_NAME}")
    make_assistant(ASSISTANT2_NAME, ASSISTANT2_PROMPT, file_search=False)