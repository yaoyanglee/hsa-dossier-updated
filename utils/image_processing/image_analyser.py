import base64
import configparser
import cv2
import hashlib
import logging
import numpy as np
import os
import pandas as pd
import time

from alive_progress import alive_bar
from azure.data.tables import TableServiceClient
from azure.core.credentials import AzureNamedKeyCredential
from io import BytesIO
from langchain_community.callbacks import get_openai_callback
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from openai import AzureOpenAI
from requests.exceptions import RequestException
from typing import List
from unstructured.partition.pdf import partition_pdf
from utils.table import azure_table_client
import configparser


# -- Constants for image quality checks --
MIN_FILE_SIZE_KB = 10
MIN_FILE_SIZE_BYTES = MIN_FILE_SIZE_KB * 1024
MIN_RESOLUTION = (150, 500)  # (height, width)
BLUR_THRESHOLD = 1500
BRIGHTNESS_RANGE = (50, 500)  # Min and max mean brightness
CONTRAST_THRESHOLD = 50      # Intensity percentile difference

# Azure HSA Store
config = configparser.ConfigParser()
# config_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.prop")
# config.read(config_path)
config.read("config.prop")
azure_hsa_store_config = config["azure_hsa_store"]
account_name = azure_hsa_store_config["account_name"]
account_key = azure_hsa_store_config["account_key"]

credential = AzureNamedKeyCredential(account_name, account_key)
endpoint = f"https://{account_name}.table.core.windows.net"

table_service_client = TableServiceClient(
    endpoint=endpoint, credential=credential)
table_name = "docmap"


# --- Configure logger ---
# Create a named logger
logger = logging.getLogger(__name__)

# Configure the logger
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# Load configuration
azure_llm_config = config["azure_openai_gpt4o-mini"]

# Set environment variables
os.environ["AZURE_OPENAI_ENDPOINT"] = azure_llm_config["endpoint"]
os.environ["AZURE_OPENAI_API_KEY"] = azure_llm_config["api_key"]

# Initialize the AzureChatOpenAI instance
llm = AzureChatOpenAI(
    azure_deployment=azure_llm_config["deployment"],
    api_version=azure_llm_config["api_version"],
    temperature=float(azure_llm_config["temperature"]),
    max_retries=int(azure_llm_config["max_retries"]),
)

# Message template
MESSAGE_TEMPLATE = [
    {
        "role": "system",
        "content": """You are an AI content specialist tasked with analyzing and fully describing images into precise, technical text representations. Your audience cannot see the image and relies entirely on your explanation. While the description must be derived purely from the image content, you are also provided with a context to guide your interpretation. Use the context only to focus and refine the explanation, but do not include assumptions or external information beyond what is explicitly present in the image.

Key Guidelines:

1. Content Analysis:
   - Focus on technical accuracy and comprehensive detail.
   - Provide a natural and logical flow of information.
   - Describe the main title, key elements, and relationships between components, processes, or data.
   - Maintain a tone appropriate for a professional technical report.

2. Content Types and Specific Requirements:
   a) Process Flows:
      - Explain sequences, relationships, and outcomes.
      - Highlight causes, effects, and decision points.
   b) Numerical Data:
      - Include all numbers, comparisons, and patterns.
      - Explain their significance and trends in context.
   c) Technical Diagrams:
      - Describe all components and their relationships.
      - Clarify dependencies, interactions, and system functions.
   d) Tables:
      - Detailed Description: Clearly articulate the purpose and context of the table, based on its title or header content (if available).
      - Structural Clarity:
         i) Define all columns, rows, sub-columns, and sub-rows, including any headers or categories.
         ii) Identify groupings, hierarchies, or subdivisions within the table.
      - Data Interpretation:
         i) Explain the meaning of the data, emphasizing patterns, relationships, and key comparisons.
         ii) Highlight any calculated values, trends, or outliers explicitly shown in the table.
      - Cross-References: If the table links to other elements (e.g., diagrams or captions), ensure these relationships are clearly described.

3. Context Usage:
   - Use the provided context to guide your analysis by understanding the focus or importance of certain elements.
   - Ensure that the context is used only as a reference and does not replace or overshadow the actual details derived from the image.
   - Highlight any elements in the image that are particularly relevant to the provided context.

4. Prohibited Descriptions:
   - Avoid format references (e.g., "This diagram shows…").
   - Exclude layout or positional cues (e.g., "top left corner").
   - Refrain from visual references unrelated to technical meaning.

5. Mandatory Details:
   - Use precise technical terminology and provide comprehensive descriptions.
   - Include relevant measurements, quantities, and identifiers.
   - Accurately describe relationships, dependencies, and interactions among elements.
   - Ensure the description is exhaustive, addressing every significant detail.

Expected Output Format:
- Structure the output logically, beginning with the title (if present).
- Provide a detailed technical description derived solely from the image content.
- Maintain a professional tone, emphasizing clarity and precision.

Critical Reminder:
You must rely entirely on the content of the image for your description. The provided context is a supplementary guide to help refine focus but should not introduce additional assumptions or meanings not explicitly present in the image. Avoid:
- Adding context or meaning that is not explicitly present.
- Summarizing details in ways that omit meaningful information.
- Speculating or using external knowledge.

Your goal is to produce an exhaustive, accurate text description as if the reader's understanding depends solely on your explanation. Double-check all relationships, values, and connections to ensure no omission or misrepresentation.
"""
    },
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Analyze the content of this image and provide a complete and precise text representation. Include all meaningful details while maintaining logical flow and technical accuracy."
            },
            {
                "type": "image_url",
                "image_url": None,  # Placeholder to be updated with actual base64 image during the iteration
                "detail": "high"
            },
            {
                "type": "text",
                "text": None  # Placeholder for context of image
            },
        ]
    }
]

# --- Helper Functions ---


def encode_image(image_path):
    """
    Encodes an image file to a base64 string.

    Parameters:
        image_path (str): Path to the image file to be encoded.

    Returns:
        str: The base64-encoded string representation of the image.
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def hash_document_name(doc_name):
    """
    Generates a SHA-256 hash for the document name and truncates it to 8 characters for brevity.

    Parameters:
        doc_name (str): The document name to be hashed.

    Returns:
        str: The first 8 characters of the SHA-256 hash of the document name.
    """
    return hashlib.sha256(doc_name.encode('utf-8')).hexdigest()[:8]


def check_image_quality(filepath):
    """
    Checks the quality of an image based on specific rules (file size, resolution, and sharpness).

    Parameters:
        filepath (str): The path to the image file to be checked.

    Returns:
        bool: True if the image meets the quality criteria, False otherwise.
              - False if the image is invalid, too small, low resolution, or blurry.
    """
    try:
        img = cv2.imread(filepath)
        if img is None:
            logger.warning(f"Invalid image: {filepath}")
            return False

        # Rule 1: File size
        file_size_bytes = os.path.getsize(filepath)
        if file_size_bytes < MIN_FILE_SIZE_BYTES:
            logger.warning(f"File size too small: {filepath}")
            return False

        # Rule 2: Resolution
        height, width = img.shape[:2]
        if height < MIN_RESOLUTION[0] or width < MIN_RESOLUTION[1]:
            logger.warning(f"Low resolution: {filepath}")
            return False

        # Rule 3: Sharpness
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if cv2.Laplacian(gray_img, cv2.CV_64F).var() < BLUR_THRESHOLD:
            logger.warning(f"Image too blurry: {filepath}")
            return False

        return True
    except Exception as e:
        logger.error(f"Error checking quality of {filepath}: {e}")
        return False


def get_verified_image_files(image_folder):
    """
    Retrieves a list of verified image files from a specified folder and its subfolders.

    The function searches for image files with extensions (.png, .jpg, .jpeg, .bmp) in folders
    containing the word 'verified' in their path. It checks the image quality using the
    check_image_quality function and includes only those that meet the quality criteria.

    Parameters:
        image_folder (str): The path to the folder containing the images to be checked.

    Returns:
        list: A list of verified image file paths that meet the quality criteria.
              - Includes only images with valid file size, resolution, and sharpness.
    """
    verified_files = []
    for root, _, files in os.walk(image_folder):
        if "verified" in root.lower():
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    filepath = os.path.join(root, file)
                    if check_image_quality(filepath):
                        verified_files.append(filepath)
    return verified_files


def insert_data_with_check(table_name, project_name, hashed_doc_name, doc_name):
    """
    Checks if a document with the given hash already exists in the specified table before inserting.

    The function retrieves existing data using the project name and document hash. If the data does
    not already exist, it inserts the new document data into the table. If the data already exists, 
    it logs a message and skips the insertion.

    Parameters:
        table_name (str): The name of the table to check and insert the document data.
        project_name (str): The name of the project associated with the document.
        hashed_doc_name (str): The SHA-256 hash of the document name used for checking existence.
        doc_name (str): The original document name to be inserted if the hash does not exist.

    Returns:
        None: The function does not return a value but performs an insert or logs a message.
    """
    existing_data = azure_table_client.retrieve_by_hashed_doc_name(
        table_name, project_name, hashed_doc_name)

    if not existing_data:  # If data doesn't exist
        azure_table_client.insert_data(
            table_name, project_name, hashed_doc_name, doc_name)
    else:
        azure_table_client.logger.info(
            f"Data with hash value '{hashed_doc_name}' already exists. Skipping insert.")


def generate_output_filename(image_path, table_name, output_folder):
    """
    Generates a unique output filename based on the image path and inserts the document data into 
    the specified table if it does not already exist.

    The function extracts relevant information such as the parent folder, document name, and 
    creates a unique hash for the document. It checks if the table exists and ensures the document 
    is not already inserted into the table using the `insert_data_with_check` function. The 
    function then generates a base output filename for the processed image.

    Parameters:
        image_path (str): The path of the image file to base the output filename on.
        table_name (str): The name of the table to check and insert the document data.
        output_folder (str): The folder where the generated output file will be saved.

    Returns:
        str: The full path of the generated output filename.
    """
    parts = image_path.split(os.sep)
    if "literature" in parts:
        subfolder_type = "lr"
    elif "ifu" in parts:
        subfolder_type = "ifu"
    else:
        raise ValueError(
            "Invalid folder structure for generating output filename.")

    # Extract parent folder (e.g., "HOLOGIC GENIUS AI")
    parent_folder = parts[parts.index("images") + 1]
    parent_folder = parent_folder.lower().replace(" ", "_")

    # Extract document name
    document_folder = parts[-3]

    # Hash the document name
    hashed_document_folder = hash_document_name(document_folder)

    if azure_table_client.check_table_exists(table_name):
        azure_table_client.logger.info(f"Table '{table_name}' exists")

        # Insert
        project_name = parent_folder
        hashed_doc_name = hashed_document_folder
        doc_name = document_folder

        insert_data_with_check(table_name, project_name,
                               hashed_doc_name, doc_name)

    # Generate base output filename
    file_name = os.path.splitext(os.path.basename(image_path))[0]
    output_filename = f"{parent_folder}-{hashed_document_folder}-{subfolder_type}-{file_name}_gen_desc.txt"

    # Full output path
    os.makedirs(output_folder, exist_ok=True)
    return os.path.join(output_folder, output_filename)


def analyse_image(image_path, llm):
    """
    Analyse the given image by converting it to base64 and sending it to the LLM for processing.
    The function generates a message with the image data, invokes the LLM, and logs the generated text 
    along with the cost of the API call.

    Parameters:
        image_path (str): The path of the image file to be analyzed.
        llm (object): The large language model object used to generate a response based on the image.

    Returns:
        str: The content of the generated response from the LLM, or None if an error occurred.
    """
    try:
        # extract image_url
        base64_image = encode_image(image_path)
        message = MESSAGE_TEMPLATE.copy()
        message[1]["content"][1]["image_url"] = {
            "url": f"data:image/png;base64,{base64_image}"}

        # extract context of image
        base, _ = os.path.splitext(image_path)
        context_path = f"{base}-context.txt"
        with open(context_path, "r") as file:
            context = file.read()
        message[1]["content"][2]["text"] = context

        # # CHANGE MADE HERE
        # # extract context of image
        # base, _ = os.path.splitext(image_path)
        # context_path = f"{base}-context.txt"
        # with open(context_path, "r", encoding="utf-8") as file:
        #     context = file.read()
        # message[1]["content"][2]["text"] = context
        # # END OF CHANGES

        with get_openai_callback() as cb:
            ai_message = llm.invoke(message)
            logger.info(
                f"Generated text for {image_path}:\n{ai_message.content}")
            logger.info(f"Total Cost (USD): ${format(cb.total_cost, '.6f')}")

        return ai_message.content
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return None


def process_images(image_folder, llm, table_name, output_folder="output_images"):
    """
    Processes the images in the given folder by analyzing each verified image, generating a description 
    using a large language model (LLM), and saving the description to an output file. 
    The function avoids overwriting existing files and updates the progress using a progress bar.

    Parameters:
        image_folder (str): The folder containing the images to be processed.
        llm (object): The large language model object used to generate descriptions for the images.
        table_name (str): The name of the table used for checking and inserting data.
        output_folder (str, optional): The folder to save the generated descriptions. Defaults to "output_images".

    Returns:
        None: The function performs side effects (i.e., processing and saving descriptions) but does not return a value.
    """
    image_files = get_verified_image_files(image_folder)
    if not image_files:
        logger.info("No valid images found for processing.")
        return

    with alive_bar(len(image_files)) as bar:
        for image_path in image_files:
            output_file = generate_output_filename(
                image_path, table_name, output_folder)
            # Avoid overwriting existing files
            if not os.path.exists(output_file):
                description = analyse_image(image_path, llm)
                if description:
                    with open(output_file, "w", errors="replace") as f:
                        f.write(description)
                    logger.info(f"Saved description to {output_file}")
            bar()


def analyse_images():
    # Paths to access the input folder and specify the output folder
    # IMAGE_FOLDER = os.path.join(os.path.dirname(__file__), "..", "..", "images")
    # OUTPUT_FOLDER = os.path.join(os.path.dirname(__file__), "..", "..", "output_images")
    IMAGE_FOLDER = "images"
    OUTPUT_FOLDER = "output_images"

    table_name = "docmap"

    process_images(IMAGE_FOLDER, llm, table_name, OUTPUT_FOLDER)


if __name__ == "__main__":
    analyse_images()
