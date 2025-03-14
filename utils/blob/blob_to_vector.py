import configparser
import os

from azure.storage.blob import BlobServiceClient
from alive_progress import alive_bar
from openai import AzureOpenAI
from utils.table import azure_table_client

# --- Functions ---
def upload_files_to_vector_stores(client, container_client, supported_extensions, logger):
    """
    Upload files from blob storage to corresponding vector stores based on their naming conventions.

    Args:
        client: AzureOpenAI client instance
        container_client: Azure Blob Container client instance
        supported_extensions (list): List of supported file extensions
        logger: Logger instance
    """
    logger.info("Starting file upload process to vector stores.")
    blob_upload_stats = {'successful': 0, 'failed': 0, 'skipped': 0}

    # Group files by vector store name
    vector_store_files = {}

    # Dictionary to store unique mappings
    vstore_mappings = {} 

    # Process each blob
    blobs = list(container_client.list_blobs())
    for blob in blobs:
        try:
            if any(blob.name.endswith(ext) for ext in supported_extensions):
                # Extract vector store details from blob name
                file_name = os.path.basename(blob.name)
                parts = file_name.split('-')
                
                if len(parts) < 4:
                    logger.warning(f"Invalid file naming convention: {file_name}")
                    blob_upload_stats['skipped'] += 1
                    continue

                # Extract source, document name and subfolder type
                source_name = parts[0]
                doc_name = parts[1]
                subfolder_type = parts[2]

                entity = azure_table_client.retrieve_by_hashed_doc_name("docmap", source_name, doc_name)
                original_doc_name = entity[0].get('doc_name', '')

                # Formulate vector store name
                vector_store_name = f"{doc_name}-{subfolder_type}"

                # Store the unique mapping
                vstore_mappings[vector_store_name] = {
                    "source_name": source_name,
                    "vector_store_name": vector_store_name,
                    "original_doc_name": original_doc_name
                }

                if vector_store_name not in vector_store_files:
                    vector_store_files[vector_store_name] = []

                # Fetch blob content and store in the vector store files group
                blob_client = container_client.get_blob_client(blob.name)
                file_bytes = blob_client.download_blob().readall()

                logger.info(f"Appending {file_name} to {vector_store_name}")
                vector_store_files[vector_store_name].append((file_name, file_bytes))
            else:
                blob_upload_stats['skipped'] += 1
                logger.info(f"Skipped {blob.name}: Unsupported file type")
        except Exception as e:
            blob_upload_stats['failed'] += 1
            logger.error(f"Error processing {blob.name}: {e}")

    # Insert mappings into the Azure Table Storage (vecstoremap table)
    for vector_store, mapping in vstore_mappings.items():
        try:
            azure_table_client.create_vs_mapping(
                table_name="vstoremap",
                project_name=mapping['source_name'],
                vs_name=vector_store,  # Using vector_store_name as hashed_doc_name
                doc_name=mapping['original_doc_name']
            )
            logger.info(f"Mapping inserted for vector store: {vector_store}")
        except Exception as e:
            logger.error(f"Error inserting mapping for vector store {vector_store}: {e}")

    # Create and upload to vector stores
    for vector_store_name, files in vector_store_files.items():
        logger.info(f"Creating and uploading to vector store: {vector_store_name}")
        try:
            vector_store = client.beta.vector_stores.create(name=vector_store_name)
            logger.info(f"Vector store {vector_store_name} created successfully with ID: {vector_store.id}")
            
            # Upload files in batches
            for file_name, file_content in files:
                try:
                    file_batch = client.beta.vector_stores.file_batches.upload_and_poll(
                        vector_store_id=vector_store.id,
                        files=[(file_name, file_content)]
                    )
                    if file_batch.status == "completed":
                        blob_upload_stats['successful'] += file_batch.file_counts.completed
                        logger.info(f"Uploaded {file_name} to vector store {vector_store_name}")
                    else:
                        blob_upload_stats['failed'] += file_batch.file_counts.failed
                        logger.warning(f"Failed to upload {file_name} to vector store {vector_store_name}")
                except Exception as e:
                    blob_upload_stats['failed'] += 1
                    logger.error(f"Error uploading {file_name} to vector store {vector_store_name}: {e}")
        except Exception as e:
            logger.error(f"Failed to create vector store {vector_store_name}: {e}")

    logger.info("File upload process to vector stores completed.")
    logger.info(f"Upload stats: {blob_upload_stats}")


def main(logger):
    """
    Main function for handling file uploads from Azure Blob Storage to vector stores.

    Args:
        logger: Logger instance
    """
    # Load configuration
    config = configparser.ConfigParser()
    config.read("config.prop")
    azure_llm_config = config["azure_openai_gpt4o-mini"]
    azure_blob_config = config["azure_blob"]

    # Set environment variables
    os.environ["AZURE_OPENAI_ENDPOINT"] = azure_llm_config["endpoint"]
    os.environ["AZURE_OPENAI_API_KEY"] = azure_llm_config["api_key"]

    # Additional variables
    CONNECTION_STRING = azure_blob_config["connection_string"]
    CONTAINER_NAME = azure_blob_config["container_name_docs"]

    # Initialize AzureOpenAI client
    client = AzureOpenAI(
        azure_endpoint=azure_llm_config["endpoint"],
        api_key=azure_llm_config["api_key"],
        api_version=azure_llm_config["api_version"]
    )

    # Initialize Blob Service Client for storage operations
    logger.info("Initializing Blob Service Client...")
    try:
        blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)
        logger.info(f"Successfully connected to container: {CONTAINER_NAME}")
    except Exception as e:
        logger.error(f"Failed to initialize Blob Service Client: {e}")
        raise

    # Define supported file extensions
    supported_extensions = [".txt"]

    # Upload files to vector stores
    logger.info("Starting file upload process")
    try:
        upload_files_to_vector_stores(client, container_client, supported_extensions, logger)
        logger.info("File upload process completed successfully")
    except Exception as e:
        logger.error(f"An error occurred during the file upload process: {e}")
