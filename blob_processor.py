import os
import logging

from utils.blob.local_to_blob import upload_local_to_blob
from utils.blob.blob_to_vector import main as upload_blob_to_vector

# --- Configure logger ---


def setup_logger(name='blob_processor', log_file='blob_processor.log', level=logging.DEBUG):
    logger = logging.getLogger(name)

    if not logger.handlers:  # Avoid duplicate handlers
        logger.setLevel(level)

        # Define log folder relative to this script
        log_dir = os.path.join(os.path.dirname(
            os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file_path = os.path.join(log_dir, log_file)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler
        file_handler = logging.FileHandler(log_file_path, mode='w')
        file_handler.setLevel(logging.WARNING)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# if __name__ == "__main__":
#     # Centralised logger for all modules
#     logger = setup_logger()

#     try:
#         logger.info("Start Blob Processor workflow")

#         # Step 1: Upload local files to Azure Blob Storage
#         logger.info("Executing local_to_blob: Uploading local files to Azure Blob Storage")
#         upload_local_to_blob(logger)
#         logger.info("local_to_blob execution completed")

#         # Step 2: Push files from Blob Storage to Vector Store
#         logger.info("Executing blob_to_vector: Uploading all blob files to vector store")
#         upload_blob_to_vector(logger)
#         logger.info("blob_to_vector execution completed")
#         logger.info("Blob Processor completed successfully")
#     except Exception as e:
#         logger.error(f"Blob Processor encountered an error: {e}")

def blob_processor_run(project_name, clean_project_name):
    # Centralised logger for all modules
    logger = setup_logger()

    try:
        logger.info("Start Blob Processor workflow")

        # Step 1: Upload local files to Azure Blob Storage
        logger.info(
            "Executing local_to_blob: Uploading local files to Azure Blob Storage")
        upload_local_to_blob(logger, project_name, clean_project_name)
        logger.info("local_to_blob execution completed")

        # Step 2: Push files from Blob Storage to Vector Store
        logger.info(
            "Executing blob_to_vector: Uploading all blob files to vector store")
        upload_blob_to_vector(logger, clean_project_name)
        logger.info("blob_to_vector execution completed")
        logger.info("Blob Processor completed successfully")
    except Exception as e:
        logger.error(f"Blob Processor encountered an error: {e}")
