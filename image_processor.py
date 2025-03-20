import logging
import os

from utils.image_processing.image_extractor import extract_images
from utils.image_processing.image_analyser import analyse_images

# --- Configure logger ---


def setup_logger(name='image_processor', log_file='image_processor.log', level=logging.DEBUG):
    """
    Set up a logger with console and file handlers.

    Parameters:
        name (str): Name of the logger.
        log_file (str): Log file name.
        level (int): Log level.

    Returns:
        logger (logging.Logger): Configured logger instance.
    """
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


def process_images(logger, project_name):
    try:
        logger.info("Starting the image extraction process...")
        extract_images(project_name)
        logger.info("Image extraction completed successfully.")

        logger.info("Starting the image analysis process...")
        analyse_images()
        logger.info("Image analysis completed successfully.")
    except Exception as e:
        logger.error(
            f"An error occurred during processing: {e}", exc_info=True)


def image_processor_run(project_name):
    logger = setup_logger()
    logger.info("Image processing pipeline initiated.")
    process_images(logger, project_name)
    logger.info("Image processing pipeline completed.")


if __name__ == "__main__":
    logger = setup_logger()
    logger.info("Image processing pipeline initiated.")
    process_images(logger)
    logger.info("Image processing pipeline completed.")
