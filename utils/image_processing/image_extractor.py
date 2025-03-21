import base64
import cv2
import logging
import numpy as np
import os
import re
import shutil

from alive_progress import alive_bar
from pathlib import Path
from typing import List
from unstructured.partition.pdf import partition_pdf

# -- Constants to prevent image cropping --
HORIZONTAL_PAD = 50
VERTICAL_PAD = 50

# Set environment variables for image extraction padding
os.environ["EXTRACT_IMAGE_BLOCK_CROP_HORIZONTAL_PAD"] = str(HORIZONTAL_PAD)
os.environ["EXTRACT_IMAGE_BLOCK_CROP_VERTICAL_PAD"] = str(VERTICAL_PAD)

# --- Configure logger ---


def setup_logger(name='image_extractor', log_file='image_extractor.log', level=logging.DEBUG):
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
            os.path.abspath(__file__)), "../../logs")
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


def get_image_context(elements, image_index, context_span=5):
    """
    Extracts the surrounding text (captions, list items, narrative, titles) around an image.

    Parameters:
        elements (List): List of extracted elements from the PDF.
        image_index (int): Index of the current image.
        context_span (int): Number of surrounding elements to include in the context.

    Returns:
        str: Combined context as a single string.
    """
    narrative_texts_before = []
    for i in range(image_index - 1, max(image_index - context_span - 1, -1), -1):
        if elements[i].category in ["FigureCaption", "ListItem", "NarrativeText", "Title"]:
            narrative_texts_before.append(elements[i].text)

    narrative_texts_after = []
    for i in range(image_index + 1, min(image_index + context_span + 1, len(elements))):
        if elements[i].category in ["FigureCaption", "ListItem", "NarrativeText", "Title"]:
            narrative_texts_after.append(elements[i].text)

    combined_context = narrative_texts_before + narrative_texts_after
    return ' '.join(combined_context).capitalize()


def save_image_context(image_context, logger, verified_dir, parent_label):
    """
    Saves the extracted image context to a text file.

    Parameters:
        image_context (str): Context text to save.
        verified_dir (str): Directory to save the context file.
        parent_label (str): Label for the image (e.g., figure number).
    """
    context_file_path = os.path.join(
        verified_dir, f"{parent_label}-context.txt")
    with open(context_file_path, "w", encoding='utf-8') as context_file:
        context_file.write(image_context)
    logger.info(f"Saved context for {parent_label} to: {context_file_path}")


def save_verified_image(image_path, logger, verified_dir, parent_label):
    """
    Copies the image to the verified directory after checking for its existence.

    Parameters:
        image_path (str): Path to the image to save.
        verified_dir (str): Directory to save the image.
        parent_label (str): Label for the image (e.g., figure number).
    """
    if os.path.exists(image_path):
        Path(verified_dir).mkdir(parents=True, exist_ok=True)
        filename = os.path.basename(image_path)
        file_stem, file_ext = os.path.splitext(filename)
        dest_path = os.path.join(verified_dir, f"{parent_label}{file_ext}")

        if os.path.abspath(image_path) != os.path.abspath(dest_path):
            shutil.copy(image_path, dest_path)
            logger.info(f"Saved verified image to: {dest_path}")
        else:
            logger.warning(
                f"Source and destination are the same: {image_path}")


def determine_subfolder_type(subfolder_name):
    """
    Determines the type of subfolder based on its name.

    Parameters:
        subfolder_name (str): Name of the subfolder to evaluate.

    Returns:
        str: Subfolder type ('literature', 'ifu', or None).
    """
    subfolder_name = subfolder_name.lower()
    if "literature" in subfolder_name:
        return "literature"
    elif "manual" in subfolder_name or "instructions for use" in subfolder_name or "ifu" in subfolder_name:
        return "ifu"
    return None


def process_all_pdfs_with_structure(directory, logger, output_dir_base="images"):
    """
    Processes all PDFs in the directory, extracting and saving images and contexts.

    Parameters:
        directory (str): Directory containing PDF files.
        output_dir_base (str): Base directory for saving images and contexts.
        logger (logging.Logger): Logger for tracking progress and errors.
    """
    for root, dirs, files in os.walk(directory):
        parent_folder = os.path.basename(os.path.dirname(root))
        subfolder_type = determine_subfolder_type(os.path.basename(root))

        if subfolder_type:  # Only process literature or ifu subfolders
            output_parent_dir = os.path.join(
                output_dir_base, parent_folder, subfolder_type)
            os.makedirs(output_parent_dir, exist_ok=True)

            for file in files:
                if file.endswith(".pdf"):
                    pdf_file_path = os.path.join(root, file)
                    file_base_name = generate_output_dir_from_filename(file)
                    output_dir = os.path.join(
                        output_parent_dir, file_base_name)

                    raw_dir = os.path.join(output_dir, "raw")
                    verified_dir = os.path.join(output_dir, "verified")
                    os.makedirs(raw_dir, exist_ok=True)
                    os.makedirs(verified_dir, exist_ok=True)

                    try:
                        logger.info(f"Processing file: {pdf_file_path}")
                        extract_pdf_images(
                            pdf_file_path, logger, raw_dir, verified_dir)
                    except Exception as e:
                        logger.error(
                            f"Error processing file {pdf_file_path}: {e}")


def generate_output_dir_from_filename(filename):
    """
    Generates an output directory name based on the filename.

    Parameters:
        filename (str): Name of the file to generate the output directory for.

    Returns:
        str: The generated directory name.
    """
    base_name = os.path.splitext(os.path.basename(filename))[0]
    folder_name = base_name.lower().replace(" ", "_")
    final_output_dir = folder_name
    return final_output_dir


def extract_pdf_images(filename, logger, raw_dir, verified_dir):
    """
    Extracts images from a PDF file, processes them, and saves verified images and contexts.

    Parameters:
        filename (str): Path to the PDF file.
        raw_dir (str): Directory to store raw extracted images.
        verified_dir (str): Directory to store verified images and contexts.
    """
    raw_pdf_elements = partition_pdf(
        filename=filename,
        extract_images_in_pdf=True,
        strategy="hi_res",
        infer_table_structure=True,
        extract_image_block_types=["Image", "Table"],
        extract_image_block_output_dir=raw_dir
    )

    # Process images
    image_indexes = [i for i, element in enumerate(
        raw_pdf_elements) if element.category == "Image"]
    image_count = 1
    for image_index in image_indexes:
        image_dict = raw_pdf_elements[image_index].metadata.to_dict()
        image_path = image_dict.get("image_path", "")
        detection_class_prob = float(
            image_dict.get("detection_class_prob", 0.0))

        if detection_class_prob >= 0.8:
            logger.info(f"Processing verified image: {image_path}")
            logger.info(f"Detection probability: {detection_class_prob}")
            parent_label = f"figure{image_count}"

            image_context = get_image_context(raw_pdf_elements, image_index)
            save_image_context(image_context, logger,
                               verified_dir, parent_label)
            save_verified_image(image_path, logger, verified_dir, parent_label)
            image_count += 1
        logger.info("-" * 153)

    # Process tables
    image_indexes = [i for i, element in enumerate(
        raw_pdf_elements) if element.category == "Table"]
    table_count = 1
    for image_index in image_indexes:
        image_dict = raw_pdf_elements[image_index].metadata.to_dict()
        image_path = image_dict.get("image_path", "")
        detection_class_prob = float(
            image_dict.get("detection_class_prob", 0.0))

        if detection_class_prob >= 0.8:
            logger.info(f"Processing verified table: {image_path}")
            logger.info(f"Detection probability: {detection_class_prob}")
            parent_label = f"table{table_count}"

            image_context = get_image_context(raw_pdf_elements, image_index)
            save_image_context(image_context, logger,
                               verified_dir, parent_label)
            save_verified_image(image_path, logger, verified_dir, parent_label)
            table_count += 1
        logger.info("-" * 153)


def extract_images():
    # -- Setup logger --
    logger = setup_logger()

    # -- Start processing PDFs from the specified directory --
    directory = r"docs/"
    process_all_pdfs_with_structure(
        directory, logger, output_dir_base="images")


if __name__ == "__main__":
    extract_images()
