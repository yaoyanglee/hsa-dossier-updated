import configparser
import logging
import os

from utils.text_processing.chunk_analyser import ChunkAnalyser
from utils.text_processing.chunk_refiner import ChunkRefiner
from unstructured.partition.pdf import partition_pdf
from unstructured.chunking.title import chunk_by_title

# Load configuration
config = configparser.ConfigParser()
config.read("config.prop")
azure_llm_config = config["azure_openai_gpt4o-mini"]

# --- Configure logger ---
# Create a named logger
logger = logging.getLogger(__name__)

# Configure the logger
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


# if __name__ == "__main__":
# logging.basicConfig(level=logging.INFO)
# docs_directory = "docs/"
# output_dir_base = "text_sections"
# analyser = ChunkAnalyser()

# subfolder_type = ""

# # Get all PDF files from the directory
# pdf_files = [
#     os.path.join(root, file)
#     for root, _, files in os.walk(docs_directory)
#     for file in files if file.endswith(".pdf")
# ]

# if not pdf_files:
#     logger.info("No PDF files found in the directory.")
# else:
#     for filename in pdf_files:
#         try:
#             logger.info(f"Processing file: {filename}")

#             # Analyze chunks for the file
#             settings = analyser.analyse_chunks(
#                 filename, max_chars_options=[2400, 3200, 4000, 4800, 5600, 6400, 7200]
#             )

#             # Extract and chunk text
#             elements = partition_pdf(filename=filename, strategy="hi_res")
#             chunks = chunk_by_title(
#                 elements,
#                 max_characters=settings.max_characters,
#                 combine_text_under_n_chars=settings.combine_under_chars,
#                 overlap=settings.overlap
#             )

#             # Extract parent folder and move up two levels
#             parent_path = os.path.dirname(os.path.dirname(filename))  # Go up two levels
#             project_name = os.path.basename(parent_path).lower().replace(" ", "_")  # Convert to lowercase and replace spaces with underscores

#             # Derive source_folder name from the filename (remove extension and apply formatting)
#             base_filename = os.path.basename(filename)
#             document_name = os.path.splitext(base_filename)[0].lower().replace(" ", "_")  # Convert to lowercase and replace spaces with underscores

#             # Derive subfolder name based on parent folder
#             subfolder_name = os.path.basename(os.path.dirname(filename)).lower().replace(" ", "_")

#             # Check subfolder and categorize
#             if "literature" in subfolder_name:
#                 subfolder_type = "lr"
#             elif any(keyword in subfolder_name for keyword in ["manual", "instructions_for_use", "ifu"]):
#                 subfolder_type = "ifu"
#             else:
#                 subfolder_type = "others"  # Default to others if no match

#             # Refine chunks and save output
#             llm_config = {
#                 'deployment_name': azure_llm_config["deployment"],
#                 'api_version': azure_llm_config["api_version"]
#             }

#             refiner = ChunkRefiner(**llm_config)
#             refiner.refine_chunks_and_save(chunks, filename, project_name, document_name, subfolder_type, output_dir_base)

#         except Exception as e:
#             logger.error(f"Error processing file {filename}: {e}")

def text_processor_run(project_files):
    logging.basicConfig(level=logging.INFO)
    docs_directory = "docs/"
    output_dir_base = "text_sections"
    analyser = ChunkAnalyser()

    subfolder_type = ""

    # # Get all PDF files from the directory
    # pdf_files = [
    #     os.path.join(root, file)
    #     for root, _, files in os.walk(docs_directory)
    #     for file in files if file.endswith(".pdf")
    # ]

    if not project_files:
        logger.info("No PDF files found in the directory.")
    else:
        for filename in project_files:
            try:
                logger.info(f"Processing file: {filename}")

                # Analyze chunks for the file
                settings = analyser.analyse_chunks(
                    filename, max_chars_options=[
                        2400, 3200, 4000, 4800, 5600, 6400, 7200]
                )

                # Extract and chunk text
                elements = partition_pdf(filename=filename, strategy="hi_res")
                chunks = chunk_by_title(
                    elements,
                    max_characters=settings.max_characters,
                    combine_text_under_n_chars=settings.combine_under_chars,
                    overlap=settings.overlap
                )

                # Extract parent folder and move up two levels
                parent_path = os.path.dirname(
                    os.path.dirname(filename))  # Go up two levels
                # Convert to lowercase and replace spaces with underscores
                project_name = os.path.basename(
                    parent_path).lower().replace(" ", "_")
                print("project name: ", project_name)

                # Derive source_folder name from the filename (remove extension and apply formatting)
                base_filename = os.path.basename(filename)
                document_name = os.path.splitext(base_filename)[0].lower().replace(
                    " ", "_")  # Convert to lowercase and replace spaces with underscores
                print("document name: ", document_name)

                # Derive subfolder name based on parent folder
                subfolder_name = os.path.basename(
                    os.path.dirname(filename)).lower().replace(" ", "_")
                print("subfolder name: ", subfolder_name)

                # Check subfolder and categorize
                if "literature" in subfolder_name:
                    subfolder_type = "lr"
                elif any(keyword in subfolder_name for keyword in ["manual", "instructions_for_use", "ifu"]):
                    subfolder_type = "ifu"
                else:
                    subfolder_type = "others"  # Default to others if no match

                # Refine chunks and save output
                llm_config = {
                    'deployment_name': azure_llm_config["deployment"],
                    'api_version': azure_llm_config["api_version"]
                }

                refiner = ChunkRefiner(**llm_config)
                refiner.refine_chunks_and_save(
                    chunks, filename, project_name, document_name, subfolder_type, output_dir_base)

            except Exception as e:
                logger.error(f"Error processing file {filename}: {e}")
