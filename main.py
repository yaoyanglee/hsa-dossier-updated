import os
import time
import hashlib
import logging
import configparser

from utils.table import azure_table_client
from text_processor import text_processor_run
from image_processor import image_processor_run
from blob_processor import blob_processor_run
from answer_generator_new import AnswerGenerator
from report_generator_new import ReportGenerator

# --- Configure logger ---
# Create a named logger
logger = logging.getLogger(__name__)
# Configure the logger
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


class Dossier:
    def __init__(self, project_name, table_name="docmap"):
        self.project_name = project_name

        config = configparser.ConfigParser()
        config.read("config.prop")
        azure_hsa_store_config = config["azure_hsa_store"]

        self.account_name = azure_hsa_store_config["account_name"]
        self.account_key = azure_hsa_store_config["account_key"]

        self.azure_table_client = azure_table_client
        self.table_name = table_name
        # Creates an azure table for the document name to hashed name mapping if it does not exist
        azure_table_client.create_table_if_not_exists(self.table_name)

    # Helper function/s

    def get_project_names(self, dir):
        folders = [name for name in os.listdir(
            dir) if os.path.isdir(os.path.join(dir, name))]

        if not folders:
            print("No projects available.")
            return None

        print("Select a project:")
        for i, folder in enumerate(folders, 1):  # Start index from 1
            print(f"{i}. {folder}")

        while True:
            try:
                choice = int(input("Enter the project of your choice: "))
                if 1 <= choice <= len(folders):
                    # Return selected folder
                    return folders[choice - 1], folders[choice - 1].lower().replace(" ", "_")
                else:
                    print("Invalid choice. Please select a number from the list.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def hash_document_name(self, doc_name):
        """
        Generates a SHA-256 hash for the document name and truncates it to 8 characters for brevity.
        Parameters:
            doc_name (str): The document name to be hashed.
        Returns:
            str: The first 8 characters of the SHA-256 hash of the document name.
        """
        return hashlib.sha256(doc_name.encode('utf-8')).hexdigest()[:8]

    def insert_data_with_check(self, table_name, project_name, hashed_doc_name, doc_name):
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

        existing_data = self.azure_table_client.retrieve_by_hashed_doc_name(
            table_name, project_name, hashed_doc_name)

        if not existing_data:  # If data doesn't exist
            self.azure_table_client.insert_data(
                table_name, project_name, hashed_doc_name, doc_name)
        else:
            self.azure_table_client.logger.info(
                f"Data with hash value '{hashed_doc_name}' already exists. Skipping insert.")

    def workflow_setup(self):
        # Retrieving all the file names in the folder.
        docs_directory = "docs/"
        # Get all PDF files from the directory
        pdf_files = [
            os.path.join(root, file)
            for root, _, files in os.walk(docs_directory)
            for file in files if file.endswith(".pdf")
        ]
        # print("PDF Files: ", pdf_files)
        # print("Len PDF Files: ", len(pdf_files))

        if not pdf_files:
            logger.info("No PDF files found in the directory.")
        else:
            for filepath in pdf_files:
                filepath_parts = filepath.split(os.sep)
                filename, _ = os.path.splitext(filepath_parts[-1])
                filename = filename.lower().replace(" ", "_")
                folder_name = filepath_parts[-3]

                hashed_filename = self.hash_document_name(filename)
                # print("Path parts: ", folder_name)
                # print("Filename: ", filename)
                # print("Hashed filename: ", hashed_filename)

                self.insert_data_with_check(
                    self.table_name, self.project_name, hashed_filename, filename)

            return pdf_files

    def text_processor(self):
        text_processor_run()

    def image_processor(self):
        image_processor_run()

    def blob_processor(self):
        blob_processor_run()

    def answer_generator(self):
        ans_generator = AnswerGenerator(self.project_name)
        ans_generator.answer_generator_run()

    def report_generator(self):
        report_generator = ReportGenerator()
        report_generator.generate_report_run()

    def run_workflow(self):
        start_time = time.time()

        logger.info("Setting up workflow requirements")
        project_pdf_files = self.workflow_setup()
        # print("Project PDF files: ", project_pdf_files)
        # print("Len Project PDF files: ", len(project_pdf_files))
        logger.info("Set up complete\n")

        logger.info("Starting workflow...")
        logger.info("Step 1: Running image processor...")
        self.image_processor()
        logger.info("Image processing complete. Moving to text processing.\n")

        # logger.info("Step 2: Running text processor...\n")
        # self.text_processor()
        # logger.info("Text processing complete. Moving to blob processing.\n")

        # logger.info("Step 3: Running blob processor...")
        # self.blob_processor()
        # logger.info("Blob processing complete. Moving to answer generation.\n")

        # logger.info("Step 4: Running answer generator...")
        # self.answer_generator()
        # logger.info(
        #     "Answer generation complete. Moving to report generation.\n")

        # logger.info("Step 5: Running report generator...")
        # self.report_generator()
        # logger.info("Report generation complete. Workflow finished!\n")

        end_time = time.time()
        elapsed_time = end_time - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        logger.info(
            f"Total execution time: {int(hours)} hrs {int(minutes)} mins {seconds:.2f} secs")


if __name__ == "__main__":
    app = Dossier("nox_medical_nox_a1_and_t3")
    app.run_workflow()
