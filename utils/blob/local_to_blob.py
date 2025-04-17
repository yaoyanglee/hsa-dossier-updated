import configparser
import os

from azure.storage.blob import BlobServiceClient
from alive_progress import alive_bar
from pathlib import Path
from utils.table import azure_table_client

# --- Functions ---


def create_container(blob_service_client, container_name, logger):
    """
    Create a container in Azure Blob Storage if it doesn't exist.

    Args:
        blob_service_client: Azure BlobServiceClient instance
        container_name (str): The name of the container to create
        logger: Centralized logger instance
    """
    try:
        container_client = blob_service_client.get_container_client(
            container_name)
        container_client.create_container()
        logger.info(f"Container '{container_name}' created successfully")
    except Exception as e:
        if "ContainerAlreadyExists" in str(e):
            logger.info(f"Container '{container_name}' already exists")
        else:
            logger.error(f"Error creating container '{container_name}': {e}")
            raise


def get_all_files_with_custom_blob_name(root_directories, file_type, logger, clean_project_name):
    """
    Fetch all files of a specific type from the given root directories and generate custom blob names.
    For '.jpg' files, include only files within 'verified' subfolders.

    Args:
        root_directories (list): List of root directories to scan for files.
        file_type (str): Type of file to search for (e.g., '.txt', '.jpg').
        logger: Centralized logger instance
        azure_table_client: Azure table client instance for retrieving document mappings.

    Returns:
        list: List of tuples containing (local_file_path, blob_name).
    """
    all_files = []
    logger.info(
        f"Scanning directories: {root_directories} for files of type '{file_type}'")

    for root_dir in root_directories:
        for dirpath, _, files in os.walk(root_dir):
            for file in files:
                if file.endswith(file_type):  # Filter by file type
                    local_file_path = os.path.join(dirpath, file)

                    if file_type == ".txt":
                        # Allow all .txt files
                        # Direct file upload for text files
                        blob_name = os.path.basename(file)
                        all_files.append((local_file_path, blob_name))
                    elif file_type == ".jpg":
                        # Include only files within 'verified' subfolders
                        if "verified" in dirpath.lower():
                            doc_folder = os.path.basename(
                                os.path.dirname(dirpath)).lower().replace(" ", "_")

                            # Determine two and three levels up for subfolder type and source
                            two_levels_up = Path(dirpath).parents[1]
                            subfolder_name = os.path.basename(
                                two_levels_up).lower().replace(" ", "_")
                            three_levels_up = Path(dirpath).parents[2]
                            source_name = three_levels_up.name.lower().replace(" ", "_")

                            logger.debug(f"DEBUG: Doc Folder: {doc_folder}")
                            logger.debug(
                                f"DEBUG: Subfolder Name (two levels up): {subfolder_name}")

                            # Determine subfolder type
                            if "literature" in subfolder_name:
                                subfolder_type = "lr"
                            elif "manual" in subfolder_name or "instructions for use" in subfolder_name or "ifu" in subfolder_name:
                                subfolder_type = "ifu"
                            else:
                                subfolder_type = "others"

                            logger.debug(
                                f"DEBUG: Subfolder Type: {subfolder_type}")

                            hashed_doc = azure_table_client.retrieve_by_doc_name(
                                table_name="docmap",
                                project_name=source_name,
                                doc_name=doc_folder
                            ) or "unknown_document"

                            if clean_project_name == source_name:
                                # Generate custom blob name
                                blob_name = f"{source_name}-{hashed_doc}-{subfolder_type}-{file}"
                                logger.info(f"Adding {blob_name}")
                                all_files.append((local_file_path, blob_name))
                            else:
                                logger.info(
                                    f"{source_name} is not part of the the current project scope: {clean_project_name}")

    logger.info(f"Found {len(all_files)} files of type '{file_type}'")
    return all_files


def upload_files_to_blob(storage_connection_string, container_name, files_to_upload, logger):
    """
    Uploads files to a specified Azure Blob Storage container, with detailed upload statistics.

    Args:
        storage_connection_string (str): The connection string for Azure Storage account.
        container_name (str): The name of the container in Azure Blob Storage.
        files_to_upload (list): List of tuples containing (local_file_path, blob_name).
        logger: Centralized logger instance
    """
    logger.info(f"Starting upload process to container '{container_name}'")

    try:
        blob_service_client = BlobServiceClient.from_connection_string(
            storage_connection_string)
        container_client = blob_service_client.get_container_client(
            container_name)
        logger.info(
            f"Successfully initialized blob service client for container '{container_name}'")

        create_container(blob_service_client, container_name, logger)

        upload_stats = {'successful': 0, 'failed': 0,
                        'total_size': 0}  # Track statistics

        with alive_bar(len(files_to_upload), title="Uploading files", force_tty=True) as bar:
            for local_file_path, blob_name in files_to_upload:
                try:
                    blob_client = container_client.get_blob_client(blob_name)
                    file_size = os.path.getsize(local_file_path)

                    with open(local_file_path, "rb") as data:
                        blob_client.upload_blob(data, overwrite=True)

                    upload_stats['successful'] += 1
                    upload_stats['total_size'] += file_size
                    logger.info(
                        f"Uploaded: {local_file_path} → {container_name}/{blob_name}")
                except Exception as e:
                    upload_stats['failed'] += 1
                    logger.error(f"Failed to upload {local_file_path}: {e}")
                bar()

        total_size_mb = upload_stats['total_size'] / (1024 * 1024)
        logger.info(
            f"Upload stats: {upload_stats['successful']} successful, {upload_stats['failed']} failed")
        logger.info(f"Total data transferred: {total_size_mb:.2f} MB")

    except Exception as e:
        logger.error(f"Error uploading to container '{container_name}': {e}")
        raise


def upload_local_to_blob(logger, project_name, clean_project_name):
    """
    Main function to upload files from local directories to Azure Blob Storage.
    """
    config = configparser.ConfigParser()
    config.read("config.prop")
    azure_blob_config = config["azure_blob"]

    connection_string = azure_blob_config["connection_string"]
    container_name_docs = azure_blob_config["container_name_docs"]
    container_name_images = azure_blob_config["container_name_images"]

    directory_container_mapping = {
        "text_sections": container_name_docs,
        "output_images": container_name_docs,
        "images": container_name_images,
    }

    logger.info("Starting local to blob upload process")

    for directory, container_name in directory_container_mapping.items():
        logger.info(
            f"Processing directory '{directory}' for container '{container_name}'")

        # print("Directory: ", directory)
        file_type = ".jpg" if directory == "images" else ".txt"
        files_to_upload = get_all_files_with_custom_blob_name(
            [directory], file_type, logger, clean_project_name)

        # Filter files that contain clean_project_name (case-insensitive + handles spaces vs underscores)
        filtered_files_to_upload = [
            (file_path, file_name) for file_path, file_name in files_to_upload
            if clean_project_name in file_path.lower().replace(" ", "_")
        ]
        # print("Filtered files: ", filtered_files_to_upload)

        # Debugging loop
        # for file_path, file_name in files_to_upload:
        #     if clean_project_name in file_path:
        #         print(f"{file_path} contains the project name.")
        #     else:
        #         print(f"{file_path} does NOT contain the project name.")

        # print("Files to upload: ", files_to_upload)
        # print("Container name: ", container_name)
        # print("Files to upload: ", files_to_upload)

        # Only upload filtered files. Was files_to_upload previously
        if filtered_files_to_upload:
            # upload_files_to_blob(
            #     connection_string, container_name, files_to_upload, logger)
            upload_files_to_blob(
                connection_string, container_name, filtered_files_to_upload, logger)
        else:
            logger.warning(
                f"No files found in directory '{directory}' for upload")
