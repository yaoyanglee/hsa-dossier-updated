import configparser
import logging
import os

from azure.data.tables import TableServiceClient
from azure.core.credentials import AzureNamedKeyCredential


class AzureTableClient:
    def __init__(self, account_name, account_key):
        self.account_name = account_name
        self.account_key = account_key
        self.endpoint = f"https://{account_name}.table.core.windows.net"
        self.credential = AzureNamedKeyCredential(account_name, account_key)
        self.table_service_client = TableServiceClient(
            endpoint=self.endpoint, credential=self.credential)
        self.logger = self.configure_logger()

    def configure_logger(self):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def get_table_client(self, table_name):
        """
        Retrieve the Azure Table client for a specific table.

        Args:
            table_name (str): The name of the Azure Table.

        Returns:
            TableClient: An instance of the Azure Table client for the specified table.
        """
        return self.table_service_client.get_table_client(table_name)

    def create_table_if_not_exists(self, table_name):
        """
        Create a table if it does not already exist.

        This method attempts to create the specified table and logs the result. If the table already exists,
        it handles the error gracefully without failing the operation.

        Args:
            table_name (str): The name of the Azure Table to create.

        Logs:
            Success or failure of the table creation.
        """
        try:
            table_client = self.get_table_client(table_name)
            table_client.create_table()
            self.logger.info(f"Table '{table_name}' created successfully")
        except Exception as e:
            self.logger.error(f"Error creating table: {e}")

    def check_table_exists(self, table_name):
        """
        Check if a table exists in Azure Table Storage.

        This method attempts to query entities from the specified table to check its existence.
        If the table exists, it returns True; otherwise, it returns False.

        Args:
            table_name (str): The name of the Azure Table to check.

        Returns:
            bool: True if the table exists, False otherwise.
        """
        try:
            table_client = self.get_table_client(table_name)
            table_client.list_entities(
                query_filter="PartitionKey eq ''", select='PartitionKey')
            return True
        except Exception:
            return False

    def get_entities(self, table_client, filter_query=None):
        """
        Retrieve entities from a specified Azure Table using an optional filter query.

        Args:
            table_client: The Azure Table client instance.
            filter_query (str, optional): A filter query to narrow down the results (default is None).

        Returns:
            list: A list of entities retrieved from the table.
        """
        entities = table_client.query_entities(query_filter=filter_query)
        return list(entities)

    def retrieve_by_hashed_doc_name(self, table_name, project_name, hashed_doc_name):
        """
        Retrieve entities from an Azure Table using the hashed document name as the RowKey.

        Args:
            table_name (str): The name of the Azure Table.
            project_name (str): The project name, used as the PartitionKey.
            hashed_doc_name (str): The hashed document name, used as the RowKey.

        Returns:
            list: A list of retrieved entities matching the query.
        """
        table_client = self.get_table_client(table_name)

        filter_query = f"PartitionKey eq '{project_name}' and RowKey eq '{hashed_doc_name}'"
        retrieved_entities = self.get_entities(table_client, filter_query)

        return retrieved_entities

    def retrieve_by_doc_name(self, table_name, project_name, doc_name):
        """
        Retrieve the hashed document name (RowKey) from an Azure Table using the document name.

        Args:
            table_name (str): The name of the Azure Table.
            project_name (str): The project name, used as the PartitionKey.
            doc_name (str): The document name to search for.

        Returns:
            str or None: The hashed document name (RowKey) if found, else None.
        """
        try:
            table_client = self.get_table_client(table_name)

            # Start searching from the beginning of the partition
            filter_query = f"PartitionKey eq '{project_name}' and RowKey ge ''"

            # Limit the results to a reasonable number
            retrieved_entities = self.get_entities(table_client, filter_query)

            for entity in retrieved_entities:
                if entity.get('doc_name', '').lower() == doc_name.lower():
                    # Return the RowKey which contains the hashed_doc_name
                    return entity['RowKey']

            return None

        except Exception as e:
            logging.error(f"An error occurred: {str(e)}")
            return None

    def insert_entity(self, table_client, entity):
        """
        Insert a single entity into an Azure Table.

        Args:
            table_client: The Azure Table client instance.
            entity (dict): The entity to insert.

        Logs:
            Success or failure of the operation.
        """
        try:
            table_client.create_entity(entity)
            self.logger.info("Entity inserted successfully")
        except Exception as e:
            self.logger.error(f"Error inserting entity: {e}")

    def insert_data(self, table_name, project_name, hashed_doc_name, doc_name):
        """
        Insert document metadata into an Azure Table.

        Args:
            table_name (str): The name of the Azure Table.
            project_name (str): The project name, used as the PartitionKey.
            hashed_doc_name (str): The hashed document name, used as the RowKey.
            doc_name (str): The original document name.

        Logs:
            Success or failure of the insertion operation.
        """
        table_client = self.get_table_client(table_name)

        entity = {
            'PartitionKey': project_name,
            'RowKey': f"{hashed_doc_name}",
            'project_name': project_name,
            'hashed_doc_name': hashed_doc_name,
            'doc_name': doc_name,
            'description': "This is a document mapper for hashed documents"
        }

        self.insert_entity(table_client, entity)

    def insert_with_schema(self, table_name, schema_dict):
        """
        Generic method to insert an entity into a table based on a given schema.

        Args:
            table_name (str): The name of the Azure Table.
            schema_dict (dict): The schema of the entity to insert.
        """
        table_client = self.get_table_client(table_name)
        try:
            self.insert_entity(table_client, schema_dict)
            self.logger.info(
                f"Entity inserted successfully into table '{table_name}'")
        except Exception as e:
            self.logger.error(f"Error inserting entity: {e}")

    def create_vs_mapping(self, table_name, project_name, vs_name, doc_name):
        """
        Create a mapping for a vector store in an Azure Table.

        Args:
            table_name (str): The name of the Azure Table.
            project_name (str): The project name, used as the PartitionKey.
            vs_name (str): The name of the vector store, used as the RowKey.
            doc_name (str): The original document name.

        Logs:
            Success or failure of the mapping creation.
        """
        schema_dict = {
            'PartitionKey': project_name,
            'RowKey': f"{vs_name}",
            'project_name': project_name,
            'vs_name': vs_name,
            'doc_name': doc_name,
            'description': "This is a document mapper for vector stores"
        }
        self.insert_with_schema(table_name, schema_dict)


config = configparser.ConfigParser()
# config_path = os.path.join(os.path.dirname(__file__), "..", "config.prop")
# config.read(config_path)
config.read("config.prop")
azure_hsa_store_config = config["azure_hsa_store"]
account_name = azure_hsa_store_config["account_name"]
account_key = azure_hsa_store_config["account_key"]

# Initialise the AzureTableClient
azure_table_client = AzureTableClient(
    account_name=account_name,
    account_key=account_key
)
