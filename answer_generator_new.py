# --- Imports and Dependencies ---
import argparse
import configparser
import logging
import pickle
import prompts  # Custom prompts module containing predefined prompts
import os

from openai import AzureOpenAI
from alive_progress import alive_bar
from datetime import datetime
import pandas as pd
from utils.table import azure_table_client


class AnswerGenerator:
    def __init__(self, project_name):
        # Defining project name
        self.project_name = project_name

        # --- Logging Configuration ---
        # Create a named logger for tracking script execution
        self.logger = logging.getLogger(__name__)

        # Set up logging configuration with timestamp, level, and message format
        self.logger.setLevel(logging.INFO)
        self.handler = logging.StreamHandler()
        self.formatter = logging.Formatter(
            "%(asctime)s - %(levelname)s - %(message)s")
        self.handler.setFormatter(self.formatter)
        self.logger.addHandler(self.handler)

        # --- Configuration Setup ---
        # Load configuration settings from config.prop file
        self.config = configparser.ConfigParser()
        self.config.read("config.prop")

        # Azure Configuration
        self.azure_llm_config = self.config["azure_openai_gpt4o-mini"]
        self.azure_blob_config = self.config["azure_blob"]
        self.azure_assistant_config = self.config["azure_assistant"]

        # Set required Azure OpenAI environment variables
        os.environ["AZURE_OPENAI_ENDPOINT"] = self.azure_llm_config["endpoint"]
        os.environ["AZURE_OPENAI_API_KEY"] = self.azure_llm_config["api_key"]

        # Defining project variables
        self.PROJECT_NAME = self.project_name
        self.ASSISTANT1_NAME = self.azure_assistant_config["assistant1_name"]
        self.ASSISTANT2_NAME = self.azure_assistant_config["assistant2_name"]
        self.API_VERSION = self.azure_llm_config["api_version"]
        self.MODEL_NAME = self.azure_llm_config["deployment"]
        self.TEMPERATURE = float(self.azure_llm_config["temperature"])
        self.PROMPTS_DICT = prompts.PROMPTS_DICT

        # --- Initialize Azure OpenAI Client ---
        self.client = AzureOpenAI(
            azure_endpoint=self.azure_llm_config["endpoint"],
            api_key=self.azure_llm_config["api_key"],
            api_version=self.API_VERSION
        )

    def get_vector_store_id_by_name(self, client, vector_store_name):
        """
        Retrieve the vector store ID based on its name.

        Args:
            client: Azure OpenAI client instance
            vector_store_name (str): Name of the vector store to find

        Returns:
            str or None: Vector store ID if found, None otherwise
        """
        try:
            vector_stores = list(client.beta.vector_stores.list())

            for vector_store in vector_stores:
                if vector_store.name == vector_store_name:
                    return vector_store.id

            return None
        except Exception as e:
            self.logger.info(f"Error retrieving vector store ID: {e}")
            return None

    def get_assistant_by_name(self, client, assistant_name):
        """
        Retrieve an existing assistant by its name.

        Args:
            client: Azure OpenAI client instance
            assistant_name (str): Name of the assistant to find

        Returns:
            Assistant object if found, None otherwise
        """
        try:
            # List all assistants
            # The issue seems to be a limit in the number of assistants they can display. The max is 20
            assistants = client.beta.assistants.list(limit=50)
            # print("\n(Get assistant) assistant name: ", assistant_name)
            # print("(Get assistant) assistant list: ", assistants.data)

            # Find the assistant with matching name
            for assistant in assistants.data:
                if assistant.name == assistant_name:
                    return assistant

            return None
        except Exception as e:
            self.logger.error(f"Error retrieving assistant: {e}")
            return None

    def generate_answer(self, user_prompt, assistant, thread):
        """
        Generate an answer and citations for a given prompt using the Azure OpenAI assistant.

        Args:
            user_prompt (str): The prompt to send to the assistant
            assistant: Azure OpenAI assistant instance
            thread: Conversation thread instance

        Returns:
            dict: Contains generated assessment, citations, record ID, token usage
        """
        # Create a new message in the thread
        message = self.client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=user_prompt
        )

        self.logger.info(
            f"The thread now has a vector store with that file in its tool resources. {thread.tool_resources.file_search}")

        try:
            # print("\nThread: ", thread)
            # print("\nAssistant: ", assistant)

            # Create and wait for the completion of the assistant's run
            run = self.client.beta.threads.runs.create_and_poll(
                thread_id=thread.id,
                assistant_id=assistant.id
            )
            # print("\nRun: ", run)
            self.logger.info(f"run_id: {run.id}")
        except Exception as e:
            print("\nException assistant run: ", e)

        try:
            # Retrieve and process messages from the completed run
            messages = list(self.client.beta.threads.messages.list(
                thread_id=thread.id, run_id=run.id))
            self.logger.info(f"messages: {messages}")
        except Exception as e:
            print(
                "\nException at retrieve and process messages from the completed run: ", e)

        try:
            # Extract and format message content and citations
            message_content = messages[0].content[0].text
            annotations = message_content.annotations
            citations = []

            # Process annotations and collect citations
            annotation_map, id = {}, 0
            for annotation in annotations:
                if annotation.text not in annotation_map:
                    annotation_map[annotation.text] = id
                    id += 1

                    # cite actual filename
                    if file_citation := getattr(annotation, "file_citation", None):
                        cited_file = self.client.files.retrieve(
                            file_citation.file_id)
                        citations.append(
                            f"[{annotation_map[annotation.text]}] {cited_file.filename}")

                # replace citation as index in response
                message_content.value = message_content.value.replace(
                    annotation.text, f"[{annotation_map[annotation.text]}]")

            answer = message_content.value
            self.logger.info(
                f"Answer and citation generated for run.id {run.id}")

            # Track and log token usage
            token_usage = self.track_token_usage(
                self.client, thread.id, run.id)
        except Exception as e:
            print("\nException at answer and citation processing: ", e)

        self.logger.info(f"Input tokens used: {token_usage['input_tokens']}")
        self.logger.info(f"Output tokens used: {token_usage['output_tokens']}")
        self.logger.info(f"Total tokens used: {token_usage['total_tokens']}")

        # Prepare and return the final response and token usage
        return {
            'assessment': answer,
            'citations': citations,
            'records_id': {
                'assistant_id': assistant.id,
                'thread_id': thread.id,
                'run_id': run.id,
                'message_id': message.id
            },
            'token_usage': {
                'input_tokens': token_usage['input_tokens'],
                'output_tokens': token_usage['output_tokens'],
                'total_tokens': token_usage['total_tokens']
            }
        }

    def track_token_usage(self, client, thread_id, run_id):
        """
        Track and aggregate token usage for a specific thread and run.

        Args:
            client: Azure OpenAI client instance
            thread_id (str): ID of the thread
            run_id (str): ID of the run

        Returns:
            dict: Dictionary containing token usage statistics
        """
        try:
            # Retrieve run steps
            steps_detail = client.beta.threads.runs.steps.list(
                thread_id=thread_id,
                run_id=run_id
            )

            # Initialize token counters
            total_tokens = 0
            total_input_tokens = 0
            total_output_tokens = 0

            # Process each step
            for step in steps_detail.data:
                # Skip steps without usage information
                if not hasattr(step, 'usage'):
                    continue

                # Accumulate token counts
                total_tokens += step.usage.total_tokens
                total_input_tokens += step.usage.prompt_tokens
                total_output_tokens += step.usage.completion_tokens

            # Prepare token usage report
            token_report = {
                'total_tokens': total_tokens,
                'input_tokens': total_input_tokens,
                'output_tokens': total_output_tokens
            }
            return token_report

        except Exception as e:
            self.logger.error(f"Error tracking token usage: {e}")
            return {
                'total_tokens': 0,
                'input_tokens': 0,
                'output_tokens': 0
            }

    def attach_vectorstore_to_thread(self, client, vector_store_name):
        """
        Create a thread and attach a specific vector store to it.

        Args:
            client: Azure OpenAI client instance
            vector_store_name (str): Name of the vector store to attach

        Returns:
            Thread object with vector store attached
        """
        # Get vector store ID for the specified store name
        vector_store_id = self.get_vector_store_id_by_name(
            client, vector_store_name)

        # Create a new thread with vector store configuration
        thread = client.beta.threads.create(
            tool_resources={
                "file_search": {
                    "vector_store_ids": [vector_store_id]
                }
            }
        )
        return thread

    def get_vectorstores(self, azure_table_client, table_name="vstoremap"):
        """
        Retrieve vector stores associated with a specific project.

        This function queries an Azure table to find vector stores, and sorts stores with 'ifu' in their name first.

        Args:
            azure_table_client: Azure table client
            table_name (str): Name of the table to query

        Returns:
            list: Sorted list of vector store names
        """

        table_client = azure_table_client.get_table_client(
            table_name=table_name)
        filter_query = f"PartitionKey eq '{self.project_name}'"
        entities = azure_table_client.get_entities(
            table_client=table_client, filter_query=filter_query)
        vectorstores = sorted(
            entities, key=lambda x: not x['vs_name'].endswith('ifu'))
        vectorstores = [ent['vs_name']
                        # sort those with ifu in front
                        for ent in vectorstores]

        return vectorstores

    def combine_dicts(self, dicts):
        """
        Combine multiple dictionaries into a single nested dictionary.

        Args:
            dicts (list): List of dictionaries to combine

        Returns:
            dict: Combined dictionary with nested structure
        """
        # Create a dictionary to store the combined result
        combined = {}

        # Get all top-level keys (SN1, SN2, etc.)
        keys = sorted(set().union(*dicts))

        # Iterate through each top-level key
        for key in keys:
            # Create a nested dictionary for this key
            combined[key] = {}

            # Iterate through each input dictionary
            for d in dicts:
                # Update the nested dictionary with the current dictionary's values for this key
                if key in d:
                    combined[key].update(d[key])

        return combined

    def extract_assessments(self, data, key):
        """
        Extract and format assessments from a nested dictionary.
        This function helps in consolidating assessments from different vector stores,
        and adds suffixes like '(User Manual)' where appropriate.

        Args:
            data (dict): Nested dictionary containing assessments
            key (str): Key to extract assessments from

        Returns:
            str: Formatted string of assessments
        """
        # Find all key-value pairs in the nested dictionary that end with "assessment"
        assessments = {k.replace('_assessment', '') + (' (User Manual)' if k.endswith('ifu_assessment') else ''): v
                       for k, v in data[key].items()
                       if k.endswith('_assessment')}

        # Format the assessments into a string with an extra newline after each
        output_string = '\n'.join(
            [f"{k}:\n{v}\n" for k, v in assessments.items()])

        return output_string

    def run_assessment(self, output_folder, output_filename):
        """
        Comprehensive assessment generation function.

        This function orchestrates the entire assessment process:
        1. Process prompts across vector stores and criterion (Step 1)
        2. Summarize results across vector stores by criterion (Step 2)
        3. Save results to pickle and token usage to CSV

        Args:
            output_folder (str): Directory to save output files
            output_filename (str): Base name for output files

        Returns:
            dict: Dictionary containing all responses and summaries
        """
        prompts_dict = self.PROMPTS_DICT

        token_logs = pd.DataFrame()

        # STEP 1 - RUN THROUGH EACH VECTOR STORE AND GO THROUGH EACH PROMPT (use assistant 1)
        # Retrieve Assistant 1
        assistant = self.get_assistant_by_name(
            self.client, self.ASSISTANT1_NAME)
        # print("\n (Run Assessment) Assistant: ", assistant)

        # Get list of vector stores for the project
        vector_stores = self.get_vectorstores(
            azure_table_client=azure_table_client)
        print("\nVectore Stores: ", vector_stores)
        responses_step1 = []
        # Process each vector store and each prompt with progress bar
        with alive_bar(len(prompts_dict)*len(vector_stores), title="Generating answer and citation", force_tty=True) as bar:
            for store in vector_stores:
                self.logger.info(
                    f"Processing assessment for vector store {store}.")
                # Attach vectore store to thread
                thread = self.attach_vectorstore_to_thread(self.client, store)

                for i, prompt in enumerate(prompts_dict):
                    self.logger.info(
                        f"Processing assessment SN_{i+1} for vector store {store}.")
                    try:
                        # Generate response
                        response = self.generate_answer(
                            user_prompt=prompts_dict[prompt], assistant=assistant, thread=thread)
                        assessment = {
                            f'{store}_assessment': response['assessment'], f'{store}_citations': response['citations']}
                        responses_step1.append({f'SN_{i+1}': assessment})

                        token_usage = pd.DataFrame({'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                    'assistant_id': response['records_id']['assistant_id'],
                                                    'thread_id': response['records_id']['thread_id'],
                                                    'run_id': response['records_id']['run_id'],
                                                    'message_id': response['records_id']['message_id'],
                                                    'input_tokens': response['token_usage']['input_tokens'],
                                                    'output_tokens': response['token_usage']['output_tokens'],
                                                    'total_tokens': response['token_usage']['total_tokens'],
                                                    'prompt': f"vectorstore_{store}--SN_{i+1}"}, index=[0])
                        token_logs = pd.concat(
                            [token_logs, token_usage], ignore_index=True)
                        self.logger.info(
                            f"Processing assessment SN_{i+1} complete.")
                    except Exception as e:
                        self.logger.error(
                            f"Error processing assessment SN_{i+1}: {e}")
                    bar()
            # create final dictionary for STEP 1 assessment
            responses_step1 = self.combine_dicts(responses_step1)

        # STEP 2 - RUN THROUGH EACH CRITERION AND SUMMARISE (use assistant 2)
        # Retrieve Assistant 2
        assistant = self.get_assistant_by_name(
            self.client, self.ASSISTANT2_NAME)
        responses_step2 = {}
        # Process each criterion with progress bar
        with alive_bar(len(responses_step1), title="Summarising answer", force_tty=True) as bar:
            for i, criterion in enumerate(prompts_dict):
                self.logger.info(f"Processing summary for SN_{i+1}.")
                try:
                    # Create thread
                    thread = self.client.beta.threads.create()
                    assessment_string = self.extract_assessments(
                        responses_step1, f'SN_{i+1}')
                    user_prompt = f"Criterion: {criterion}\nCompare using only these assessment reports:\n" + assessment_string
                    # Generate response
                    response = self.generate_answer(
                        user_prompt=user_prompt, assistant=assistant, thread=thread)
                    responses_step2[f'SN_{i+1}'] = {
                        'summary': response['assessment']}

                    token_usage = pd.DataFrame({'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                                'assistant_id': response['records_id']['assistant_id'],
                                                'thread_id': response['records_id']['thread_id'],
                                                'run_id': response['records_id']['run_id'],
                                                'message_id': response['records_id']['message_id'],
                                                'input_tokens': response['token_usage']['input_tokens'],
                                                'output_tokens': response['token_usage']['output_tokens'],
                                                'total_tokens': response['token_usage']['total_tokens'],
                                                'prompt': f"Summary--SN_{i+1}"}, index=[0])
                    token_logs = pd.concat(
                        [token_logs, token_usage], ignore_index=True)
                    self.logger.info(
                        f"Processing summary for SN_{i+1} complete.")
                except Exception as e:
                    self.logger.error(
                        f"Error processing summary for SN_{i+1}: {e}")
                bar()
        final_responses = self.combine_dicts(
            [responses_step1, responses_step2])  # create final dictionary
        # Create output directory if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)

        # pickle file name
        output_pkl = output_filename + ".pkl"
        output_path = os.path.join(output_folder, output_pkl)

        # Save results to pickle file
        with open(output_path, "wb") as output_file:
            pickle.dump(final_responses, output_file)
            self.logger.info(
                f"Generated assessment in pkl saved to {output_path}")

        # Save token logs to a CSV file
        token_logs_csv = f"{output_filename}_{datetime.now().strftime("%Y-%m-%d_%H%M")}.csv"
        token_logs_csv_path = os.path.join(output_folder, token_logs_csv)
        token_logs.to_csv(token_logs_csv_path, index=False)
        self.logger.info(
            f"Generated token logs saved to {token_logs_csv_path}")

        return final_responses

    def answer_generator_run(self):
        # Generate assessment and save results
        self.run_assessment(output_folder="assessment_reports",
                            output_filename="report")
        self.logger.info("Processing complete.")
