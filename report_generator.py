# --- Imports and Dependencies ---
import os
import xlsxwriter
from PIL import Image 
import pandas as pd
import re
import pickle
from azure.storage.blob import BlobServiceClient
import logging
import configparser
from alive_progress import alive_bar
from utils.table import azure_table_client

# --- Logging Configuration ---
# Create a named logger for tracking script execution
logger = logging.getLogger(__name__)

# Set up logging configuration with timestamp, level, and message format
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)

# --- Configuration Setup ---
# Load configuration settings from config.prop file
config = configparser.ConfigParser()
config.read("config.prop")
azure_llm_config = config["azure_openai_gpt4o-mini"]
azure_blob_config = config["azure_blob"]
azure_assistant_config = config["azure_assistant"]
blob_config = config["azure_blob"]

# Set required Azure OpenAI environment variables
os.environ["AZURE_OPENAI_ENDPOINT"] = azure_llm_config["endpoint"]
os.environ["AZURE_OPENAI_API_KEY"] = azure_llm_config["api_key"]

# --- Configuration Variables ---
CONNECTION_STRING = blob_config["connection_string"]
CONTAINER_NAME = blob_config["container_name_docs"]
CONTAINER_NAME_IMAGES = blob_config["container_name_images"]

# --- Helper Functions ---

def get_entities(table_client, filter_query=None):
    entities = table_client.query_entities(query_filter=filter_query)
    return list(entities)

def get_table_client(table_service_client, table_name):
    return table_service_client.get_table_client(table_name)

def retrieve(table_service_client, table_name, project_name, query_key, query_value):
    table_client = get_table_client(table_service_client, table_name)
    filter_query = f"PartitionKey eq '{project_name}' and {query_key} eq '{query_value}'"
    retrieved_entities = get_entities(table_client, filter_query)
    
    if retrieved_entities:
        return retrieved_entities[0]
    else:
        print("No matching entities found")
        return {}

def get_project_name(obj):
    try:
        citation_pattern = r'(.*?)_citations'
        for a,b in obj['SN_1'].items():
            match = re.findall(citation_pattern, a)
            #get vector store name
            if len(match) >0:
                #vector_store_name = match[0]
                #print(f'vector store name: {vector_store_name}')

                #get an example of citation document name 
                if len(b)>0:
                    citation_doc = b[0]
                    #print(citation_doc)
                    
                    #get project name
                    citation_number_pattern = r'(\[\d+\]).*'
                    file_match = re.findall(citation_number_pattern, citation_doc)
                    start_index = len(file_match[0]) + 1
                    file_name = citation_doc[start_index:] 
                    project_name_pattern = r'(.*?)-.*'
                    pn_match = re.findall(project_name_pattern, file_name)
                    project_name = pn_match[0]
                    return project_name
    except:
        print('project name not found in SN_1')
        return 'NOT FOUND'

#replace hashed document name with original document name
def dehash(file):
    table_name = "docmap"
    query_key = "hashed_doc_name"

    #remove [\d]
    citation_number_pattern = r'(\[\d+\]).*'
    file_match = re.findall(citation_number_pattern, file)
    if len(file_match)>0:
        citation_number = file_match[0]
        #print(f'citation is: {citation_number}')
        start_index = len(citation_number) + 1
        hash_file_name = file[start_index:] 
        #print(f'hash file name: {hash_file_name}')

    #find the hash
    hash_pattern = rf'{project_name}-([a-zA-Z0-9]{{8}})-(.*?)-(.*)'
    match = re.findall(hash_pattern, hash_file_name)
    try: 
        if len(match) >0:
            #print(match)
            hash_value = match[0][0]
            remaining_value = match[0][-1]
            #print(remaining_value)

            #find original doc name
            h_d_map = retrieve(azure_table_client.table_service_client, table_name, project_name, query_key, hash_value)
            original_document_name = h_d_map['doc_name']
            #print(original_document_name)

            #replace hash with original file name
            original_chunk_name = f'{project_name}-{original_document_name}-{remaining_value}'
            #print(f'original chunk name: {original_chunk_name}')

            #add back [\d]
            citation_chunk_name = f'{citation_number} {original_chunk_name}'
            #print(f'citation chunk name: {citation_chunk_name}')
            return citation_chunk_name

    except:
        #return hash file name
        return file
    

def process_pkl_to_excel(excel_data):
    citation_number_pattern = r'(\[\d+\]).*'
    assessment_pattern = r'(.*?)_assessment'

    with alive_bar(len(obj), title="Processing pkl data to excel data", force_tty=True) as bar:
        for criterion, d in obj.items():
            logger.info(f"Processing items in {criterion}.")
            excel_data[criterion] = {}
            for key in d.keys():
                logger.info(f"Processing {key}.")
                match = re.findall(assessment_pattern, key)
                if len(match) >0:
                    #print(match[0])
                    answer_k = f"{match[0]+'_assessment'}"
                    citation_k = f"{match[0]+'_citations'}"

                    dup_hash_cite_list = d[citation_k]
                    
                    #replace with original doc names
                    dup_ori_cite_list = [dehash(cite) for cite in dup_hash_cite_list]

                    answer_with_text = d[answer_k] + "\n" + "\n".join(dup_ori_cite_list)
                    unique = set()

                    #remove [\d] and de-duplicate citation list with hash file name (for retrieval from blob)
                    for file in dup_hash_cite_list:
                        file_match = re.findall(citation_number_pattern, file)
                        start_index = len(file_match[0]) + 1

                        if file[start_index:] not in unique:
                            hash_file_name = file[start_index:]
                            ori_file_name_with_citation = dehash(file)
                            ori_file_name = ori_file_name_with_citation[start_index:]

                            unique.add((hash_file_name,ori_file_name)) #create a tuple so that can retrieve blob from hash file name and reflect in excel the original file name

                    excel_data[criterion][answer_k] = answer_with_text
                    excel_data[criterion][citation_k] = list(unique)
                elif key == 'summary':
                    excel_data[criterion]['summary'] = d['summary']

            bar()

def generate_excel(excel_data):
    #Final report 
    output_images_folder = f'{project_name}_supp_images'
    output_file_name = "output.xlsx"
    workbook = xlsxwriter.Workbook(output_file_name)

    # Add formats
    bold = workbook.add_format({'bold': True})

    response_format = workbook.add_format({'color': 'blue', 
                                    'text_wrap': True,
                                        'align': 'top'})

    criteria_format = workbook.add_format({'align': 'top',
                                        'text_wrap': True,
                                        'bold': True})

    sn_format = workbook.add_format({'align': 'top',
                                        'text_wrap': True})

    citation_format = workbook.add_format({'color': 'black', 
                                    'text_wrap': True,
                                        'align': 'top'})

    overview_worksheet = workbook.add_worksheet('Overview')
    overview_worksheet.write('A1', 'SN', bold)
    overview_worksheet.write('B1', 'Item', bold)
    overview_worksheet.set_column('B:B', 50)

    #column header for each journal + summary column
    assessment_pattern = r'(.*?)_assessment'
    header_count = 1
    headers = []
    for a in obj['SN_1'].keys():
        match = re.findall(assessment_pattern, a)

        #get vector store name
        if len(match) >0:
            vector_store_name = match[0]
            #print(f'vector store name: {vector_store_name}')
            table_name = "vstoremap"
            query_key = "vs_name"

            # Retrieve document name
            v_d_map = retrieve(azure_table_client.table_service_client, table_name, project_name, query_key, vector_store_name)

            #write actual doc names as headers
            header_title = v_d_map['doc_name']
            headers.append(f'{vector_store_name}: {header_title}')
            overview_worksheet.write(0, (header_count+1), f'{vector_store_name}: {header_title}', bold)
            logger.info(f"Wrote header '{vector_store_name}: {header_title}' to excel.")
            header_count+=1

    overview_worksheet.write(0, (header_count+1), 'Summary', bold)
    overview_worksheet.set_column(2,header_count+1, 150)

    #draw table border
    border_format=workbook.add_format({'border':1, 'border_color': '#000000'})
    overview_worksheet.conditional_format( 1,1,7,(header_count+1), {'type' : 'no_errors' ,'format' : border_format} )

    with alive_bar(len(excel_data), title="Writing to excel", force_tty=True) as bar:
        for i,(k,v) in enumerate(excel_data.items()):
            #write to first tab (overview)
            overview_worksheet.write(i+1,0, i+1, sn_format)
            criterion = mapping[k]
            overview_worksheet.write(i+1, 1, criterion, criteria_format)

            # write to individual tabs
            detailed_worksheet = workbook.add_worksheet(f'{k}')

            detailed_worksheet.write(0,0, f'{k} citations', bold)
            #headers in indiv tab
            for header in headers:
                detailed_worksheet.write(1, (headers.index(header)), header, bold)


            #check whether it is citation
            reponse_col_count = 1
            citation_col_count = 0
            for key in v.keys():
                citation_pattern = r'(.*?)_citations'
                citation_match = re.findall(citation_pattern, key)
                
                #not citation list
                if len(citation_match) == 0:
                    logger.info(f"Writing response: {k}, {key}.")
                    overview_worksheet.write(i+1, 1+reponse_col_count, v[key], response_format)
                    if key != 'summary':
                        detailed_worksheet.write(2, reponse_col_count-1, v[key], response_format)
                    reponse_col_count +=1

                #else if citation list, then retrieve from blob and append to excel
                elif len(citation_match) >0:
                    logger.info(f"Writing citations: {k}, {key}.")
                    citation_k = f"{citation_match[0]+'_citations'}"
                    citation_col_count +=1

                    for n,item in enumerate(v[citation_k]):
                        if '_gen_desc' in item[0]:
                            #print(item[0])
                            try:
                                original_file_name = item[0][:-13]
                                fig_path = str(original_file_name) + '.jpg'
                                #print(original_file_name)
        
                                blob_client = container_client_images.get_blob_client(fig_path)
                                file_bytes = blob_client.download_blob().readall()

                                # Create output directory if it doesn't exist
                                os.makedirs(output_images_folder, exist_ok=True)
                                output_path = os.path.join(output_images_folder, fig_path)

                                #convert file bytes to jpeg file locally to read and insert into excel
        
                                with open(output_path, 'wb') as file:
                                    file.write(file_bytes)

                                img = Image.open(output_path)
                                width,height = img.size 

                                cell_width = 300
                                cell_height = 300
            
                                detailed_worksheet.set_row_pixels(n+3, cell_height) #rows start from index 0. set the row height based on pixels units instead of excel character units (to fit the image)
                                
                                x_scale = cell_width/width
                                y_scale = cell_height/height
                                
                                detailed_worksheet.insert_image(n+3,citation_col_count-1, f'{output_path}', {'x_scale': x_scale, 'y_scale': y_scale})
                            except:
                                text = 'could not retrieve image '
                                detailed_worksheet.write_rich_string(n+3, citation_col_count-1, bold, item[1], '\n' + text, citation_format)
                        else:
                            try:
                                item_path = f'{item[0]}'
                                #print(item_path)
                                content = container_client.download_blob(item_path).readall()
                                text = content.decode()
                                detailed_worksheet.write_rich_string(n+3, citation_col_count-1, bold, item[1], '\n' + text, citation_format)
                            except:
                                text = 'could not retrieve from blob storage'
                                detailed_worksheet.write_rich_string(n+3, citation_col_count-1, bold, item[1], '\n' + text, citation_format)
  
            detailed_worksheet.set_column(0,citation_col_count-1, 150)
            bar()
        workbook.close()

# --- Initialize Blob Service Client ---
blob_service_client = BlobServiceClient.from_connection_string(CONNECTION_STRING)
container_client = blob_service_client.get_container_client(CONTAINER_NAME) 
container_client_images = blob_service_client.get_container_client(CONTAINER_NAME_IMAGES)


mapping = {'SN_1': 'Input Data and Output Result',
           'SN_2': 'Training Dataset Labelling',
           'SN_3': 'Evaluation of the Artificial Intelligence Model Using a Separate Test Dataset',
           'SN_4': 'Intended Workflow During Deployment',
           'SN_5': 'Performance Claims in the Instructions For Use (IFU)',
           'SN_6': 'Interval for Training Data Update Cycle',
           'SN_7': 'Final Assessment and Conclusion'}

# --- Main Execution Block ---
if __name__ == "__main__":

    # read pickle file of generated assessments
    obj = pd.read_pickle(r'assessment_reports/report.pkl')

    # Get project name
    project_name = get_project_name(obj)
    logger.info(f"Project name is {project_name}")

    # Get data to write to excel
    excel_data = {}
    process_pkl_to_excel(excel_data)

    #save excel_data
    os.makedirs("excel_folder", exist_ok=True)
    output_pkl = "excel_data" + ".pkl"
    excel_path = os.path.join("excel_folder", output_pkl)
    with open(excel_path, "wb") as output_file:
        pickle.dump(excel_data, output_file)

    #read excel data
    excel_data = pd.read_pickle(r'excel_folder/excel_data.pkl')

    generate_excel(excel_data)

    logger.info("Excel report generated.")
