ASSISTANT1_PROMPT = """
You are an expert in healthcare regulation, specializing in evaluating submissions of Artificial Intelligence medical devices (AI-MD).
Your task is to search through all the sources provided and evaluate the information based on the specified criterion.
Answer the questions relating to the criterion and support your assessment by citing the relevant information.
Be as critical as possible in your assessment and scrutinize the documents provided.
"""

ASSISTANT2_PROMPT = """
You are an expert in healthcare regulation, specializing in evaluating submissions of Artificial Intelligence medical devices (AI-MD).
Your task is to read through all the content provided and generate a summary report.
The summary report should consists of:
1) Are the content enough to support the criterion being addressed?
2) Are the claims mentioned in the user manual been substantiated by the supporting documents?
3) Which are the supporting documents that agree with the claims in the user manual? Concisely explain how each supporting document agree with the claims in the following format:
document: <reason>
4) Which are the supporting documents that contradict? Concisely explain how each supporting document contradict the claims in the following format:
document: <reason>
5) Which are the supporting documents that are unclear or not used at all to support the criterion? Concisely explain how each supporting document is unclear in the following format:
document: <reason>

Be as critical as possible in your assessment and scrutinize the documents provided.
"""


PROMPT_Q1 = """ 
The criterion is as follows:
Criterion: Input data and features to generate corresponding output
Questions:
1) What are the various input data types and features/attributes used by the Artificial Intelligence Medical Device to generate its output?
2) What is the rationale for selecting the input data and features/attributes?
3) How does the device handle different types of input data, and are there specific requirements for each input data type?
4) Is any pre-processing of input data required? If yes, what are the specific pre-processing steps and their rationale? 
5) What are the expected input data formats, and how does the system manage data that doesn't meet these specifications? 
6) What Quality Control (QC) measures are built into the medical device to ensure only relevant and high-quality input data is processed?
7) How does the medical device validate the completeness and accuracy of the input data before processing? 
8) What is the expected output from the Artificial Intelligence Medical Device?
9) How are other data sources (e.g., patient historical records, physiological signals, medication records, handwritten text, literature reviews) incorporated as input data, if applicable?

First, check if the provided sources are sufficient to address the above criteria. 
If not, answer \"There is not enough information to address this criterion.\"
If yes, please proceed to answer the questions and providing the relevant sources.

Remember to cite the information when providing your assessment.
Remember to only answer using information from the documents provided.
"""

PROMPT_Q2 = """
The criterion is as follows:
Criterion: Source, size and attribution of training, validation and test datasets
Questions:
1) What are the sources and sizes of the training, validation, and test datasets?
2) How is the training dataset being labelled, curated, and annotated?
3) What is the process for dataset cleaning, and how is missing data imputation handled?
4) What control measures are in place to ensure all potential sources of biases in selecting the training, validation, and test datasets are adequately addressed and managed?
5) How is the separation between training and validation datasets maintained to ensure no duplication?
6) Are the choice and sufficiency of the datasets used justified? 
7) What methods are used to compare the performance of the AI model to state-of-the-art techniques or reference standards in the field?
8) What quality control measures are in place to ensure consistent and accurate labelling across the datasets?
9) How is the representativeness of the datasets assessed to ensure they cover the intended use population and conditions?
10) Are there any limitations in the datasets that could affect the AI model's performance or generalisability?

First, check if the provided sources are sufficient to address the above criteria. 
If not, answer \"There is not enough information to address this criterion.\"
If yes, please proceed to answer the questions and providing the relevant sources.

Remember to cite the information when providing your assessment.
Remember to only answer using information from the documents provided.
"""

PROMPT_Q3 = """ 
The criterion is as follows:
Criterion: Artificial Intelligence Model description and selection
Questions:
1) Which AI algorithm is employed, and does it build upon any pre-existing model? 
2) What evidence supports the suitability of the selected model for its intended application? 
3) What are the known limitations of the model, and what mitigating measures have been implemented?
4) How is the Artificial Intelligence model evaluation performed?
5) Which performance indicators are used to gauge the model's effectiveness, and what is the rationale behind their selection? 
6) What are the results of the model evaluation using the selected metrics?
7) How does the size of the test dataset compare to the training dataset? (e.g. 80% / 20% split between training and validation)

First, check if the provided sources are sufficient to address the above criteria. 
If not, answer \"There is not enough information to address this criterion.\"
If yes, please proceed to answer the questions and providing the relevant sources.

Remember to cite the information when providing your assessment.
Remember to only answer using information from the documents provided.
"""

PROMPT_Q4 = """ 
The criterion is as follows:
Criterion: Device Workflow including how the output result should be used
Questions:
1) What is the intended or recommended clinical workflow during the deployment of the device? Provide a detailed explanation of each step.
2) Does the system require human intervention? If so, at which specific stage(s) of the workflow does it occur?
3) What is the degree or extent of human intervention required, and how does it integrate with the automated processes of the device?

First, check if the provided sources are sufficient to address the above criteria. 
If not, answer \"There is not enough information to address this criterion.\"
If yes, please proceed to answer the questions and providing the relevant sources.

Remember to cite the information when providing your assessment.
Remember to only answer using information from the documents provided.
"""

PROMPT_Q5 = """ 
The criterion is as follows:
Criterion: Test protocol for performance verification and validation of the Artificial Intelligence Medical Devices (AI-MD)
Questions:
1) What performance claims are made for the Artificial Intelligence model?
2) What is the test protocol to validate the performance of machine learning function? Provide the test results and the clinical validation of the machine learning function.
3) What control measures are in place to detect extremes or outliers in the AI model's performance?
4) What are the known limitations of the AI Medical Device and its operating system, and how are these communicated to the end user?
5) What evidence demonstrates a valid clinical association between the AI Medical Device's output and the targeted clinical condition?

First, check if the provided sources are sufficient to address the above criteria. 
If not, answer \"There is not enough information to address this criterion.\"
If yes, please proceed to answer the questions and providing the relevant sources.

Remember to cite the information when providing your assessment.
Remember to only answer using information from the documents provided.
"""

PROMPT_Q6 = """ 
The criterion is as follows:
Criterion: Interval for training data update cycle
Questions:
1) What is the interval for the training data update cycle, if applicable?
2) What plans are in place for continuous performance monitoring and updates, if applicable?
3) How will feedback from users be collected and used for improvements, if applicable?
4) What is the process for updating the Machine Learning model post-deployment?

First, check if the provided sources are sufficient to address the above criteria. 
If not, answer \"There is not enough information to address this criterion.\"
If yes, please proceed to answer the questions and providing the relevant sources.

Remember to cite the information when providing your assessment.
Remember to only answer using information from the documents provided.
"""

PROMPT_FINAL = """ 
The criterion is as follows:
Criterion: Final assessment and conclusion.
Questions:
1) Are the input data and features/ attributes used to generate the corresponding output sufficient?
2) Are the source, size and attribution of training, validation and test datasets valid?
3) Are the Artificial Intelligence model description and selection complete? 
4) Is the device workflow including how the output result should be used complete?
5) Is the Test protocol for performance verification and validation of the Artificial Intelligence Medical Devices valid?
6) Is the Interval for training data update cycle mentioned?

Give a short conclusion based on the questions above and overall assessment.
Remember to cite the information when providing your assessment.
"""

PROMPTS_DICT = {"Input data and features/ attributes used to generate the corresponding output": PROMPT_Q1,
                "Source, size and attribution of training, validation and test datasets": PROMPT_Q2,
                "Artificial Intelligence model description and selection": PROMPT_Q3, 
                "Device workflow including how the output result should be used": PROMPT_Q4,
                "Test protocol for performance verification and validation of the Artificial Intelligence Medical Devices": PROMPT_Q5,
                "Interval for training data update cycle": PROMPT_Q6,
                "Assessment/ Remarks, if any": PROMPT_FINAL}