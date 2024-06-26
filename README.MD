# Test Bedrock Models via Local ChatBot using Streamlit

### Step 1
Install boto3 (the AWS SDK for Python), Langchain, and Streamlit:

`pip install boto3 streamlit langchain`

### Step 2
Make sure you have [access](https://us-east-1.console.aws.amazon.com/bedrock/home?region=us-east-1#/modelaccess) to Bedrock models you're testing by requesting in the console. Then, configure the `modelID` with the model of [your choice](https://docs.aws.amazon.com/bedrock/latest/userguide/model-ids-arns.html) and set the region with `region_name`.

### Step 3
Run the Bedrock model locally as a chatbot: `streamlit run <yourfile.py>`
