import os
import boto3
import streamlit as st
from langchain.llms.bedrock import Bedrock
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import time

os.environ["AWS_PROFILE"] = "default"

# Initialize the Bedrock client
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')

# Add model selection here
model_options = {
    "Claude 2": "anthropic.claude-v2:1",
    "Titan Text Express": "amazon.titan-text-express-v1"
}
selected_model_name = st.sidebar.selectbox("Choose Model", list(model_options.keys()))
modelID = model_options[selected_model_name]

if selected_model_name == "Claude 2":
    model_kwargs = {"temperature": 0.4}
elif selected_model_name == "Titan Text Express":
    model_kwargs = {"temperature": 0.4, "topP": 0.4, "maxTokenCount": 2048}

llm = Bedrock(model_id=modelID, client=bedrock_client, model_kwargs=model_kwargs)

def my_chatbot(lang, txt):
    prompt_template = "You are a chatbot. You are in {language}.\n\n{freeform_text}"
    prompt = PromptTemplate(input_variables=["language", "freeform_text"], template=prompt_template)

    
    start_time = time.time()

    bedrock_chain = LLMChain(llm=llm, prompt=prompt)
    result = bedrock_chain({'language': lang, 'freeform_text': txt})['text']

    end_time = time.time()
    
    elapsed_time = end_time - start_time

    return result, elapsed_time

# Streamlit stuff
st.title("Ask a legally-compliant AWS friend")

lang = st.sidebar.selectbox("Choose Language", ["english", "spanish", "chinese"])
user_question = st.sidebar.text_area("What do you want to ask?", max_chars=1000)

if user_question:
    answer, elapsed_time = my_chatbot(lang, user_question)
    st.write(answer)
    st.write(f"Response time: {elapsed_time:.2f} seconds")
