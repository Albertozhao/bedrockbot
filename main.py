import os
import boto3
import streamlit as st
from langchain.llms.bedrock import Bedrock
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate

os.environ["AWS_PROFILE"] = "default"  # gotta tell it which AWS profile to use

# let's get the Bedrock client ready
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')

modelID = "amazon.titan-text-express-v1"  # this is the cool model we're using

# setting up our Bedrock model, tweaking some stuff
llm = Bedrock(model_id=modelID, client=bedrock_client,
              model_kwargs={"temperature": 0.4, "topP": 0.4, "maxTokenCount": 2048})

def my_chatbot(lang, txt):  # shortened variable names, cuz why not?
    prompt_template = "You are a chatbot. You are in {language}.\n\n{freeform_text}"
    prompt = PromptTemplate(input_variables=["language", "freeform_text"], template=prompt_template)

    bedrock_chain = LLMChain(llm=llm, prompt=prompt)

    return bedrock_chain({'language': lang, 'freeform_text': txt})['text']

# Streamlit stuff starts here
st.title("Ask a legally-compliant AWS friend")

lang = st.sidebar.selectbox("Choose Language", ["english", "spanish", "chinese"])

if lang:
    user_question = st.sidebar.text_area("What do you want to ask?", max_chars=1000)

if user_question:
    answer = my_chatbot(lang, user_question)
    st.write(answer)
