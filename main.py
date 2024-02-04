from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
import streamlit as st
import os

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets['api_key']

st.title('Skimlit')
txt = st.text_area('Text to analyze', '''Ask the Question''')

# Load saved model
vectordb = None
persist_dir = "MyTextEmbedding"
embedding = OpenAIEmbeddings(model="text-embedding-3-large")
vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding)
chain = RetrievalQA.from_chain_type(OpenAI(), retriever=vectordb.as_retriever(), chain_type="stuff")

def get_results(abs_text):
    list1 = chain.run(abs_text)
    return list1

if st.button('Ask'):
    get_results(txt)
