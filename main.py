# from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
import os
# from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import streamlit as st

# Load environment variables from .env file
load_dotenv()

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets['api_key']



# vectordb = None
# persist_dir = "MyTextEmbedding"
# embedding = OpenAIEmbeddings()
# vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding)
# chain = RetrievalQA.from_chain_type(OpenAI(), retriever=vectordb.as_retriever(), chain_type="stuff")

st.title('Skimlit')
txt = st.text_area('Text to analyze', '''Ask the Question''')

print("hi")

# Load saved model
vectordb = None
persist_dir = "MyTextEmbedding"
embedding = OpenAIEmbeddings(model="gpt-4")
vectordb = Chroma(persist_directory=persist_dir, embedding_function=embedding)
chain = RetrievalQA.from_chain_type(OpenAI(), retriever=vectordb.as_retriever(), chain_type="stuff")

def get_results(abs_text):
    list1 = chain.run(abs_text)
    list1['result'] = list1['result'].replace('\n', '')
    list1['result']

print("\n\nAI formatted abstract is given below:\n")

if st.button('Ask'):
    get_results(txt)
