import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI



# Load environment variables
load_dotenv()
groq_api_key = os.getenv('GROQ_API_KEY')

# Initialize session state if not already initialized

if "vectors" not in st.session_state:
    st.session_state.embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    website_urls = [
        "https://sallysbakingaddiction.com/chewy-chocolate-chip-cookies/",
        "https://www.loveandlemons.com/oatmeal-cookies/"
        "https://sallysbakingaddiction.com/soft-peanut-butter-cookie-recipe/"
        "https://www.modernhoney.com/the-best-snickerdoodle-cookie-recipe/"
        "https://bromabakery.com/double-chocolate-chip-cookies/"
        "https://sallysbakingaddiction.com/best-gingerbread-cookies/"
        "https://www.recipetineats.com/christmas-cookies-vanilla-biscuits/"
        "https://handletheheat.com/glazed-lemon-cookies/"
    ]
    st.session_state.loader = WebBaseLoader(website_urls)
    st.session_state.docs = st.session_state.loader.load()

    # Initialize and split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    st.session_state.final_documents = text_splitter.split_documents(st.session_state.docs)

    # Create FAISS vectors from documents and embeddings
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

# Streamlit interface
st.title("Cookie Recipes")
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma-7b-It")

prompt_template = """provide the recipe based on the context provided and make sure that it is accurate and the steps are 
given in detail.Provide only the recipe nothing else.

<context>
{context}
<context>
Questions:{input}
"""

prompt = ChatPromptTemplate.from_template(prompt_template)

document_chain = create_stuff_documents_chain(llm, prompt)
retriever = st.session_state.vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)
 
prompt_input = st.text_input("Enter your question")

if prompt_input:
    response = retrieval_chain.invoke({"input": prompt_input})
    st.write(response['answer'])
