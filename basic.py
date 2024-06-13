import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from PyPDF2 import PdfReader
import streamlit as st

def get_text(doc_name):
    text = ""
    pdf_reader = PdfReader(doc_name)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def convert_to_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)
    return chunks 

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in
    the provided context, just say, "answer is not available in the context". Don't provide a wrong answer.
    
    Context: {context}
    Question: {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.load_local("faiss_index", embeddings,allow_dangerous_deserialization=True)

    docs = db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])

def main():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")

    path = r"C:\Users\DELL\Desktop\Langchain\Rag_Pipeline\SAMPLE.pdf"
    text_string = get_text(path)
    converted_chunks = convert_to_chunks(text_string)

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    db = FAISS.from_texts(converted_chunks, embedding=embeddings)
    db.save_local("faiss_index")

    st.title("Basic retrieval model")
    user_question = st.text_input("Ask a question:")
    if user_question:
        user_input(user_question)

if __name__ == "__main__":
    main()
