import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import JinaEmbeddings
#import faiss
from langchain_community.vectorstores import FAISS 
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_text_splitters import  RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough 
from langchain_core.prompts import ChatPromptTemplate

# from dotenv import load_dotenv
# load_dotenv()


def vector_embedding():

    if "vectors" not in st.session_state:
        st.session_state.embeddings=embeddings

        temp_file = "./temp.pdf"
        with open(temp_file, "wb") as file:
            file.write(uploaded_file.getvalue())
            file_name = uploaded_file.name

        st.session_state.loader=PyPDFLoader(temp_file)
        st.session_state.docs=st.session_state.loader.load_and_split() ## Document Loading
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=700,chunk_overlap=50) ## Chunk Creation
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs) #splitting
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) #vector OpenAI embeddings

# import os
# Load the groq API KEY
# groq_api_key = os.getenv('GROQ_API_KEY')
# jina_api_key = os.getenv('JINA_API_KEY')


groq_api_key = st.secrets['GROQ_API_KEY']
jina_api_key = st.secrets['JINA_API_KEY']

st.title("ChatGroq with Llama3.2")

llm = ChatGroq(groq_api_key=groq_api_key,
               model = "llama-3.1-8b-instant")

embeddings = JinaEmbeddings(jina_api_key=jina_api_key, model_name="jina-embeddings-v2-base-en")

#print(retriever.invoke(question))
prompt = """
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, just say that you don't know.
    Answer in bullet points. Make sure your answer is relevant to the question and it is answered from the context only.
    Question: {question} 
    Context: {context} 
    Answer:
"""


question=st.text_input("Enter Your Question From Doduments")
uploaded_file = st.file_uploader("Choose a pdf file", type="pdf")

st.session_state.flag = False


if st.button("Documents Embedding") and uploaded_file is not None:
    vector_embedding()
    st.session_state.flag = True
    st.write("Vector Store DB Is Ready")


prompt = ChatPromptTemplate.from_template(prompt)

def format_docs(docs):
    return '\n\n'.join([doc.page_content for doc in docs])

import time

if question:

    retriever = st.session_state.vectors.as_retriever(search_type = 'similarity', 
                                      search_kwargs = {'k': 3})
    
    docs = retriever.invoke(question)
    rag_chain = (
        {"context": retriever|format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
        )
    start=time.process_time()
    response=rag_chain.invoke(question)
    print("Response time :",time.process_time()-start)
    st.write(response)

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for doc in docs:
            st.write(doc['page_content'])
            st.write("--------------------------------")

