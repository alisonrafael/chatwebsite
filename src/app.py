# instale as bibliotecas de requirements.txt com pip install
# crie o arquivo .streamlit/secrets.toml dentro da pasta src com a variável OPENAI_API_KEY="???", PERSISTENT_VECTORSTORE = "False" e  FILE_PATH = "./"

import streamlit as st
import os

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from chromadb.config import Settings
from langchain_community.document_loaders import OnlinePDFLoader


load_dotenv()


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PERSISTENT_VECTORSTORE = os.environ["PERSISTENT_VECTORSTORE"]
PERSISTENT_VECTORSTORE_DIR = "./"
FILE_PATH = os.environ["FILE_PATH"]

def init_llm():
    return ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.6, model_name="gpt-3.5-turbo")


def init_embeddings():
    return OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


def get_documents_from_pdfs_from_file():
    text_splitter = RecursiveCharacterTextSplitter()

    print("Processando PDFs de arquivo...")
    document_chunks = None
    file_pdfs = open(FILE_PATH + 'pdfs.txt', 'r')
    lines = file_pdfs.readlines()
    for line in lines:
        print("Carregando o conteúdo do PDF {}...".format(line.strip()))
        loader = OnlinePDFLoader(line.strip())
        document = loader.load()
        if document_chunks is None:
            document_chunks = text_splitter.split_documents(document)
        else:
            document_chunks += text_splitter.split_documents(document)

    print("Processamento de PDFs finalizado")
    return document_chunks


def get_documents_from_urls_from_file():
    text_splitter = RecursiveCharacterTextSplitter()

    print("Processando URLs de arquivo...")
    document_chunks = None
    file_urls = open(FILE_PATH + 'urls.txt', 'r')
    lines = file_urls.readlines()
    for line in lines:
        print("Carregando o conteúdo da URL {}...".format(line.strip()))
        loader = WebBaseLoader(line.strip())
        document = loader.load()
        if document_chunks is None:
            document_chunks = text_splitter.split_documents(document)
        else:
            document_chunks += text_splitter.split_documents(document)

    print("Processamento de URLs finalizado")
    return document_chunks


def get_context_retriever_chain(vector_store):
    llm = init_llm()

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user",
         "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain


def get_conversational_rag_chain(retriever_chain):
    llm = init_llm()

    # question_to_chatgpt = "Answer the user's questions based on the below context and do not give me any information about procedures and service features that are not mentioned in the provided context:"
    # question_to_chatgpt = "Responda a pergunta do usuário baseado no contexto abaixo. Caso a pergunta seja de conhecimentos que não pertence à Diretoria de Assuntos Acadêmicos da UEM diga que você só sabe questões relacionadas à Diretoria de Assuntos Acadêmicos da UEM: "
    # question_to_chatgpt = "You are a support resource that answers questions about X and the integration of X. If the question is not about X, how to use X, or cannot be answered based on the context, return the specific message saying that you know only about Diretoria de Assuntos Acadêmicos, do not make up an answer"
    question_to_chatgpt = "You are a support resource that answers questions about X and the integration of X based on the below context: "

    prompt = ChatPromptTemplate.from_messages([
        ("system", question_to_chatgpt + "\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])

    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)
    return create_retrieval_chain(retriever_chain, stuff_documents_chain)


def get_response(user_input):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })

    return response['answer']


print("Iniciando...")
st.set_page_config(page_title="Converse com a DAA", page_icon="🤖")
st.title("Converse com a DAA")
st.markdown("""
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">
            """, unsafe_allow_html=True)
st.markdown("""
           <div class="text-muted mb-3">
           Desenvolvido com OpenAI como parte integrante do SigaUEM (Sistema de Informações Gerenciais Acadêmicas) do NPD (Núcleo de Processamentos de dados) da UEM (Universidade Estadual de Maringá)
           </div>
           """, unsafe_allow_html=True)

# exemplo de sidebar
# with st.sidebar:
#     st.header("DAA")
#     st.info("Diretoria de Assuntos Acadêmicos")
#     website_url = "http://www.daa.uem.br" #st.text_input("Endereço do site")
#
# if website_url is None or website_url == "":
#     st.info("Por favor, digite o endereço web do site no campo ao lado")
#
# else:

if "chat_history" not in st.session_state:
    print("Criando histórico de conversas...")
    st.session_state.chat_history = [
        AIMessage(content="Olá, eu sou o robô da DAA. Como posso te ajudar?"),
    ]
else:
    print("Usando histórico de conversas da sessão")

if "vector_store" not in st.session_state:
    print("Criando conhecimento...")
    if PERSISTENT_VECTORSTORE.lower() == "true":
        print("Carregando base de conhecimento local...")
        st.session_state.vector_store = Chroma(persist_directory=PERSISTENT_VECTORSTORE_DIR,
                                               embedding_function=init_embeddings())
        print("A base de conhecimento foi carregada de uma já existente")
    else:
        print("Populando uma nova base de conhecimento...")
        document_chunks_from_urls = get_documents_from_urls_from_file()
        document_chunks_from_pdfs = get_documents_from_pdfs_from_file()
        document_chunks = document_chunks_from_urls + document_chunks_from_pdfs

        st.session_state.vector_store = Chroma.from_documents(documents=document_chunks,
                                                              embedding=init_embeddings(),
                                                              client_settings=Settings(chroma_db_impl='duckdb+parquet',
                                                                                       persist_directory=PERSISTENT_VECTORSTORE_DIR,
                                                                                       anonymized_telemetry=False))

        st.session_state.vector_store.persist()
        print("Nova base de conhecimento criada")
else:
    print("Usando base de conhecimento da sessão")

# input do usuário
user_query = st.chat_input("Escreva sua pergunta aqui...")
if user_query is not None and user_query != "":
    response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

# conversação
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)


