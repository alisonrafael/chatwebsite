# instale as bibliotecas com pip install streamlit langchain lanchain-openai beautifulsoup4 python-dotenv chromadb
# crie o arquivo .streamlit/secrets.toml dentro da pasta src com a variável OPENAI_API_KEY="???" e PERSISTENT_VECTORSTORE = "False"

import streamlit as st
import os

from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain, VectorDBQA
from langchain.chains.combine_documents import create_stuff_documents_chain
from chromadb.config import Settings
from langchain_openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.retrievers.self_query.base import SelfQueryRetriever

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PERSISTENT_VECTORSTORE = os.environ["PERSISTENT_VECTORSTORE"]
PERSISTENT_VECTORSTORE_DIR = "./"

def get_vectorstore_from_urls_from_file():
    text_splitter = RecursiveCharacterTextSplitter()

    print("Processando URLs de arquivo...")
    document_chunks = None
    file1 = open('urls.txt', 'r')
    lines = file1.readlines()
    for line in lines:
        print("Obtendo dados de {}".format(line.strip()))
        loader = WebBaseLoader(line.strip())
        document = loader.load()
        if document_chunks is None:
            document_chunks = text_splitter.split_documents(document)
        else:
            document_chunks += text_splitter.split_documents(document)

    vector_store = Chroma.from_documents(documents=document_chunks,
                                         embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
                                         client_settings=Settings(chroma_db_impl='duckdb+parquet',
                                                                  persist_directory=PERSISTENT_VECTORSTORE_DIR,
                                                                  anonymized_telemetry=False))
    print("Fim do processamento de URLs de arquivo")
    return vector_store


def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

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
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

    # question = "Answer the user's questions based on the below context and do not give me any information about procedures and service features that are not mentioned in the provided context:"
    # question_to_chatopenai = "Responda a pergunta do usuário baseado no contexto abaixo. Caso a pergunta seja de conhecimentos que não pertence à Diretoria de Assuntos Acadêmicos da UEM diga que você só sabe questões relacionadas à Diretoria de Assuntos Acadêmicos da UEM: "
    question_to_chatopenai = "You are a support resource that answers questions about X and the integration of X. If the question is not about X, how to use X, or cannot be answered based on the context, return the specific message saying that you know only about Diretoria de Assuntos Acadêmicos, do not make up an answer"

    prompt = ChatPromptTemplate.from_messages([
        ("system", question_to_chatopenai + "\n\n{context}"),
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

st.markdown("""<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-QWTKZyjpPEjISv5WaRU9OFeRpok6YctnYmDr5pNlyT2bRjXh0JMhjY6hW+ALEwIH" crossorigin="anonymous">""", unsafe_allow_html=True)
aviso = """
       <div class="text-muted mb-3">
       Desenvolvido com OpenAI como parte integrante do SigaUEM (Sistema de Informações Gerenciais Acadêmicas) do NPD (Núcleo de Processamentos de dados) da UEM (Universidade Estadual de Maringá)
       </div>
       """
st.markdown(aviso, unsafe_allow_html=True)

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
    if PERSISTENT_VECTORSTORE:
        print("Carregando base de conhecimento local...")
        st.session_state.vector_store = Chroma(persist_directory=PERSISTENT_VECTORSTORE_DIR,
                                               embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))
        print("A base de conhecimento foi carregada de uma já existente")
    else:
        print("Populando uma nova base de conhecimento...")
        st.session_state.vector_store = get_vectorstore_from_urls_from_file()
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


