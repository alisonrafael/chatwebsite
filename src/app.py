# instale as bibliotecas com pip install streamlit langchain lanchain-openai beautifulsoup4 python-dotenv chromadb
# crie o arquivo .streamlit/secrets.toml dentro da pasta src com a variável OPENAI_API_KEY="???"

import streamlit as st
import os
import dill as pickle
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain, ConversationalRetrievalChain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
PERSIST = False

def get_vectorstore_from_url_daa():
    text_splitter = RecursiveCharacterTextSplitter()

    loader = WebBaseLoader("http://www.daa.uem.br")
    document = loader.load()
    document_chunks = text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/estude-na-uem/transferencia-externa")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/academicos/graduacao/transferencia-interna-de-turno-campus-polo-e-curso")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/academicos/graduacao/informacoes-sobre-secretaria/transferencia-da-uem-para-outra-instituicao-de-ensino-superior")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/servidores-equipe-daa")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/horarios-2023")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/calendario-academico")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/estude-na-uem")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/academicos")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/editais-e-portarias/editais-e-portarias-2024")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/taxas-e-requerimentos")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/taxas-e-requerimentos/graduacao/requerimentos-academicos")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/revalidacao-de-diploma-estrangeiro-graduacao-1")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/contatos-e-servicos")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/academicos/graduacao/informacoes-sobre-secretaria/aproveitamento-de-estudos-de-disciplinas-cursadas-na-uem")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/academicos/graduacao/informacoes-sobre-secretaria/aproveitamento-de-estudos-de-disciplinas-cursadas-em-outra-instituicao")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/estude-na-uem/pas-vestibular")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/estude-na-uem/sisu")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/estude-na-uem/vestibular-ead")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/estude-na-uem/vagas-remanescentes")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/estude-na-uem/programa-de-estudantes-convenio-de-graduacao-peg-g")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/estude-na-uem/aluno-indigena")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/estude-na-uem/ex-officio")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/estude-na-uem/reingresso-de-alunos-desligados")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/estude-na-uem/portador-de-diploma")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/academicos/graduacao/solicitacoes/informacoes-sobre-solicitacao-de-carteirinha")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/academicos/graduacao/formatura/colacao-de-grau")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/academicos/graduacao/formatura/solicitacao-de-colacao-de-grau-especial-e-antecipada")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/academicos/graduacao/solicitacoes/historico-escolar")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/taxas-e-requerimentos/graduacao")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/academicos/graduacao/informacoes-sobre-secretaria/historico-escola-oficial")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/academicos/graduacao/informacoes-sobre-secretaria/permuta-de-turno")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/revalidacao-de-diploma-estrangeiro-graduacao-1")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/academicos/graduacao/ajustes-de-matricula/ajuste-dematricula-em-disciplinas-turmas")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/academicos/graduacao/horarios-de-aulas-orientacoes")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/academicos/graduacao/renovacao-de-matricula-1/renovacao-de-matricula")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.scs.uem.br/2019/cep/022cep2019.htm")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/academicos/graduacao/reserva-esporadica-de-sala-de-aula/reserva-esporadica-de-sala-de-aula-orientacoes")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/academicos/graduacao/renovacao-de-retardatario/renovacao-de-retardatario-2021-orientacoes")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/academicos/graduacao/informacoes-sobre-secretaria/atividade-domiciliar")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/academicos/graduacao/informacoes-sobre-secretaria/dispensa-para-jogos")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/academicos/graduacao/informacoes-sobre-secretaria/emissao-de-programas-de-disciplinas-ementas-curriculares-criterios-de-avaliacao")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/academicos/graduacao/informacoes-sobre-secretaria/plano-de-acompanhamento-de-estudos-pae")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/academicos/pos-graduacao/orientacoes-gerais-para-solicitacoes-1/historico-escolar-oficial")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/academicos/pos-graduacao/orientacoes-gerais-para-solicitacoes-1/atestado-de-matricula-oficial")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/academicos/pos-graduacao/orientacoes-gerais-para-solicitacoes-1/certificados-de-curso-de-especializacao-e-residencias")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/academicos/pos-graduacao/orientacoes-gerais-para-solicitacoes-1/diplomas-de-mestrado-e-doutorado")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    loader = WebBaseLoader("http://www.daa.uem.br/diploma-digital")
    document = loader.load()
    document_chunks += text_splitter.split_documents(document)

    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())

    return vector_store

def get_vectorstore_from_url(url):
    # get the text in document form
    loader = WebBaseLoader(url)
    document = loader.load()
    
    # split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)
    
    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY))

    return vector_store

def get_context_retriever_chain(vector_store):
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_messages([
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
      ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    
    return retriever_chain
    
def get_conversational_rag_chain(retriever_chain):
    
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY)

    #question = "Answer the user's questions based on the below context and do not give me any information about procedures and service features that are not mentioned in the provided context:"
    question_to_chatopenai = "Responda a pergunta do usuário baseado no contexto abaixo e se a resposta não estiver no contexto diga que você só sabe coisas da Diretoria de Assuntos Acadêmicos da UEM: "

    prompt = ChatPromptTemplate.from_messages([
      ("system", question_to_chatopenai + "\n\n{context}"),
      MessagesPlaceholder(variable_name="chat_history"),
      ("user", "{input}"),
    ])
    
    stuff_documents_chain = create_stuff_documents_chain(llm,prompt)
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

# sidebar
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
    print("Usando histórico de conversas da sessão...")

#if "vector_store" not in st.session_state:
#    st.session_state.vector_store = get_vectorstore_from_url_daa()

# se não temos na sessão a base de conhecimento, carregamos do disco ou criamos novamente
if "vector_store" not in st.session_state:
    print("Criando conhecimento...")
    if PERSIST and os.path.exists(PERSIST):
        #st.session_state.vector_store = from disk
        print("A base de conhecimento foi carregada de uma já existente")
    else:
        print("Populando uma nova base de conhecimento...")
        st.session_state.vector_store = get_vectorstore_from_url_daa()
        # save st.session_state.vector_store to disk
        print("Nova base de conhecimento criada")
else:
    print("Usando base de conhecimento da sessão")

# user input
user_query = st.chat_input("Escreva sua pergunta aqui...")
if user_query is not None and user_query != "":
    response = get_response(user_query)
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=response))

# conversation
for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message("AI"):
            st.write(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human"):
            st.write(message.content)
