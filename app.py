# ============================================================
# app.py — Sistema de Auditoria Inteligente TCE/PR
# Trabalho de Conclusão de Curso
# Arquitetura: RAG (Retrieval-Augmented Generation)
# ============================================================

import os
import logging

import streamlit as st
from dotenv import load_dotenv
import PyPDF2

from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# ============================================================
# LOGS (Passo 7 — Auditoria)
# ============================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("auditoria.log")
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

# ============================================================
# CSS PERSONALIZADO — Tema claro institucional TCE/PR
# ============================================================
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=Source+Sans+3:wght@300;400;600&display=swap');

/* === BASE === */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #f5f2eb !important;
    color: #2c2a24 !important;
    font-family: 'Source Sans 3', sans-serif !important;
}

[data-testid="stHeader"] {
    background-color: #f5f2eb !important;
    border-bottom: 1px solid #ddd8cc !important;
}

/* === SIDEBAR === */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 2px solid #e8e0d0 !important;
}

[data-testid="stSidebar"] * {
    color: #2c2a24 !important;
}

/* === SIDEBAR CONTEÚDO === */
.sb-logo {
    font-family: 'Playfair Display', serif;
    font-size: 1.05rem;
    color: #8a6a1a;
    font-weight: 700;
    border-bottom: 2px solid #e8e0d0;
    padding-bottom: 0.75rem;
    margin-bottom: 1rem;
}

.sb-label {
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #9a8a6a;
    margin-bottom: 0.5rem;
    margin-top: 1.2rem;
}

.exemplo {
    background: #faf7f0;
    border-left: 3px solid #c4a050;
    padding: 0.4rem 0.7rem;
    margin: 0.3rem 0;
    border-radius: 0 4px 4px 0;
    font-size: 0.8rem;
    color: #5a4a2a;
    font-style: italic;
}

.sb-footer {
    font-size: 0.65rem;
    color: #b0a080;
    text-align: center;
    padding-top: 0.5rem;
    border-top: 1px solid #e8e0d0;
    margin-top: auto;
}

/* === BOTÃO PROCESSAR === */
.stButton > button {
    background: linear-gradient(135deg, #c4a050 0%, #a07830 100%) !important;
    color: #ffffff !important;
    font-family: 'Source Sans 3', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    border: none !important;
    border-radius: 4px !important;
    padding: 0.65rem 1.5rem !important;
    transition: all 0.2s ease !important;
    width: 100% !important;
    box-shadow: 0 2px 8px rgba(160,120,48,0.25) !important;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #d4b060 0%, #b08840 100%) !important;
    box-shadow: 0 4px 16px rgba(160,120,48,0.35) !important;
    transform: translateY(-1px) !important;
}

/* === FILE UPLOADER — legível === */
[data-testid="stFileUploader"] {
    background: #faf7f0 !important;
    border: 2px dashed #c4a050 !important;
    border-radius: 6px !important;
}

[data-testid="stFileUploader"] * {
    color: #5a4a2a !important;
    font-size: 0.85rem !important;
}

[data-testid="stFileUploaderDropzone"] {
    background: #faf7f0 !important;
    padding: 0.75rem !important;
}

/* === CABEÇALHO PRINCIPAL === */
.tce-header {
    background: linear-gradient(135deg, #ffffff 0%, #faf7f0 100%);
    border: 1px solid #e8e0d0;
    border-radius: 8px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}

.tce-logo-linha {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 0.75rem;
}

.tce-brasao {
    font-size: 2.8rem;
}

.tce-titulo {
    font-family: 'Playfair Display', serif !important;
    font-size: 2rem !important;
    font-weight: 900 !important;
    color: #8a6a1a !important;
    letter-spacing: -0.5px;
    line-height: 1.1;
    margin: 0 !important;
}

.tce-subtitulo {
    font-size: 0.75rem;
    color: #9a8a6a;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    margin-top: 0.3rem;
}

.tce-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #faf7f0;
    border: 1px solid #ddd0a0;
    color: #8a6a1a;
    font-size: 0.68rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    padding: 0.3rem 0.85rem;
    border-radius: 3px;
}

/* === CHAT MESSAGES === */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.5rem 0 !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
    background: #faf7f0 !important;
    border-left: 3px solid #c4a050 !important;
    padding-left: 1rem !important;
    border-radius: 0 6px 6px 0 !important;
    margin-bottom: 0.5rem !important;
}

[data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
    background: #ffffff !important;
    border: 1px solid #e8e0d0 !important;
    border-left: 3px solid #8a6a1a !important;
    padding-left: 1rem !important;
    border-radius: 0 6px 6px 0 !important;
    margin-bottom: 0.5rem !important;
}

/* === CHAT INPUT === */
[data-testid="stChatInput"] {
    background: #ffffff !important;
    border: 2px solid #e8e0d0 !important;
    border-radius: 6px !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #c4a050 !important;
}

[data-testid="stChatInput"] textarea {
    background: transparent !important;
    color: #2c2a24 !important;
    font-family: 'Source Sans 3', sans-serif !important;
}

/* === ALERTS === */
[data-testid="stAlert"] {
    border-radius: 4px !important;
    font-size: 0.85rem !important;
}

/* === BOAS VINDAS === */
.boas-vindas {
    text-align: center;
    padding: 3.5rem 2rem;
    background: #ffffff;
    border: 1px solid #e8e0d0;
    border-radius: 8px;
}

.boas-vindas-icone {
    font-size: 3.5rem;
    margin-bottom: 1rem;
}

.boas-vindas-titulo {
    font-family: 'Playfair Display', serif;
    font-size: 1.4rem;
    color: #8a6a1a;
    margin-bottom: 0.75rem;
}

.boas-vindas-texto {
    color: #7a6a4a;
    font-size: 0.95rem;
    line-height: 1.7;
    max-width: 480px;
    margin: 0 auto;
}

.passos {
    display: flex;
    justify-content: center;
    gap: 1.5rem;
    margin-top: 1.5rem;
    flex-wrap: wrap;
}

.passo {
    background: #faf7f0;
    border: 1px solid #e8e0d0;
    border-radius: 6px;
    padding: 0.6rem 1rem;
    font-size: 0.78rem;
    color: #8a6a1a;
    font-weight: 600;
}

.linha-divisoria {
    border: none;
    border-top: 1px solid #e8e0d0;
    margin: 1rem 0;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""

# ============================================================
# FUNÇÃO 1: get_texto — Extrai texto do PDF (Passo 2)
# ============================================================
def get_texto(arquivo_pdf):
    texto_completo = ""
    try:
        leitor_pdf = PyPDF2.PdfReader(arquivo_pdf)
        numero_paginas = len(leitor_pdf.pages)
        logger.info(f"PDF carregado. Páginas: {numero_paginas}")
        for i in range(numero_paginas):
            texto_completo += leitor_pdf.pages[i].extract_text() or ""
    except PyPDF2.errors.PdfReadError as e:
        logger.error(f"Erro ao ler PDF: {e}")
        st.error("Erro: PDF corrompido ou protegido por senha.")
        return None
    except Exception as e:
        logger.error(f"Erro inesperado: {e}")
        st.error(f"Erro ao processar o PDF: {e}")
        return None

    if not texto_completo.strip():
        logger.warning("Nenhum texto extraído do PDF.")
        st.warning("Nenhum texto encontrado. O PDF pode ser composto por imagens.")
        return None

    return texto_completo


# ============================================================
# FUNÇÃO 2: get_chunks — Divide em blocos (Passo 3)
# ============================================================
def get_chunks(texto):
    try:
        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = splitter.split_text(texto)
        logger.info(f"Texto segmentado em {len(chunks)} blocos.")
        return chunks
    except Exception as e:
        logger.error(f"Erro ao segmentar texto: {e}")
        st.error(f"Erro ao segmentar o documento: {e}")
        return None


# ============================================================
# FUNÇÃO 3: get_vectorstore — Embeddings + FAISS (Passo 4)
# ============================================================
def get_vectorstore(chunks):
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY não encontrada.")
            st.error("Chave de API não encontrada. Verifique o arquivo .env")
            return None

        logger.info("Gerando embeddings com text-embedding-ada-002...")
        embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002",
            openai_api_key=api_key
        )
        banco = FAISS.from_texts(texts=chunks, embedding=embeddings)
        logger.info("Banco vetorial FAISS criado com sucesso.")
        return banco
    except Exception as e:
        logger.error(f"Erro ao criar banco vetorial: {e}")
        st.error(f"Erro ao gerar embeddings: {e}")
        return None


# ============================================================
# FUNÇÃO 4: get_chat — Monta a cadeia RAG (Passos 5 e 6)
# ============================================================
def get_chat(banco_vetorial):
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        llm = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=api_key)
        retriever = banco_vetorial.as_retriever(search_kwargs={"k": 4})

        prompt = ChatPromptTemplate.from_messages([
            ("system", """Você é um auditor especialista do Tribunal de Contas do \
Estado do Paraná (TCE/PR). Analise os trechos abaixo extraídos de uma peça processual \
e responda à pergunta do auditor com precisão técnica.

Identifique indícios de irregularidades como superfaturamento, dano ao erário, \
fraudes em licitações ou desvio de recursos públicos quando presentes no texto.
Se não houver informação suficiente no documento, informe claramente.

Trechos relevantes do documento:
{context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ])

        def formatar_docs(docs):
            return "\n\n---\n\n".join(doc.page_content for doc in docs)

        cadeia = (
            {
                "context": retriever | formatar_docs,
                "question": RunnablePassthrough(),
                "chat_history": RunnablePassthrough(),
            }
            | prompt
            | llm
            | StrOutputParser()
        )

        logger.info("Cadeia RAG criada com sucesso.")
        return cadeia, retriever

    except Exception as e:
        logger.error(f"Erro ao criar cadeia: {e}")
        st.error(f"Erro ao configurar o assistente: {e}")
        return None, None


# ============================================================
# FUNÇÃO 5: entrada_usuario — Processa pergunta (Passos 5 e 6)
# ============================================================
def entrada_usuario(pergunta):
    try:
        logger.info(f"Consulta: '{pergunta}'")

        docs = st.session_state.retriever.invoke(pergunta)
        contexto = "\n\n---\n\n".join(doc.page_content for doc in docs)

        resposta = st.session_state.cadeia.invoke({
            "context": contexto,
            "question": pergunta,
            "chat_history": st.session_state.chat_history,
        })

        st.session_state.chat_history.append(HumanMessage(content=pergunta))
        st.session_state.chat_history.append(AIMessage(content=resposta))

        for msg in st.session_state.chat_history:
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    st.write(msg.content)
            else:
                with st.chat_message("assistant"):
                    st.write(msg.content)

        logger.info("Resposta entregue com sucesso.")

    except Exception as e:
        logger.error(f"Erro na consulta: {e}")
        st.error(f"Erro ao processar sua pergunta: {e}")


# ============================================================
# INTERFACE PRINCIPAL
# ============================================================
def main():
    st.set_page_config(
        page_title="Auditoria Inteligente — TCE/PR",
        page_icon="⚖️",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown(CSS, unsafe_allow_html=True)

    if "cadeia" not in st.session_state:
        st.session_state.cadeia = None
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # --------------------------------------------------------
    # BARRA LATERAL
    # --------------------------------------------------------
    with st.sidebar:
        st.markdown('<div class="sb-logo">⚖️ TCE/PR — Auditoria IA</div>', unsafe_allow_html=True)

        st.markdown('<div class="sb-label">📄 Peça Processual</div>', unsafe_allow_html=True)
        arquivo_pdf = st.file_uploader(
            "Arraste ou clique para selecionar o PDF",
            type=["pdf"],
            label_visibility="visible"
        )

        st.markdown("<br>", unsafe_allow_html=True)

        if st.button("🔍  PROCESSAR DOCUMENTO"):
            if arquivo_pdf is None:
                st.warning("Faça o upload de um PDF primeiro.")
            else:
                logger.info(f"Upload: {arquivo_pdf.name}")
                texto = chunks = banco = None

                with st.spinner("Extraindo texto do PDF..."):
                    texto = get_texto(arquivo_pdf)

                if texto:
                    with st.spinner("Segmentando em blocos..."):
                        chunks = get_chunks(texto)

                if chunks:
                    with st.spinner("Gerando embeddings..."):
                        banco = get_vectorstore(chunks)

                if banco:
                    with st.spinner("Configurando assistente de IA..."):
                        cadeia, retriever = get_chat(banco)
                        st.session_state.cadeia = cadeia
                        st.session_state.retriever = retriever
                        st.session_state.chat_history = []

                    if st.session_state.cadeia:
                        st.success("✅ Documento pronto para análise!")
                        logger.info(f"Processamento completo: {arquivo_pdf.name}")

        st.markdown('<hr class="linha-divisoria">', unsafe_allow_html=True)
        st.markdown('<div class="sb-label">💡 Exemplos de perguntas</div>', unsafe_allow_html=True)

        for p in [
            "Há indícios de superfaturamento?",
            "Existe dano ao erário?",
            "Quais os valores contratados?",
            "Há irregularidades na licitação?",
        ]:
            st.markdown(f'<div class="exemplo">{p}</div>', unsafe_allow_html=True)

        st.markdown('<hr class="linha-divisoria">', unsafe_allow_html=True)
        st.markdown('<div class="sb-footer">Protótipo — TCC 2026<br>RAG · GPT-4o · FAISS</div>', unsafe_allow_html=True)

    # --------------------------------------------------------
    # ÁREA PRINCIPAL
    # --------------------------------------------------------
    st.markdown("""
    <div class="tce-header">
        <div class="tce-logo-linha">
            <span class="tce-brasao">⚖️</span>
            <div>
                <div class="tce-titulo">Sistema de Auditoria Inteligente</div>
                <div class="tce-subtitulo">Tribunal de Contas do Estado do Paraná</div>
            </div>
        </div>
        <span class="tce-badge">🔒 Governança · IA · RAG · Auditoria Governamental</span>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.cadeia is None:
        st.markdown("""
        <div class="boas-vindas">
            <div class="boas-vindas-icone">📋</div>
            <div class="boas-vindas-titulo">Pronto para analisar peças processuais</div>
            <div class="boas-vindas-texto">
                Faça o upload de um documento PDF na barra lateral e clique em
                <strong style="color:#8a6a1a">Processar Documento</strong> para iniciar.
                Em seguida, faça perguntas em linguagem natural para identificar
                indícios de irregularidades.
            </div>
            <div class="passos">
                <div class="passo">① Upload do PDF</div>
                <div class="passo">② Processar</div>
                <div class="passo">③ Fazer perguntas</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            if isinstance(msg, HumanMessage):
                with st.chat_message("user"):
                    st.write(msg.content)
            else:
                with st.chat_message("assistant"):
                    st.write(msg.content)

        pergunta = st.chat_input("Digite sua pergunta sobre o documento...")
        if pergunta:
            entrada_usuario(pergunta)


if __name__ == "__main__":
    main()
