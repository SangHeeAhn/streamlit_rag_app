import streamlit as st
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms.ollama import Ollama
from langchain.prompts import ChatPromptTemplate
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# 색상 팔레트 정의
primary_color = "#1E90FF"
secondary_color = "#FF6347"
background_color = "#F5F5F5"
text_color = "#333333"
highlight_color = "#FFD700"

# 사용자 정의 CSS
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {background_color};
        color: {text_color};
    }}
    .stButton>button {{
        background-color: {primary_color};
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }}
    .stTextInput>div>div>input {{
        border: 2px solid {primary_color};
        padding: 10px;
    }}
    .response-container {{
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
    }}
    .highlight {{
        color: {highlight_color};
        font-weight: bold;
    }}
    </style>
""", unsafe_allow_html=True)

# 앱 타이틀
st.title("📚 RAG 시스템 (DeepSeek R1 & Ollama 기반)")

# 파일 업로드
uploaded_file = st.file_uploader("📂 PDF 파일 업로드", type="pdf")

if uploaded_file is not None:
    # 파일 저장
    with open("uploaded_file.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    # PDF 로더 초기화
    loader = PDFPlumberLoader("uploaded_file.pdf")
    docs = loader.load()

    # 문서 분할 및 임베딩 생성
    text_splitter = SemanticChunker(HuggingFaceEmbeddings())
    documents = text_splitter.split_documents(docs)

    # 임베딩 모델 초기화
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector = FAISS.from_documents(documents, embedder)
    retriever = vector.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    # LLM 설정
    llm = Ollama(model="deepseek-r1:32b")

    # 시스템 프롬프트 정의
    system_prompt = (
        "주어진 문맥을 바탕으로 반드시 한국어로 답변하세요. "
        "답변을 모를 경우 '모르겠습니다'라고만 답하세요. "
        "답변은 간결하고 명확해야 하며, 반드시 한국어로 작성하세요. "
        "문맥: {context}"
    )

    # 프롬프트 생성
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}")
        ]
    )

    # QA 체인 생성
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # 사용자 입력
    user_input = st.text_input("💬 질문을 입력하세요:")

    if user_input:
        with st.spinner("🔍 검색 중..."):
            response = rag_chain.invoke({"input": user_input})
            answer = response.get("answer", "응답을 처리할 수 없습니다.")

            # 결과 출력
            formatted_answer = f"""
            <div class="response-container">
                <p><b>📌 응답:</b></p>
                <p style="color:{text_color}; font-size:16px;">{answer}</p>
            </div>
            """
            st.markdown(formatted_answer, unsafe_allow_html=True)

else:
    st.write("📂 **PDF 파일을 업로드해주세요.**")