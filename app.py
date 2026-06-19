import os
import streamlit as st
import fitz  # PyMuPDF
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

# .env 파일에서 환경변수 로드
load_dotenv()

# 시스템 환경변수에서 API Key 가져오기
openai_api_key = os.getenv("OPENAI_API_KEY")

# 1. 페이지 기본 설정 및 디자인
st.set_page_config(page_title="보험 약관 챗봇", page_icon="📄", layout="centered")

st.title("📄 보험 약관 QA 시스템")
st.markdown("보험 약관 PDF를 업로드하고 궁금한 점을 질문해 보세요!")
st.divider()

# API Key 검증
if not openai_api_key:
    st.error("❌ `.env` 파일에 `OPENAI_API_KEY`가 설정되지 않았거나 올바르지 않습니다.")
    st.stop()

# 2. PDF 텍스트 추출 및 벡터스토어 생성 함수
def process_pdf(uploaded_file):
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text("text", sort=True)
        text += '\n\n'

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(text)

    # 환경변수에 등록되어 있으므로 별도의 api_key 매개변수 전달이 필요 없음
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

# 3. 파일 업로드 UI
uploaded_file = st.file_uploader("보험 약관 PDF 파일을 업로드하세요", type=["pdf"])

if uploaded_file:
    # 세션 상태를 활용해 한 번만 임베딩 하도록 최적화
    if "vectorstore" not in st.session_state:
        with st.spinner("PDF 문서를 분석하고 인덱싱하는 중입니다... 잠시만 기다려주세요."):
            st.session_state.vectorstore = process_pdf(uploaded_file)
        st.success("✅ 문서 분석 완료! 이제 아래에서 질문할 수 있습니다.")

    st.divider()

    # 4. 사용자 질문 입력 및 RAG 실행 UI
    query = st.text_input("💡 궁금한 점을 입력하세요", placeholder="예: 암보험의 월 납입해야 하는 보험료는 얼마인지?")

    if query:
        with st.spinner("답변을 생성하는 중..."):
            # 검색기 작동
            retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
            docs = retriever.invoke(query)
            context = "\n\n".join([doc.page_content for doc in docs])

            # 프롬프트 및 LLM 설정
            template = """다음 배경 지식을 사용해서 질문에 대답해 주세요.

            배경지식
            {context}
            ============
            질문
            {question}"""

            prompt = ChatPromptTemplate.from_template(template)
            llm = ChatOpenAI(temperature=0, model='gpt-4o')
            chain = prompt | llm | StrOutputParser()

            # 체인 실행
            response = chain.invoke({"context": context, "question": query})

            # 결과 출력
            st.subheader("🤖 답변")
            st.info(response)

            # 참고한 문서 출처 접기/펼치기 UI
            with st.expander("🔍 참고한 약관 내용 보기"):
                for idx, doc in enumerate(docs):
                    st.markdown(f"**[참고 내용 {idx+1}]**")
                    st.code(doc.page_content, language="text")
