import os
import time
import json
from datetime import date
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains.question_answering import load_qa_chain


# =========================
# CẤU HÌNH
# =========================
APP_TITLE = "Chat bot hỗ trợ cho sinh viên HCMUE"
APP_SUBTITLE = (
    "Tư vấn Quy chế cho Sinh viên hệ Chính quy "
    "Trường Đại học Sư phạm TP.HCM"
)

APP_DIR = Path(__file__).resolve().parent
KB_JSON_PATH = APP_DIR / "chunks.json"

MODEL_NAME = "gemini-2.5-flash-lite"
EMBED_MODEL = "models/gemini-embedding-001"

MIN_SECONDS_BETWEEN_REQUESTS = 2
MAX_REQUESTS_PER_DAY = 30

CHUNK_SIZE = 1600
CHUNK_OVERLAP = 200
TOP_K = 4

MAX_OUTPUT_TOKENS = 512
TEMPERATURE = 0.2


# =========================
# HEADER
# =========================
def render_header():
    st.markdown(
        """
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
            
            html, body, [class*="css"] {
                font-family: 'Inter', sans-serif;
            }

            .stApp { background-color: #f8f9fa; }
            
            /* Header Style */
            .hcmue-header {
                background-color: #124874;
                padding: 2rem;
                border-radius: 0 0 24px 24px;
                color: white;
                text-align: center;
                margin-bottom: 30px;
                shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1);
            }

            /* Message Container */
            .chat-msg-container {
                display: flex;
                width: 100%;
                margin-bottom: 1.5rem;
            }
            .justify-start { justify-content: flex-start; }
            .justify-end { justify-content: flex-end; }

            /* Bubble Style */
            .msg-bubble {
                max-width: 85%;
                display: flex;
                flex-direction: column;
            }
            .items-start { align-items: flex-start; }
            .items-end { align-items: flex-end; }

            /* Avatar & Label */
            .msg-info {
                display: flex;
                align-items: center;
                gap: 10px;
                margin-bottom: 6px;
            }
            .flex-row-reverse { flex-direction: row-reverse; }
            
            .avatar {
                width: 35px;
                height: 35px;
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 14px;
                font-weight: bold;
            }
            .bot-avatar { bg-color: #124874; color: white; background-color: #124874; }
            .user-avatar { bg-color: #e2e8f0; color: #475569; background-color: #e2e8f0; }
            
            .role-label {
                font-size: 11px;
                font-weight: 700;
                color: #64748b;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }

            /* Content Bubble */
            .content-bubble {
                padding: 12px 20px;
                border-radius: 18px;
                font-size: 15px;
                line-height: 1.6;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .bot-content {
                background-color: white;
                color: #1e293b;
                border: 1px solid #f1f5f9;
                border-top-left-radius: 2px;
            }
            .user-content {
                background-color: #124874;
                color: white;
                border-top-right-radius: 2px;
            }

            /* Animation cho thinking */
            .dot-flashing {
                display: flex;
                gap: 4px;
                padding: 4px 0;
            }
            .dot {
                width: 6px;
                height: 6px;
                border-radius: 50%;
                background-color: #60a5fa;
                animation: bounce 1.5s infinite linear;
            }
            @keyframes bounce {
                0%, 100% { transform: translateY(0); }
                50% { transform: translateY(-5px); }
            }
        </style>
        <div class="hcmue-header">
            <h1 style="margin:0; font-size: 24px;">HCMUE ASSISTANT</h1>
            <p style="margin:5px 0 0 0; opacity: 0.8; font-size: 14px;">Hỗ trợ & tư vấn quy chế sinh viên</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
#===============tạo hàm helper=======================
def display_chat_message(role, content, thinking=False):
    is_bot = role == "assistant"
    justify = "justify-start" if is_bot else "justify-end"
    items = "items-start" if is_bot else "items-end"
    row_dir = "" if is_bot else "flex-row-reverse"
    avatar_class = "bot-avatar" if is_bot else "user-avatar"
    icon = '<i class="fas fa-robot"></i>' if is_bot else '<i class="fas fa-user-graduate"></i>'
    label = "Trợ lý HCMUE" if is_bot else "Sinh viên"
    bubble_class = "bot-content" if is_bot else "user-content"

    if thinking:
        inner_content = """
        <div class="dot-flashing">
            <div class="dot" style="animation-delay: 0s"></div>
            <div class="dot" style="animation-delay: 0.2s"></div>
            <div class="dot" style="animation-delay: 0.4s"></div>
            <span style="font-size: 12px; color: #94a3b8; margin-left: 10px; font-style: italic;">Đang kiểm tra sổ tay...</span>
        </div>
        """
    else:
        inner_content = content

    html = f"""
    <div class="chat-msg-container {justify}">
        <div class="msg-bubble {items}">
            <div class="msg-info {row_dir}">
                <div class="avatar {avatar_class}">{icon}</div>
                <span class="role-label">{label}</span>
            </div>
            <div class="content-bubble {bubble_class}">
                {inner_content}
            </div>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
#=============================

# =========================
# CHỐNG SPAM
# =========================
def allow_request():
    now = time.time()
    today = str(date.today())

    st.session_state.setdefault("last_req", 0.0)
    st.session_state.setdefault("count_today", 0)
    st.session_state.setdefault("day", today)

    if st.session_state["day"] != today:
        st.session_state["day"] = today
        st.session_state["count_today"] = 0

    if now - st.session_state["last_req"] < MIN_SECONDS_BETWEEN_REQUESTS:
        return False, "Bạn gửi hơi nhanh, vui lòng chờ một chút nhé."

    if st.session_state["count_today"] >= MAX_REQUESTS_PER_DAY:
        return False, "Bạn đã sử dụng hết lượt hỏi hôm nay."

    st.session_state["last_req"] = now
    st.session_state["count_today"] += 1
    return True, ""


# =========================
# LOAD KB
# =========================
@st.cache_data
def load_kb_texts():
    if not KB_JSON_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {KB_JSON_PATH}")

    with open(KB_JSON_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    texts = [item["content"] for item in data if "content" in item]
    if not texts:
        raise ValueError("File JSON không có nội dung hợp lệ.")

    return texts


@st.cache_resource(show_spinner=True)
def load_kb_vectorstore(api_key: str):
    texts = load_kb_texts()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBED_MODEL,
        google_api_key=api_key,
    )

    return FAISS.from_texts(chunks, embedding=embeddings)


@st.cache_resource
def load_qa_chain_cached(api_key: str):
    prompt_template = """
Bạn là trợ lý hỗ trợ sinh viên.
Trả lời ngắn gọn, rõ ràng, đúng trọng tâm.

NGỮ CẢNH:
{context}

CÂU HỎI:
{question}

TRẢ LỜI:
""".strip()

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=api_key,
        temperature=TEMPERATURE,
        max_output_tokens=MAX_OUTPUT_TOKENS,
    )

    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
    )

    return load_qa_chain(
        llm=llm,
        chain_type="stuff",
        prompt=prompt,
    )


# =========================
# QUICK ANSWER
# =========================
def quick_answer(option: str) -> str:
    keyword_map = {
        "Cách xét học bổng": ["học bổng"],
        "Điều kiện xét học bổng": ["điều kiện", "học bổng"],
        "Điều kiện cần và đủ để tốt nghiệp": ["tốt nghiệp"],
        "Điều kiện xét học ngành thứ hai": ["ngành", "thứ hai"],
    }

    keywords = keyword_map.get(option, [])
    if not keywords:
        return "Chưa có thông tin."

    texts = load_kb_texts()

    for text in texts:
        content = text.lower()
        if all(k in content for k in keywords):
            sentences = [
                s.strip()
                for s in text.split(".")
                if s.strip()
            ]
            bullets = [
                "- " + " ".join(s.split()[:18])
                for s in sentences[:3]
            ]
            return "\n".join(bullets)

    return "Không tìm thấy nội dung phù hợp trong quy chế."


# =========================
# RESET CHAT
# =========================
def reset_chat():
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Tôi có thể hỗ trợ gì cho các bạn?",
        }
    ]


# =========================
# MAIN
# =========================
def main():
    st.set_page_config(page_title=APP_TITLE, layout="centered")
    render_header()

    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get(
        "GOOGLE_API_KEY"
    )

    if not api_key:
        st.error("Chưa cấu hình GOOGLE_API_KEY.")
        st.stop()

    st.session_state.setdefault(
        "messages",
        [
            {
                "role": "assistant",
                "content": "Tôi có thể hỗ trợ gì cho các bạn?",
            }
        ],
    )

    with st.sidebar:
        st.subheader("Hỏi nhanh")
        for item in [
            "Cách xét học bổng",
            "Điều kiện xét học bổng",
            "Điều kiện cần và đủ để tốt nghiệp",
            "Điều kiện xét học ngành thứ hai",
        ]:
            if st.button(item, use_container_width=True):
                st.session_state.messages.extend(
                    [
                        {"role": "user", "content": item},
                        {
                            "role": "assistant",
                            "content": quick_answer(item),
                        },
                    ]
                )

        st.divider()
        st.button("Xóa cuộc hội thoại", on_click=reset_chat)

    for m in st.session_state.messages:
        display_chat_message(m["role"], m["content"])

    # ... các bước xử lý vector store ...

# 1. Khởi tạo Vector Store và Chain (Làm trước khi nhận input)
    vs = load_kb_vectorstore(api_key)
    chain = load_qa_chain_cached(api_key)

    # 2. CHỈ DÙNG MỘT THANH NHẬP LIỆU DUY NHẤT
    if question := st.chat_input("Nhập câu hỏi của bạn tại đây..."):
        # Kiểm tra giới hạn yêu cầu (Spam)
        ok, msg = allow_request()
        if not ok:
            st.warning(msg)
        else:
            # A. Thêm câu hỏi của User vào lịch sử và hiển thị ngay
            st.session_state.messages.append({"role": "user", "content": question})
            display_chat_message("user", question)

            # B. Tạo khung trống (placeholder) để xử lý hiệu ứng "Thinking"
            # Lưu ý: Avatar=None để không hiện khung mặc định của Streamlit chồng lên CSS của bạn
            with st.chat_message("assistant", avatar=None):
                 placeholder = st.empty()
                 
                 # Hiển thị trạng thái đang xử lý
                 with placeholder:
                     display_chat_message("assistant", "", thinking=True)
                
                 # C. Logic xử lý AI (RAG)
                 try:
                     # Tìm kiếm nội dung liên quan trong file JSON
                     docs = vs.similarity_search(question, k=TOP_K)
                     
                     # Chạy Chain để trả lời dựa trên ngữ cảnh
                     out = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
                     answer = out.get("output_text", "Xin lỗi, tôi không tìm thấy thông tin phù hợp trong quy chế.")
                 except Exception as e:
                     answer = f"Đã xảy ra lỗi khi xử lý: {str(e)}"

                 # D. Xóa hiệu ứng Thinking và thay bằng câu trả lời thực tế
                 placeholder.empty()
                 with placeholder:
                     display_chat_message("assistant", answer)
                
                 # E. Lưu câu trả lời của Bot vào lịch sử hội thoại
                 st.session_state.messages.append({"role": "assistant", "content": answer})

if __name__ == "__main__":
    main()
