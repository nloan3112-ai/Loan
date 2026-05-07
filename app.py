import streamlit as st
import pandas as pd
import google.generativeai as genai
import os
from datetime import datetime

# --- 1. 페이지 설정 및 초기화 ---
st.set_page_config(page_title="쇼핑몰 고객 응대 AI 상담사", layout="centered")
st.title("🛍️ 쇼핑몰 고객 응대 AI 상담사")
st.caption("불편 사항을 말씀해 주시면 정성을 다해 도와드리겠습니다.")

# 세션 상태(대화 기록) 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- 2. 사이드바 설정 (모델 선택 및 설정) ---
with st.sidebar:
    st.header("⚙️ 설정")
    
    # 모델 선택 UI
    model_options = ["gemini-2.5-flash-lite", "gemini-2.5-flash", "gemini-2.0-flash"]
    selected_model = st.selectbox("사용할 모델을 선택하세요", model_options, index=0)
    st.info(f"현재 모델: {selected_model}")

    # API 키 관리 (Secrets 우선, 없으면 입력창 제공)
    api_key = st.secrets.get("GEMINI_API_KEY")
    if not api_key:
        api_key = st.text_input("Gemini API Key를 입력하세요", type="password")
    
    # 대화 초기화 버튼
    if st.button("대화 기록 초기화"):
        st.session_state.messages = []
        st.rerun()

# --- 3. CSV 데이터 로드 및 시스템 프롬프트 구성 ---
def load_system_instruction():
    base_instruction = (
        "1. 당신은 쇼핑몰의 전문 고객 상담사입니다. 사용자의 불편/불만에 대해 정중하고 공감 어린 말투로 응답하세요.\n"
        "2. 사용자의 불편 사항을 구체적(무엇이/언제/어디서/어떻게)으로 정리하여 수집하고, 이를 사내 담당자에게 전달한다는 취지를 안내하세요.\n"
        "3. 대화의 마지막에는 담당자가 회신할 수 있도록 이메일 주소를 요청하세요. 거부 시 안내가 어렵다는 점을 정중히 알리세요.\n"
    )
    
    faq_path = "faq_data.csv"
    if os.path.exists(faq_path):
        try:
            df = pd.read_csv(faq_path)
            faq_md = df.to_markdown(index=False)
            # CSV 데이터가 있을 경우 추가 지침 삽입
            base_instruction += (
                f"\n[CSV 참조 데이터]\n{faq_md}\n\n"
                "4. 위 데이터를 우선 확인하여 안내하세요. 데이터에 없는 내용은 임의로 지어내지 말고 "
                "'담당 부서 확인 후 안내해 드리겠습니다'라고 답변하세요."
            )
        except Exception as e:
            st.error(f"CSV 로드 중 오류 발생: {e}")
            
    return base_instruction

# --- 4. 챗봇 엔진 동작 함수 ---
def ask_gemini(api_key, model_name, messages):
    if not api_key:
        st.warning("API 키를 설정해 주세요.")
        return
    
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=load_system_instruction()
        )
        
        # 메모리 관리: 최근 6턴(12개 메시지)만 추출하여 전달
        history_to_send = messages[-12:]
        
        # Gemini 형식으로 대화 기록 변환 (role: user/model)
        chat_session = model.start_chat(history=[
            {"role": m["role"], "parts": [m["content"]]} for m in history_to_send[:-1]
        ])
        
        # 최신 메시지에 대해 응답 생성
        response = chat_session.send_message(history_to_send[-1]["content"])
        return response.text

    except Exception as e:
        # 에러 처리 (ResourceExhausted 포함)
        error_msg = str(e)
        if "429" in error_msg or "ResourceExhausted" in error_msg:
            st.error("⚠️ 현재 사용량이 많아 응답이 지연되고 있습니다. 1분 뒤에 다시 시도해 주세요.")
        else:
            st.error(f"An error occurred: {e}")
        return None

# --- 5. 채팅 UI 렌더링 ---
# 이전 대화 출력
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("불편하신 점을 입력해 주세요."):
    # 사용자 메시지 저장 및 표시
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 모델 응답 생성 및 표시
    with st.chat_message("model"):
        with st.spinner("상담사가 답변을 작성 중입니다..."):
            response_text = ask_gemini(api_key, selected_model, st.session_state.messages)
            if response_text:
                st.markdown(response_text)
                st.session_state.messages.append({"role": "model", "content": response_text})

# --- 6. 로그 저장 및 다운로드 ---
if st.session_state.messages:
    st.divider()
    chat_df = pd.DataFrame(st.session_state.messages)
    csv_data = chat_df.to_csv(index=False).encode('utf-8-sig')
    
    st.download_button(
        label="📥 전체 대화 내역 다운로드 (CSV)",
        data=csv_data,
        file_name=f"chat_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv"
    )
