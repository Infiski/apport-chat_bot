import os
import time
import fitz 
import joblib
import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
from gtts import gTTS
from deep_translator import GoogleTranslator

load_dotenv()
GOOGLE_API_KEY = "AIzaSyC5U3mXbw93v923i9QTQK4EVkgT5ym3las"
genai.configure(api_key=GOOGLE_API_KEY)

new_chat_id = f'{time.time()}'
MODEL_ROLE = 'ai'
AI_AVATAR_ICON = '✨'
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'

LANGUAGES = {
    'English': 'en',
    'Hindi': 'hi',
    'Bengali': 'bn',
    'Tamil': 'ta',
    'Telugu': 'te',
    'Marathi': 'mr',
}

os.makedirs('data/', exist_ok=True)

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)

embedding_model = load_embedding_model()
translator = GoogleTranslator()

try:
    past_chats: dict = joblib.load('data/past_chats_list')
except FileNotFoundError:
    past_chats = {}

with st.sidebar:
    st.write('# Settings')
    selected_language = st.selectbox('Select Language', options=list(LANGUAGES.keys()), index=0)
    language_code = LANGUAGES[selected_language]

    N_value = st.slider('Set N (Number of Top Chunks to Highlight)', min_value=1, max_value=20, value=10, step=1)
    enable_audio = st.checkbox("Enable Audio Responses", value=True)

    st.write('# Past Chats')
    if st.session_state.get('chat_id') is None:
        st.session_state.chat_id = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id] + list(past_chats.keys()),
            format_func=lambda x: past_chats.get(x, 'New Chat'),
            placeholder='_',
        )
    else:
        st.session_state.chat_id = st.selectbox(
            label='Pick a past chat',
            options=[new_chat_id, st.session_state.chat_id] + list(past_chats.keys()),
            index=1,
            format_func=lambda x: past_chats.get(x, 'New Chat' if x != st.session_state.chat_id else st.session_state.chat_title),
            placeholder='_',
        )
    st.session_state.chat_title = f'ChatSession-{st.session_state.chat_id}'

    st.write('# Upload PDF')
    uploaded_pdf = st.file_uploader("Choose a PDF file", type="pdf")
    if uploaded_pdf is not None:
        with st.spinner('Extracting text from PDF...'):
            def extract_text_from_pdf(pdf_file):
                pdf_file.seek(0)
                doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
                text_chunks = []
                page_numbers = []
                for page_num, page in enumerate(doc):
                    blocks = page.get_text("blocks")
                    if not blocks:
                        text_chunks.append('')
                        page_numbers.append(page_num)
                    else:
                        for block in blocks:
                            text = block[4].strip()
                            if text:
                                text_chunks.append(text)
                                page_numbers.append(page_num)
                return text_chunks, page_numbers, doc

            pdf_text_chunks, pdf_page_numbers, pdf_doc = extract_text_from_pdf(uploaded_pdf)
            st.session_state.pdf_text_chunks = pdf_text_chunks
            st.session_state.pdf_page_numbers = pdf_page_numbers
            st.session_state.pdf_doc = pdf_doc
            st.session_state.uploaded_pdf_name = uploaded_pdf.name
            st.success("PDF text extracted successfully!")

st.write('# Pdf par charcha! AI-powered')

if 'messages' not in st.session_state:
    st.session_state.messages = []
    st.session_state.gemini_history = []
else:
    try:
        st.session_state.messages = joblib.load(f'data/{st.session_state.chat_id}-st_messages')
        st.session_state.gemini_history = joblib.load(f'data/{st.session_state.chat_id}-gemini_messages')
    except FileNotFoundError:
        st.session_state.messages = []
        st.session_state.gemini_history = []

@st.cache_resource(show_spinner=False)
def load_chat_model():
    return genai.GenerativeModel('gemini-pro')

st.session_state.model = load_chat_model()
st.session_state.chat = st.session_state.model.start_chat(
    history=st.session_state.gemini_history,
)

for message in st.session_state.messages:
    with st.chat_message(
        name=message['role'],
        avatar=message.get('avatar'),
    ):
        if message['role'] == 'user':
            st.markdown(message['content'])
        else:
            st.markdown(message['content'])
            if enable_audio and 'audio_path' in message and message['audio_path'] is not None:
                try:
                    with open(message['audio_path'], 'rb') as audio_file:
                        audio_bytes = audio_file.read()
                        st.audio(audio_bytes, format='audio/mp3')
                except Exception as e:
                    st.error(f"Error loading audio: {e}")
            else:
                pass

user_input = st.text_input("Type your message:", "")

if user_input:
    with st.spinner('Processing your input...'):
        if language_code != 'en':
            try:
                user_input_translated = GoogleTranslator(source=language_code, target='en').translate(user_input)
            except Exception as e:
                st.error(f"Sorry, I didn’t understand your question. Do you want to connect with a live agent?")
                user_input_translated = user_input
        else:
            user_input_translated = user_input

        if st.session_state.chat_id not in past_chats.keys():
            past_chats[st.session_state.chat_id] = st.session_state.chat_title
            joblib.dump(past_chats, 'data/past_chats_list')

        with st.chat_message('user'):
            st.markdown(user_input)

        st.session_state.messages.append(
            dict(
                role='user',
                content=user_input,
            )
        )

    if 'pdf_text_chunks' in st.session_state:
        context = f"The following is the content of the PDF:\n{ ' '.join(st.session_state.pdf_text_chunks) }\n\n"
        full_prompt = context + user_input_translated
    else:
        full_prompt = user_input_translated

    with st.spinner('Generating response...'):
        response = st.session_state.chat.send_message(
            full_prompt,
            stream=True,
        )

    full_response = ''
    for chunk in response:
        if hasattr(chunk, 'text'):
            full_response += chunk.text

    st.session_state['full_response'] = full_response

    if language_code != 'en':
        try:
            full_response_translated = GoogleTranslator(source='en', target=language_code).translate(full_response)
        except Exception as e:
            st.error(f"Sorry, I didn’t understand your question. Do you want to connect with a live agent?")
            full_response_translated = full_response
    else:
        full_response_translated = full_response

    if enable_audio:
        with st.spinner('Generating audio...'):
            try:
                tts = gTTS(full_response_translated, lang=language_code)
                audio_file_path = f"data/{st.session_state.chat_id}response_audio{len(st.session_state.messages)}.mp3"
                tts.save(audio_file_path)
            except Exception as e:
                st.error(f"Text-to-speech error: {e}")
                audio_file_path = None
    else:
        audio_file_path = None

    with st.chat_message(
        name=MODEL_ROLE,
        avatar=AI_AVATAR_ICON,
    ):
        st.markdown(full_response_translated)
        if enable_audio and audio_file_path:
            try:
                with open(audio_file_path, "rb") as audio_file:
                    audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format='audio/mp3')
            except Exception as e:
                st.error(f"Sorry, I didn’t understand your question. Do you want to connect with a live agent?\nError loading audio: {e}")
        else:
            pass

    st.session_state.messages.append(
        dict(
            role=MODEL_ROLE,
            content=full_response_translated,
            avatar=AI_AVATAR_ICON,
            audio_path=audio_file_path,
        )
    )
    st.session_state.gemini_history = st.session_state.chat.history

    with st.spinner('Saving chat history...'):
        joblib.dump(
            st.session_state.messages,
            f'data/{st.session_state.chat_id}-st_messages',
        )
        joblib.dump(
            st.session_state.gemini_history,
            f'data/{st.session_state.chat_id}-gemini_messages',
        )

if 'pdf_doc' in st.session_state:
    download_highlighted = st.checkbox("Download highlighted relevant sections")

    if download_highlighted:
        with st.spinner('Processing...'):
            full_response = st.session_state.get('full_response', '')
            if not full_response:
                st.error("Sorry, I didn’t understand your question. Do you want to connect with a live agent?")
            else:
                with st.spinner('Computing embeddings for PDF text...'):
                    @st.cache_data(show_spinner=False)
                    def get_pdf_embeddings(pdf_text_chunks):
                        return embedding_model.encode(pdf_text_chunks)

                    pdf_embeddings = get_pdf_embeddings(st.session_state.pdf_text_chunks)

                with st.spinner('Computing similarity...'):
                    response_embedding = embedding_model.encode(full_response)

                similarities = np.dot(pdf_embeddings, response_embedding) / (
                    np.linalg.norm(pdf_embeddings, axis=1) * np.linalg.norm(response_embedding)
                )

                N = min(N_value, len(st.session_state.pdf_text_chunks))
                top_n_indices = similarities.argsort()[-N:][::-1]

                for idx in top_n_indices:
                    page_num = st.session_state.pdf_page_numbers[idx]
                    page = st.session_state.pdf_doc[page_num]
                    text_to_highlight = st.session_state.pdf_text_chunks[idx]
                    text_instances = page.search_for(text_to_highlight)
                    if text_instances:
                        for inst in text_instances:
                            try:
                                page.add_highlight_annot(inst)
                            except ValueError as e:
                                st.warning(f"Could not highlight text on page {page_num + 1}: {e}")
                    else:
                        pass

                original_filename = os.path.splitext(st.session_state.uploaded_pdf_name)[0]
                highlighted_pdf_name = f"{original_filename}_highlighted.pdf"
                highlighted_pdf_path = f"data/{highlighted_pdf_name}"
                st.session_state.pdf_doc.save(highlighted_pdf_path)

                st.markdown("Here is the relevant highlighted PDF section from where I took the context from:")
                with open(highlighted_pdf_path, "rb") as file:
                    btn = st.download_button(
                        label="Download highlighted PDF",
                        data=file,
                        file_name=highlighted_pdf_name,
                        mime="application/pdf"
                    )
