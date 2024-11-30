# AI Chatbot for PDF Interaction and Multilingual Audio Responses

Welcome to the AI Chatbot project! This repository contains a Streamlit-based application designed to interact with users, process PDF documents, highlight relevant content, and provide audio responses in multiple Indic languages.

## Features
- **PDF Text Extraction**: Upload PDF files and extract their content for processing.
- **Contextual Highlighting**: Automatically highlight relevant sections of the PDF and download the updated document.
- **Multilingual Audio Responses**: Get chatbot responses in six Indic languages (English, Hindi, Bengali, Tamil, Telugu, Marathi).
- **Customizable Highlighting**: Choose the number of text chunks to highlight in the PDF.
- **Intuitive UI**: User-friendly Streamlit interface with support for past chat history.

## Installation

1. Clone the repository:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment:
   - Create a `.env` file in the root directory.
   - Add your Google API Key:
     ```plaintext
     GOOGLE_API_KEY=your_google_api_key
     ```

## Usage

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```

2. In the app:
   - Upload a PDF file to use as a knowledge base.
   - Configure settings such as language, number of highlights, and audio response preferences.
   - Ask questions in natural language, and the chatbot will respond based on the PDF content.

## Project Structure
- **`app.py`**: Main application script.
- **`data/`**: Stores chat history and generated files.
- **`requirements.txt`**: Dependencies required to run the project.
- **`.env`**: Environment file to store API keys (not included in the repository).

## Dependencies
The project relies on the following Python libraries:
- Streamlit
- PyMuPDF (`fitz`)
- Joblib
- Sentence-Transformers
- Google Generative AI
- Dotenv
- Google Text-to-Speech (`gTTS`)
- Deep Translator
- NumPy

For a complete list, refer to `requirements.txt`.

## Key Functionalities
### PDF Text Processing
- Extracts text from PDF files and processes it into chunks for efficient querying.
- Highlights the most relevant content based on user queries and generates a downloadable PDF.

### Multilingual Audio Output
- Converts chatbot responses to audio using `gTTS`.
- Supports six Indic languages, ensuring accessibility for diverse users.

### Contextual Chat
- Uses Google Generative AI (`gemini-pro`) to generate responses based on PDF content and user inputs.
- Caches embeddings and past chats for optimized performance.
