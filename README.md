**Web-RAG Chatbot**
This is a tool that lets you chat with any website. It reads the page content, understands the context, and answers your questions in a chat format. Only use the websites that has no ai bot blocker. example: https://lilianweng.github.io/posts/    (you can chose any of these posts for experimenting the code.)

**Main Features**
**URL Input:** Just paste a link to read its content.
**Memory:** It remembers your previous questions during the session.
**Alignment:** Your questions appear on the right, and AI answers appear on the left.
**Auto-Reset:** The session clears after 2 minutes of inactivity or if you enter a new URL.
**Fast Responses:** Uses Groq and Llama 3.1 for high-speed performance.

**Setup Instructions**
Install the required libraries:
pip install streamlit langchain langchain-groq langchain-community langchain-chroma langchain-huggingface beautifulsoup4 python-dotenv

**Create a .env file and add your keys:**
GROQ_API_KEY=your_key_here
HF_TOKEN=your_token_here

**Run the application:**
streamlit run webpageQA.py

**How to Use**
    -Paste a website link in the sidebar.
    -Click Process Content.
    -Once it finishes, type your questions in the chat box at the bottom.
    -If you want to talk about a different website, just enter the new URL and the old chat will be cleared automatically.
