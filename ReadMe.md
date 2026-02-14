# 🎙️ Hindi-to-English Voice RAG Chatbot

This project is a complete, voice-enabled Retrieval-Augmented Generation (RAG) pipeline designed for Google Colab. It allows a user to speak a query in **Hindi**, which is then transcribed, translated, and answered using a custom knowledge base built dynamically from Wikipedia.

### 🌟 Project Workflow
1.  **Dynamic Knowledge Base:** The system scrapes Wikipedia based on a user-defined topic to build a searchable database.
2.  **Hindi ASR:** Uses AI4Bharat's `indic-conformer` model to accurately transcribe Hindi speech.
3.  **Sarvam AI Translation:** Converts the Hindi transcription into English for processing.
4.  **RAG Retrieval:** Uses ChromaDB and LangChain to find the most relevant "facts" from the scraped data.
5.  **Gemma 3 LLM:** Synthesizes the final answer based on the retrieved facts and the user's question.

---

### 📂 File Structure Explained

To make the project modular and easier to maintain, I have split the code into three main files:

* **`main.py` (The Backend):** This is the core engine. It handles the Wikipedia scraping, sets up the Vector Database (ChromaDB), loads the ASR model, and runs the FastAPI server. It exposes a `/process_query` endpoint that the UI interacts with.
* **`app.py` (The UI):** This file contains the Gradio interface. It handles the microphone input from the user, sends the audio file to the backend, and displays the final transcription, translation, and answer in a clean "Chat" format.
* **`requirements.txt`:** Contains all the library versions needed to run the project. This ensures that anyone (including the evaluators) can set up the environment with a single command.

---

### 🛠️ Setup & Secrets (Google Colab)

Since this project relies on sensitive API keys, I have used **Google Colab Secrets** (the 🔑 icon) for security. You must add the following keys to your Colab environment:
* `SERPAPI_KEY` (Search results)
* `SARVAM_KEY` (Translation)
* `HF_TOKEN` (HuggingFace model access)
* `GOOGLE_API_KEY` (Gemma 3 model)
* `NGROK_TOKEN` (Public API tunnel)

---

### 🚀 How to Run

1.  **Install Dependencies:**
    `pip install -r requirements.txt`

2.  **Start the Backend (`main.py`):**
    Run this first to initialize the knowledge base and start the server. Look for the output: `API DEPLOYED AT: https://xxxx.ngrok-free.dev`.

3.  **Start the UI (`app.py`):**
    Copy the Ngrok URL from the previous step and paste it into the `API_URL` variable in `app.py`. Run the script.

4.  **Use the External Gradio Link:**
    For the microphone to work properly, **click the public Gradio link** (the one ending in `.gradio.live`). Using it in a separate browser tab ensures the microphone permissions are handled correctly.

---

### 🧠 Challenges I Addressed
* **Kernel Arguments:** I used `parse_known_args()` to prevent the script from crashing due to Colab's default background arguments.
* **Audio Sampling:** The AI4Bharat ASR model is very strict about a **16kHz** sample rate. I added a resampling step in the backend to ensure any recorded audio is converted correctly before processing.
* **API Tunneling:** Using Ngrok with FastAPI required a specific header (`ngrok-skip-browser-warning`) to ensure the frontend could talk to the backend without being blocked by warning pages.

---
*Built for the AI Assignment. Includes all bonus UI tasks.*
