# 🎙️ Hindi-to-English Voice RAG Chatbot

This project is a modular, voice-enabled Retrieval-Augmented Generation (RAG) pipeline designed for Google Colab. It allows users to speak a query in **Hindi**, which is then transcribed, translated, and answered using a custom knowledge base built dynamically from Wikipedia.



---

### 🌟 Project Workflow
1.  **Dynamic Knowledge Base:** Scrapes Wikipedia based on a user-defined topic to build a searchable database.
2.  **Hindi ASR:** Uses AI4Bharat's `indic-conformer` model to accurately transcribe Hindi speech.
3.  **Sarvam AI Translation:** Converts the Hindi transcription into English for processing.
4.  **RAG Retrieval:** Uses ChromaDB and LangChain to find relevant facts from the scraped data.
5.  **Gemma 3 LLM:** Synthesizes the final answer based on the retrieved facts and the user's question.

---

### 📂 File Structure Explained

To follow best practices in modularization, the project is split into three files:

* **`main.py` (The Backend):** The core engine. Handles Wikipedia scraping, Vector Database setup, ASR model loading, and the FastAPI server.
* **`app.py` (The UI):** The Gradio interface. It captures microphone input, sends it to the backend, and displays the response in a "Chat" format.
* **`requirements.txt`:** Lists all library versions needed to ensure a seamless environment setup.

---

### 🛠️ Setup & Secrets (Google Colab)

Since this project relies on sensitive API keys, I have used **Google Colab Secrets** (the 🔑 icon) for security. You must add the following keys to your Colab environment:
* `SERPAPI_KEY` (Search results)
* `SARVAM_KEY` (Translation)
* `HF_TOKEN` (HuggingFace model access)
* `GOOGLE_API_KEY` (Gemma 3 model)
* `NGROK_TOKEN` (Public API tunnel)



---

### ⚠️ IMPORTANT: Connection Guidelines
To ensure the Frontend and Backend communicate correctly, please note:

* **Copy the Link:** When you run `main.py`, it will generate a unique Ngrok URL in the output. **Make sure** you copy this entire URL (including the `https://`).
* **Update the Variable:** You must manually paste this link into the `API_URL` variable inside `app.py`. Without this step, the UI will not be able to "find" the backend server.
* **Refresh on Restart:** Remember that every time you restart the backend, a new Ngrok link is generated. **Make sure** to update `app.py` with the fresh link every time.

---

### 🚀 How to Run

1.  **Install Dependencies:**
    `pip install -r requirements.txt`

2.  **Start the Backend (`main.py`):**
    Run this first to initialize the knowledge base and start the server. Look for the output line: `API DEPLOYED AT: https://xxxx.ngrok-free.dev`. **Copy this URL.**

3.  **Start the UI (`app.py`):**
    Open `app.py`, paste your copied URL into the `API_URL` variable, and run the script.

4.  **Use the External Gradio Link:**
    For the microphone to work properly, **click the public Gradio link** (ending in `.gradio.live`). Using it in a separate browser tab ensures microphone permissions are handled correctly by your browser.

---

### 🧠 Challenges I Addressed
* **Kernel Arguments:** I used `parse_known_args()` to prevent the script from crashing due to the hidden background arguments Colab passes to its own kernel.
* **Audio Sampling:** The AI4Bharat ASR model is very strict about a **16kHz** sample rate. I implemented a resampling step in the backend using `torchaudio` to ensure user microphone input is always compatible.
* **API Tunneling:** Using Ngrok with FastAPI required a specific header (`ngrok-skip-browser-warning`) to ensure the frontend could talk to the backend without being blocked by Ngrok's security landing page.

---
