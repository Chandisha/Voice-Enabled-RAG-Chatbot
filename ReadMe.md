# Voice-Enabled Hindi-to-English RAG Chatbot

This repository contains a modular pipeline designed for Google Colab that implements a Retrieval-Augmented Generation (RAG) system with voice capabilities. The system allows users to provide a topic, builds a local knowledge base from Wikipedia, and answers voice queries spoken in Hindi by transcribing and translating them before generating a response.



### Project Workflow
1. **Knowledge Acquisition:** The system uses SerpApi to identify relevant Wikipedia pages for a given topic and scrapes the content using BeautifulSoup.
2. **Speech Recognition:** Spoken Hindi is captured and processed using the AI4Bharat Indic-Conformer model.
3. **Translation:** The transcribed Hindi text is converted to English via the Sarvam AI translation API.
4. **Vector Indexing:** English text is split into chunks and stored in a ChromaDB vector database using HuggingFace embeddings.
5. **Contextual Generation:** Relevant document chunks are retrieved based on the query, and the Gemma 3 LLM synthesizes a final answer grounded in that context.

### File Structure
* **main.py (Backend):** Contains the FastAPI server logic, Wikipedia scraping functions, vector database initialization, and the core ASR/inference pipeline.
* **app.py (Frontend):** A Gradio-based interface that handles audio recording and communicates with the backend API.
* **requirements.txt:** A comprehensive list of all dependencies required to replicate the environment.

### Setup and Secrets (Google Colab)
To maintain security, I utilized Google Colab Secrets for API key management. Ensure the following keys are configured in your environment:
* SERPAPI_KEY: Used for Google Search queries.
* SARVAM_KEY: Used for the translation API.
* HF_TOKEN: Required for accessing HuggingFace models.
* GOOGLE_API_KEY: Required for the Gemma 3 generative model.
* NGROK_TOKEN: Used to create the public tunnel for the FastAPI server.



### Technical Connection Guidelines
The communication between the frontend and backend relies on a dynamic tunnel. Please note:
* **Ngrok URL:** When main.py is executed, it generates a unique public URL. This URL is required for the frontend to locate the backend.
* **Variable Update:** The generated URL must be manually updated in the API_URL variable within app.py.
* **Session Persistence:** Restarting the backend will generate a new URL, necessitating an update to the frontend configuration to maintain the connection.

### Execution Steps
1. **Dependencies:** Install the required packages using pip install -r requirements.txt.
2. **Initialize Backend:** Run main.py. Once the models load and the knowledge base is built, copy the public Ngrok URL from the console output.
3. **Initialize Frontend:** Update the API_URL in app.py with the copied link and execute the script.
4. **Interface Access:** Open the provided .gradio.live link in a new browser tab. Using an external tab is necessary to ensure the browser correctly prompts for microphone permissions.

### Development Challenges
* **Colab Kernel Management:** I implemented parse_known_args to handle the default JSON arguments passed by the Colab kernel, which otherwise interfere with standard command-line argument parsing.
* **Audio Processing:** The Indic-Conformer model requires a specific 16kHz sampling rate. I added a resampling layer using torchaudio to normalize inputs from various microphones.
* **Tunnel Interstitials:** Ngrok often displays a warning page for free-tier users. I resolved this by including the ngrok-skip-browser-warning header in all API requests to allow seamless automated communication.

This project was built to demonstrate the integration of multi-lingual ASR, machine translation, and RAG in a cloud-hosted environment.
