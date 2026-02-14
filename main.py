import os
import sys
import argparse  
import glob
import requests
import threading
import uvicorn
import nest_asyncio
import torch
import torchaudio
import time
from bs4 import BeautifulSoup
from pyngrok import ngrok
from fastapi import FastAPI, UploadFile, File
from google.colab import userdata
from huggingface_hub import login
from transformers import AutoModel
import google.generativeai as genai
from sarvamai import SarvamAI

# LangChain and ChromaDB modern imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. API CONFIGURATION
SERPAPI_KEY = userdata.get('SERPAPI_KEY')
SARVAM_KEY = userdata.get('SARVAM_KEY')
HF_TOKEN = userdata.get('HF_TOKEN')
GOOGLE_API_KEY = userdata.get('GOOGLE_API_KEY')
NGROK_TOKEN = userdata.get('NGROK_TOKEN')

if HF_TOKEN:
    login(HF_TOKEN)

genai.configure(api_key=GOOGLE_API_KEY)
llm_model = genai.GenerativeModel('gemma-3-27b-it')

# 2. DATA COLLECTION & VECTOR DB LOGIC
def setup_knowledge_base(topic_query):
    print(f"Building knowledge base for: {topic_query}")
    
    endpoint = "https://serpapi.com/search"
    params = {"engine": "google", "q": f"{topic_query} site:en.wikipedia.org", "api_key": SERPAPI_KEY}
    
    try:
        response = requests.get(endpoint, params=params).json()
        wiki_link = None
        if "organic_results" in response:
            for result in response["organic_results"]:
                if "en.wikipedia.org/wiki/" in result.get("link", ""):
                    wiki_link = result["link"]
                    break
        
        if not wiki_link:
            print(f"FAILED: No Wikipedia page found for '{topic_query}'")
            return None

        res = requests.get(wiki_link, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        content = soup.find("div", class_="mw-parser-output")
        text = "\n".join([p.get_text() for p in content.find_all("p") if p.get_text().strip()])
        
        os.makedirs("wiki_data", exist_ok=True)
        with open("wiki_data/kb.txt", "w", encoding="utf-8") as f:
            f.write(text)

        loader = TextLoader("wiki_data/kb.txt", encoding="utf-8")
        docs = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = splitter.split_documents(docs)
        
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        vectordb = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory="chromadb_store")
        print("[SUCCESS] Vector database ready.")
        return vectordb
    except Exception as e:
        print(f"Error in setup_knowledge_base: {e}")
        return None

# 3. FASTAPI SERVER
app = FastAPI()
nest_asyncio.apply()

asr_model = None
vector_store = None

@app.post("/process_query")
async def process_query(file: UploadFile = File(...)):
    global vector_store
    try:
        # Check if vector store exists before proceeding
        if vector_store is None:
            return {"error": "Knowledge base not initialized. Check your API keys and topic."}

        temp_audio = "temp_query.wav"
        with open(temp_audio, "wb") as buffer:
            buffer.write(await file.read())

        wav, sr = torchaudio.load(temp_audio)
        wav = torch.mean(wav, dim=0, keepdim=True)
        if sr != 16000:
            wav = torchaudio.transforms.Resample(sr, 16000)(wav)
        
        with torch.no_grad():
            asr_output = asr_model(wav, "hi", "rnnt")
            hindi_text = asr_output[0] if isinstance(asr_output, list) else asr_output

        client = SarvamAI(api_subscription_key=SARVAM_KEY)
        translate_response = client.text.translate(
            input=str(hindi_text),
            source_language_code="hi-IN",
            target_language_code="en-IN",
            model="sarvam-translate:v1"
        )
        english_query = translate_response.translated_text

        # RAG Retrieval
        docs = vector_store.similarity_search(english_query, k=2)
        context = "\n".join([doc.page_content for doc in docs])

        prompt = (
            f"Context: {context}\n\n"
            f"Question: {english_query}\n"
            f"Answer:"
        )
        response = llm_model.generate_content(prompt)

        return {
            "hindi_transcription": str(hindi_text),
            "english_translation": str(english_query),
            "answer": response.text.strip()
        }

    except Exception as e:
        print(f"Pipeline Error: {e}")
        return {"error": str(e)}

# 4. EXECUTION
def run_app(topic):
    global asr_model, vector_store
    
    # Kill any existing ngrok tunnels and check for background processes
    try: ngrok.kill() 
    except: pass
    
    vector_store = setup_knowledge_base(topic)
    if vector_store is None:
        print("Stopping: Knowledge base could not be created.")
        return None

    print("Loading ASR Model...")
    asr_model = AutoModel.from_pretrained("ai4bharat/indic-conformer-600m-multilingual", trust_remote_code=True)
    asr_model.eval()

    ngrok.set_auth_token(NGROK_TOKEN)
    public_url = ngrok.connect(8000).public_url
    print(f"\n API DEPLOYED AT: {public_url}")
    
    # We use a random port or try-except to handle the "Address in use" error
    try:
        threading.Thread(target=lambda: uvicorn.run(app, host="0.0.0.0", port=8000), daemon=True).start()
    except:
        print("Server already running on port 8000. Reusing existing instance.")
        
    return public_url

def test_chatbot(api_url):
    if not api_url: return
    time.sleep(5) 
    audio_files = glob.glob("*.mp4") + glob.glob("*.wav") + glob.glob("*.m4a")
    
    if not audio_files:
        print("No audio file found to test.")
        return

    print(f"Testing with: {audio_files[0]}")
    url = f"{api_url}/process_query"
    with open(audio_files[0], 'rb') as f:
        files = {'file': (audio_files[0], f, 'audio/mp4')}
        resp = requests.post(url, files=files, headers={"ngrok-skip-browser-warning": "true"})

    if resp.status_code == 200:
        data = resp.json()
        print("\n" + "═"*40)
        print(f"🇮🇳 Hindi: {data.get('hindi_transcription')}")
        print(f"🇬🇧 English: {data.get('english_translation')}")
        print(f"Answer: {data.get('answer')}")
        print("═"*40)
    else:
        print(f"Request failed: {resp.text}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice-Enabled RAG Chatbot")
    
    # We change "topic" to "--topic" (an optional flag).
    # This ensures it ignores positional junk arguments passed by Colab.
    parser.add_argument(
        "--topic", 
        type=str, 
        default="Machine Learning", 
        help="The search query/topic for Wikipedia scraping"
    )
    
    # parse_known_args tells it to ignore the --f= kernel arguments
    args, unknown = parser.parse_known_args()
    
    # If the user used a positional argument by mistake (like in !python main.py Topic),
    # we check 'unknown' and take the first item if it exists.
    search_topic = args.topic
    if unknown and not search_topic or search_topic == "Machine Learning":
        # Check if the first unknown argument looks like a real string and not a flag
        if unknown and not unknown[0].startswith('-'):
            search_topic = unknown[0]

    print(f"--- SUCCESS! TARGET TOPIC: {search_topic} ---")
    
    deployed_url = run_app(search_topic)
    test_chatbot(deployed_url)
