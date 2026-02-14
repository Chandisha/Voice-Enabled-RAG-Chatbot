import gradio as gr
import requests
import json

# Replace this with the URL printed by your previous cell
# It should look like: https://xxxx-xx-xx-xx-xx.ngrok-free.app
API_URL = "INSERT_YOUR_NGROK_URL_HERE"
def chat_with_rag(audio_path):
    if not audio_path:
        return "Please record or upload an audio file."
    
    url = f"{API_URL}/process_query"
    
    try:
        # Open the audio file recorded by Gradio
        with open(audio_path, 'rb') as f:
            files = {'file': (audio_path, f, 'audio/wav')}
            # ngrok-skip-browser-warning header is required to bypass the ngrok landing page
            headers = {"ngrok-skip-browser-warning": "true"}
            
            response = requests.post(url, files=files, headers=headers)
        
        if response.status_code == 200:
            data = response.json()
            
            # Extract data from your FastAPI response
            hindi = data.get('hindi_transcription', 'N/A')
            english = data.get('english_translation', 'N/A')
            answer = data.get('answer', 'No answer generated.')
            
            # Format as a chat-style string
            formatted_chat = (
                f"**🎤 You (Hindi):** {hindi}\n\n"
                f"**🔤 Translated:** {english}\n\n"
                f"**🤖 Bot:** {answer}"
            )
            return formatted_chat
        else:
            return f"Error: API returned status code {response.status_code}\n{response.text}"
            
    except Exception as e:
        return f"UI Error: {str(e)}"

# Define the Gradio Interface
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎙️ Voice-Enabled RAG Chatbot")
    gr.Markdown("Record your query in **Hindi** about the topic you scraped. The bot will transcribe, translate, and answer using the Wikipedia knowledge base.")
    
    with gr.Row():
        with gr.Column(scale=1):
            audio_input = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Speak here")
            submit_btn = gr.Button("Send to RAG Bot", variant="primary")
            
        with gr.Column(scale=2):
            chat_output = gr.Markdown("### Response will appear here...")

    submit_btn.click(
        fn=chat_with_rag,
        inputs=audio_input,
        outputs=chat_output
    )

# Launch the UI
demo.launch(share=True, debug=True)
