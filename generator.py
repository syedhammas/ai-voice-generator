import os
import torch
import gradio as gr
from TTS.api import TTS
import shutil
import time 

# --- Configuration ---
PROJECT_NAME = "Hammas Voice Cloner"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SPEAKER_FOLDER = os.path.join(BASE_DIR, "speakers")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs")
os.makedirs(SPEAKER_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


# --- Model Loading and---

device = "cpu"
torch.set_num_threads(4) 
print(f"🚀 {PROJECT_NAME} is booting up...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

# --- Functions ---
def get_speakers():
    files = [f for f in os.listdir(SPEAKER_FOLDER) if f.endswith('.wav')]
    return files if files else ["No voices found"]

def upload_new_voice(file):
    if file is None: 
        return gr.update(choices=get_speakers()), "❌ File upload fail ho gayi."
    
    filename = os.path.basename(file.name)
    dest_path = os.path.join(SPEAKER_FOLDER, filename)
    shutil.copy(file.name, dest_path)
    
    # Refresh logic: List ko foran update karna
    new_choices = get_speakers()
    return gr.update(choices=new_choices, value=filename), f"✅ {filename} database mein add ho gayi!"

def generate_voice(text, speaker_name, language, speed_val, progress=gr.Progress()):
    if not text.strip(): return None, "⚠️ Pehle text likhein!"
    if speaker_name == "No voices found": return None, "⚠️ Please select Voice before Start."
    
    speaker_path = os.path.join(SPEAKER_FOLDER, speaker_name)
    output_path = os.path.join(OUTPUT_FOLDER, "hammas_output.wav")
    
    try:
        progress(0.1, desc="Hammas Engine Starting...")
        # Speed parameter ko float mein convert karke pass kiya hai
        tts.tts_to_file(
            text=text,
            speaker_wav=speaker_path,
            language=language,
            file_path=output_path,
            speed=float(speed_val), 
            split_sentences=True
        )
        progress(1.0, desc="Generation Complete!")
        return output_path, f"✨ {PROJECT_NAME} ne awaaz tayyar kar di!"
    except Exception as e:
        return None, f"❌ System Error: {str(e)}"

# --- Ultra Modern UI Design ---
modern_css = """
body { background: #050505; color: white; }
.gradio-container { 
    background: #0a0a0c; 
    border: 1px solid rgba(255, 136, 0, 0.3); 
    border-radius: 25px !important; 
    box-shadow: 0 10px 40px rgba(255, 136, 0, 0.1);
}
#hero-title { 
    background: linear-gradient(90deg, #ff8c00, #ff4d00);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    font-size: 40px; font-weight: 900; text-align: center; margin-bottom: 0px;
}
.glass-panel { 
    background: rgba(255, 255, 255, 0.03) !important; 
    border: 1px solid rgba(255, 255, 255, 0.1) !important; 
    border-radius: 15px !important; 
}
#main-button { 
    background: linear-gradient(135deg, #ff8c00 0%, #ff4d00 100%) !important; 
    border: none !important; color: white !important; font-weight: bold !important;
    box-shadow: 0 5px 15px rgba(255, 77, 0, 0.3) !important;
}
footer { display: none !important; }
"""

with gr.Blocks(title=PROJECT_NAME) as demo:
    gr.HTML(f"<div id='hero-title'>{PROJECT_NAME.upper()}</div>")
    gr.Markdown("<p style='text-align:center; color:#666; margin-top:-10px;'>Powered by Hammas Neural Engine</p>")
    
    with gr.Row():
        # Left Side
        with gr.Column(scale=1, elem_classes=["glass-panel"]):
            gr.Markdown("### 🎙️ Voice Library")
            voice_file = gr.File(label="Upload New Sample (.wav)", file_types=[".wav"])
            upload_btn = gr.Button("✨ Sync to Cloud", variant="secondary")
            
            gr.Markdown("---")
            
            speaker_select = gr.Dropdown(choices=get_speakers(), label="Select Persona", value=get_speakers()[0] if get_speakers() else None)
            speed_slider = gr.Slider(0.5, 2.0, 1.0, step=0.1, label="⏩ Speed Control")
            lang_select = gr.Dropdown(choices=["en", "ur", "hi", "ar", "es"], value="en", label="Language")

        # Right Side
        with gr.Column(scale=2):
            gr.Markdown("### ✍️ Scripting Zone")
            text_input = gr.Textbox(placeholder="Yahan apna script likhein...", lines=12, label=None)
            
            gen_btn = gr.Button("⚡ GENERATE CLONE VOICE", elem_id="main-button")
            
            gr.Markdown("---")
            
            with gr.Group(elem_classes=["glass-panel"]):
                audio_out = gr.Audio(label="Playback Studio", type="filepath")
                status_out = gr.Textbox(label="System Feedback", interactive=False)

    # UI Logic
    upload_btn.click(upload_new_voice, inputs=[voice_file], outputs=[speaker_select, status_out])
    
    gen_btn.click(
        generate_voice, 
        inputs=[text_input, speaker_select, lang_select, speed_slider], 
        outputs=[audio_out, status_out]
    )

if __name__ == "__main__":
    demo.launch(inbrowser=True, css=modern_css)