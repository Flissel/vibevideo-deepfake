"""Clone voice from WhatsApp video and generate TTS."""
import requests
import os

API_KEY = os.environ.get("ELEVENLABS_API_KEY", "sk_d213401526da11762046dfa179f8f6f7d0890e51981b6a6e")
headers = {"xi-api-key": API_KEY}

# Step 1: Clone voice
print("Cloning voice from WhatsApp audio...")
with open("whatsapp_voice_sample.wav", "rb") as f:
    resp = requests.post(
        "https://api.elevenlabs.io/v1/voices/add",
        headers=headers,
        data={"name": "WhatsApp Test Voice", "description": "Cloned from WhatsApp video"},
        files={"files": ("sample.wav", f, "audio/wav")}
    )

print(f"Clone status: {resp.status_code}")
if resp.status_code != 200:
    print(resp.text[:500])
    exit(1)

voice_id = resp.json()["voice_id"]
print(f"Voice cloned! ID: {voice_id}")

# Step 2: Generate TTS
print("Generating TTS: Hey, it's your mum, get the shit done.")
tts_resp = requests.post(
    f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
    headers={**headers, "Content-Type": "application/json"},
    json={
        "text": "Hey, it's your mum, get the shit done.",
        "model_id": "eleven_multilingual_v2",
        "voice_settings": {"stability": 0.5, "similarity_boost": 0.85}
    }
)

print(f"TTS status: {tts_resp.status_code}")
if tts_resp.status_code != 200:
    print(tts_resp.text[:500])
    exit(1)

os.makedirs("tts", exist_ok=True)
out_path = "tts/WhatsApp_Test.mp3"
with open(out_path, "wb") as f:
    f.write(tts_resp.content)

print(f"TTS saved: {out_path} ({len(tts_resp.content)/1024:.0f} KB)")

# Save voice_id for later
with open("whatsapp_voice_id.txt", "w") as f:
    f.write(voice_id)
print(f"Voice ID saved to whatsapp_voice_id.txt")
