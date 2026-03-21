import pyttsx3

engine = pyttsx3.init('sapi5')   # 🔥 force Windows voice

engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

def speak(text):
    try:
        print("Speaking:", text)
        engine.stop()
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        print("Voice error:", e)