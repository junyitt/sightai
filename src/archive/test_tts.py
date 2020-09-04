import gtts
from playsound import playsound
import pyttsx3

# tts = gtts.gTTS("Hello world")
# tts.save("hello.mp3")
# playsound("hello.mp3")


# tts = gtts.gTTS("Hola Mundo", lang="es")
# tts.save("hola.mp3")
# playsound("hola.mp3")


engine = pyttsx3.init()
text = "Python is a great programming language"
engine.say(text)
# play the speech
engine.runAndWait()

rate = engine.getProperty("rate")
print(rate)

voices = engine.getProperty("voices")
engine.setProperty("voice", voices[1].id)
engine.say(text)
engine.runAndWait()