import pyttsx3

engine = pyttsx3.init()

# Configure voice
engine.setProperty("rate", 150)
engine.setProperty("volume", 1.0)

text = input("Enter something you want me to speak: ")

# Save speech into an audio file
engine.save_to_file(text, "output.wav")

engine.runAndWait()

print("Audio file generated successfully!")