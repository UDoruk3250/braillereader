from gtts import gTTS

ttsA = gTTS("A",lang="tr")
ttsA.save("A.mp3")
ttsA = gTTS("B",lang="tr")
ttsA.save("B.mp3")
ttsA = gTTS("C",lang="tr")
ttsA.save("C.mp3")
ttsA = gTTS("D",lang="tr")
ttsA.save("D.mp3")
print("ALL DONE")
