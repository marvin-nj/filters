import scipy
import librosa
import soundfile


audio, sr = librosa.load("recording-20210122.wav")
#see  https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.wiener.html
result=scipy.signal.wiener(audio,len(audio))     
soundfile.write("result-wiener.wav", result, sr)
