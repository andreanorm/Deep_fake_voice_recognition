import requests

from main import predict

files = {"file":open("linus-original-DEMO.mp3", "rb")}

results = requests.post("https://deepfakevoicerecognition-fy2s5a5k2q-ew.a.run.app/predict/", files=files)

print(results.json())
