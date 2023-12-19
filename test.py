import requests

files = {"file":open("Test_Elise.m4a", "rb")}

# results = requests.post("https://audioauthenticator-fy2s5a5k2q-ew.a.run.app/predict/", files=files)
results = requests.post("http://127.0.0.1:8000/predict/", files=files)

print(results.json())
