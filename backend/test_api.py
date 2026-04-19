import requests

url = "http://127.0.0.1:5000/detect"

files = {
    "image": open("../dataset/test/images/potholes12_png_jpg.rf.8adae25d6550c012131dc0d0f468d572.jpg", "rb")
}

response = requests.post(url, files=files)

print(response.json())