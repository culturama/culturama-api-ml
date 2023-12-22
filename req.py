import requests

# Ganti URL dengan URL tempat Flask berjalan
API_URL = 'http://127.0.0.1:5000/recommend/'

# Contoh input data penilaian dari pengguna
user_ratings = {
    "Place_Id": 1,
    "Place_Ratings": 4.5
}

# Membuat permintaan POST ke API
response = requests.post(API_URL, json={"ratings": user_ratings})

# Mengecek apakah permintaan sukses (status kode 200)
if response.status_code == 200:
    data = response.json()
    recommendation = data.get("recommendations", {})
    print(f"- Tempat: {recommendation.get('Place_Name')}")
    print(f"- Kategori: {recommendation.get('Category')}")
    print(f"- Prediksi Rating: {recommendation.get('Predicted_Rating')}")
else:
    print(f"Error: {response.status_code}, {response.text}")
