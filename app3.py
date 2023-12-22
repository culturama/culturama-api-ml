from flask import Flask, jsonify, request
import tensorflow as tf
import pandas as pd
import numpy as np

app = Flask(__name__)

from recommended_net import RecommenderNet

# Muat kembali model dengan menambahkan custom_objects
loaded_model = tf.keras.models.load_model('Collaborative_filtering_keras.h5', custom_objects={'RecommenderNet': RecommenderNet})

# Load user and place encodings
user_to_user_encoded = pd.read_pickle("user_to_user_encoded.pkl")
place_encoded_to_place = pd.read_pickle("place_encoded_to_place.pkl")
place_not_rated = pd.read_pickle("place_not_rated.pkl")
place_to_place_encoded = pd.read_pickle("place_to_place_encoded.pkl")

@app.route('/recommend/', methods=['POST'])
def get_recommendations():
    try:
        # Menerima input data penilaian dari pengguna
        user_ratings = request.get_json()["ratings"]

        place_id = user_ratings.get("Place_Id")
        place_ratings = user_ratings.get("Place_Ratings")

        if place_id is None or place_ratings is None:
            return jsonify({"error": "Invalid input format"}), 400

        # Cek apakah place_id ada di dalam data yang dimiliki
        if place_id not in place_to_place_encoded:
            return jsonify({"error": "Place ID not found"}), 404

        # Buat data penilaian user dan tempat untuk membuat prediksi
        user_place_array = np.array([[0, place_to_place_encoded[place_id]]])  # Anda bisa menentukan nilai default untuk User_Id

        # Gunakan model untuk membuat prediksi
        rating_prediction = loaded_model.predict(user_place_array).flatten()[0]

        # Tentukan kategori berdasarkan rating_prediction (disesuaikan dengan model dan data yang Anda miliki)
        if rating_prediction >= 0.5:
            predicted_category = "Bahari"
        else:
            predicted_category = "Budaya"

        # Format output rekomendasi
        recommendations = {
            "Place_Name": place_encoded_to_place[place_to_place_encoded[place_id]],
            "Category": predicted_category,
            "Predicted_Rating": float(rating_prediction)  # Konversi ke float
        }

        return jsonify({"recommendations": recommendations})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
