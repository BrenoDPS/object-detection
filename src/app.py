from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Carregar o modelo pré-treinado ResNet50
model = tf.keras.applications.ResNet50(weights="imagenet")

# Função para processar a imagem
def preprocess_image(image):
    image = Image.open(io.BytesIO(image)).convert('RGB')
    image = image.resize((224, 224))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=0)
    return tf.keras.applications.resnet50.preprocess_input(image_array)

# Rota de predição
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400
    
    image_file = request.files['image'].read()
    processed_image = preprocess_image(image_file)

    # Realizar a predição
    predictions = model.predict(processed_image)
    decoded_predictions = tf.keras.applications.resnet50.decode_predictions(predictions, top=5)[0]

    # Formatando a resposta
    response = {label: float(score) for (_, label, score) in decoded_predictions}

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
