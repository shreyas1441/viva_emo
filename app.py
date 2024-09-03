from flask import Flask, request, jsonify, render_template
import joblib

app = Flask(__name__)

# Load the trained model
model_filename = 'viva_emotion_prediction_model.pkl'
model = joblib.load(model_filename)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the text from the request
    data = request.json
    if 'text' not in data:
        return jsonify({"error": "No text provided"}), 400

    text = data['text']
    
    # Predict the emotion
    predicted_emotion = model.predict([text])
    
    # Return the prediction as a JSON response
    return jsonify({"text": text, "predicted_emotion": predicted_emotion[0]})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
