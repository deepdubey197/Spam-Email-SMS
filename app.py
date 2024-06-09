from flask import Flask, request, render_template
import pickle
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the saved model and vectorizer
model = load_model('spam_classifier_model.h5')
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Function to classify new text
def classify_text(text):
    # Preprocess the text
    text_features = vectorizer.transform([text]).toarray()
    # Predict using the loaded model
    prediction = model.predict(text_features)
    return 'Spam' if prediction[0][0] > 0.5 else 'Not Spam'

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        user_input = request.form['text']
        result = classify_text(user_input)
        return render_template('index.html', input_text=user_input, prediction=result)
    return render_template('index.html', input_text='', prediction='')

if __name__ == '__main__':
    app.run(debug=True)
