from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))


@app.route('/')
def home():
    return render_template('index.html')  # Simple HTML page for input

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']

    # Preprocess and vectorize
    processed_message = tfidf.transform([message])

    prediction = model.predict(processed_message)
    result = 'Spam' if prediction[0] == 1 else 'Not Spam'

    return render_template('index.html', prediction_text=f'Message is {result}')

if __name__ == '__main__':
    app.run(debug=True)