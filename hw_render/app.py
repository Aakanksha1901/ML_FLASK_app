from flask import Flask, request, render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    Temperature = float(request.form['Temperature'])

    prediction = model.predict([[Temperature]])

    return render_template(
        'index.html',
        prediction_text=f"Predicted Sales: {prediction[0]}"
    )

if __name__ == '__main__':
    app.run(debug=True)