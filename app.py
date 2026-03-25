from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

#try:
 #   model = pickle.load(open('model.pkl', 'rb'))
#except:
 #   model = None


# HOME

@app.route('/')
def home():
    return render_template('index.html')


# USE CASES
@app.route('/use-cases')
def use_cases():
    return render_template('use_cases.html')


# LINEAR REGRESSION PAGE

@app.route('/regression')
def regression():
    return render_template('regression.html')


# PREDICTION

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return "Model not trained yet"

    try:
        year = int(request.form['year'])
        mileage = int(request.form['mileage'])

        prediction = model.predict([[year, mileage]])
        price = round(prediction[0], 2)

        return render_template('result.html', price=price)

    except:
        return "Error in input data"


# RUN APP

if __name__ == "__main__":
    app.run(debug=True)