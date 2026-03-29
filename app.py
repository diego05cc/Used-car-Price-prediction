from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

#LOAD MODELS

#linear regression
try:
    model = pickle.load(open('model.pkl', 'rb'))
except:
    model = None

#logistic regression
try:
    logistic_model = pickle.load(open('logistic_model.pkl', 'rb'))
except:
    logistic_model = None

#decision tree
try:
    tree_model = pickle.load(open('tree_model.pkl', 'rb'))
except:
    tree_model = None



#HOME
@app.route('/')
def home():
    return render_template('index.html')



#USE CASES
@app.route('/use-cases')
def use_cases():
    return render_template('use_cases.html')

@app.route('/use-case1')
def use_case1():
    return render_template('use_case1.html')

@app.route('/use-case2')
def use_case2():
    return render_template('use_case2.html')

@app.route('/use-case3')
def use_case3():
    return render_template('use_case3.html')

@app.route('/use-case4')
def use_case4():
    return render_template('use_case4.html')



#LINEAR REGRESSION
@app.route('/regression')
def regression():
    return render_template('regression.html')


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



#LOGISTIC REGRESSION
#concepts
@app.route('/logistic')
def logistic():
    return render_template('logistic.html')

#application
@app.route('/logistic-app')
def logistic_app():
    return render_template('logistic_app.html')

#prediction
@app.route('/logistic-predict', methods=['POST'])
def logistic_predict():
    if logistic_model is None:
        return "Logistic model not trained"

    try:
        year = int(request.form['year'])
        mileage = int(request.form['mileage'])

        prediction = logistic_model.predict([[year, mileage]])

        result = "High Price Car" if prediction[0] == 1 else "Low Price Car"

        return render_template('logistic_result.html', result=result)

    except:
        return "Error in input data"



#DECISION TREE
#concepts
@app.route('/tree')
def tree():
    return render_template('tree.html')

#application
@app.route('/tree-app')
def tree_app():
    return render_template('tree_app.html')

#pediction
@app.route('/tree-predict', methods=['POST'])
def tree_predict():
    if tree_model is None:
        return "Tree model not trained"

    try:
        year = int(request.form['year'])
        mileage = int(request.form['mileage'])

        prediction = tree_model.predict([[year, mileage]])

        result = "High Price Car" if prediction[0] == 1 else "Low Price Car"

        return render_template('tree_result.html', result=result)

    except:
        return "Error in input data"


#METRICS
@app.route('/metrics')
def metrics():
    return render_template('metrics.html')

@app.route('/tree-metrics')
def tree_metrics():
    return render_template('tree_metrics.html')


#RUN APP
if __name__ == "__main__":
    app.run(debug=True)