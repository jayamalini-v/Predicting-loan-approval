        import pandas
import numpy as np
import pickle
import os
from flask import Flask, request, render_template

app = Flask(__name__)
model1 = pickle.load(open(r'rdf.pkl', 'rb'))
scale = pickle.load(open(r'scale1.pkl', 'rb'))


@app.route('/')  # rendering the html template
def home():
    return render_template('index.html')


@app.route('/predict', methods=["POST", "GET"])  # rendering the html template
def predict():
    return render_template("prediction.html")


@app.route('/submit', methods=["POST", "GET"])
def submit(values=None):
    # reading the inputs given by the user
    input_feature = [int(x) for x in request.form.values()]
    input_feature = np.transpose(input_feature)
    input_feature = [np.array(input_feature)]
    print(input_feature)
    names = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
             'ApplicantIncome', 'Co-applicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History',
             'Property_Area']
    data = pandas.DataFrame(input_feature, columns=names)
    print(data)

    data_scaled = scale.fit_transfrom(data)
    data = pandas.DataFrame(data, columns=names)

    # predictions using the loaded model1 file
    prediction = model1.predict(data)
    print(prediction)
    prediction = int(prediction)
    print(type(prediction))

    if prediction == 0:
        return render_template("prediction.html", result="Loan will not be Approved")
    else:
        return render_template("prediction.html", result="Loan will be Approved")
    # showing the prediction results in a UI


if __names__ == "__main__":
    app.run(debug=False)  # running the app
