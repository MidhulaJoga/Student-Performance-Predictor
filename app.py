from flask import Flask, render_template, request, redirect, session
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

app = Flask(__name__)
app.secret_key = "secret"

lr_model = pickle.load(open("student_model.pkl", "rb"))
rf_model = pickle.load(open("rf_model.pkl", "rb"))

users = {}

@app.route("/")
def login():
    return render_template("index.html")

@app.route("/register", methods=["GET","POST"])
def register():
    if request.method == "POST":
        users[request.form["username"]] = request.form["password"]
        return redirect("/")
    return render_template("register.html")

@app.route("/dashboard", methods=["POST"])
def dashboard():
    username = request.form["username"]
    password = request.form["password"]

    if username in users and users[username] == password:
        session["user"] = username
        return render_template("dashboard.html")
    return "Invalid Credentials"

@app.route("/predict", methods=["POST"])
def predict():
    attendance = float(request.form["attendance"])
    study_hours = float(request.form["study_hours"])
    previous_marks = float(request.form["previous_marks"])

    features = np.array([[attendance, study_hours, previous_marks]])

    # Use only Random Forest for final prediction
    prediction = rf_model.predict(features)[0]

    result = "Pass" if prediction == 1 else "Fail"

    return render_template("dashboard.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=False,host='0.0.0.0',port=5000)