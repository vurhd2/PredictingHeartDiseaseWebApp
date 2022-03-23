import re
from turtle import setx
from flask import Flask, flash, redirect, render_template, request, session 
from flask_session import Session
from tempfile import mkdtemp
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MaxAbsScaler 
from sklearn.ensemble import RandomForestClassifier

# creating flask object
app = Flask(__name__)

# makes sure the app auto reloads
app.config["TEMPLATES_AUTO_RELOAD"] = True

# makes sure responses are not cached
@app.after_request
def after_request(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    response.headers["Expires"] = 0
    response.headers["Pragma"] = "no-cache"
    return response

# Configure session to use filesystem (instead of signed cookies)
app.config["SESSION_FILE_DIR"] = mkdtemp()
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

@app.route("/", methods=["GET", "POST"])
def redirecting():
    return redirect("/index")

global age
global sex
global chestpaintype
global restingbp
global cholesterol
global fastingbs
global maxhr
global exerciseangina
global oldpeak
global stslope

submitted = False
@app.route("/index", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        global age
        age = request.form.get("age")
        if not age:
            flash("Missing Age")
            return render_template("index.html")
        
        global sex
        sex = request.form.get("sex")
        if not sex:
            flash("Missing Sex")
            return render_template("index.html")
        elif sex != "M" and sex != "F" and sex != "I":
            flash("Invalid Sex")
            return render_template("index.html")
        
        global chestpaintype
        chestpaintype = request.form.get("chestpaintype")
        if not chestpaintype:
            flash("Missing Chest Pain Type")
            return render_template("index.html")
        elif chestpaintype != "TA" and chestpaintype != "ATA" and chestpaintype != "NAP" and chestpaintype != "ASY":
            flash("Invalid Chest Pain Type")
            return render_template("index.html")
        
        global restingbp
        restingbp = request.form.get("restingbp")
        if not restingbp:
            flash("Missing Resting Blood Pressure")
            return render_template("index.html")

        global cholesterol
        cholesterol = request.form.get("cholesterol")
        if not cholesterol:
            flash("Missing Cholesterol")
            return render_template("index.html")

        global fastingbs
        fastingbs = request.form.get("fastingbs")
        if not fastingbs:
            flash("Missing Fasting Blood Sugar")
            return render_template("index.html")

        global maxhr
        maxhr = request.form.get("maxhr")
        if not maxhr:
            flash("Missing Maximum Heart Rate")
            return render_template("index.html")
        
        global exerciseangina
        exerciseangina = request.form.get("exerciseangina")
        if not exerciseangina:
            flash("Missing Exercise Angina")
            return render_template("index.html")
        elif exerciseangina != "Y" and exerciseangina != "N":
            flash("Invalid Exercise Angina")
            return render_template("index.html")
        
        global oldpeak
        oldpeak = request.form.get("oldpeak")
        if not oldpeak:
            flash("Missing Oldpeak")
            return render_template("index.html")
        
        global stslope
        stslope = request.form.get("stslope")
        if not stslope:
            flash("Missing ST Slope")
            return render_template("index.html")
        elif stslope != "Up" and stslope != "Flat" and stslope != "Down":
            flash("Invalid ST Slope")
            return render_template("index.html")
        
        global submitted 
        submitted = True

        return redirect("/result")
    return render_template("index.html")

@app.route("/result", methods=["GET", "POST"])
def result():
    global submitted
    if not submitted:
        return redirect("/")
    submitted = False
    stats = output()
    spec = stats[0] * 100
    sens = stats[1] * 100
    total = stats[2] * 50
    if (stats[3] == 1):
        result = "Yes"
    else:
        result = "No"

    return render_template("result.html",result=result,spec=spec,sens=sens,total=total)


def output():
    data = pd.read_csv('heart.csv')
    data = data.drop(columns=["RestingECG"])
    df = pd.get_dummies(data)
    #df = pd.get_dummies(data, columns=["Sex", "ChestPainType", "ExerciseAngina", "ST_Slope"])
    
    scaler = MaxAbsScaler()
    
    x = scaler.fit_transform(df.drop("HeartDisease", axis=1))
    y = df["HeartDisease"]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

    model = RandomForestClassifier()
    model.fit(x_train, y_train.ravel())

    predictions = model.predict(x_test)

    skf = StratifiedKFold(shuffle=True, n_splits=20)
    SKFpredictions = cross_val_predict(model, x_test, y_test.ravel(), cv=skf)
    TN, FP, FN, TP = confusion_matrix(y_test, SKFpredictions, labels=[0,1]).ravel()

    specificity = float(TN / (TN + FP))

    sensitivity = float(TP / (FN + TP))

    global age
    global sex
    global chestpaintype
    global restingbp
    global cholesterol
    global fastingbs
    global maxhr
    global exerciseangina
    global oldpeak
    global stslope

    x_input = [age, sex, chestpaintype, restingbp, cholesterol, fastingbs, maxhr, exerciseangina, oldpeak, stslope]
    x_input = pd.get_dummies(x_input)

    x_input = scaler.fit_transform(x_input)

    #new_predictions = cross_val_predict(model, x_input, y_output, cv=skf)
    new_predictions = model.predict(x_input)

    return [specificity, sensitivity, specificity+sensitivity, new_predictions[0]]



