from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle
model = pickle.load(open('Alcohol.pkl', 'rb'))
Scale_1 = pickle.load(open('Al_obj.obj', 'rb'))
app = Flask(__name__)
@app.route('/')
def fun():
    return render_template("one.html")

@app.route('/one', methods=["POST"])
def one():
    beer_servings = request.form["beer_servings"]
    spirit_servings = request.form["spirit_servings"]
    wine_servings = request.form["wine_servings"]

    array1 = np.array([[beer_servings,spirit_servings,wine_servings]])

    df1 = pd.DataFrame(array1)

    Scale_2 = Scale_1.transform(df1)
    pred = model.predict(Scale_2)
    return render_template("output.html",data = pred)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5007,debug=False)