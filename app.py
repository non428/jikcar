from flask import Flask, render_template, request
from model import recommend_places, df, model

app = Flask(__name__)

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        user_input = request.form['name']
        # top_n を 1 に設定して、1つの推薦結果のみを取得
        recommendations = recommend_places(user_input, df, model, top_n=1)
        return render_template('predict.html', recommendations=recommendations)
    return render_template('predict.html', recommendations=None)

if __name__ == "__main__":
    app.run(debug=True)
