from flask import Flask, render_template
import integrated_model as im
import pandas as pd

app = Flask(__name__,template_folder='template')
#data = im.ProcessData("Data Orbit - Data Gabungan.csv")
#data.run_preprocessing()
#data.getData()

@app.route('/')
def hello_world():
    return render_template("index.html")

@app.route('/rekomendasi')
def rekomendasi():
    return render_template("rekomendasi.html")

@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/detail/<detail>', methods=['GET'])
def detail(detail):
    json = pd.read_json("static/rekomendasi_restauran.json")
    data = json[json["id"] == detail]
    data.to_json("static/detail_restauran.json",orient = 'records')
    return render_template("detail_tempat.html")

if __name__ == "__main__":
    app.run(debug = True, port = 5000)