from flask import Flask, jsonify, render_template
import pandas as pd
import os

app = Flask(__name__)

CSV_FILE = "esp_live_output.csv"

@app.route("/")
def index():
    # This assumes you have templates/index.html
    return render_template("index.html")

@app.route("/api/latest")
def latest():
    """Return the most recent row as JSON"""
    if not os.path.exists(CSV_FILE):
        return jsonify({"error": "No data yet"}), 404
    
    df = pd.read_csv(CSV_FILE)
    if df.empty:
        return jsonify({"error": "No data yet"}), 404
    
    latest_row = df.iloc[-1].to_dict()
    return jsonify(latest_row)

@app.route("/api/history")
def history():
    """Return full history (all rows) as JSON"""
    if not os.path.exists(CSV_FILE):
        return jsonify({"error": "No data yet"}), 404
    
    df = pd.read_csv(CSV_FILE)
    return df.to_json(orient="records")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
