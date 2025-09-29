from flask import Flask, render_template, request
import xgboost as xgb
import numpy as np

app = Flask(__name__)

# Load model
model = xgb.XGBClassifier()
model.load_model("xgb_model.json")

# Mapping hasil prediksi ke label
label_mapping = {
    0: "Tidak Terdiagnosis",
    1: "Terdiagnosis"
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Ambil semua field dari form
            fields = [
                "ketergantungan",
                "batuk",
                "sesak",
                "demam",
                "diagnosa_awal",
                "riwayat",
                "jenis_kelamin",
                "usia"
            ]

            # Ganti kosong dengan np.nan
            values = []
            for f in fields:
                val = request.form.get(f, "")
                if val.strip() == "":
                    values.append(np.nan)
                else:
                    values.append(float(val))

            features = np.array([values])

            # Prediksi
            raw_pred = model.predict(features)[0]
            prediction = label_mapping.get(int(raw_pred), f"Label {raw_pred}")

            return render_template("results.html",
                       pred=prediction,
                       data=zip(fields, values))

        except Exception as e:
            return f"Error: {e}"

    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
