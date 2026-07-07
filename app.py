from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

# Load trained model
model = pickle.load(open("bangluru_house_price_model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        location = request.form["location"]
        total_sqft = float(request.form["total_sqft"])
        bath = int(request.form["bath"])
        bhk = int(request.form["bhk"])

        # Input Validation
        if total_sqft < 300:
            return render_template(
                "index.html",
                prediction_text="❌ Total area must be at least 300 sqft."
            )

        if bath < 1:
            return render_template(
                "index.html",
                prediction_text="❌ Bathrooms must be at least 1."
            )

        if bhk < 1:
            return render_template(
                "index.html",
                prediction_text="❌ Bedrooms must be at least 1."
            )

        # Create DataFrame with the same columns used during training
        data = pd.DataFrame({
            "location": [location],
            "total_sqft": [total_sqft],
            "bath": [bath],
            "BHK": [bhk]
        })

        # Predict
        prediction = model.predict(data)[0]

        return render_template(
            "index.html",
            prediction_text=f"₹ {prediction:.2f} Lakhs"
        )

    except ValueError:
        return render_template(
            "index.html",
            prediction_text="❌ Please enter valid numeric values."
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"⚠ Error: {str(e)}"
        )


@app.errorhandler(404)
def page_not_found(e):
    return "<h1>404 - Page Not Found</h1>", 404


@app.errorhandler(500)
def server_error(e):
    return "<h1>500 - Internal Server Error</h1>", 500


if __name__ == "__main__":
    app.run(debug=True)