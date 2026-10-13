from flask import Flask, request, render_template
import joblib
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)

# Load the trained models
model_tesla = joblib.load('tesla_model.pkl')
model_adv = joblib.load('adv_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict_tesla', methods=['POST'])
def predict_tesla():
    year = float(request.form['year'])
    km = float(request.form['km'])
    
    pred = model_tesla.predict([[year, km]])
    result = f"${pred[0]:,.2f}"
    
    return render_template('index.html', result_tesla=result)

@app.route('/predict_sales', methods=['POST'])
def predict_sales():
    tv = float(request.form['tv'])
    radio = float(request.form['radio'])
    newspaper = float(request.form['newspaper'])
    
    pred = model_adv.predict([[tv, radio, newspaper]])
    result = f"{pred[0]:.2f} Units"
    
    return render_template('index.html', result_sales=result)

if __name__ == "__main__":
    app.run(debug=True, port=5000)