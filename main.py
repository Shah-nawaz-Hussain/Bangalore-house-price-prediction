from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

data = pd.read_csv('cleaned_data.csv')
pipe = pickle.load(open('LinearRegressionModel (1).pkl', 'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', locations=locations)

@app.route('/predict', methods=['POST'])
def predict():
    location = request.form.get('location')
    bhk = float(request.form.get('bhk'))
    bath = float(request.form.get('bath'))
    total_sqft = float(request.form.get('sq_feet'))

    input_df = pd.DataFrame([[location, total_sqft, bath, bhk]],
                            columns=['location', 'total_sqft', 'bath', 'bhk'])

    prediction = pipe.predict(input_df)[0] * 100000

    return str(round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)
