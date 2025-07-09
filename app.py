from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)
model = pickle.load(open('workout_model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        data = {
            'Age': int(request.form.get('age', 0)),
            'Gender': request.form.get('gender', 'Male'),
            'Weight (kg)': float(request.form.get('weight', 0)),
            'Height (m)': float(request.form.get('height', 0)),
            'Max_BPM': int(request.form.get('max_bpm', 0)),
            'Avg_BPM': int(request.form.get('avg_bpm', 0)),
            'Resting_BPM': int(request.form.get('resting_bpm', 0)),
            'Session_Duration (hours)': float(request.form.get('duration', 0)),
            'Fat_Percentage': float(request.form.get('fat', 0)),
            'Water_Intake (liters)': float(request.form.get('water', 0)),
            'Workout_Frequency (days/week)': int(request.form.get('freq', 0)),
            'Experience_Level': int(request.form.get('exp', 0)),
            'BMI': float(request.form.get('bmi', 0)),
        }

        input_df = pd.DataFrame([data])
        prediction = model.predict(input_df)[0]

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)