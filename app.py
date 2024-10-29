import pandas as pd # type: ignore
from flask import Flask, render_template, request, redirect, url_for # type: ignore
import os

app = Flask(__name__)

# Path to the CSV file
csv_file_path = 'customers.csv'

# Create CSV file with headers if it doesn't exist
if not os.path.isfile(csv_file_path):
    df = pd.DataFrame(columns=["age", "purchase_frequency", "average_spent", "spending_score"])
    df.to_csv(csv_file_path, index=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    # Get form data
    age = int(request.form['age'])
    purchase_frequency = int(request.form['purchase_frequency'])
    average_spent = float(request.form['average_spent'])
    
    # Calculate spending score (example calculation, modify as needed)
    spending_score = purchase_frequency * average_spent
    
    # Create a new DataFrame with the new data
    new_data = pd.DataFrame([[age, purchase_frequency, average_spent, spending_score]], 
                             columns=["age", "purchase_frequency", "average_spent", "spending_score"])
    
    # Append to the existing CSV file
    try:
        new_data.to_csv(csv_file_path, mode='a', header=False, index=False)
    except Exception as e:
        print("Error while writing to CSV:", e)  # Debug print to check for issues
    
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
