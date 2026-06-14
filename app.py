from flask import Flask, request, render_template, jsonify
import joblib

app = Flask(__name__)

# 1. Load the ML "Brain" when the server starts
print("Loading model and vectorizer...")
model = joblib.load('phishing_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# 2. Route for the Home Page
@app.route('/')
def home():
    # This serves the index.html file to the user's browser
    return render_template('index.html')

# 3. Route for the API (Receives the email and returns the prediction)
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the email text from the frontend
        data = request.get_json()
        raw_email = data.get('email_text', '')

        if not raw_email.strip():
            return jsonify({'error': 'Please enter some text to analyze.'})

        # Clean the text (match your training logic)
        clean_email = raw_email.replace('Subject:', '')

        # Vectorize and Predict
        email_vector = vectorizer.transform([clean_email])
        prediction = model.predict(email_vector)[0]
        
        # Get exact confidence scores
        probabilities = model.predict_proba(email_vector)[0]
        phishing_score = round(probabilities[1] * 100, 2)
        safe_score = round(probabilities[0] * 100, 2)

        # Format the response
        if prediction == 1:
            result = 'PHISHING'
            confidence = phishing_score
        else:
            result = 'SAFE'
            confidence = safe_score

        # Send the data back to the website
        return jsonify({
            'status': 'success',
            'result': result,
            'confidence': confidence
        })

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # Run the server on port 5000
    app.run(debug=True, port=5000)