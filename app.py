from flask import Flask, request, render_template, redirect, url_for, session
import pickle

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a secure, random key

# Load model and utilities
model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
label_encoder_sentiment = pickle.load(open("label_encoder_sentiment.pkl", "rb"))

@app.route('/')
def welcome():
    return render_template('welcome.html')

@app.route('/input', methods=['GET', 'POST'])
def user_input():
    if request.method == 'POST':
        year = request.form['year']
        month = request.form['month']
        day = request.form['day']
        time_of_tweet = request.form['time_of_tweet']
        platform = request.form['platform']
        text = request.form['text']
        session['input'] = {'year': year, 'month': month, 'day': day,
                            'time_of_tweet': time_of_tweet,
                            'platform': platform, 'text': text}
        return redirect(url_for('output'))
    return render_template('input.html')

@app.route('/output')
def output():
    data = session.get('input', {})
    if not data:
        return redirect(url_for('user_input'))
    text = data['text']
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    sentiment = int(prediction)
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    label = sentiment_map.get(sentiment, "Unknown")

    return render_template('output.html', prediction=sentiment, label=label, text=text)

if __name__ == '__main__':
    app.run(debug=True)
