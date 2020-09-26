from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer


# load the model from disk
filename1 = 'random_forest.pkl'
filename2 = 'naive_bayes.pkl'

r_clf = pickle.load(open(filename1, 'rb'))

cv = pickle.load(open('transform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():


	if request.method == 'POST':
		text = request.form['text']
		data = [text]
		vect = cv.transform(data)
		my_prediction = r_clf.predict(vect)
	return render_template('result.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)
    
    
