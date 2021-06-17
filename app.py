from flask import Flask,render_template,url_for,request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import pickle

# load the model from disk
clf = joblib.load('XGB_model.pkl')
cv=joblib.load('TF_model.pkl')
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('quora index.html')

@app.route('/predict',methods=['POST'])
def predict():
#	train= pd.read_csv("train.csv", encoding="latin-1")
#   train.head()
#   train= pd.read_csv("train.csv", encoding="latin-1")
#   drop_cols = ['qid']
#   train =train.drop(drop_cols,axis = 1)
#   train.head()
#   train['target'] = train['target'].map({'ham': 0, 'spam': 1})
#   qs = train['question_text']
#   Y = train['target']

#	
#	# Extract Feature With tfidfVectorizer
#	import pickle
#   tf = TfidfVectorizer(lowercase=False, min_df=0.01, max_df=0.999)
#   X = tf.fit_transform(qs) # Fit the Data
  
#pickle.dump(tf, open('TF_model.pkl', 'wb'))
#    
#    
#	from sklearn.model_selection import train_test_split
#	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
#	from xgboost import XGBClassifier
#   import xgboost as xgb2
#   clf = xgb2.XGBClassifier(objective="binary:logistic")
#   clf.fit(X_train,y_train)
#   clf.score(X_test,y_test)
#   filename = 'xgb_wow.pkl'
#   pickle.dump(clf, open(filename, 'wb'))

	if request.method == 'POST':
		message = request.form['review_text']
		data = [message]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('predict.html',prediction = my_prediction)



if __name__ == '__main__':
	app.run(debug=True)