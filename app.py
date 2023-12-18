from subprocess import check_output
from flask import Flask, render_template, request
import joblib
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

app = Flask(__name__)



@app.route('/')
def index():
    return render_template('main.html') 


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get user input for age and gender from the form
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        fav = str(request.form['fav'])

        # Make predictions based on user input
        user_input = [[age, gender, fav]]
        
        model=joblib.load( 'our_pridction.joblib')

        predictions = model.predict(user_input)

        return render_template('predict.html', predictions=predictions[0])

    return render_template('predict.html')


@app.route('/learn', methods=['GET','POST'])
def learn():
    if request.method == 'POST':
        age = int(request.form['age'])
        gender = int(request.form['gender'])
        fav = int(request.form['fav'])
        genre = str(request.form['genre'])
        
        music_dt = pd.read_csv('music.csv') # Load existing CSV file into a DataFrame
        new_row_dict = {'age': age, 'gender': gender,'fav': fav, 'genre': genre}# Create a new row as a dictionary or list
        music_dt = music_dt._append  (new_row_dict, ignore_index=True)# Append the new row to the DataFrame
        music_dt.to_csv('music.csv', index=False)# Save the updated DataFrame back to the CSV file
        
        # prepare 2 groups (features, output)
        X=music_dt.drop(columns=['genre']) # sample features
        Y=music_dt['genre'] # sample output

        # create ant train the model 
        model = DecisionTreeClassifier()
        model.fit(X,Y)

        joblib.dump(model, 'our_pridction.joblib') #save pridcions to binary file for quiq reuse
                
        done = True
        
        return render_template('learn.html', done=done)


    return render_template('learn.html')



if __name__ == '__main__':
    app.run(debug=True)