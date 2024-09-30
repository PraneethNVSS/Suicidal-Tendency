from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
from tkinter import filedialog
from tkinter.filedialog import askopenfilename
import numpy as np 
import matplotlib.pyplot as plt

from sklearn.preprocessing import normalize
from sklearn.ensemble import RandomForestClassifier

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import normalize
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgbm
import pandas as pd
from imblearn.over_sampling import SMOTE
from tensorflow.keras.utils import to_categorical

from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D
from keras.models import Sequential
from keras.models import model_from_json
import pickle

global filename

global X,Y
global classifier
global dataset
global X_train, X_test, y_train, y_test
accuracy = []
precision = []
recall = []
fscore = []
global le

main = tkinter.Tk()
main.title("Suicidal Tendency Detection") #designing main screen
main.geometry("1300x1200")


def uploadDataset():
    global filename
    global dataset
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    text.insert(END,filename+" loaded\n\n")

    dataset = pd.read_csv(filename)
    text.insert(END,"Dataset before applying machine translation\n\n")
    text.insert(END,str(dataset.head()))
    
    

def processDataset():
    global X,Y
    global dataset
    text.delete('1.0', END)
    label = dataset.groupby('attempt_suicide').size()
    label.plot(kind="bar")
    dataset.fillna(0, inplace = True)
    text.insert(END,"All missing values are replaced with 0\n")
    text.insert(END,"Total processed records found in dataset : "+str(dataset.shape[0])+"\n\n")
    plt.show()

def translation():
    global X_train, X_test, y_train, y_test
    global X,Y
    global le
    global dataset
    text.delete('1.0', END)

    dataset.drop(['time'], axis = 1,inplace=True)
    dataset.drop(['income'], axis = 1,inplace=True)

    Y = dataset.values[:,2]

    cols = ['gender','sexuallity','race','bodyweight','virgin','prostitution_legal','pay_for_sex','social_fear','stressed','what_help_from_others',
            'attempt_suicide','employment','job_title','edu_level','improve_yourself_how']
    le = LabelEncoder()
    dataset[cols[0]] = pd.Series(le.fit_transform(dataset[cols[0]].astype(str)))
    dataset[cols[1]] = pd.Series(le.fit_transform(dataset[cols[1]].astype(str)))
    dataset[cols[2]] = pd.Series(le.fit_transform(dataset[cols[2]].astype(str)))
    dataset[cols[3]] = pd.Series(le.fit_transform(dataset[cols[3]].astype(str)))
    dataset[cols[4]] = pd.Series(le.fit_transform(dataset[cols[4]].astype(str)))
    dataset[cols[5]] = pd.Series(le.fit_transform(dataset[cols[5]].astype(str)))
    dataset[cols[6]] = pd.Series(le.fit_transform(dataset[cols[6]].astype(str)))
    dataset[cols[7]] = pd.Series(le.fit_transform(dataset[cols[7]].astype(str)))
    dataset[cols[8]] = pd.Series(le.fit_transform(dataset[cols[8]].astype(str)))
    dataset[cols[9]] = pd.Series(le.fit_transform(dataset[cols[9]].astype(str)))
    dataset[cols[10]] = pd.Series(le.fit_transform(dataset[cols[10]].astype(str)))
    dataset[cols[11]] = pd.Series(le.fit_transform(dataset[cols[11]].astype(str)))
    dataset[cols[12]] = pd.Series(le.fit_transform(dataset[cols[12]].astype(str)))
    dataset[cols[13]] = pd.Series(le.fit_transform(dataset[cols[13]].astype(str)))
    dataset[cols[14]] = pd.Series(le.fit_transform(dataset[cols[14]].astype(str)))
    text.insert(END,"Dataset after applying machine translation\n\n")
    text.insert(END,str(dataset)+"\n\n")
    Y = dataset.values[:,12]
    print(Y)
    dataset.drop(['attempt_suicide'], axis = 1,inplace=True)
    X = dataset.values
    sm = SMOTE(random_state = 42)
    X, Y = sm.fit_resample(X, Y)
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    text.insert(END,"Total records used to train machine learning LITE GBM Algorithm is : "+str(X_train.shape[0])+"\n") 
    text.insert(END,"Total records used to test machine learning LITE GBM Algorithm is  : "+str(X_test.shape[0])+"\n")

def trainCNN():
    global X_train, X_test, y_train, y_test
    global X,Y
    global classifier
    global accuracy
    global precision
    global recall
    global fscore
    accuracy.clear()
    precision.clear()
    recall.clear()
    fscore.clear()
    text.delete('1.0', END)
    #classifier = lgbm.LGBMClassifier()
    #classifier.fit(X,Y)
    XX = X.reshape(X.shape[0],X.shape[1],1,1)
    YY = to_categorical(Y)
    X_train1 = X_train.reshape(X_train.shape[0],X_train.shape[1],1,1)
    X_test1 = X_test.reshape(X_test.shape[0],X_test.shape[1],1,1)
    y_train1 = to_categorical(y_train)
    y_test1 = to_categorical(y_test)
    classifier = Sequential()
    classifier.add(Convolution2D(32, 1, 1, input_shape = (X_train1.shape[1], X_train1.shape[2], X_train1.shape[3]), activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1, 1)))
    classifier.add(Convolution2D(32, 1, 1, activation = 'relu'))
    classifier.add(MaxPooling2D(pool_size = (1, 1)))
    classifier.add(Flatten())
    classifier.add(Dense(output_dim = 256, activation = 'relu'))
    classifier.add(Dense(output_dim = y_train1.shape[1], activation = 'softmax'))
    print(classifier.summary())
    classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    hist = classifier.fit(XX, YY,batch_size=16, epochs=70, shuffle=True, verbose=2)
    predict = classifier.predict(X_test1)
    predict = np.argmax(predict, axis=1)
    y_test1 = np.argmax(y_test1, axis=1)
    a = accuracy_score(y_test1,predict)*100
    p = precision_score(y_test1, predict,average='macro') * 100
    r = recall_score(y_test1, predict,average='macro') * 100
    f = f1_score(y_test1, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(r)
    text.insert(END,"Propose CNN Accuracy on Test Data : "+str(a)+"\n")
    text.insert(END,"Propose CNN Precision on Test Data : "+str(p)+"\n")
    text.insert(END,"Propose CNN GBM Recall on Test Data    : "+str(r)+"\n")
    text.insert(END,"Propose CNN GBM FSCORE on Test Data    : "+str(f)+"\n\n")
    
    
def RFTraining():
    global accuracy
    global X_train, X_test, y_train, y_test
    global X,Y
    rf = RandomForestClassifier(class_weight='balanced')
    rf.fit(X_train,y_train)
    predict = rf.predict(X_test)
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    accuracy.append(a)
    precision.append(p)
    recall.append(r)
    fscore.append(r)
    text.insert(END,"Existing Random Forest Accuracy on Test Data : "+str(a)+"\n")
    text.insert(END,"Existing Random Forest Precision on Test Data : "+str(p)+"\n")
    text.insert(END,"Existing Random Forest Recall on Test Data    : "+str(r)+"\n")
    text.insert(END,"Existing Random Forest FSCORE on Test Data    : "+str(f)+"\n\n")
    
def predict():
    global classifier
    text.delete('1.0', END)
    filename = filedialog.askopenfilename(initialdir="Dataset")
    testData = pd.read_csv(filename)
    temp = testData.values
    testData.fillna(0, inplace = True)
    testData.drop(['time'], axis = 1,inplace=True)
    testData.drop(['income'], axis = 1,inplace=True)
    cols = ['gender','sexuallity','race','bodyweight','virgin','prostitution_legal','pay_for_sex','social_fear','stressed','what_help_from_others',
            'employment','job_title','edu_level','improve_yourself_how']
    testData[cols[0]] = pd.Series(le.fit_transform(testData[cols[0]].astype(str)))
    testData[cols[1]] = pd.Series(le.fit_transform(testData[cols[1]].astype(str)))
    testData[cols[2]] = pd.Series(le.fit_transform(testData[cols[2]].astype(str)))
    testData[cols[3]] = pd.Series(le.fit_transform(testData[cols[3]].astype(str)))
    testData[cols[4]] = pd.Series(le.fit_transform(testData[cols[4]].astype(str)))
    testData[cols[5]] = pd.Series(le.fit_transform(testData[cols[5]].astype(str)))
    testData[cols[6]] = pd.Series(le.fit_transform(testData[cols[6]].astype(str)))
    testData[cols[7]] = pd.Series(le.fit_transform(testData[cols[7]].astype(str)))
    testData[cols[8]] = pd.Series(le.fit_transform(testData[cols[8]].astype(str)))
    testData[cols[9]] = pd.Series(le.fit_transform(testData[cols[9]].astype(str)))
    testData[cols[10]] = pd.Series(le.fit_transform(testData[cols[10]].astype(str)))
    testData[cols[11]] = pd.Series(le.fit_transform(testData[cols[11]].astype(str)))
    testData[cols[12]] = pd.Series(le.fit_transform(testData[cols[12]].astype(str)))
    testData[cols[13]] = pd.Series(le.fit_transform(testData[cols[13]].astype(str)))
    
    testData = testData.values
    testData = testData.reshape(testData.shape[0],testData.shape[1],1,1)
    #testData = normalize(testData)
    predict = classifier.predict(testData)
    predict = np.argmax(predict, axis=1)
    print(predict)
    for i in range(len(predict)):
        if predict[i] == 1:
            text.insert(END,str(temp[i])+" ====> SUICIDAL Depression Detected\n\n")
        if predict[i] == 0:
            text.insert(END,str(temp[i])+" ====> NO SUICIDAL Depression Detected\n\n")

def graph():
    df = pd.DataFrame([['CNN','Precision',precision[0]],['CNN','Recall',recall[0]],['CNN','F1 Score',fscore[0]],['CNN','Accuracy',accuracy[0]],
                       ['Random Forest','Precision',precision[1]],['Random Forest','Recall',recall[1]],['Random Forest','F1 Score',fscore[1]],['Random Forest','Accuracy',accuracy[1]],
                        
                                               
                      ],columns=['Parameters','Algorithms','Value'])
    df.pivot("Parameters", "Algorithms", "Value").plot(kind='bar')
    plt.show()

font = ('times', 16, 'bold')
title = Label(main, text='Suicidal Tendency Detection')
title.config(bg='dark goldenrod', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=30,width=110)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=10,y=100)
text.config(font=font1)

font1 = ('times', 13, 'bold')
uploadButton = Button(main, text="Upload Suicide Attempt & Stressed Dataset", command=uploadDataset, bg='#ffb3fe')
uploadButton.place(x=900,y=100)
uploadButton.config(font=font1)  

processButton = Button(main, text="Preprocess Dataset", command=processDataset, bg='#ffb3fe')
processButton.place(x=900,y=150)
processButton.config(font=font1) 

translationButton = Button(main, text="Machine Translation & Features Extraction", command=translation, bg='#ffb3fe')
translationButton.place(x=900,y=200)
translationButton.config(font=font1) 

gbmButton = Button(main, text="Train Propose CNN Algorithm", command=trainCNN, bg='#ffb3fe')
gbmButton.place(x=900,y=250)
gbmButton.config(font=font1)

gbmButton = Button(main, text="Train Existing Random Forest Algorithm", command=RFTraining, bg='#ffb3fe')
gbmButton.place(x=900,y=300)
gbmButton.config(font=font1) 

predictButton = Button(main, text="Predict Suicidal Attempt from Test Data", command=predict, bg='#ffb3fe')
predictButton.place(x=900,y=350)
predictButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph", command=graph, bg='#ffb3fe')
graphButton.place(x=900,y=400)
graphButton.config(font=font1)

main.config(bg='RoyalBlue2')
main.mainloop()



    
