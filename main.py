import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, ConfusionMatrixDisplay
import joblib


from sklearn.metrics import accuracy_score  #оценка предсказаний
# Сравнение моделей на основе кросс-валидации

if __name__ == "__main__":
    #Загрузка, чтение и разбиение данных на тестовые и обучающие
    data  = pd.read_csv("D:\The Science\PGU\артюхина\Нейронка\CovidData.csv", sep=';')
    bool_prizn = ['PNEUMONIA','PREGNANT','DIABETES','COPD','ASTHMA','INMSUPR','HIPERTENSION','CARDIOVASCULAR','RENAL_CHRONIC','OTHER_DISEASE','OBESITY','TOBACCO','ICU','INTUBED']
    all_prizn = ['USMER', 'SEX', 'PATIENT_TYPE', 'DATE_DIED', 'PNEUMONIA', 'AGE',  'DIABETES', 'COPD', 'ASTHMA', 'INMSUPR', 'HIPERTENSION', 'OTHER_DISEASE', 'CARDIOVASCULAR', 'OBESITY','RENAL_CHRONIC','TOBACCO','CLASIFFICATION_FINAL']
    
    data = data.drop(['MEDICAL_UNIT'], axis = 1)
    
   
    
    data['CLASIFFICATION_FINAL'] = data['CLASIFFICATION_FINAL'].apply(lambda x: 0 if x in [1, 2, 3] else (1 if x in [4, 5, 6, 7, 8] else 97))
    data[bool_prizn] = data[bool_prizn].replace(1,0).replace(2,1)
    data[all_prizn] = data[all_prizn].replace(97,np.nan).replace(99,np.nan)
    data['AGE'] = data['AGE'].fillna(data['AGE'].median())

    pn_c = data['PNEUMONIA'].value_counts()
    data['PNEUMONIA'] = data['PNEUMONIA'].fillna(1) #Заполняем единицами потому что value_count значений 1 больше чем 0
    data.loc[data['DATE_DIED'] == "9999-99-99",'DATE_DIED'] = int(0)
    data.loc[data['DATE_DIED'] != "0",'DATE_DIED'] = int(1)
    
    missing_values = data.isna().sum()

    #Удаляем тк слишком много пропущенных значений, > 50%
    data = data.drop(['PREGNANT','ICU','INTUBED'], axis = 1)

    X = data[all_prizn]
    y = data['CLASIFFICATION_FINAL']

    X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size=0.75, random_state=41) #
    Y_train = Y_train.astype(int)
    y_test = y_test.astype(int)
    #стандартизация данных /  нормализация
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    # Создание модели нейронной сети
    model = MLPClassifier(hidden_layer_sizes=(50,30), max_iter=500)#, random_state=41 //50 ,30
    #Обучение модели на обучающем наборе
    model.fit(X_train_scaled, Y_train)
    #Сохранение модели:
    #joblib.dump(model, 'D:\The Science\PGU\Нейронка\Trained_model.joblib')
    #model = joblib.load('D:\The Science\PGU\Нейронка\Trained_model.joblib')
    
    #Предсказание на тестовом наборе
    y_pred = model.predict(X_test_scaled)
    
    #Оценка точности
    accuracy = accuracy_score(y_test, y_pred)
    cross_val = model.score(X_test_scaled,y_test)
    # #Матрица ошибок (confusion matrix)
    confusion_mat = confusion_matrix(y_test, y_pred)

    # #Отчет о классификации (classification report)
    report = classification_report(y_test, y_pred)

    # #Вывод результатов
    print(f'cross-validation: {cross_val}')
    print("predict: ", y_pred)
    print(f'Accuracy: {accuracy}')
    print(f'Confusion Matrix:\n{confusion_mat}')
    print(f'Classification Report:\n{report}')
#/
#Графики     
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mat, display_labels=model.classes_)
disp.plot()
plt.show() 

par = ["cross-validation", "Accuracy"]
counts = [cross_val, accuracy]
plt.bar(par, counts)
plt.title("accuracy")
plt.xlabel("parametr")
plt.ylabel("Count")
plt.show()  

