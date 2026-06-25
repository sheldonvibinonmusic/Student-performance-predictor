import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

df=pd.read_csv("student_performance.csv") #importing csv file

#INITIAL train_test_split 

X=df[["Hours"]]
y=df["Marks"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.4,random_state=42)


print("\n")
#DATASET OVERVIEW
print("---------____DATA SET OVERVIEW___---------")
df.info()
print(df.describe())
print("\n")


#VISUALIZING DATASET

import matplotlib.pyplot as plt

plt.scatter(df["Hours"],df["Marks"])
plt.title("Hours vs Marks from data")
plt.xlabel("Hours")
plt.ylabel("Marks")
plt.show()

#MODEL CREATION AND ANALYSIS

model=LinearRegression()

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

print("\n")
print(y_test)
print(y_pred)
print("\n")


#SHOWING mean absoloute error and r2

mae=mean_absolute_error(y_test,y_pred)
print("Mean Absolout error is: ",mae)

r2=r2_score(y_test,y_pred)
print("r2 score is: ",r2)
print("\n")

#USER INPUT FOR PREDICTION

print("\n")
print("---------____USER INPUT___---------")
print("\n")


while(True):
    try:
      n=int(input("how many values to test for: "))

      if n>0:
        break

      else:
        print("Number of test cases cannot be negative") 

    except:
       print("Input cannot be character or string")      

print("\n")
hrs=[]

for i in range (1,n+1):

    while(True):
    
     try:
         hr=float(input("Enter value: "))
         if hr>0:
          hrs.append(hr)
          break

         else:
          print("Hours cannot be negative") 
    
     except:
       print("Input cannot be character or string")     

#creating a dictionary type set up

hr_df=pd.DataFrame({"Hours":hrs})#converting it to data frame
print(hr_df)    

print("---------____MODEL PREDICTION___---------")
print("\n")

prediction_user=model.predict(hr_df)

for i in range(len(hr_df)):
  print("Input hour: ",hr_df["Hours"][i])
  print("Predicted Marks: ",prediction_user[i]) #the reason for zero is that
    #we dont want to print [answer] but just answer


print("---------------------------------------------")
print('''
      ---------____SUMMARY___---------
The model shows a strong positive relationship
between study hours and marks.

MAE is low, indicating accurate predictions.

R² is close to 1, indicating the model explains
most of the variation in marks.
''')  
