import numpy as np
import pandas as pd


df=pd.read_csv("student_performance.csv")

num_cols = df.select_dtypes(include=['int64', 'float64']).columns
for col in num_cols:
    Q1 = df[col].quantile(0.25) #25th percentile
    Q3 = df[col].quantile(0.75) #75th percentile
    IQR = Q3 - Q1 #spread of middle 50% of data is seen
    lower = Q1 - 1.5 * IQR #standard statistical rule
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]

def result_score_calc(row):
    a = 0
    if (row['total_score'] < 75 ):
        a=a+1
    if (row['attendance_percentage'] < 75 ):
        a=a+1
    if(row['weekly_self_study_hours'] < 18):
        a=a+1

    if(a>=2):
        score=0
    else:
        score=1
    return score

df['result'] = df.apply(result_score_calc, axis=1)

X = df.values
print(X.shape)

df = df.drop(columns=["student_id", "class_participation","grade"])

X = df.drop("result", axis=1)
y = df["result"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

model.fit(X,y)

y_pred=model.predict(X_test)
y_pred

df.to_csv("student_10k_dataset.csv", index=False)

print("Dataset generated")