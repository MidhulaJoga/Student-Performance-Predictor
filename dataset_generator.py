import pandas as pd
import numpy as np

np.random.seed(42)

data = {
    "attendance": np.random.randint(20, 100, 200),
    "study_hours": np.random.uniform(0.5, 6, 200),
    "previous_marks": np.random.randint(30, 100, 200)
}

df = pd.DataFrame(data)

# Create realistic result condition
df["result"] = np.where(
    (df["attendance"] > 60) & 
    (df["study_hours"] > 2) & 
    (df["previous_marks"] > 50), 1, 0
)

df.to_csv("student_data.csv", index=False)

print("Dataset Created Successfully!")