# numerical data entry from raman spectroscopy/ FTIR resulting in detection of microplastic or not

import numpy
import pandas
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# creating dummy FTIR data for scanning purposes
# Example: Peak near 2915 cm-1 (C-H stretch), 1730 cm-1 (C=O stretch), 1450 cm-1 (C-H bending)...
# Labels: 1 = Plastic, 0 = Non-Plastic

data = [
    [0.9, 0.1, 0.7, 1],   # Plastic data
    [0.85, 0.15, 0.65, 1],
    [0.88, 0.12, 0.72, 1],
    [0.2, 0.8, 0.1, 0],   # Non-Plastic data
    [0.25, 0.75, 0.12, 0],
    [0.18, 0.82, 0.09, 0]
]
df = pandas.DataFrame(data, columns=["Peak_2915", "Peak_1730", "Peak_1450", "Label"])

x = df[["Peak_2915", "Peak_1730", "Peak_1450"]]
y = df["Label"]
# the dataset is divided for both training and testing.
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = RandomForestClassifier()
model.fit(x_train, y_train)

print("Model has been trained successfully")

# taking one data from the user which is the spectroscope.
print("\nEnter FTIR intensity values for 3 peaks:")
p1 = float(input("Intensity at 2915 cm-1: "))
p2 = float(input("Intensity at 1730 cm-1: "))
p3 = float(input("Intensity at 1450 cm-1: "))

# convert the data came from spectroscope to setting up an array
test_input = numpy.array([p1, p2, p3]).reshape(1, -1)  # Reshape to 2D array
# prediction using trained model to guess or tell whether it is a plastic or not
prediction = model.predict(test_input)

# final solution that it is microplastic or not
if prediction[0] == 1:
    print("The sample is predicted as Plastic")
else:
    print("The sample is predicted as Non-Plastic")