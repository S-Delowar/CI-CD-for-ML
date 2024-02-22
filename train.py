# %%
import warnings
warnings.simplefilter('ignore', category=DeprecationWarning)

# %%
import pandas as pd

df = pd.read_csv('Data/drug200.csv')

# %%
df.head(3)

# %%
from sklearn.model_selection import train_test_split

X = df.drop("Drug", axis=1).values
y = df.Drug.values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# %%
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

cat_col = [1,2,3]
num_col = [0,4]

transform = ColumnTransformer(
    [
        ("encoder", OrdinalEncoder(), cat_col),
        ("num_imputer", SimpleImputer(strategy="median"), num_col),
        ("num_scaler", StandardScaler(), num_col),
    ]
)
pipe = Pipeline(
    steps=[
        ("preprocessing", transform),
        ("model", RandomForestClassifier(n_estimators=100, random_state=125)),
    ]
)
pipe.fit(X_train, y_train)

# %%
from sklearn import set_config
set_config(display='diagram')

# %%
pipe

# %% [markdown]
# ### Model Evaluation

# %%
from sklearn.metrics import accuracy_score, f1_score

predictions = pipe.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test,predictions, average='macro')

# %%
print(f'Accuracy: {round(accuracy, 2) * 100} %, F1: {round(f1, 2)}')

# %% [markdown]
# ### Create the metrics file and save into Results folder 

# %%
with open("Results/metrics.txt", "w") as outfile:
    outfile.write(f'\nAccuracy: {round(accuracy, 2)}, F1: {round(f1, 2)}')

# %%
# Confusion Matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

cm = confusion_matrix(y_test, predictions, labels=pipe.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipe.classes_)
disp.plot()
plt.savefig('Results/model_results.png', dpi=120)

# %% [markdown]
# ### Saving the Model

# %%
import skops.io as sio 
sio.dump(pipe, 'Model/drug_pipeline.skops')

# %% [markdown]
# ### Load the Model

# %%
sio.load("Model/drug_pipeline.skops", trusted=True)

# %%



