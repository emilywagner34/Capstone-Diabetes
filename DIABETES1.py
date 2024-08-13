import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, RocCurveDisplay, PrecisionRecallDisplay, precision_score, recall_score
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import imblearn as imb
from imblearn.over_sampling import _random_over_sampler


warnings.filterwarnings("ignore")


data = pd.read_csv('C:\\Users\\emwag\\Downloads\\archive (31)\diabetes_012_health_indicators_BRFSS2015.csv')
print(data)

print(data.shape)

print(data.columns)

print(data.info())

print(data.describe())

print(data.isnull().sum())

data_1 = data.rename(columns={'Diabetes_012':'Diabetes_Class'})

print(data_1)

print(data_1.head)


columns = list(data_1.columns)
print(columns)

print("Missing values distribution: ")
print(data_1.isnull().mean())
print("")

print("Column datatypes: ")
print(data_1.dtypes)

data_2 = data_1.astype('int64')

print(data_2)

print("Column datatypes: ")
print(data_2.dtypes)

variable_descriptions = {
    'Diabetes_Class': ['Diabetes status', '0 = no diabetes 1 = pre-diabetes 2 = diabetes'],
    'HighBP': ['High blood pressure?', 'Yes/No'],
    'HighChol': ['High cholesterol? (>240 mg/dL)', 'Yes/No'],
    'CholCheck': ['Checked cholesterol in the past 5 years?', 'Yes/No'],
    'BMI': ['Body mass index', 'Continuous'],
    'Smoker': ['Smoked at least 100 cigarettes? (5 packs)', 'Yes/No'],
    'Stroke': ['Had a stroke or been told so?', 'Yes/No'],
    'HeartDiseaseorAttack': ['Had a coronary heart disease (CHD) or myocardial infarction (MI)?', 'Yes/No'],
    'PhysActivity': ['Done physical activity in past 30 days? (not including job)', 'Yes/No'],
    'Fruits': ['Consumes at least 1 fruit per day', 'Yes/No'],
    'Veggies': ['Consumes at least 1 vegetable per day', 'Yes/No'],
    'HvyAlcoholConsump': ['Heavy drinker?', 'Yes/No'],
    'AnyHealthcare': ['Have any kind of health care coverage', 'Yes/No'],
    'NoDocbcCost': ['Unable to see doctor because of cost in the past year?', 'Yes/No'],
    'GenHlth': ['General health description', 'Excellent/Very good/Good/Fair/Poor'],
    'MentHlth': ['Days with poor mental health in last month', 'Discrete scale: 1-30 days'],
    'PhysHlth': ['Days with poor physical health in last month', 'Discrete scale: 1-30 days'],
    'DiffWalk': ['Difficulty walking or climbing stairs?', 'Yes/No'],
    'Sex': ['Sex', 'Male/Female'],
    'Age': ['13-level age category (Split increments of 5 years)', 'Age groups description'],
    'Education': ['Education level (categorized 1-6)', 'Education levels description'],
    'Income': ['Income scale (1-8)', 'Income levels description'],
}

# Create a DataFrame from the dictionary
df_desc = pd.DataFrame.from_dict(variable_descriptions, orient='index', columns=['Description', 'Responses'])

# Add a count of each variable's occurrences in the dataset
df_desc['Data Length'] = data_2.count

# Styling the DataFrame
df_styled = df_desc.fillna(0).style.format({"Data Length": "{:,.0f}"}).set_properties(**{
    'text-align': 'left',
    'white-space': 'pre-wrap',
}).set_table_styles([
    dict(selector='th', props=[('text-align', 'left')])
])

# Output the styled DataFrame
print(df_desc)

class_count = data_2['Diabetes_Class'].value_counts()
print("Diabetes Class Count: ", class_count)

fig,ax = plt.subplots(figsize=[14,6])
class_count.plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightgreen'], startangle=90, wedgeprops={'edgecolor': 'black'})
plt.title('Class Counts')
plt.ylabel('')
plt.xlabel('')
plt.show()

ros = imb.over_sampling.RandomOverSampler(random_state=42)
data_resampled, target_resampled = ros.fit_resample(data_2, data_2['Diabetes_Class'])


print("Target Resampled")
print(target_resampled)

resampled_counts = pd.Series(target_resampled).value_counts()
print("Counts After Random Over-Sampling:")
print(resampled_counts)


fig1,ax = plt.subplots(figsize=[14,6])
resampled_counts.plot(kind='pie', autopct='%1.1f%%', colors=['skyblue', 'lightgreen'], startangle=90, wedgeprops={'edgecolor': 'black'})
plt.title('Resampled Counts')
plt.ylabel('')
plt.xlabel('')
plt.show()

corr_df=data_resampled.corr()
print(corr_df)

fig2,ax = plt.subplots(figsize=[14,6])
sns.heatmap(corr_df, annot=True, fmt = '.2f', ax=ax)
sns.color_palette("rocket", as_cmap=True)
ax.set_title("Correlation Heatmap", fontsize=12)
plt.show()

Selected_df = data_resampled.drop(['CholCheck', 'AnyHealthcare', 'NoDocbcCost', 'DiffWalk', 'HvyAlcoholConsump', 'MentHlth', 'Fruits', 'Veggies', 'Education', 'Income' ], axis=1)

fig3,ax = plt.subplots(4,2, figsize=(10,16))
sns.distplot(Selected_df.HighBP, bins=20, ax=ax[0,0], color="red")
sns.distplot(Selected_df.HighChol, bins=20, ax=ax[0,1], color="red")
sns.distplot(Selected_df.BMI, bins=20, ax=ax[1,0], color="blue")
sns.distplot(Selected_df.PhysActivity, bins=20, ax=ax[1,1], color="blue")
sns.distplot(Selected_df.Age, bins=20, ax=ax[2,1], color="green")
sns.distplot(Selected_df.HeartDiseaseorAttack, bins=20, ax=ax[2,1], color="green")
sns.distplot(Selected_df.GenHlth, bins=20, ax=ax[3,0], color="#b640d4")
sns.distplot(Selected_df.Diabetes_Class, bins=20, ax=ax[3,1], color="#b640d4")
plt.show()

fig4, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(Selected_df['BMI'])
plt.show()

Q1_BMI = Selected_df['BMI'].quantile(0.25)
Q3_BMI = Selected_df['BMI'].quantile(0.75)
IQR = Q3_BMI - Q1_BMI
lower_bmi = Q1_BMI - 1.5*IQR
upper_bmi = Q3_BMI + 1.5*IQR

print("Lower BMI: ", lower_bmi)
print("Upper BMI: ", upper_bmi)

upper_array_BMI = np.where(Selected_df['BMI'] >= upper_bmi)[0]
lower_array_BMI = np.where(Selected_df['BMI'] <= lower_bmi)[0]

Selected_df.drop(index=upper_array_BMI, inplace=True)
Selected_df.drop(index=lower_array_BMI, inplace=True)

X = Selected_df.iloc[:, 1:]
y = Selected_df.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)


print(X)
print(y)
print(X_train.shape, X_test.shape,   y_train.shape,  y_test.shape)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

classifier_one = KNeighborsClassifier(n_neighbors=5)
classifier_one.fit(X_train, y_train)

y_pred_one = classifier_one.predict(X_test)

print(classification_report(y_test, y_pred_one))

cm_KNN = confusion_matrix(y_test,y_pred_one)

fig5, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(cm_KNN,
            annot=True,
            fmt='g')
plt.ylabel('Actual', fontsize=13)
plt.title('Confusion Matrix', fontsize=17, pad=20)
plt.gca().xaxis.set_label_position('top')
plt.xlabel('Prediction', fontsize=13)
plt.gca().xaxis.tick_top()

plt.gca().figure.subplots_adjust(bottom=0.2)
plt.gca().figure.text(0.5, 0.05, 'Prediction', ha='center', fontsize=13)
plt.show()



classifier_two = RandomForestClassifier(n_estimators=10)
classifier_two.fit(X_train, y_train)

y_pred_two = classifier_two.predict(X_test)

print(classification_report(y_test, y_pred_two))


cm_RF = confusion_matrix(y_test,y_pred_two)
fig6, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(cm_RF,
            annot=True,
            fmt='g')
plt.ylabel('Actual', fontsize=13)
plt.title('Confusion Matrix', fontsize=17, pad=20)
plt.gca().xaxis.set_label_position('top')
plt.xlabel('Prediction', fontsize=13)
plt.gca().xaxis.tick_top()

plt.gca().figure.subplots_adjust(bottom=0.2)
plt.gca().figure.text(0.5, 0.05, 'Prediction', ha='center', fontsize=13)
plt.show()

classifier_three = MLPClassifier(activation='logistic', solver='adam', alpha=0.0001, max_iter=1000, hidden_layer_sizes=(10,))
classifier_three.fit(X_train, y_train)

y_pred_three = classifier_three.predict(X_test)

print(classification_report(y_test, y_pred_three))

cm_MLP = confusion_matrix(y_test,y_pred_three)
fig7, ax = plt.subplots(figsize=(10, 5))
sns.heatmap(cm_MLP,
            annot=True,
            fmt='g')
plt.ylabel('Actual', fontsize=13)
plt.title('Confusion Matrix', fontsize=17, pad=20)
plt.gca().xaxis.set_label_position('top')
plt.xlabel('Prediction', fontsize=13)
plt.gca().xaxis.tick_top()
plt.show()



st.title('Diabetes Classifier')

st.info("The goal of this GUI is to explore the data and display the results of the classification models using the 2015 CDC BRFSS data")
st.caption("HELP")
st.caption("When the dropdown menus are selected the corresponding graph will be displayed")

with st.expander('Data Description'):
    st.caption("Descritption of features")
    chart_data1 = pd.DataFrame(df_desc)
    chart_data1

with st.expander('Dataframe'):
    st.caption("Cleaned, Balanced Data with feature selection")
    st.caption("For security purposes all identifying patient information has been removed")
    chart_data = pd.DataFrame(Selected_df)
    chart_data

with st.expander('Unbalanced Class Data'):
    st.caption("Distribution of classes before over sampling")
    visual = st.pyplot(fig)

with st.expander('Balanced Class Data'):
    st.caption("Distribution of classes after over sampling")
    visual1 = st.pyplot(fig1)

with st.expander('Correlation Matrix'):
    st.caption("Correlation matrix used to determine feature selection")
    visual2 = st.pyplot(fig2)

with st.expander('Density Barcharts'):
    st.caption("Density charts for highly correlated data points")
    visual3 = st.pyplot(fig3)

st.sidebar.subheader("Choose classifier")
st.sidebar.caption("HELP")
st.sidebar.caption("Click the corresponding button to run the classifier and display the precision metrics.")

if st.sidebar.button("Classify KNN", key="classify KNN"):
        classifier_one = KNeighborsClassifier(n_neighbors=5)
        classifier_one.fit(X_train, y_train)
        accuracy = classifier_one.score(X_test, y_test)
        y_pred_two = classifier_one.predict(X_test)

        st.write("Classification Report KNN: ",classification_report(y_test, y_pred_one))
        st.write("Accuracy KNN: ", accuracy_score(y_test, y_pred_one))
        st.write("Precision KNN: ", precision_score(y_test, y_pred_one, average='micro'))
        st.write("Recall KNN: ", recall_score(y_test, y_pred_one,  average='weighted'))
        visual4 = st.pyplot(fig5)

if st.sidebar.button("Classify RF", key="classify RF"):
        classifier_two = RandomForestClassifier(n_estimators=10)
        classifier_two.fit(X_train, y_train)
        accuracy = classifier_two.score(X_test, y_test)
        y_pred_two = classifier_two.predict(X_test)

        st.write("Classification Report Random Forest: ",classification_report(y_test, y_pred_two))
        st.write("Accuracy Random Forest: ", accuracy_score(y_test, y_pred_two))
        st.write("Precision Random Forest: ", precision_score(y_test, y_pred_two, average='micro'))
        st.write("Recall Random Forest: ", recall_score(y_test, y_pred_two,  average='weighted'))
        visual5 = st.pyplot(fig6)

if st.sidebar.button("Classify MLP Classifier", key="classify MLP Classifier"):
        classifier_three = MLPClassifier(activation='logistic', solver='adam', alpha=0.0001, max_iter=1000,
                                     hidden_layer_sizes=(10,))
        classifier_three.fit(X_train, y_train)
        accuracy = classifier_three.score(X_test, y_test)
        y_pred_two = classifier_three.predict(X_test)

        st.write("Classification Report MLP Classifier: ",classification_report(y_test, classifier_three))
        st.write("Accuracy MLP Classifier: ", accuracy_score(y_test, classifier_three))
        st.write("Precision MLP Classifier: ", precision_score(y_test, classifier_three, average='micro'))
        st.write("Recall MLP Classifier: ", recall_score(y_test, classifier_three,  average='weighted'))
        visual6 = st.pyplot(fig7)