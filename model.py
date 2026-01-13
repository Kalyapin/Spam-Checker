import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pickle




df = pd.read_csv('spam.csv', encoding='latin-1')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)




def recover_text(row):
    result = row['v2']

    if pd.notnull(row['Unnamed: 2']):
        result += row['Unnamed: 2']

    elif pd.notnull(row['Unnamed: 3']):
        result += row['Unnamed: 3']

    elif pd.notnull(row['Unnamed: 4']):
        result += row['Unnamed: 4']

    return result

df['Text'] = df.apply(recover_text, axis=1)
df = df.drop(['v2', 'Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)


def recover_v1(spam):
    if spam == 'ham':
        return 0
    return 1

df['Is spam'] = df['v1'].apply(recover_v1)
df = df.drop(['v1'], axis=1)
df.info()

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['Text'])
# print(vectorizer.get_feature_names_out())
y = df['Is spam']

#x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

mnb = MultinomialNB(alpha=0.001)
mnb.fit(X, y)
#y_pred = mnb.predict(x_test)
#print('Confusion matrix:', confusion_matrix(y_test, y_pred))
#print('Accuracy score:', accuracy_score(y_test, y_pred))
#print('Precision score:', precision_score(y_test, y_pred))
#print('Recall score:', recall_score(y_test, y_pred))
#print('F1 score:', f1_score(y_test, y_pred))


with open('model.pkl', 'wb') as f:
    pickle.dump(mnb, f)

with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)