import pandas as pd
import regex as re
import numpy as np
from nltk import download
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import pickle
import warnings


def warn(*args, **kwargs):
    pass


class TruffleRevenge:
    data = pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/NLP-project-tutorial/main/url_spam.csv')
    lemmings = WordNetLemmatizer()
    vect = TfidfVectorizer(max_features=5000, max_df=0.8, min_df=5)
    model = SVC(kernel='poly', random_state=42)
    grid = GridSearchCV(param_grid={'C': np.logspace(-2, 2, 5),
                                    'degree': range(5)}, estimator=model, scoring='accuracy')

    def __init__(self):
        download('wordnet')
        download('stopwords')
        self.stop = stopwords.words('english')
        self.pre_process()
        self.watch_my_little_strut_on_the_catwalk()

    def watch_my_little_strut_on_the_catwalk(self):
        self.model.fit(self.xtr, self.ytr)
        pred = self.model.predict(self.xte)
        print(f'Accuracy (Default): {accuracy_score(self.yte, pred)}')
        self.grid.fit(self.xtr, self.ytr)
        print(
            f'Grid Search Model:\n\tBest Parameters: {self.grid.best_params_}\n\tBest Accuracy: {self.grid.best_score_}')
        mod = self.grid.best_estimator_
        print(f'Accuracy (Best Grid): {accuracy_score(self.yte, mod.predict(self.xte))}')
        with open('./models/trufflerevenge.dat', 'wb') as file:
            pickle.dump(mod, file)

    def pre_process(self):
        self.data.drop_duplicates(inplace=True)
        self.data['is_spam'] = self.data['is_spam'].map(lambda x: 1 if x == True else 0)
        self.data['url'] = self.data['url'].apply(self.hooked_on_tokens)
        self.data['url'] = self.data['url'].apply(self.disney_hates_lemmings)
        self.data['url'] = self.the_world_in_black_and_white(self.data['url'])
        self.xtr, self.xte, self.ytr, self.yte = train_test_split(pd.DataFrame(self.data['url']),
                                                                  pd.DataFrame(self.data['is_spam']), test_size=0.2,
                                                                  random_state=42)

    def the_world_in_black_and_white(self, df):
        tokes = [' '.join(tokens) for tokens in df]
        return self.vect.fit_transform(tokes).toarray()

    def hooked_on_tokens(self, txt):
        txt = re.sub(r'[^a-z]', ' ', txt)
        return txt.split()

    def disney_hates_lemmings(self, txt):
        tokens = [self.lemmings.lemmatize(word) for word in txt if word not in self.stop and len(word) > 3]
        return tokens


if __name__ == '__main__':
    warnings.warn = warn
    TruffleRevenge()
