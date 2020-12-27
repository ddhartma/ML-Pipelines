[image1]: git_images/image1.png "image1"
# ML Pipelines

Let's build up pipelines to automate Machine Learning Workflows on ETL processed data.

These topics will be covered:
1. [Build the Machine Learning Workflow](#Build_the_Machine_Learning_Workflow)
2. [Estimators - Transformers - Predictors](#Estimators_Transformers_Predictors)
3. [Without a Pipeline](#Without_a_Pipeline)
4. [With a Pipeline](#With_a_Pipeline)
5. [Advantages of Using Pipeline](#Advantages_of_Using_Pipeline)
6. [Coding with ML Pipelines](#Coding_with_ML_Pipelines)
7. [Pipelines and Feature Unions](#Pipelines_and_Feature_Unions)
8. [Using Feature Union](#Using_Feature_Union)
9. [Creating Customer Transformer](#Creating_Customer_Transformer)
10. [Pipelines and Grid Search](#Pipelines_and_Grid_Search)

- Scikit-learn Pipeline
- Scikit-learn Feature Union
- Pipelines and Grid Search
- Case Study


## Build the Machine Learning Workflow <a name="Build_the_Machine_Learning_Workflow"></a>
Open notebook ***./ml_notebooks/ml_workflow.ipynb*** for the ml workflow
- The necessary code:
  ```
  import nltk
  nltk.download(['punkt', 'wordnet'])

  import re
  import numpy as np
  import pandas as pd
  from nltk.tokenize import word_tokenize
  from nltk.stem import WordNetLemmatizer
  from sklearn.metrics import confusion_matrix
  from sklearn.model_selection import train_test_split
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

  url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

  def load_data():
      df = pd.read_csv('corporate_messaging.csv', encoding='latin-1')
      df = df[(df["category:confidence"] == 1) & (df['category'] != 'Exclude')]
      X = df.text.values
      y = df.category.values
      return X, y

  def tokenize(text):
      detected_urls = re.findall(url_regex, text)
      for url in detected_urls:
          text = text.replace(url, "urlplaceholder")

      tokens = word_tokenize(text)
      lemmatizer = WordNetLemmatizer()

      clean_tokens = []
      for tok in tokens:
          clean_tok = lemmatizer.lemmatize(tok).lower().strip()
          clean_tokens.append(clean_tok)

      return clean_tokens

  def display_results(y_test, y_pred):
      labels = np.unique(y_pred)
      confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
      accuracy = (y_pred == y_test).mean()

      print("Labels:", labels)
      print("Confusion Matrix:\n", confusion_mat)
      print("Accuracy:", accuracy)

  def main():
      X, y = load_data()
      X_train, X_test, y_train, y_test = train_test_split(X, y)

      vect = CountVectorizer(tokenizer=tokenize)
      tfidf = TfidfTransformer()
      clf = RandomForestClassifier()

      # train classifier
      X_train_counts = vect.fit_transform(X_train)
      X_train_tfidf = tfidf.fit_transform(X_train_counts)
      clf.fit(X_train_tfidf, y_train)

      # predict on test data
      X_test_counts = vect.transform(X_test)
      X_test_tfidf = tfidf.transform(X_test_counts)
      y_pred = clf.predict(X_test_tfidf)

      # display results
      display_results(y_test, y_pred)

  main()
  ```

## Estimators - Transformers - Predictors <a name="Estimators_Transformers_Predictors"></a>
Below, you'll find a simple example of a machine learning workflow where we generate features from text data using count vectorizer and tf-idf transformer, and then fit it to a random forest classifier. Before we get into using pipelines, let's first use this example to go over some scikit-learn terminology.

- ***ESTIMATOR***: An estimator is any object that learns from data, whether it's a classification, regression, or clustering algorithm, or a transformer that extracts or filters useful features from raw data. Since estimators learn from data, they each must have a ***fit method*** that takes a dataset. In the example below, the
  - CountVectorizer,
  - TfidfTransformer and
  - RandomForestClassifier

  are all estimators, and each have a fit method.

- ***TRANSFORMER***: A transformer is a specific type of estimator that has a ***fit method*** to learn from training data, and then a ***transform method*** to apply a transformation model to new data. These transformations can include cleaning, reducing, expanding, or generating features. In the example below,
  - CountVectorizer and
  - TfidfTransformer

  are transformers.

- ***PREDICTOR***: A predictor is a specific type of estimator that has a ***predict method*** to predict on test data based on a supervised learning algorithm, and has a fit method to train the model on training data. The final estimator,   
  - RandomForestClassifier,

  in the example below is a predictor.


## Without a Pipeline <a name="Without_a_Pipeline"></a>
- In machine learning tasks, it's pretty common to have a very specific sequence of transformers to fit to data before applying a final estimator, such as this classifier. And normally, we'd have to initialize all the estimators, fit and transform the training data for each of the transformers, and then fit to the final estimator. Next, we'd have to call transform for each transformer again to the test data, and finally call predict on the final estimator.
  ```
  vect = CountVectorizer()
  tfidf = TfidfTransformer()
  clf = RandomForestClassifier()

  # train classifier
  X_train_counts = vect.fit_transform(X_train)
  X_train_tfidf = tfidf.fit_transform(X_train_counts)
  clf.fit(X_train_tfidf, y_train)

  # predict on test data
  X_test_counts = vect.transform(X_test)
  X_test_tfidf = tfidf.transform(X_test_counts)
  y_pred = clf.predict(X_test_tfidf)
  ```


## With a Pipeline <a name="With_a_Pipeline"></a>
- Fortunately, you can actually automate all of this fitting, transforming, and predicting, by chaining these estimators together into a single estimator object. That single estimator would be scikit-learn's Pipeline. To create this pipeline, we just need a list of (key, value) pairs, where the key is a string containing what you want to name the step, and the value is the estimator object.
  ```
  pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', RandomForestClassifier()),
  ])

  # train classifier
  pipeline.fit(Xtrain)

  # predict on test data
  predicted = pipeline.predict(Xtest)
  ```
- Now, by fitting our pipeline to the training data, we're accomplishing exactly what we would by fitting and transforming each of these steps to our training data one by one. Similarly, when we call predict on our pipeline to our test data, we're accomplishing what we would by calling transform on each of our transformer objects to our test data and then calling predict on our final estimator. Not only does this make our code much shorter and simpler.

- Note that every step of this pipeline has to be a transformer, except for the last step, which can be of an estimator type. Pipeline takes on all the methods of whatever the last estimator in its sequence is. For example, here, since the final estimator of our pipeline is a classifier, the pipeline object can be used as a classifier, taking on the fit and predict methods of its last step. Alternatively, if the last estimator was a transformer, then pipeline would be a transformer.


## Advantages of Using Pipeline <a name="Advantages_of_Using_Pipeline"></a>
1. ***Simplicity and Convencience***
  - Automates repetitive steps - Chaining all of your steps into one estimator allows you to fit and predict on all steps of your sequence automatically with one call. It handles smaller steps for you, so you can focus on implementing higher level changes swiftly and efficiently.
  - Easily understandable workflow - Not only does this make your code more concise, it also makes your workflow much easier to understand and modify. Without Pipeline, your model can easily turn into messy spaghetti code from all the adjustments and experimentation required to improve your model.
  - Reduces mental workload - Because Pipeline automates the intermediate actions required to execute each step, it reduces the mental burden of having to keep track of all your data transformations. Using Pipeline may require some extra work at the beginning of your modeling process, but it prevents a lot of headaches later on.


2. ***Optimizing Entire Workflow***
  - GRID SEARCH: Method that automates the process of testing different hyper parameters to optimize a model.
  - By running grid search on your pipeline, you're able to optimize your entire workflow, including data transformation and modeling steps. This accounts for any interactions among the steps that may affect the final metrics.
  - Without grid search, tuning these parameters can be painfully slow, incomplete, and messy.


3. ***Preventing Data leakage***
  - Using Pipeline, all transformations for data preparation and feature extractions occur within each fold of the cross validation process.
  - This prevents common mistakes where you’d allow your training process to be influenced by your test data - for example, if you used the entire training dataset to normalize or extract features from your data.

## Coding with ML Pipelines <a name="Coding_with_ML_Pipelines"></a>
Open notebook ***./ml_notebooks/pipeline.ipynb*** for working with ML pipeline
  ```
  import nltk
  nltk.download(['punkt', 'wordnet'])

  import re
  import numpy as np
  import pandas as pd
  from nltk.tokenize import word_tokenize
  from nltk.stem import WordNetLemmatizer

  from sklearn.pipeline import Pipeline
  from sklearn.metrics import confusion_matrix
  from sklearn.model_selection import train_test_split
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

  url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

  def load_data():
      df = pd.read_csv('corporate_messaging.csv', encoding='latin-1')
      df = df[(df["category:confidence"] == 1) & (df['category'] != 'Exclude')]
      X = df.text.values
      y = df.category.values
      return X, y

  def tokenize(text):
      detected_urls = re.findall(url_regex, text)
      for url in detected_urls:
          text = text.replace(url, "urlplaceholder")

      tokens = word_tokenize(text)
      lemmatizer = WordNetLemmatizer()

      clean_tokens = []
      for tok in tokens:
          clean_tok = lemmatizer.lemmatize(tok).lower().strip()
          clean_tokens.append(clean_tok)

      return clean_tokens

  def display_results(y_test, y_pred):
      labels = np.unique(y_pred)
      confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
      accuracy = (y_pred == y_test).mean()

      print("Labels:", labels)
      print("Confusion Matrix:\n", confusion_mat)
      print("Accuracy:", accuracy)

  def main():
      X, y = load_data()
      X_train, X_test, y_train, y_test = train_test_split(X, y)

      pipeline = Pipeline([
          ('vect', CountVectorizer(tokenizer=tokenize)),
          ('tfidf', TfidfTransformer()),
          ('clf', RandomForestClassifier())
      ])

      # train classifier
      pipeline.fit(X_train, y_train)

      # predict on test data
      y_pred = pipeline.predict(X_test)

      # display results
      display_results(y_test, y_pred)

  main()

  ```

## Pipelines and Feature Unions <a name="Pipelines_and_Feature_Unions"></a>
- FEATURE UNION: Feature union is a class in scikit-learn’s Pipeline module that allows us to perform steps in parallel and take the union of their results for the next step.
- A pipeline performs a list of steps in a linear sequence, while a FEATURE UNION performs a ***list of steps in parallel*** and then ***combines their results***.
- In more complex workflows, multiple feature unions are often used within pipelines, and multiple pipelines are used within feature unions.
- In the image below one would like to engineer a feature  (extract tfidf) from the dataset but simultaneously wants to engineer another feature (number of characters for each document)
  ![image1]


## Using Feature Union <a name="Using_Feature_Union"></a>
Open notebook ***./ml_notebooks/feature_union.ipynb*** for working with feature unions
- Feature unions are super helpful for handling these situations, where we need to run two steps in parallel on the same data and combine their results to pass into the next step.
- Like pipelines, feature unions are built using a list of (key, value) pairs, where the key is the string that you want to name a step, and the value is the estimator object. Also like pipelines, feature unions combine a list of estimators to become a single estimator. However, a feature union runs its estimators in parallel, rather than in a sequence as a pipeline does. In this example, the estimators run in parallel are nlp_pipeline and text_length. Notice we use a pipeline in this feature union to make sure the count vectorizer and tfidf transformer steps are still running in sequence.
  ```
  from sklearn.pipeline import Pipeline, FeatureUnion

  X = df['text'].values
  y = df['label'].values
  X_train, X_test, y_train, y_test = train_test_split(X, y)

  pipeline = Pipeline([
      ('features', FeatureUnion([

          ('nlp_pipeline', Pipeline([
              ('vect', CountVectorizer()),
              ('tfidf', TfidfTransformer())
          ])),

          ('txt_len', TextLengthExtractor())
      ])),

      ('clf', RandomForestClassifier())
  ])

  # train classifier
  pipeline.fit(Xtrain)

  # predict on test data
  predicted = pipeline.predict(Xtest)

  ```

- Read more about feature unions in Scikit-learn's [user guide](scikit-learn.org/stable/modules/pipeline.html#feature-union).

## Creating Customer Transformer <a name="Creating_Customer_Transformer"></a>
Open notebook ***./ml_notebooks/custom_transformer.ipynb*** for creating customer transformer
- One can implement a custom transformer ***by extending the base class in Scikit-Learn***.
- Let's take a look at a a very simple example that multiplies the input data by ten.
  ```
  import numpy as np
  from sklearn.base import BaseEstimator, TransformerMixin

  class TenMultiplier(BaseEstimator, TransformerMixin):
      def __init__(self):
          pass

      def fit(self, X, y=None):
          return self

      def transform(self, X):
          return X * 10

  ```

- ALL ESTIMATORS have a ***fit method***, and since this is a transformer, it also has ***a transform method***.
  - ***FIT METHOD***: INPUT a ***2d array X*** for the feature data and a ***1d array y*** for the target labels. Inside the fit method, we simply return self. This allows us to chain methods together, since the result on calling fit on the transformer is still the transformer object. This method is required to be compatible with scikit-learn.
  - ***TRANSFORM METHOD***: The function to include the code that transforms the data. Return (here): the data in X multiplied by 10. This transform method also takes a 2d array X.


- Test the transformer via
  ```
  multiplier = TenMultiplier()

  X = np.array([6, 3, 7, 4, 7])
  multiplier.transform(X)

  OUTPUT:
  array([60, 30, 70, 40, 70])
  ```

- Let's build a case normalizer, which simply converts all text to lowercase. We aren't setting anything in our init method, so we can actually remove that. We can leave our fit method as is, and focus on the transform method. We can lowercase all the values in X by applying a lambda function that calls lower on each value. We'll have to wrap this in a pandas Series to be able to use this apply function. With the values attribute we transform it to an numpy array.
  ```
  import numpy as np
  import pandas as pd
  from sklearn.base import BaseEstimator, TransformerMixin

  class CaseNormalizer(BaseEstimator, TransformerMixin):
      def fit(self, X, y=None):
          return self

      def transform(self, X):
          return pd.Series(X).apply(lambda x: x.lower()).values

  case_normalizer = CaseNormalizer()

  X = np.array(['Implementing', 'a', 'Custom', 'Transformer', 'from', 'SCIKIT-LEARN'])
  case_normalizer.transform(X)

  OUTPUT:
  array(['implementing', 'a', 'custom', 'transformer', 'from',
       'scikit-learn'], dtype=object)
  ```

- Another option: Another way to create custom transformers is by usingthe FunctionTransformer ([link1](scikit-learn.org/stable/modules/preprocessing.html#custom-transformers) and [link2](scikit-learn.org/stable/modules/generated/sklearn.preprocessing.FunctionTransformer.html#sklearn.preprocessing.FunctionTransformer)) from scikit-learn's preprocessing module. This allows you to wrap an existing function to become a transformer. This provides less flexibility, but is much simpler.

- Here is a full solution (check also notebook)
  ```
  import nltk
  nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

  import re
  import numpy as np
  import pandas as pd
  from nltk.tokenize import word_tokenize
  from nltk.stem import WordNetLemmatizer

  from sklearn.metrics import confusion_matrix
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import train_test_split
  from sklearn.pipeline import Pipeline, FeatureUnion
  from sklearn.base import BaseEstimator, TransformerMixin
  from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

  url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

  class StartingVerbExtractor(BaseEstimator, TransformerMixin):

      def starting_verb(self, text):
          sentence_list = nltk.sent_tokenize(text)
          for sentence in sentence_list:
              pos_tags = nltk.pos_tag(tokenize(sentence))
              first_word, first_tag = pos_tags[0]
              if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                  return True
          return False

      def fit(self, X, y=None):
          return self

      def transform(self, X):
          X_tagged = pd.Series(X).apply(self.starting_verb)
          return pd.DataFrame(X_tagged)

  def load_data():
      df = pd.read_csv('corporate_messaging.csv', encoding='latin-1')
      df = df[(df["category:confidence"] == 1) & (df['category'] != 'Exclude')]
      X = df.text.values
      y = df.category.values
      return X, y

  def tokenize(text):
      detected_urls = re.findall(url_regex, text)
      for url in detected_urls:
          text = text.replace(url, "urlplaceholder")

      tokens = word_tokenize(text)
      lemmatizer = WordNetLemmatizer()

      clean_tokens = []
      for tok in tokens:
          clean_tok = lemmatizer.lemmatize(tok).lower().strip()
          clean_tokens.append(clean_tok)

      return clean_tokens

  def model_pipeline():
      pipeline = Pipeline([
          ('features', FeatureUnion([

              ('text_pipeline', Pipeline([
                  ('vect', CountVectorizer(tokenizer=tokenize)),
                  ('tfidf', TfidfTransformer())
              ])),

              ('starting_verb', StartingVerbExtractor())
          ])),

          ('clf', RandomForestClassifier())
      ])

      return pipeline

  def display_results(y_test, y_pred):
      labels = np.unique(y_pred)
      confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
      accuracy = (y_pred == y_test).mean()

      print("Labels:", labels)
      print("Confusion Matrix:\n", confusion_mat)
      print("Accuracy:", accuracy)

  def main():
      X, y = load_data()
      X_train, X_test, y_train, y_test = train_test_split(X, y)

      model = model_pipeline()
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)

      display_results(y_test, y_pred)

  main()

  OUTPUT:
  -------
  Labels: ['Action' 'Dialogue' 'Information']
  Confusion Matrix:
   [[ 82   0  19]
   [  3  27   6]
   [  6   1 457]]
  Accuracy: 0.941763727121
  ```

## Pipelines and Grid Search <a name="Pipelines_and_Grid_Search"></a>
Open notebook ***./ml_notebooks/grid_search.ipynb*** for implementing Grid Search with pipelines.
- A powerful benefit of pipelines is the ability to perform a grid search on your entire workflow.
- Most machine learning algorithms have a set of parameters that need tuning. Grid search is a tool that allows you to define a “grid” of parameters, or a set of values to check. Your computer automates the process of trying out all possible combinations of values. Grid search scores each combination with cross validation, and uses the cross validation score to determine the parameters that produce the most optimal model.
- Running grid search on your pipeline allows you to try many parameter values thoroughly and conveniently, for both your data transformations and estimators.
- Although one can also run grid search on just a single classifier, running it on your whole pipeline helps to test multiple parameter combinations across your entire pipeline. This accounts for interactions among parameters not just in the model, but data preparation steps as well.
- All you need to do is create a dictionary of parameters to search, using keys for the names of the parameters and values for the list of parameter values to check. Then, pass the model and parameter grid to the grid search object. Now when you call fit on this grid search object, it will run cross validation on all different combinations of these parameters to find the best combination of parameters for the model.

  ```
  parameters = {
    'kernel': ['linear', 'rbf'],
    'C':[1, 10]
  }

  svc = SVC()
  clf = GridSearchCV(svc, parameters)
  clf.fit(X_train, y_train)
  ```

- Now consider if we had a data preprocessing step, where we standardized the data using StandardScaler like this.
  ```
  scaler = StandardScaler()
  scaled_data = scaler.fit_transform(X_train)

  parameters = {
      'kernel': ['linear', 'rbf'],
      'C':[1, 10]
  }

  svc = SVC()
  clf = GridSearchCV(svc, parameters)
  clf.fit(scaled_data, y_train)
  ```

  This may seem okay at first, but if you standardize your whole training dataset, and then use cross validation in grid search to evaluate your model, you've got data leakage. Why? Grid search uses cross validation to score the model, meaning it splits training data into folds of train and validation sets, trains the model on the train set, and scores it on the validation set, and does this multiple times.

  However, each time, or fold, that this happens, the model already has knowledge of the validation set because all the data was rescaled based on the distribution of the whole training dataset. Important factors like the mean and standard deviation are influenced by the whole dataset. This means the model perform better than it really should on unseen data, since information about the validation set is always baked into the rescaled values of your train dataset.

  The way to fix this, would be to make sure you run standard scaler only on the training set, and not the validation set within each fold of cross validation. Pipelines allow you to do just this.

  ```
  pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC())
  ])

  parameters = {
      'scaler__with_mean': [True, False]
      'clf__kernel': ['linear', 'rbf'],
      'clf__C':[1, 10]
  }

  cv = GridSearchCV(pipeline, param_grid=parameters)

  cv.fit(X_train, y_train)
  y_pred = cv.predict(X_test)
  ```

  Now, since the rescaling is included as part of the pipeline, the standardization doesn't happen until we run grid search. Meaning in each fold of cross validation, the rescaling is done only on the data that the model is trained on, preventing leakage from the validation set.

- Here is a full solution on Grid Search with Pipelines (check also the notebook)
  ```
  import nltk
  nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

  import re
  import numpy as np
  import pandas as pd
  from nltk.tokenize import word_tokenize
  from nltk.stem import WordNetLemmatizer

  from sklearn.metrics import confusion_matrix
  from sklearn.model_selection import GridSearchCV
  from sklearn.ensemble import RandomForestClassifier
  from sklearn.model_selection import train_test_split
  from sklearn.pipeline import Pipeline, FeatureUnion
  from sklearn.base import BaseEstimator, TransformerMixin
  from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

  url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

  class StartingVerbExtractor(BaseEstimator, TransformerMixin):

      def starting_verb(self, text):
          sentence_list = nltk.sent_tokenize(text)
          for sentence in sentence_list:
              pos_tags = nltk.pos_tag(tokenize(sentence))
              first_word, first_tag = pos_tags[0]
              if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                  return True
          return False

      def fit(self, x, y=None):
          return self

      def transform(self, X):
          X_tagged = pd.Series(X).apply(self.starting_verb)
          return pd.DataFrame(X_tagged)

  def load_data():
      df = pd.read_csv('corporate_messaging.csv', encoding='latin-1')
      df = df[(df["category:confidence"] == 1) & (df['category'] != 'Exclude')]
      X = df.text.values
      y = df.category.values
      return X, y

  def tokenize(text):
      detected_urls = re.findall(url_regex, text)
      for url in detected_urls:
          text = text.replace(url, "urlplaceholder")

      tokens = word_tokenize(text)
      lemmatizer = WordNetLemmatizer()

      clean_tokens = []
      for tok in tokens:
          clean_tok = lemmatizer.lemmatize(tok).lower().strip()
          clean_tokens.append(clean_tok)

      return clean_tokens

  def build_model():
      pipeline = Pipeline([
          ('features', FeatureUnion([

              ('text_pipeline', Pipeline([
                  ('vect', CountVectorizer(tokenizer=tokenize)),
                  ('tfidf', TfidfTransformer())
              ])),

              ('starting_verb', StartingVerbExtractor())
          ])),

          ('clf', RandomForestClassifier())
      ])

      parameters = {
          'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
          'features__text_pipeline__vect__max_df': (0.5, 0.75, 1.0),
          'features__text_pipeline__vect__max_features': (None, 5000, 10000),
          'features__text_pipeline__tfidf__use_idf': (True, False),
          'clf__n_estimators': [50, 100, 200],
          'clf__min_samples_split': [2, 3, 4],
          'features__transformer_weights': (
              {'text_pipeline': 1, 'starting_verb': 0.5},
              {'text_pipeline': 0.5, 'starting_verb': 1},
              {'text_pipeline': 0.8, 'starting_verb': 1},
          )
      }

      cv = GridSearchCV(pipeline, param_grid=parameters)

      return cv

  def display_results(cv, y_test, y_pred):
      labels = np.unique(y_pred)
      confusion_mat = confusion_matrix(y_test, y_pred, labels=labels)
      accuracy = (y_pred == y_test).mean()

      print("Labels:", labels)
      print("Confusion Matrix:\n", confusion_mat)
      print("Accuracy:", accuracy)
      print("\nBest Parameters:", cv.best_params_)

  def main():
      X, y = load_data()
      X_train, X_test, y_train, y_test = train_test_split(X, y)

      model = build_model()
      model.fit(X_train, y_train)
      y_pred = model.predict(X_test)

      display_results(model, y_test, y_pred)

  main()

  OUTPUT:
  ------
  Labels: ['Action' 'Dialogue' 'Information']
  Confusion Matrix:
   [[ 92   1  15]
   [  2  32   3]
   [ 14   2 440]]
  Accuracy: 0.9384359401

  Best Parameters: {'features__transformer_weights': {'text_pipeline': 1, 'verb_feature': 0.5}}
  ```
