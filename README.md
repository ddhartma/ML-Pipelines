# ML Pipelines

Let's build up pipelines to automate Machine Learning Workflows on ETL processed data.

These topics will be covered:
- Advantages of Machine Learning Pipelines
- Scikit-learn Pipeline
- Scikit-learn Feature Union
- Pipelines and Grid Search
- Case Study



## Build the Machine Learning Workflow
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

## Estimators - Transformers - Predictors
Below, you'll find a simple example of a machine learning workflow where we generate features from text data using count vectorizer and tf-idf transformer, and then fit it to a random forest classifier. Before we get into using pipelines, let's first use this example to go over some scikit-learn terminology.

- ***ESTIMATOR***: An estimator is any object that learns from data, whether it's a classification, regression, or clustering algorithm, or a transformer that extracts or filters useful features from raw data. Since estimators learn from data, they each must have a fit method that takes a dataset. In the example below, the
  - CountVectorizer,
  - TfidfTransformer and
  - RandomForestClassifier

  are all estimators, and each have a fit method.

- ***TRANSFORMER***: A transformer is a specific type of estimator that has a fit method to learn from training data, and then a transform method to apply a transformation model to new data. These transformations can include cleaning, reducing, expanding, or generating features. In the example below,
  - CountVectorizer and
  - TfidfTransformer

  are transformers.

- ***PREDICTOR***: A predictor is a specific type of estimator that has a predict method to predict on test data based on a supervised learning algorithm, and has a fit method to train the model on training data. The final estimator,   
  - RandomForestClassifier,

  in the example below is a predictor.


## Without a Pipeline
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


## With a Pipelines
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


## Advantages of Using Pipeline
1. Simplicity and Convencience
  - Automates repetitive steps - Chaining all of your steps into one estimator allows you to fit and predict on all steps of your sequence automatically with one call. It handles smaller steps for you, so you can focus on implementing higher level changes swiftly and efficiently.
  - Easily understandable workflow - Not only does this make your code more concise, it also makes your workflow much easier to understand and modify. Without Pipeline, your model can easily turn into messy spaghetti code from all the adjustments and experimentation required to improve your model.
  - Reduces mental workload - Because Pipeline automates the intermediate actions required to execute each step, it reduces the mental burden of having to keep track of all your data transformations. Using Pipeline may require some extra work at the beginning of your modeling process, but it prevents a lot of headaches later on.


2. Optimizing Entire Workflow
  - GRID SEARCH: Method that automates the process of testing different hyper parameters to optimize a model.
  - By running grid search on your pipeline, you're able to optimize your entire workflow, including data transformation and modeling steps. This accounts for any interactions among the steps that may affect the final metrics.
  - Without grid search, tuning these parameters can be painfully slow, incomplete, and messy.


3. Preventing Data leakage
  - Using Pipeline, all transformations for data preparation and feature extractions occur within each fold of the cross validation process.
  - This prevents common mistakes where you’d allow your training process to be influenced by your test data - for example, if you used the entire training dataset to normalize or extract features from your data.

## Coding with ML Pipelines
Open notebook ***./ml_notebooks/pipeline.ipynb*** for the ml workflow
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

  i
