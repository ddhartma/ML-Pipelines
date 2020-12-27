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
Open notebook ***./ml_notebooks/customer_transformer.ipynb*** for creating customer transformer
