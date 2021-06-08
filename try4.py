import tensorflow as tf
import tensorflow.compat.v1 as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import json
import os
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from tensorflow.train import SequenceExample, FeatureLists
from tensorflow import feature_column
from tensorflow.keras import layers




csv_file = 'train_anxiety.csv'
csv_data = pd.read_csv(csv_file, low_memory = False)
csv_df = pd.DataFrame(csv_data)

test_file = 'test_anxiety.csv'
# test_file = 'test_positive0.csv'
test_data = pd.read_csv(test_file, low_memory = False)
# test_df = pd.DataFrame(test_data)


CONTI_FEATURES = ['Age']
CATE_FEATURES = ['Gender', 'ICD9Code', 'Smoking']


# create the feature column:
# continuous_features = [tf.feature_column.numeric_column(k) for k in CONTI_FEATURES]
Age_Fea = tf.feature_column.numeric_column('Age')
Gender_Fea = tf.feature_column.categorical_column_with_hash_bucket('Gender', hash_bucket_size = 2)
ICD9Code_Fea = tf.feature_column.categorical_column_with_hash_bucket('ICD9Code', hash_bucket_size = 1000)
Smoking_Fea = tf.feature_column.categorical_column_with_hash_bucket('Smoking', hash_bucket_size = 2)
categorical_features = [Gender_Fea, ICD9Code_Fea, Smoking_Fea]
# categorical_features = [tf.feature_column.categorical_column_with_hash_bucket(k, hash_bucket_size = 1000) for k in CATE_FEATURES]

Age_buck = tf.feature_column.bucketized_column(source_column = Age_Fea, boundaries = [20, 30, 40, 50, 60, 70, 80, 90,100])

continuous_features = [tf.feature_column.embedding_column(categorical_column = Age_buck, dimension= 2)]
gender_embedding = tf.feature_column.embedding_column(categorical_column = Gender_Fea, dimension= 2)
smoking_embedding = tf.feature_column.embedding_column(categorical_column=Smoking_Fea, dimension= 2)
ICD9Code_embedding = tf.feature_column.embedding_column(categorical_column=ICD9Code_Fea, dimension= 6)
categorical_features = [gender_embedding, smoking_embedding, ICD9Code_embedding]


FEATURES = ['Age', 'Gender', 'ICD9Code', 'Smoking']
LABEL = 'Condition'


# input function:
def get_input_fn(data_set, n_batch, num_epochs, shuffle):
    input = tf.compat.v1.estimator.inputs.pandas_input_fn(
       x = pd.DataFrame({k: data_set[k].values for k in FEATURES}),
       y = pd.Series(data_set[LABEL].values),
       batch_size = n_batch,
       num_epochs = num_epochs,
       shuffle = shuffle
     )

    return input

model = tf.estimator.LinearClassifier(
  n_classes = 2,
  model_dir = "ongoing/train5 ",
  feature_columns = categorical_features + continuous_features
)

# model = tf.estimator.DNNClassifier(
#   model_dir = 'dnn_1',
#   feature_columns = categorical_features + continuous_features,
#   hidden_units = [1024, 520],
#   dropout = 0.1,
#   # optimizer=tf.train.AdamOptimizer(1e-4),
#
#   # optimizer=tf.compat.v1.train.ProximalAdagradOptimizer(
#   #     learning_rate=0.1,
#   #     l1_regularization_strength=0.001
#   #   ),
#   optimizer = tf.train.AdamOptimizer(learning_rate = 0.1),
#   # config=tf.estimator.RunConfig().replace(save_summary_steps=10)
# )

# model1.train(
#   input_fn = get_input_fn(csv_data, 5000, 10, True),
#   steps = 1000
# )


# train the model
model.train(
  input_fn = get_input_fn(csv_data, 5000, 10, True),
  steps = 500
)

def predict_input_fn(dataset):
    input = tf.compat.v1.estimator.inputs.pandas_input_fn(
    # predict_input_fn = tf.estimator.inputs.numpy_input_fn(

                          # x = {k: dataset[k].values for k in FEATURES},
                          x = pd.DataFrame({k: dataset[k].values for k in FEATURES}),
                          y = None,
                          batch_size = 1,
                          num_epochs = 1,
                          shuffle = True,
                          num_threads = 1
                       )
    return input


# iterate every data in test dataset and make a prediction:
row_pre = 0
true_positive = []
true_negative = []
false_positive = []
false_negative = []
list = []



# for i in test_data.loc[:,'PatientGuid']:
for i in range(1500):

    dict = {'Age': test_data.loc[i]['Age'],
            'Gender': test_data.loc[i]['Gender'],
            'ICD9Code': test_data.loc[i]['ICD9Code'],
            'Smoking': test_data.loc[i]['Smoking']
    }
    df = pd.DataFrame(dict, index = [1,2,3,4])



    predict_results = model.predict(predict_input_fn(df))

    proba_positive = next(predict_results)['probabilities'][1]
    proba_negative = next(predict_results)['probabilities'][0]
    #
    # if proba_positive > proba_negative:
    #     print('positive', proba_positive)
    # else:
    #     print('negative', proba_negative)

    if proba_positive > proba_negative:

        if test_data.loc[i]['Condition'] == 1:
            true_positive.append(1)
        else:
            false_positive.append(1)
    else:
        if test_data.loc[i]['Condition'] == 0:
            true_negative.append(0)
        else:
            false_negative.append(0)





print('this is true positive: ', len(true_positive))
print('this is false positive: ', len(false_positive))
print('this is true negative', len(true_negative) )
print('this is false negative', len(false_negative) )

evaluate = model.evaluate(input_fn = get_input_fn(test_data, n_batch = 4978, num_epochs = 1, shuffle = False))
print(evaluate)
