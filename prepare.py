# ===== Module Description:

"""
# Data Preparation Pipeline
This module provides preparation pipeline of data from the covid-19
dataset.
Example:
--------
data_prep = data_preparation(training_data, new_data)
data_prep.prepare()
"""


# ====================
# ===== Imports:

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler,StandardScaler

# ====================

class data_preparation():
  """
  data_preparation class.
  -------
  Attributes
  -------
  - new_dataset : pandas.DataFrame
      Dataframe to prepare

  - training_data_ref : pandas.DataFrame
      Dataframe to prepare by

  - symptoms_per_patient : list
       List of possible features from training_data_ref

  -------
  Protected Methods
  -------
  __seperate_blood_groups(dataset)
  __seperate_symptoms(dataset)
  __seperate_location(dataset)
  __replace_sex(dataset)
  __replace_pcr_date(dataset)
  __normalize()
  -------
  Public Methods
  -------
  prepare()
    Returns the prepared data.

  """
  def __init__(self, training_data, new_data):
    self.new_dataset = new_data.copy()
    self.training_data_ref = training_data.copy()
    self.symptoms_per_patient = [x.split(';') for x in self.training_data_ref["symptoms"] if x is not np.nan]

    self.__seperate_blood_groups(self.training_data_ref)
    self.__seperate_symptoms(self.training_data_ref)
    self.__seperate_location(self.training_data_ref)
    self.__replace_sex(self.training_data_ref)
    self.__replace_pcr_date(self.training_data_ref)


  def __seperate_blood_groups(self, dataset):
  # Groups we seperate by
    blood_groups = {'A': ['A+', 'A-'],
                    'O': ['O-', 'O+'],
                    'B/AB': ['B+', 'B-', 'AB+', 'AB-']}

    for type, blood_group in blood_groups.items():
        # Creating the mew features
        feature = dataset["blood_type"].isin(blood_group).astype(int)
        # Inserting the new features
        dataset['blood_group_' + type] = feature

    # Drop the old blood_type feature
    dataset.drop('blood_type', inplace=True ,axis=1)
    return dataset


  def __seperate_symptoms(self, dataset):
    symptoms = list({x for l in self.symptoms_per_patient for x in l})

    for symptom in symptoms:
      feature = []
      for sample in dataset["symptoms"]:
        if sample is np.nan or symptom not in sample:
          feature.append(0)
        else:
          feature.append(1)

      dataset['symptom_' + symptom] = feature

    # Drop the old symptoms feature   
    dataset.drop('symptoms', inplace=True ,axis=1)
    return dataset


  def __seperate_location(self, dataset):
    # Drop the old current_location feature and adding x_coor and y_coor
    dataset['x_global_coor'] = [float(x.split("'")[1]) for x in dataset['current_location']]
    dataset['y_global_coor'] = [float(x.split("'")[3]) for x in dataset['current_location']]

    dataset.drop(['current_location'], inplace=True, axis=1)
    return dataset


  def __replace_sex(self, dataset):
    # Replace sex with binary is_male & Drop the old current_location
    dataset['sex'] = dataset['sex'].isin(['F']).astype(int)
    return dataset


  def __replace_pcr_date(self, dataset):
    # Drop the old pcr_date feature and replace
    dataset['pcr_date_numeric'] = [int(datetime.strptime(date, '%Y-%m-%d').timestamp()) for date in dataset['pcr_date']]

    dataset.drop(['pcr_date'], inplace=True, axis=1)
    return dataset


  def __normalize(self):
    all_features = list(self.new_dataset.columns)
    all_features.remove('spread')
    all_features.remove('risk')

    mm_features = ['patient_id', 'num_of_siblings', 'happiness_score',
                  'sport_activity', 'pcr_date_numeric', 'PCR_01', 'PCR_02', 'PCR_03',
                  'PCR_04', 'PCR_05', 'PCR_07', 'PCR_09', 'y_global_coor']

    for feature in all_features:
      if feature in mm_features:
        mm_scaler = MinMaxScaler(feature_range=(-1, 1))
        mm_scaler.fit(self.training_data_ref[[feature]])
        self.new_dataset[[feature]] = mm_scaler.transform(self.new_dataset[[feature]])
      else:
        ss_scaler = StandardScaler()
        ss_scaler.fit(self.training_data_ref[[feature]])
        self.new_dataset[[feature]] = ss_scaler.transform(self.new_dataset[[feature]])
    return self.new_dataset


  def prepare(self):
    self.__seperate_blood_groups(self.new_dataset)
    self.__seperate_symptoms(self.new_dataset)
    self.__seperate_location(self.new_dataset)
    self.__replace_sex(self.new_dataset)
    self.__replace_pcr_date(self.new_dataset)
    self.__normalize()

    return self.new_dataset

def preprare_data(training_data, new_data):
  data_prep = data_preparation(training_data, new_data)
  return data_prep.prepare()


if __name__ == '__main__':
  dataset = pd.read_csv("virus_data.csv")
  df = preprare_data(dataset, dataset)
  print(df.head())