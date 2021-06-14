# Import packages

import pandas as pd
from h2o_wave_ml import build_model

PATH_TRAIN = './train.csv'

# Train model using path
wave_model = build_model(train_file_path=PATH_TRAIN, target_column='target', ...)

# Train model using pandas
train_data = pd.read_csv(PATH_TRAIN)
wave_model = build_model(train_df=train_data, target_column='target', ...)
