# Import packages

import pandas as pd
from h2o_wave_ml import build_model

PATH_TEST = './test.csv'

# Wave Model
wave_model = build_model(...)

# Score model using path
preds = wave_model.predict(file_path=PATH_TEST)

# Score model using pandas
test_data = pd.read_csv(PATH_TEST)
preds = wave_model.predict(test_df=test_data)
