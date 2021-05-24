# Import packages

from h2o_wave_ml import build_model, save_model

PATH_MODEL = './mymodelpath'

# Wave Model
wave_model = build_model(...)

# Save model to a path
path_model = save_model(model=wave_model, output_dir_path=PATH_MODEL)
