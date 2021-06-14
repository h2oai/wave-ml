# Import packages

from h2o_wave_ml import build_model, save_model, load_model

PATH_MODEL = './mymodelpath'

# Wave Model
wave_model = build_model(...)

# Save model to a local path
path_model = save_model(model=wave_model, output_dir_path=PATH_MODEL)

# Load model from a local path
wave_model = load_model(file_path=path_model)
