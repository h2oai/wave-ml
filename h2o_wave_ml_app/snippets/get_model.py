# Import packages

from h2o_wave_ml import build_model, get_model

# Wave Model's endpoint URL
wave_model = build_model(...)
wave_model_endpoint_url = wave_model.endpoint_url

# Get model from endpoint URL
wave_model = get_model(endpoint_url=wave_model_endpoint_url, ...)
