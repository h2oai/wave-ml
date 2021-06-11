# Import packages

from h2o_wave_ml import build_model
from h2o_wave_ml.utils import save_autodoc

ACCESS_TOKEN = ''  # ACCESS_TOKEN = q.auth.access_token (on H2O AI Hybrid Cloud)
REFRESH_TOKEN = ''  # REFRESH_TOKEN = q.auth.refresh_token (on H2O AI Hybrid Cloud)
PATH_AUTODOC = './mymodelautodoc'

# Wave Model
wave_model = build_model(...)

# Save AutoDoc using access token for Steam
path_autodoc = save_autodoc(project_id=wave_model.project_id, output_dir_path=PATH_AUTODOC, access_token=ACCESS_TOKEN)

# Save AutoDoc using refresh token for Steam
path_autodoc = save_autodoc(project_id=wave_model.project_id, output_dir_path=PATH_AUTODOC, refresh_token=REFRESH_TOKEN)
