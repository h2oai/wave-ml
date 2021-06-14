# Import packages

from h2o_wave_ml.utils import list_dai_instances

ACCESS_TOKEN = ''  # ACCESS_TOKEN = q.auth.access_token (on H2O AI Hybrid Cloud)
REFRESH_TOKEN = ''  # REFRESH_TOKEN = q.auth.refresh_token (on H2O AI Hybrid Cloud)

# List of available Driverless AI single instances using access token for Steam
dai_instances = list_dai_instances(access_token=ACCESS_TOKEN)

# List of available Driverless AI single instances using refresh token for Steam
dai_instances = list_dai_instances(refresh_token=REFRESH_TOKEN)
