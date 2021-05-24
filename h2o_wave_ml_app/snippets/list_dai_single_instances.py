# Import packages

from h2o_wave_ml.utils import list_dai_instances

ACCESS_TOKEN = ''
REFRESH_TOKEN = ''

# List of available Driverless AI single instances using access token for Steam
dai_instances = list_dai_instances(access_token=ACCESS_TOKEN)

# List of available Driverless AI single instances using refresh token for Steam
dai_instances = list_dai_instances(refresh_token=REFRESH_TOKEN)
