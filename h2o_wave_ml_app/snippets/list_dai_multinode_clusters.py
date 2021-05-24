# Import packages

from h2o_wave_ml.utils import list_dai_multinodes

ACCESS_TOKEN = ''
REFRESH_TOKEN = ''

# List of available Driverless AI multinode clusters using access token for Steam
dai_multinodes = list_dai_multinodes(access_token=ACCESS_TOKEN)

# List of available Driverless AI multinode clusters using refresh token for Steam
dai_multinodes = list_dai_multinodes(refresh_token=REFRESH_TOKEN)
