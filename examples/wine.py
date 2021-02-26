"""
Train on a Wine dataset and predict wine rating.
https://www.kaggle.com/christopheiv/winemagdata130k

Drop columns so just: country, points, price, province, region_1, variety and winery remains.
"""

from random import choice, randrange

import datatable as dt
from h2o_wave import main, app, Q, ui
from h2o_wave_ml import build_model

dataset = './winemag_edit.csv'

model = build_model(dataset, target_column='points')

df = dt.fread(dataset)

# Get a list of unique choices of particular column
countries = dt.unique(df['country']).to_list()[0]
provinces = dt.unique(df['province']).to_list()[0]
regions = dt.unique(df['region_1']).to_list()[0]
varieties = dt.unique(df['variety']).to_list()[0]
wineries = dt.unique(df['winery']).to_list()[0]

# Make a valid input for dropbox component used later, also filter blank values if any
country_choices = [ui.choice(c, c) for c in countries if c]
province_choices = [ui.choice(p, p) for p in provinces if p]
region_choices = [ui.choice(r, r) for r in regions if r]
variety_choices = [ui.choice(v, v) for v in varieties if v]
winery_choices = [ui.choice(w, w) for w in wineries if w]


@app('/demo')
async def serve(q: Q):

    # Prepare feature values or use default ones
    country = q.args.country if 'country' in q.args else choice(country_choices).name
    price = float(q.args.price) if 'price' in q.args else randrange(4, 150)
    province = q.args.province if 'province' in q.args else choice(province_choices).name
    region = q.args.region if 'region' in q.args else choice(region_choices).name
    variety = q.args.variety if 'variety' in q.args else choice(variety_choices).name
    winery = q.args.winery if 'winery' in q.args else choice(winery_choices).name

    # Prepare input data for prediction
    input_data = [['country', 'price', 'province', 'region_1', 'variety', 'winery'],
                  [country, price, province, region, variety, winery]]
    rating = model.predict(input_data)
    rating = rating[0][0]

    # Initialize page with a layout
    if not q.client.initialized:
        q.page['header'] = ui.header_card(
            box='1 1 3 1',
            title='Wine rating calculator',
            subtitle='Cheers!',
            icon='Wines',
            icon_color='$red',
        )
        q.page['result'] = ui.tall_gauge_stat_card(
            box='1 2 3 2',
            title='',
            value=str(rating),
            aux_value='points',
            plot_color='$red' if rating < 90 else '$green',
            progress=rating/100,
        )
        q.page['wine'] = ui.form_card(box='1 4 3 5', items=[
            ui.dropdown(name='country', label='Country', value=country, trigger=True, choices=country_choices),
            ui.dropdown(name='province', label='Province', value=province, trigger=True, choices=province_choices),
            ui.dropdown(name='region', label='Region', value=region, trigger=True, choices=region_choices),
            ui.dropdown(name='variety', label='Variety', value=variety, trigger=True, choices=variety_choices),
            ui.dropdown(name='winery', label='Winery', value=winery, trigger=True, choices=winery_choices),
            ui.slider(name='price', label='Price in $', min=4, max=150, step=1, value=price, trigger=True),
        ])
        q.client.initialized = True
    else:
        q.page['result'].value = str(rating)
        q.page['result'].progress = rating/100
        q.page['result'].plot_color = '$red' if rating < 90 else '$green'

    await q.page.save()
