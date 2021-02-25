"""
Train on a Wine dataset and predict wine rating.
https://www.kaggle.com/christopheiv/winemagdata130k
"""

from random import choice, randrange

import datatable as dt
from h2o_wave import main, app, Q, ui
from h2o_wave_ml import build_model, save_model, load_model

dataset = '/Users/geomodular/Datasets/winemag_edit.csv'
target_column = 'points'

# model = build_model(dataset, target_column=target_column)
# save_model(model, './')
model = load_model('./XGBoost_1_AutoML_20210225_134946')

df = dt.fread(dataset)

countries = dt.unique(df['country']).to_list()[0]
provinces = dt.unique(df['province']).to_list()[0]
regions = dt.unique(df['region_1']).to_list()[0]
variety_ = dt.unique(df['variety']).to_list()[0]
wineries = dt.unique(df['winery']).to_list()[0]

country_choices = [ui.choice(c, c) for c in countries if c]
province_choices = [ui.choice(p, p) for p in provinces if p]
region_choices = [ui.choice(r, r) for r in regions if r]
variety_choices = [ui.choice(v, v) for v in variety_ if v]
winery_choices = [ui.choice(w, w) for w in wineries if w]

template = '''
# <center>{rating} points</center>
| | |
| --- | ---  |
| Country | {country} |
| Province | {province} |
| Region | {region} |
| Variety | {variety} |
| Winery | {winery} |
| Price | ${price} |
'''


@app('/demo')
async def serve(q: Q):

    if not q.client.initialized:
        q.page['header'] = ui.header_card(
            box='1 1 3 1',
            title='Wine rating calculator',
            subtitle='Cheers!',
            icon='Wines',
            icon_color='$red',
        )
        q.client.initialized = True

    if q.args.rate:
        country = q.args.country
        price = q.args.price
        province = q.args.province
        region = q.args.region
        variety = q.args.variety
        winery = q.args.winery

        input_data = [['country', 'price', 'province', 'region_1', 'variety', 'winery'],
                      [country, float(price), province, region, variety, winery]]
        rating = model.predict(input_data)
        rating = rating[0][0]

        q.page['wine'] = ui.form_card(box='1 2 3 6', items=[
            ui.text(template.format(rating=rating, country=country, price=price, province=province, region=region,
                                    variety=variety, winery=winery)),
            ui.buttons([
                ui.button(name='next', label='Next wine please', primary=True),
            ], justify='end')
        ])
    else:
        q.page['wine'] = ui.form_card(box='1 2 3 6', items=[
            ui.dropdown(name='country', label='Country', value=choice(country_choices).name, choices=country_choices),
            ui.dropdown(name='province', label='Province', value=choice(province_choices).name, choices=province_choices),
            ui.dropdown(name='region', label='Region', value=choice(region_choices).name, choices=region_choices),
            ui.dropdown(name='variety', label='Variety', value=choice(variety_choices).name, choices=variety_choices),
            ui.dropdown(name='winery', label='Winery', value=choice(winery_choices).name, choices=winery_choices),
            ui.textbox(name='price', label='Price', value=str(randrange(1, 75)) + '.00', prefix='$'),
            ui.buttons([
                ui.button(name='rate', label='Rate', primary=True),
            ], justify='end')
        ])

    await q.page.save()
