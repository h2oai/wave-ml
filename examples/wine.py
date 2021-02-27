"""
Train on a Wine dataset and predict wine rating.
https://www.kaggle.com/christopheiv/winemagdata130k

Drop columns so just: country, points, price, province, region_1, variety and winery remains.
"""

import datatable as dt
from h2o_wave import main, app, Q, ui
from h2o_wave_ml import build_model

dataset = './winemag_edit.csv'

model = build_model(dataset, target_column='points')

df = dt.fread(dataset)

# Get a list of unique values of a particular column and prepare choices for dropdown component
features = ['country', 'price', 'province', 'region_1', 'variety', 'winery']
columns = {f: dt.unique(df[f]).to_list()[0] for f in features}
choices = {key: [ui.choice(str(item)) for item in columns[key] if item] for key in columns}


@app('/demo')
async def serve(q: Q):

    # Prepare feature values or use default ones
    country = q.args.country if 'country' in q.args else 'US'
    price = float(q.args.price) if 'price' in q.args else 14.0
    province = q.args.province if 'province' in q.args else 'Oregon'
    region = q.args.region if 'region' in q.args else 'Willamette Valley'
    variety = q.args.variety if 'variety' in q.args else 'Pinot Gris'
    winery = q.args.winery if 'winery' in q.args else 'Rainstorm'

    # Prepare input data and do the predictions
    input_data = [features, [country, price, province, region, variety, winery]]
    rating = model.predict(input_data)
    rating = rating[0][0]

    # Initialize page with a layout
    if not q.client.initialized:
        q.page['meta'] = ui.meta_card(box='', layouts=[
            ui.layout(
                breakpoint='xs',
                width='576px',
                zones=[ui.zone('base')],
            )
        ])
        q.page['header'] = ui.header_card(
            box='base',
            title='Wine rating calculator',
            subtitle='Cheers!',
            icon='Wines',
            icon_color='$red',
        )
        q.page['result'] = ui.tall_gauge_stat_card(
            box=ui.box('base', height='180px'),
            title='',
            value=str(rating),
            aux_value='points',
            plot_color='$red' if rating < 90 else '$green',
            progress=rating/100,
        )
        q.page['wine'] = ui.form_card(box='base', items=[
            ui.dropdown(name='country', label='Country', value=country, trigger=True, choices=choices['country']),
            ui.dropdown(name='province', label='Province', value=province, trigger=True, choices=choices['province']),
            ui.dropdown(name='region', label='Region', value=region, trigger=True, choices=choices['region_1']),
            ui.dropdown(name='variety', label='Variety', value=variety, trigger=True, choices=choices['variety']),
            ui.dropdown(name='winery', label='Winery', value=winery, trigger=True, choices=choices['winery']),
            ui.slider(name='price', label='Price in $', min=4, max=150, step=1, value=price, trigger=True),
        ])
        q.client.initialized = True
    else:
        q.page['result'].value = str(rating)
        q.page['result'].progress = rating/100
        q.page['result'].plot_color = '$red' if rating < 90 else '$green'

    await q.page.save()
