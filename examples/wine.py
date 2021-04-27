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

# Extract a default row
default_row = df[2, :].to_dict()
default_value = {key: cols[0] for key, cols in default_row.items()}


@app('/demo')
async def serve(q: Q):

    # Prepare feature values or use default ones
    country = q.args.country or default_value['country']
    price = q.args.price if q.args.price else default_value['price']
    province = q.args.province or default_value['province']
    region = q.args.region or default_value['region_1']
    variety = q.args.variety or default_value['variety']
    winery = q.args.winery or default_value['winery']

    # Prepare input data and do the predictions
    input_data = [features, [country, price, province, region, variety, winery]]
    rating = model.predict(input_data)
    rating = int(rating[0][0])

    # Initialize page with a layout
    if not q.client.initialized:
        q.page['meta'] = ui.meta_card(box='', layouts=[
            ui.layout(
                breakpoint='xs',
                width='576px',
                zones=[ui.zone('body')],
            )
        ])
        q.page['header'] = ui.header_card(
            box='body',
            title='Wine rating calculator',
            subtitle='Cheers!',
            icon='Wines',
            icon_color='$red',
        )
        q.page['result'] = ui.tall_gauge_stat_card(
            box=ui.box('body', height='180px'),
            title='',
            value=str(rating),
            aux_value='points',
            plot_color='$red' if rating < 90 else '$green',
            progress=rating/100,
        )
        q.page['wine'] = ui.form_card(box='body', items=[
            ui.dropdown(name='country', label='Country', value=country, trigger=True, choices=choices['country']),
            ui.dropdown(name='province', label='Province', value=province, trigger=True, choices=choices['province']),
            ui.dropdown(name='region', label='Region', value=region, trigger=True, choices=choices['region_1']),
            ui.dropdown(name='variety', label='Variety', value=variety, trigger=True, choices=choices['variety']),
            ui.dropdown(name='winery', label='Winery', value=winery, trigger=True, choices=choices['winery']),
            ui.slider(name='price', label='Price in $', min=4, max=150, step=1, value=float(price), trigger=True),
        ])
        q.client.initialized = True
    else:
        q.page['result'].value = str(rating)
        q.page['result'].progress = rating/100
        q.page['result'].plot_color = '$red' if rating < 90 else '$green'

    await q.page.save()
