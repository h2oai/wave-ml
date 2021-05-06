"""
Train on a Wine dataset and predict wine rating.
https://www.kaggle.com/christopheiv/winemagdata130k

Drop columns so just: country, points, price, province, region_1, variety and winery remains.
"""

from typing import List, Optional

import datatable as dt
from h2o_wave import main, app, Q, ui, handle_on, on
from h2o_wave_ml import build_model, get_model

DATASET = './winemag_edit.csv'
FEATURES = ['country', 'price', 'province', 'region_1', 'variety', 'winery']
TARGET_COLUMN = 'points'


def make_load_form(textbox_value: Optional[str] = '') -> List[ui.Component]:
    return [
        ui.textbox(name='project_id', label='Input the project id or the deployment endpoint url', value=textbox_value),
        ui.buttons([
            ui.button(name='train', label='Train new model'),
            ui.button(name='load', label='Use model', primary=True),
        ], justify='end')
    ]


def make_load_form_with_message(type_: str, message: str) -> List[ui.Component]:
    return [
        *make_load_form(),
        ui.message_bar(type=type_, text=message),
    ]


def make_loading() -> List[ui.Component]:
    return [ui.progress(label='Building a model')]


def get_feature_values(q: Q):

    country = q.args.country or q.app.default_value['country']
    price = q.args.price if q.args.price else q.app.default_value['price']
    province = q.args.province or q.app.default_value['province']
    region = q.args.region or q.app.default_value['region_1']
    variety = q.args.variety or q.app.default_value['variety']
    winery = q.args.winery or q.app.default_value['winery']

    return [country, price, province, region, variety, winery]


def show_body(q: Q, rating: int):

    [country, price, province, region, variety, winery] = get_feature_values(q)

    q.page['result'] = ui.tall_gauge_stat_card(
        box=ui.box('body', height='180px'),
        title='',
        value=str(rating),
        aux_value='points',
        plot_color='$red' if rating < 90 else '$green',
        progress=rating/100,
    )
    q.page['wine'] = ui.form_card(box='body', items=[
        ui.dropdown(name='country', label='Country', value=country, trigger=True, choices=q.app.choices['country']),
        ui.dropdown(name='province', label='Province', value=province, trigger=True, choices=q.app.choices['province']),
        ui.dropdown(name='region', label='Region', value=region, trigger=True, choices=q.app.choices['region_1']),
        ui.dropdown(name='variety', label='Variety', value=variety, trigger=True, choices=q.app.choices['variety']),
        ui.dropdown(name='winery', label='Winery', value=winery, trigger=True, choices=q.app.choices['winery']),
        ui.slider(name='price', label='Price in $', min=4, max=150, step=1, value=float(price), trigger=True),
    ])


def predict(q: Q) -> int:
    values = get_feature_values(q)
    input_data = [FEATURES, values]
    rating = q.client.model.predict(input_data)
    return int(rating[0][0])


@on()
async def train(q: Q):
    q.page['model'].items = make_loading()
    await q.page.save()

    model = build_model(DATASET, target_column=TARGET_COLUMN, _dai_time=1, _dai_accuracy=1,
                        _dai_interpretability=10, refresh_token=q.auth.refresh_token)

    q.client.model = model
    rating = predict(q)
    q.page['model'].items = make_load_form(textbox_value=model.endpoint_url)
    show_body(q, rating)


@on()
async def load(q: Q):
    if q.args.project_id.startswith('http'):
        model = get_model(endpoint_url=q.args.project_id)
    else:
        model = get_model(model_id=q.args.project_id, access_token=q.auth.access_token)

    if model is None:
        q.page['model'].items = make_load_form_with_message('error', 'The model does not exists.')
    else:
        q.page['model'].items = make_load_form(textbox_value=model.endpoint_url)

        q.client.model = model
        rating = predict(q)
        show_body(q, rating)


@app('/demo')
async def serve(q: Q):

    if not q.app.initialized:
        df = dt.fread(DATASET)

        # Get a list of unique values of a particular column and prepare choices for dropdown component
        columns = {f: dt.unique(df[f]).to_list()[0] for f in FEATURES}
        q.app.choices = {key: [ui.choice(str(item)) for item in columns[key] if item] for key in columns}

        # Extract a default row
        default_row = df[2, :].to_dict()
        q.app.default_value = {key: cols[0] for key, cols in default_row.items()}

        q.app.initialized = True

    # Initialize page with a layout
    if not q.client.initialized:
        q.page['meta'] = ui.meta_card(box='', layouts=[
            ui.layout(
                breakpoint='xs',
                width='576px',
                zones=[ui.zone('top'), ui.zone('middle'), ui.zone('body')],
            )
        ])
        q.page['header'] = ui.header_card(
            box='top',
            title='Wine rating calculator',
            subtitle='Cheers!',
            icon='Wines',
            icon_color='$red',
        )
        q.page['model'] = ui.form_card(box='middle', items=make_load_form())
        q.client.initialized = True

    if not await handle_on(q):
        if q.client.model:
            rating = predict(q)
            q.page['result'].value = str(rating)
            q.page['result'].progress = rating/100
            q.page['result'].plot_color = '$red' if rating < 90 else '$green'

    await q.page.save()
