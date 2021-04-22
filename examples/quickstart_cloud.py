"""
Take a Titanic dataset, train a model and show a confusion matrix based on that model.
"""

from typing import List, Optional

import datatable as dt
from h2o_wave import main, app, Q, ui, handle_on, on
from h2o_wave_ml import build_model, get_model
from sklearn.metrics import confusion_matrix

DATASET = './titanic.csv'
TARGET_COLUMN = 'Survived'

TEMPLATE = '''
#
| | | |
|:-:|:-:|:-:|
| | Survived | Not survived |
| Survived | **{tp}** | {fp} (FP) |
| Not survived | {fn} (FN) | **{tn}** |
<br><br>
'''


def make_header(textbox_value: Optional[str] = '') -> List[ui.Component]:
    return [
        ui.text_xl('Confusion Matrix for Titanic'),
        ui.textbox(name='project_id', label='Input the project id or the deployment endpoint url', value=textbox_value),
        ui.buttons([
            ui.button(name='train', label='Train new model'),
            ui.button(name='load', label='Use model', primary=True),
        ], justify='end')
    ]


def make_matrix(tn: float, fp: float, fn: float, tp: float, model_id: Optional[str]) -> List[ui.Component]:
    return [
        *make_header(model_id),
        ui.text(TEMPLATE.format(tn=tn, fp=fp, fn=fn, tp=tp)),
        ui.slider(name='slider', label='Threshold', min=0, max=1, step=0.01, value=0.5, trigger=True),
    ]


def make_message(type_: str, message: str) -> List[ui.Component]:
    return [
        *make_header(),
        ui.message_bar(type=type_, text=message),
    ]


def make_loading() -> List[ui.Component]:
    return [
        ui.text_xl('Confusion Matrix for Titanic'),
        ui.progress(label='Building a model'),
    ]


def compute_matrix(q: Q):
    threshold = q.args.slider if 'slider' in q.args else 0.5
    y_pred = [p[1] < threshold for p in q.client.prediction]
    tn, fp, fn, tp = confusion_matrix(q.app.y_true, y_pred).ravel()
    return dict(tn=tn, fp=fp, fn=fn, tp=tp)


@on()
async def train(q: Q):
    q.page['matrix'].items = make_loading()
    await q.page.save()

    model = build_model(DATASET, target_column=TARGET_COLUMN, _dai_time=1, _dai_accuracy=1,
                        _dai_interpretability=10, refresh_token=q.auth.refresh_token)

    q.client.prediction = model.predict(file_path=DATASET)
    q.page['matrix'].items = make_matrix(model_id=model.endpoint_url, **compute_matrix(q))


@on()
async def load(q: Q):
    if q.args.project_id.startswith('http'):
        model = get_model(endpoint_url=q.args.project_id)
    else:
        model = get_model(model_id=q.args.project_id, access_token=q.auth.access_token)

    if model is None:
        q.page['matrix'].items = make_message('error', 'The model does not exists.')
    else:
        q.client.prediction = model.predict(file_path=DATASET)
        q.page['matrix'].items = make_matrix(model_id=q.args.project_id, **compute_matrix(q))


@app('/demo')
async def serve(q: Q):

    if not q.app.initialized:
        df = dt.fread(DATASET)
        q.app.y_true = df[TARGET_COLUMN].to_list()[0]
        q.app.initialized = True

    if not q.client.initialized:
        q.page['matrix'] = ui.form_card(box='1 1 3 6', items=make_message('blocked', 'No model loaded yet.'))
        q.client.initialized = True

    if not await handle_on(q):
        if q.client.prediction:
            q.page['matrix'].items[3].text.content = TEMPLATE.format(**compute_matrix(q))
            q.page['matrix'].items[4].slider.value = q.args.slider

    await q.page.save()
