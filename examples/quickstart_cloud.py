"""
Take a Titanic dataset, train a model and show a confusion matrix based on that model.
"""

import datatable as dt
from h2o_wave import main, app, Q, ui
from h2o_wave_ml import build_model, get_model
from sklearn.metrics import confusion_matrix

dataset = "/Users/geomodular/Datasets/titanic.csv"
target_column = "Survived"

# Prepare the actual values from target_column
df = dt.fread(dataset)
y_true = df[target_column].to_list()[0]

template = '''
## Confusion Matrix for Titanic
| | | |
|:-:|:-:|:-:|
| | Survived | Not survived |
| Survived | **{tp}** | {fp} (FP) |
| Not survived | {fn} (FN) | **{tn}** |
<br><br>
'''

@app('/demo')
async def serve(q: Q):

    threshold = q.args.slider if 'slider' in q.args else 0.5

    if q.args.build:
        q.page['matrix'] = ui.form_card(box='1 1 3 5', items=[
            ui.progress(label='Building a model')
        ])
        await q.page.save()

        model = build_model(dataset, target_column=target_column, _dai_time=1, _dai_accuracy=1,
                            _dai_interpretability=10, refresh_token=q.auth.refresh_token)
        q.app.prediction = model.predict(file_path=dataset)
        q.page['matrix'] = ui.form_card(box='1 1 3 5', items=[
            ui.message_bar(type='success', text='Model successfully built!'),
            ui.text(model.endpoint_url),
            ui.button(name='predict', label='Next', primary=True),
        ])

    elif q.args.predict:
        if q.args.project_id:
            if q.args.project_id.startswith('http'):
                model = get_model(endpoint_url=q.args.project_id)
            else:
                model = get_model(model_id=q.args.project_id, access_token=q.auth.access_token)
            q.app.prediction = model.predict(file_path=dataset)

        y_pred = [p[1] < threshold for p in q.app.prediction]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        q.page['matrix'] = ui.form_card(box='1 1 3 5', items=[
            ui.text(template.format(tn=tn, fp=fp, fn=fn, tp=tp)),
            ui.slider(name='slider', label='Threshold', min=0, max=1, step=0.01, value=0.5, trigger=True),
            ui.button(name='get', label='Change model'),
        ])

    if q.args.get or q.app.prediction is None:
        q.page['matrix'] = ui.form_card(box='1 1 3 5', items=[
            ui.text_xl('Confusion Matrix for Titanic'),
            ui.textbox(name='project_id', label='Input the project id or the deployment endpoint url'),
            ui.buttons([
                ui.button(name='build', label='Build model'),
                ui.button(name='predict', label='Next', primary=True),
            ], justify='end')
        ])

    elif 'slider' in q.args:
        y_pred = [p[1] < threshold for p in q.app.prediction]
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        q.page['matrix'].items[0].text.content = template.format(tn=tn, fp=fp, fn=fn, tp=tp)
        q.page['matrix'].items[1].slider.value = threshold

    await q.page.save()
