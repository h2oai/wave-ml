from h2o_wave import main, app, Q, ui
from h2o_wave_ml import build_model
from utility import default_value, choices, features

model = build_model(train_file_path='__DATASET__', target_column='__TARGET__')


@app('/')
async def serve(q: Q):

    # Prepare feature values or use default ones
__DEFAULT_VALUES__

    # Prepare input data and do the predictions
__INPUTS__
    score = model.predict(input_data)
    score = score[0][0]

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
            title='__TITLE__',
            subtitle='Cheers!',
            icon='DiagnosticDataBarTooltip',
            icon_color='$red',
        )
__OUTPUT_CARD__
        q.page['wine'] = ui.form_card(box='body', items=[
__FORM_ITEMS__
        ])
        q.client.initialized = True
    else:
__SCORE_UPDATE__

    await q.page.save()
