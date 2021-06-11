import logging
import os
import pandas as pd
from pathlib import Path

from h2o_wave import Q, main, app, copy_expando, expando_to_dict, handle_on, on
from h2o_wave_ml import build_model, ModelType
from h2o_wave_ml.utils import list_dai_instances
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from . import cards

logging.basicConfig(format='%(levelname)s:\t[%(asctime)s]\t%(message)s', level=logging.INFO)

STEAM_URL = os.environ.get('STEAM_URL')
MLOPS_URL = os.environ.get('MLOPS_URL')


@app('/')
async def serve(q: Q):
    """
    Serving function.
    """

    try:
        # initialize app
        if not q.app.app_initialized:
            await initialize_app(q)

        # initialize client
        if not q.client.client_initialized:
            await initialize_client(q)
            await setup_home(q)

        # set theme
        elif q.args.theme_dark is not None and q.args.theme_dark != q.client.theme_dark:
            await update_theme(q)

        # update code snippet
        elif q.args.code_function is not None and q.args.code_function != q.client.code_function:
            await update_code_example(q)

        # handle ons
        elif await handle_on(q):
            pass

        # dummy update for edge cases
        else:
            await update_dummy(q)

    except Exception as error:
        await handle_error(q, error=str(error))


async def initialize_app(q: Q):
    """
    Initializing app.
    """

    logging.info('Initializing app')

    q.app.default_theme_dark = True
    q.app.default_code_function = 'Train Model'

    q.app.paths_architecture = {}
    q.app.paths_architecture['dark'], *_ = await q.site.upload([str(Path('static') / 'architecture_dark.png')])
    q.app.paths_architecture['light'], *_ = await q.site.upload([str(Path('static') / 'architecture_light.png')])

    wine_data = load_wine(as_frame=True)['frame']
    q.app.wine_train_df, q.app.wine_test_df = train_test_split(wine_data, train_size=0.7)

    q.app.app_initialized = True


async def initialize_client(q: Q):
    """
    Initializing client.
    """

    logging.info('Initializing client')

    q.client.theme_dark = q.app.default_theme_dark
    q.client.path_architecture = q.app.paths_architecture['dark']
    q.client.code_function = q.app.default_code_function

    q.page['meta'] = cards.meta()
    await q.page.save()
    q.page['meta'].redirect = ''

    q.page['header'] = cards.header()
    q.page['tabs'] = cards.tabs()
    q.page['misc'] = cards.misc(q.client.theme_dark)
    q.page['footer'] = cards.footer()

    q.page['dummy'] = cards.dummy()

    q.client.client_initialized = True

    await q.page.save()


async def update_theme(q: Q):
    """
    Update theme of app.
    """

    copy_expando(q.args, q.client)

    if q.client.theme_dark:
        logging.info('Updating theme to dark mode')

        q.client.path_architecture = q.app.paths_architecture['dark']

        q.page['meta'].theme = 'neon'
        q.page['header'].icon_color = 'black'
    else:
        logging.info('Updating theme to light mode')

        q.client.path_architecture = q.app.paths_architecture['light']

        q.page['meta'].theme = 'light'
        q.page['header'].icon_color = '#CDDD38'

    q.page['misc'].items[3].toggle.value = q.client.theme_dark

    if q.client['#'] == 'home':
        q.page['home'].items[2].text.content = f'''<center>
            <img src="{q.client.path_architecture}" width="540px"></center>'''
    elif q.client['#'] == 'resources':
        q.page['code_examples'] = cards.code_examples(
            code_function=q.client.code_function,
            theme_dark=q.client.theme_dark
        )

    await q.page.save()


@on('#home')
async def setup_home(q: Q):
    """
    Setup home page.
    """

    logging.info('Setting up home page')

    copy_expando(q.args, q.client)

    await drop_cards(q, cards.DROPPABLE_CARDS)

    q.page['home'] = cards.home(path_architecture=q.client.path_architecture)

    await q.page.save()


@on('back_demo')
@on('#demo')
async def setup_demo(q: Q):
    """
    Setup demo page.
    """

    logging.info('Setting up demo page')

    copy_expando(q.args, q.client)

    await drop_cards(q, cards.DROPPABLE_CARDS)

    q.page['demo_h2oaml_local'] = cards.demo_h2oaml_local()
    q.page['demo_h2oaml_cloud'] = cards.demo_h2oaml_cloud()
    q.page['demo_dai_cloud'] = cards.demo_dai_cloud()

    await q.page.save()


@on('demo_h2oaml_local')
async def demo_h2oaml_local(q: Q):
    """
    Demo for H2O AutoML (Local).
    """

    logging.info('Setting up demo for H2O AutoML (Local)')

    await drop_cards(q, ['demo_h2oaml_local', 'demo_h2oaml_cloud', 'demo_dai_cloud'])

    q.page['inputs_h2oaml_local'] = cards.inputs_h2oaml_local()

    await q.page.save()


@on('train_h2oaml_local')
async def train_h2oaml_local(q: Q):
    """
    Train H2O AutoML (Local) model.
    """

    logging.info('Training H2O AutoML (Local) model')

    copy_expando(q.args, q.client)

    wave_model = build_model(
        train_df=q.app.wine_train_df,
        target_column='target',
        model_type=ModelType.H2O3,
        _h2o3_max_runtime_secs=q.client.max_runtime_secs,
        _h2o3_max_models=q.client.max_models
    )

    model_id = wave_model.model.model_id

    preds_test = pd.DataFrame(wave_model.predict(test_df=q.app.wine_test_df))
    accuracy_test = accuracy_score(preds_test.iloc[:, 0].astype(int).values, q.app.wine_test_df.target.values)

    q.page['inputs_h2oaml_local'] = cards.inputs_h2oaml_local(
        max_runtime_secs=q.client.max_runtime_secs,
        max_models=q.client.max_models
    )
    q.page['outputs_h2oaml_local'] = cards.outputs_h2oaml_local(
        model_id=model_id,
        accuracy_test=accuracy_test,
        preds_test=preds_test
    )

    await q.page.save()


@on('demo_dai_cloud')
async def demo_dai_cloud(q: Q):
    """
    Demo for Driverless AI (Cloud).
    """

    logging.info('Setting up demo for Driverless AI (Cloud)')

    await drop_cards(q, ['demo_h2oaml_local', 'demo_h2oaml_cloud', 'demo_dai_cloud'])

    q.client.dai_instances = list_dai_instances(access_token=q.auth.access_token)

    q.page['inputs_dai_cloud'] = cards.inputs_dai_cloud(
        dai_instances=q.client.dai_instances,
        steam_url=STEAM_URL
    )

    await q.page.save()


@on('train_dai_cloud')
async def train_dai_cloud(q: Q):
    """
    Train Driverless AI (Cloud) model.
    """

    logging.info('Training Driverless AI (cloud) model')

    copy_expando(q.args, q.client)

    for dai_instance in q.client.dai_instances:
        if dai_instance['id'] == int(q.client.dai_instance_id):
            q.client.dai_instance_name = dai_instance['name']

    q.page['inputs_dai_cloud'] = cards.inputs_dai_cloud(
        dai_instances=q.client.dai_instances,
        dai_instance_id=q.client.dai_instance_id,
        dai_accuracy=q.client.dai_accuracy,
        dai_time=q.client.dai_time,
        dai_interpretability=q.client.dai_interpretability
    )
    q.page['inputs_dai_cloud'].items[6].buttons.items[0].button.disabled = True

    q.page['outputs_dai_cloud'] = cards.outputs_dai_cloud(
        dai_instance_name=q.client.dai_instance_name,
        dai_instance_id=q.client.dai_instance_id,
        steam_url=STEAM_URL
    )

    await q.page.save()

    wave_model = await q.run(
        build_model,
        train_df=q.app.wine_train_df,
        target_column='target',
        model_type=ModelType.DAI,
        refresh_token=q.auth.refresh_token,
        _steam_dai_instance_name=q.client.dai_instance_name,
        _dai_accuracy=q.client.dai_accuracy,
        _dai_time=q.client.dai_time,
        _dai_interpretability=q.client.dai_interpretability
    )

    mlops_project_id = wave_model.project_id

    preds_test = pd.DataFrame(wave_model.predict(test_df=q.app.wine_test_df))
    accuracy_test = accuracy_score(preds_test.iloc[:, 0].astype(int).values, q.app.wine_test_df.target.values)

    q.client.dai_instances = list_dai_instances(refresh_token=q.auth.refresh_token)

    q.page['inputs_dai_cloud'] = cards.inputs_dai_cloud(
        dai_instances=q.client.dai_instances,
        dai_instance_id=q.client.dai_instance_id,
        dai_accuracy=q.client.dai_accuracy,
        dai_time=q.client.dai_time,
        dai_interpretability=q.client.dai_interpretability
    )

    q.page['outputs_dai_cloud'] = cards.outputs_dai_cloud(
        dai_instance_name=q.client.dai_instance_name,
        dai_instance_id=q.client.dai_instance_id,
        steam_url=STEAM_URL,
        mlops_url=MLOPS_URL,
        mlops_project_id=mlops_project_id,
        accuracy_test=accuracy_test,
        preds_test=preds_test
    )

    await q.page.save()


@on('refresh_dai_instances')
async def refresh_dai_instances(q: Q):
    """
    Refresh DAI instances.
    """

    logging.info('Refreshing DAI instances')

    q.client.dai_instances = list_dai_instances(access_token=q.auth.access_token)

    q.page['inputs_dai_cloud'] = cards.inputs_dai_cloud(dai_instances=q.client.dai_instances)

    await q.page.save()


@on('#resources')
async def setup_resources(q: Q):
    """
    Setup resources page.
    """

    logging.info('Setting up resources page')

    copy_expando(q.args, q.client)

    await drop_cards(q, cards.DROPPABLE_CARDS)

    q.page['code_examples_heading'] = cards.code_examples_heading()
    q.page['code_examples'] = cards.code_examples(
        code_function=q.client.code_function,
        theme_dark=q.client.theme_dark
    )
    q.page['wave_examples_heading'] = cards.wave_examples_heading()
    q.page['h2oaml_examples'] = cards.h2oaml_examples()
    q.page['dai_examples'] = cards.dai_examples()

    await q.page.save()


async def update_code_example(q: Q):
    """
    Update code example.
    """

    logging.info('Updating code snippet')

    copy_expando(q.args, q.client)

    q.page['code_examples'] = cards.code_examples(
        code_function=q.client.code_function,
        theme_dark=q.client.theme_dark
    )

    await q.page.save()


async def drop_cards(q: Q, card_names: list):
    """
    Drop cards from Wave page.
    """

    for card_name in card_names:
        del q.page[card_name]


async def handle_error(q: Q, error: str):
    """
    Handle any app error.
    """

    logging.error(error)

    await drop_cards(q, cards.DROPPABLE_CARDS)

    q.page['error'] = cards.error(
        q_app=expando_to_dict(q.app),
        q_user=expando_to_dict(q.user),
        q_client=expando_to_dict(q.client),
        q_events=expando_to_dict(q.events),
        q_args=expando_to_dict(q.args)
    )

    await q.page.save()


@on('restart')
async def restart(q: Q):
    """
    Restart app.
    """

    q.page['meta'].redirect = '#home'
    q.client.client_initialized = False

    await q.page.save()


@on('report')
async def report(q: Q):
    """
    Report error details.
    """

    q.page['error'].items[4].separator.visible = True
    q.page['error'].items[5].text.visible = True
    q.page['error'].items[6].text_l.visible = True
    q.page['error'].items[7].text.visible = True
    q.page['error'].items[8].text.visible = True
    q.page['error'].items[9].text.visible = True
    q.page['error'].items[10].text.visible = True
    q.page['error'].items[11].text.visible = True
    q.page['error'].items[12].text.visible = True

    await q.page.save()


async def update_dummy(q: Q):
    """
    Dummy update for edge cases.
    """

    q.page['dummy'].items = []

    await q.page.save()
