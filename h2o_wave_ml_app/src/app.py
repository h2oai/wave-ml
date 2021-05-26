import logging
import pandas as pd
from pathlib import Path

from h2o_wave import Q, main, app, copy_expando, expando_to_dict, handle_on, on
from h2o_wave_ml import build_model, ModelType
from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from . import cards

logging.basicConfig(format='%(levelname)s:\t[%(asctime)s]\t%(message)s', level=logging.INFO)


@app('/')
async def serve(q: Q):
    """
    Serving function.
    """
    print(q.args)
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

        # handle code snippet update
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
            <img src="{q.client.path_architecture}" width="550px"></center>'''
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

    logging.info('Setting up resources page')

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

    nfolds = 3 if q.client.enable_cv else 0

    wave_model = build_model(
        train_df=q.app.wine_train_df,
        target_column='target',
        model_type=ModelType.H2O3,
        categorical_columns=q.client.categorical_columns,
        _h2o3_max_runtime_secs=q.client.max_runtime_secs,
        _h2o3_nfolds=nfolds,
    )

    model_id = wave_model.model.model_id
    accuracy_train = 1 - wave_model.model.mean_per_class_error()
    vimp = pd.DataFrame(wave_model.model.varimp())
    vimp.columns = ['feature', 'class_1', 'class_2', 'class_3']
    vimp['importance'] = (vimp.class_1 + vimp.class_2 + vimp.class_3) / 3

    test_preds = pd.DataFrame(wave_model.predict(test_df=q.app.wine_test_df))
    accuracy_test = accuracy_score(test_preds.iloc[:, 0].astype(int).values, q.app.wine_test_df.target.values)

    q.page['inputs_h2oaml_local'] = cards.inputs_h2oaml_local(
        categorical_columns=q.client.categorical_columns,
        enable_cv=q.client.enable_cv,
        max_runtime_secs=q.client.max_runtime_secs
    )
    q.page['outputs_h2oaml_local'] = cards.outputs_h2oaml_local(
        model_id=model_id,
        accuracy_train=accuracy_train,
        accuracy_test=accuracy_test,
        vimp=vimp
    )

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
        q_args=expando_to_dict(q.args)
    )

    await q.page.save()


async def update_dummy(q: Q):
    """
    Dummy update for edge cases.
    """

    q.page['dummy'].items = []

    await q.page.save()
