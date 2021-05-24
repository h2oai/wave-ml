import logging

from h2o_wave import Q, main, app, copy_expando, expando_to_dict, handle_on, on

from .utils import Utils

logging.basicConfig(format='%(levelname)s:\t[%(asctime)s]\t%(message)s', level=logging.INFO)
app_utils = Utils()


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

    q.app.app_initialized = True


async def initialize_client(q: Q):
    """
    Initializing client.
    """

    logging.info('Initializing client')

    q.client.theme_dark = q.app.default_theme_dark
    q.client.code_function = q.app.default_code_function

    q.page['meta'] = app_utils.card_meta()
    await q.page.save()
    q.page['meta'].redirect = ''

    q.page['header'] = app_utils.card_header()
    q.page['tabs'] = app_utils.card_tabs()
    q.page['misc'] = app_utils.card_misc(q.client.theme_dark)
    q.page['footer'] = app_utils.card_footer()

    q.page['dummy'] = app_utils.card_dummy()

    q.client.client_initialized = True

    await q.page.save()


async def update_theme(q: Q):
    """
    Update theme of app.
    """

    copy_expando(q.args, q.client)

    if q.client.theme_dark:
        logging.info('Updating theme to dark mode')
        q.page['meta'].theme = 'neon'
        q.page['header'].icon_color = 'black'
    else:
        logging.info('Updating theme to light mode')
        q.page['meta'].theme = 'light'
        q.page['header'].icon_color = '#CDDD38'

    q.page['misc'].items[3].toggle.value = q.client.theme_dark

    if q.client['#'] == 'resources':
        q.page['code_examples'] = app_utils.card_code_examples(
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

    await drop_cards(q, card_names=app_utils.droppable_cards)

    q.page['home'] = app_utils.card_home()

    await q.page.save()


@on('#demo')
async def setup_demo(q: Q):
    """
    Setup demo page.
    """

    logging.info('Setting up resources page')

    copy_expando(q.args, q.client)

    await drop_cards(q, card_names=app_utils.droppable_cards)

    q.page['demo_h2o3'] = app_utils.card_demo_h2o3()
    q.page['demo_dai_cloud'] = app_utils.card_demo_dai_cloud()
    q.page['demo_dai_standalone'] = app_utils.card_demo_dai_standalone()

    await q.page.save()


@on('#resources')
async def setup_resources(q: Q):
    """
    Setup resources page.
    """

    logging.info('Setting up resources page')

    copy_expando(q.args, q.client)

    await drop_cards(q, card_names=app_utils.droppable_cards)

    q.page['code_examples_heading'] = app_utils.card_code_examples_heading()
    q.page['code_examples'] = app_utils.card_code_examples(
        code_function=q.client.code_function,
        theme_dark=q.client.theme_dark
    )
    q.page['wave_examples_heading'] = app_utils.card_wave_examples_heading()
    q.page['h2o3_examples'] = app_utils.card_h2o3_examples()
    q.page['dai_examples'] = app_utils.card_dai_examples()

    await q.page.save()


async def update_code_example(q: Q):
    """
    Update code example.
    """

    logging.info('Updating code snippet')

    copy_expando(q.args, q.client)

    q.page['code_examples'] = app_utils.card_code_examples(
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

    await drop_cards(q, app_utils.droppable_cards)

    q.page['error'] = app_utils.card_error(
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
