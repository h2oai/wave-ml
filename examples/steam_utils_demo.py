from h2o_wave import ui, app, Q, main, on, handle_on
from h2o_wave_ml_utils.wave_steam_utils.steam_utils import init_steam_lib, steam_menu, clear_steam_cards


# Initialize app
def init_app(q: Q):
    q.page['meta'] = ui.meta_card(box='', layouts=[
        ui.layout(
            breakpoint='m',
            zones=[
                ui.zone('header', direction=ui.ZoneDirection.ROW),
                ui.zone('main', direction=ui.ZoneDirection.COLUMN),
                ui.zone('footer'),
            ]
        ),
        ui.layout(
            breakpoint='xl',
            width='1700px',
            zones=[
                ui.zone('header', direction=ui.ZoneDirection.ROW),
                ui.zone('main', direction=ui.ZoneDirection.COLUMN),
                ui.zone('footer'),
            ]
        )
    ])
    # Header for app
    q.page['header'] = ui.header_card(box='header', title='Steam Utils', subtitle='Demo of steam utils')
    q.page['footer'] = ui.footer_card(box='footer', caption='(c) 2021 H2O.ai. All rights reserved.')



@on('#steam')
async def steam_view(q: Q):
    q.app.steam_box_name = 'main'
    await steam_menu(q)


@on('dai_engine_connected')
async def dai_connected(q: Q):
    q.page['meta'].dialog = None
    clear_steam_cards(q)
    await q.page.save()
    # h2oai Driverless AI client object to use downstream
    h2oai = q.user.h2oai
    selected_engine = q.client.selected_steam_engine
    # DAI engine selected by user. Of type DAISteamEngine
    dai_engine = q.user.user.dai_engines[selected_engine]['engine']
    dai_url = dai_engine.openid_login_url

    ## User code after this
    q.page['main'] = ui.form_card(
        box=q.app.steam_box_name,
        items=[
            ui.text_xl('DAI Connection'),
            ui.text(f'DAI Name: {dai_engine.name}'),
            ui.text(f'DAI Status: {dai_engine.status}'),
            ui.link(
                label='Go to Driverless AI', path=dai_url, target='_blank', button=True
            ),
        ]
    )

@on('#home')
async def main_menu(q: Q):
    clear_steam_cards(q)
    q.page['menu'] = ui.tab_card(
        box=ui.box('main', height='50px'),
        items=[
            ui.tab(name='#home', label='Home', icon='Home'),
            ui.tab(name='#steam', label='Steam', icon='Table'),

        ])
    q.page['main'] = ui.form_card(
        box='main',
        items=[
            ui.text_xl('Steam Utils Demo'),
            ui.text('Use Tabs to navigate to Steam')
        ]

    )


def clean_cards(q):
    cards_to_clean = ['main']
    for card in cards_to_clean:
        del q.page[card]


# Main loop
@app('/')
async def serve(q: Q):
    init_app(q)
    if not q.app.initialized:
        init_steam_lib(q)
        q.app.initialized = True

    clean_cards(q)

    LOCAL_TEST = True
    if LOCAL_TEST:
        q.app.env_vars['DEV_STEAM_URL'] = 'https://steam.demo.h2o.ai/'
        q.app.env_vars['STEAM_API_TOKEN'] = '<steam API key here>'
        q.auth.username = '<user email>'
        q.auth.subject = '<user name>'

    if not await handle_on(q):
        await main_menu(q)

    await q.page.save()
