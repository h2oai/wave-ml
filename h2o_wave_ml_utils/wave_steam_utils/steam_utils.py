from h2o_wave import Q, on, ui
from loguru import logger

from .steam_dashboard_ui import *
from .steam_libs import *

"""
from .steam_dashboard_ui import (
    get_ai_engine_dashboard,
    get_ai_engine_dashboard_header,
    get_ai_engine_dashboard_header_items,
    get_dai_engine_details_card,
    get_h2o3_engine_details_card,
    get_new_dai_steam_engine_options,
    get_new_h2o3_steam_engine_options,
    get_wait_for_h2o3_terminate_dialog,
    get_wait_for_restart_dialog,
    get_wait_for_steam_dialog,
    get_wait_for_stop_dialog,
    get_wait_for_terminate_dialog,
    get_wait_for_connect_dialog,
    make_new_dai_steam_engine_dialog,
    make_new_h2o3_steam_engine_dialog,
    wait_for_steam_refresh_dialog,
)
from .steam_libs import (
    H2O3SteamEngine,
    DAIEngineConfig,
    DAISteamEngine,
    H2oEngineConfig,
    get_steam_engines,
    validate_steam_engine_name,
    make_user_steam_connection
)
"""

def get_progress_dialog(title):
    return ui.dialog(title=title, items=[ui.progress(label='', caption='')])


@on()
async def manage_ai_engines(q: Q):
    # Refresh Steam Connection
    q.page['meta'].dialog = get_progress_dialog(title='Refreshing Steam Connection')
    await q.page.save()
    make_user_steam_connection(q)
    q.page['meta'].dialog = None


    q.page['engine_dashboard_header'] = get_ai_engine_dashboard_header(q)
    q.page['engine_dashboard'] = get_ai_engine_dashboard(q)

    #q.page['main_nav'].value = 'manage_ai_engines'

    q.client.selected_steam_engine = None
    await q.page.save()


@on()
async def refresh_steam_engine_table(q: Q):
    q.page['meta'].dialog = wait_for_steam_refresh_dialog()
    await q.page.save()
    connected = await get_steam_engines(q)
    if not connected:
        logger.error('There was an error establishing connection to STEAM')
    q.page['engine_dashboard'] = get_ai_engine_dashboard(q)
    q.page['meta'].dialog = None
    await q.page.save()


async def create_new_dai_steam_engine(q: Q):
    q.client.selected_steam_engine = q.args.dai_steam_instance_name
    q.page['engine_dashboard'] = get_dai_engine_details_card(
        q, q.args.dai_steam_instance_name
    )
    q.page['engine_dashboard_header'].items = get_ai_engine_dashboard_header_items(
        q,
        is_dai=True,
        status=q.user.user.dai_engines[q.args.dai_steam_instance_name]['engine'].status,
    )
    await q.page.save()


@on()
async def new_dai_steam_engine_dialog(q: Q):
    q.app.callbacks['new_dai_steam_engine'] = create_new_dai_steam_engine
    await make_new_dai_steam_engine_dialog(q)


@on()
async def launch_new_dai_steam_engine(q: Q):
    dai_engine_config = DAIEngineConfig(  # type: ignore
        name=q.args.dai_steam_instance_name,
        version=q.args.dai_steam_instance_version,
        cpu_count=q.args.dai_steam_instance_cpu_count,
        gpu_count=q.args.dai_steam_instance_gpu_count,
        memory_gb=q.args.dai_steam_instance_mem,
        storage_gb=q.args.dai_steam_instance_storage,
        max_idle_h=q.args.dai_steam_instance_idle_time,
        max_uptime_h=q.args.dai_steam_instance_uptime,
        timeout_s=q.args.dai_steam_instance_timeout * 60,
        sync=False,
    )
    q.user.new_dai_instance_running_timeout_min = (
        q.args.new_dai_instance_running_timeout_min
    )
    error_msg = validate_steam_engine_name(
        name=q.args.dai_steam_instance_name, existing_engines=q.user.user.dai_engines
    )
    if error_msg:
        q.page['meta'].dialog = get_new_dai_steam_engine_options(
            q,
            dai_engine_config=dai_engine_config,
            error_msg=error_msg,
        )
        await q.page.save()
        return

    q.page['meta'].dialog = get_wait_for_steam_dialog(q)
    await q.page.save()

    new_dai_engine = DAISteamEngine(
        steam_connection=q.user.steam_connection, config=dai_engine_config
    )
    is_running = False
    if new_dai_engine is not None:
        logger.debug('new_dai_engine is not None. Waiting for "running" status.')
        is_running = new_dai_engine._wait(
            target_status='running',
            timeout=q.user.new_dai_instance_running_timeout_min * 60,
        )

    connected = await get_steam_engines(q)
    if not connected:
        logger.error('There was an error establishing connection to STEAM')

    if is_running and q.args.dai_steam_instance_name in q.user.user.dai_engines:
        q.page['meta'].dialog = get_wait_for_steam_dialog(q, done=True)
        if (
            isinstance(q.app.callbacks, dict)
            and 'new_dai_steam_engine' in q.app.callbacks
        ):
            callback_func = q.app.callbacks['new_dai_steam_engine']
            del q.app.callbacks['new_dai_steam_engine']
            if 'new_dai_steam_engine_cancel' in q.app.callbacks:
                del q.app.callbacks['new_dai_steam_engine_cancel']
        else:
            callback_func = None
    else:
        if new_dai_engine is None:
            error_msg = (
                'Error occurred while trying to create a Driverless AI Steam Engine!'
            )
        else:
            error_msg = f'Newly created Steam Engine did not get to "running" status in {q.user.new_dai_instance_running_timeout_min * 60} seconds!'

        q.page['meta'].dialog = get_wait_for_steam_dialog(
            q, done=True, error_msg=error_msg
        )
        if (
            isinstance(q.app.callbacks, dict)
            and 'new_dai_steam_engine_cancel' in q.app.callbacks
        ):
            callback_func = q.app.callbacks['new_dai_steam_engine_cancel']
            del q.app.callbacks['new_dai_steam_engine_cancel']
            if 'new_dai_steam_engine' in q.app.callbacks:
                del q.app.callbacks['new_dai_steam_engine']
        else:
            callback_func = None

    if callback_func:
        await callback_func(q)
    else:
        await q.page.save()


async def create_new_h2o3_steam_engine(q: Q):
    q.client.selected_steam_engine = q.args.h2o3_steam_name
    q.page['engine_dashboard'] = get_h2o3_engine_details_card(q, q.args.h2o3_steam_name)
    q.page['engine_dashboard_header'].items = get_ai_engine_dashboard_header_items(
        q,
        is_dai=False,
        status=q.user.user.h2o3_engines[q.args.h2o3_steam_name]['engine'].status,
    )
    await q.page.save()


@on()
async def new_h2o3_steam_engine_dialog(q: Q):
    q.app.callbacks['new_h2o3_steam_engine'] = create_new_h2o3_steam_engine
    await make_new_h2o3_steam_engine_dialog(q)


@on()
async def launch_new_h2o3_steam_engine(q: Q):
    h2o3_engine_config = H2oEngineConfig(  # type: ignore
        name=q.args.h2o3_steam_name,
        version=q.args.h2o3_steam_version,
        node_count=q.args.h2o3_steam_node_count,
        cpu_count=q.args.h2o3_steam_cpu_count,
        memory_gb=q.args.h2o3_steam_memory,
        max_idle_h=q.args.h2o3_steam_max_idle_time,
        max_uptime_h=q.args.h2o3_steam_max_uptime,
        timeout_s=q.args.h2o3_steam_timeout * 60,
    )
    error_msg = validate_steam_engine_name(
        name=q.args.h2o3_steam_name, existing_engines=q.user.user.h2o3_engines
    )
    if error_msg:
        q.page['meta'].dialog = get_new_h2o3_steam_engine_options(
            q,
            h2o3_engine_config=h2o3_engine_config,
            error_msg=error_msg,
        )
        await q.page.save()
        return

    q.page['meta'].dialog = get_wait_for_steam_dialog(q)
    await q.page.save()

    new_h2o3_engine = H2O3SteamEngine(
        steam_connection=q.user.steam_connection,
        timeout=q.app.steam_engine_timeout,
        config=h2o3_engine_config,
    )
    is_running = False
    if new_h2o3_engine is not None:
        is_running = new_h2o3_engine.status == 'running'

    connected = await get_steam_engines(q)
    if not connected:
        logger.error('There was an error establishing connection to STEAM')

    if is_running and q.args.h2o3_steam_name in q.user.user.h2o3_engines:
        q.page['meta'].dialog = get_wait_for_steam_dialog(q, done=True)
        if (
            isinstance(q.app.callbacks, dict)
            and 'new_h2o3_steam_engine' in q.app.callbacks
        ):
            callback_func = q.app.callbacks['new_h2o3_steam_engine']
            del q.app.callbacks['new_h2o3_steam_engine']
            if 'new_h2o3_steam_engine_cancel' in q.app.callbacks:
                del q.app.callbacks['new_h2o3_steam_engine_cancel']
        else:
            callback_func = None
    else:
        if new_h2o3_engine is None:
            error_msg = 'Error occurred while trying to create a H2O-3 Steam Engine!'
        else:
            error_msg = f'Newly created Steam Engine did not get to "running" status in {q.app.steam_engine_timeout} seconds!'

        q.page['meta'].dialog = get_wait_for_steam_dialog(
            q, done=True, error_msg=error_msg
        )
        if (
            isinstance(q.app.callbacks, dict)
            and 'new_h2o3_steam_engine_cancel' in q.app.callbacks
        ):
            callback_func = q.app.callbacks['new_h2o3_steam_engine_cancel']
            del q.app.callbacks['new_h2o3_steam_engine_cancel']
            if 'new_h2o3_steam_engine' in q.app.callbacks:
                del q.app.callbacks['new_h2o3_steam_engine']
        else:
            callback_func = None

    if callback_func:
        await callback_func(q)
    else:
        await q.page.save()


@on()
async def steam_engine_table(q: Q):
    if q.args.steam_engine_table.startswith('dai_steam_engine_'):
        engine_name = q.args.steam_engine_table.split('dai_steam_engine_')[-1]
        q.client.selected_steam_engine = engine_name
        q.page['engine_dashboard'] = get_dai_engine_details_card(q, engine_name)
        q.page['engine_dashboard_header'].items = get_ai_engine_dashboard_header_items(
            q,
            is_dai=True,
            status=q.user.user.dai_engines[engine_name]['engine'].status,
        )
    elif q.args.steam_engine_table.startswith('h2o3_steam_engine_'):
        engine_name = q.args.steam_engine_table.split('h2o3_steam_engine_')[-1]
        q.client.selected_steam_engine = engine_name
        q.page['engine_dashboard'] = get_h2o3_engine_details_card(q, engine_name)
        q.page['engine_dashboard_header'].items = get_ai_engine_dashboard_header_items(
            q,
            is_dai=False,
            status=q.user.user.h2o3_engines[engine_name]['engine'].status,
        )
    else:
        logger.error(f'Selection {q.args.steam_engine_table} is neither DAI or H2O-3')
        q.client.selected_steam_engine = None

    await q.page.save()


@on()
async def dai_steam_engine_start(q: Q):
    # Show waiting progress dialog
    q.page['meta'].dialog = get_wait_for_restart_dialog(q)
    await q.page.save()

    # Start the Driverless AI Steam Engine
    started = q.user.user.dai_engines[q.client.selected_steam_engine]['engine'].start()

    # Update local Steam engine dict
    connected = await get_steam_engines(q)
    if not connected:
        logger.error('There was an error establishing connection to STEAM')

    # TODO: Add Fail message
    # Update the waiting progress dialog to Done!
    q.page['meta'].dialog = get_wait_for_restart_dialog(q, done=True)

    if started:
        # Update Details Card and the Header on top of it
        q.page['engine_dashboard'] = get_dai_engine_details_card(
            q, q.client.selected_steam_engine
        )
        q.page['engine_dashboard_header'].items = get_ai_engine_dashboard_header_items(
            q,
            is_dai=True,
            status=q.user.user.dai_engines[q.client.selected_steam_engine][
                'engine'
            ].status,
        )
    else:
        q.page['engine_dashboard_header'] = get_ai_engine_dashboard_header(q)
        q.page['engine_dashboard'] = get_ai_engine_dashboard(q)
        q.client.selected_steam_engine = None

    await q.page.save()


@on()
async def dai_steam_engine_stop(q: Q):
    q.page['meta'].dialog = get_wait_for_stop_dialog(q)
    await q.page.save()

    # TODO: handle when stopped is False
    stopped = q.user.user.dai_engines[q.client.selected_steam_engine]['engine'].stop()
    connected = await get_steam_engines(q)
    if not connected:
        logger.error('There was an error establishing connection to STEAM')

    q.page['meta'].dialog = get_wait_for_stop_dialog(q, done=True)

    q.page['engine_dashboard'] = get_dai_engine_details_card(
        q, q.client.selected_steam_engine
    )
    q.page['engine_dashboard_header'].items = get_ai_engine_dashboard_header_items(
        q,
        is_dai=True,
        status=q.user.user.dai_engines[q.client.selected_steam_engine]['engine'].status,
    )
    await q.page.save()


@on()
async def dai_steam_engine_terminate(q: Q):
    q.page['meta'].dialog = get_wait_for_terminate_dialog(q)
    await q.page.save()

    # TODO: handle when terminate is False
    terminated = q.user.user.dai_engines[q.client.selected_steam_engine][
        'engine'
    ].terminate()
    connected = await get_steam_engines(q)
    if not connected:
        logger.error('There was an error establishing connection to STEAM')

    q.page['meta'].dialog = get_wait_for_terminate_dialog(q, done=True)

    q.page['engine_dashboard_header'] = get_ai_engine_dashboard_header(q)
    q.page['engine_dashboard'] = get_ai_engine_dashboard(q)

    await q.page.save()


@on()
async def dai_steam_engine_connect(q: Q):
    q.page['meta'].dialog = get_wait_for_connect_dialog(q)
    await q.page.save()
    engine_name = q.client.selected_steam_engine
    dai_engine = q.user.user.dai_engines[engine_name]['engine']
    q.user.h2oai = dai_engine.connect()
    q.page['meta'].dialog = get_wait_for_connect_dialog(q, done=True)
    await q.page.save()


@on()
async def h2o3_steam_engine_terminate(q: Q):
    q.page['meta'].dialog = get_wait_for_h2o3_terminate_dialog(q)
    await q.page.save()

    terminated = q.user.user.h2o3_engines[q.client.selected_steam_engine][
        'engine'
    ].terminate()
    connected = await get_steam_engines(q)
    if not connected:
        logger.error('There was an error establishing connection to STEAM')

    q.page['meta'].dialog = get_wait_for_h2o3_terminate_dialog(q, done=True)

    q.page['engine_dashboard_header'] = get_ai_engine_dashboard_header(q)
    q.page['engine_dashboard'] = get_ai_engine_dashboard(q)

    await q.page.save()


@on()
async def close_dialog(q: Q):
    q.page['meta'].dialog = None
    await q.page.save()


@on()
async def close_new_dai_steam_engine(q: Q):
    q.page['meta'].dialog = None
    if (
        isinstance(q.app.callbacks, dict)
        and 'new_dai_steam_engine_cancel' in q.app.callbacks
    ):
        callback_func = q.app.callbacks['new_dai_steam_engine_cancel']
        del q.app.callbacks['new_dai_steam_engine_cancel']
        if 'new_dai_steam_engine' in q.app.callbacks:
            del q.app.callbacks['new_dai_steam_engine']
    else:
        callback_func = None

    if callback_func:
        await callback_func(q)
    else:
        await q.page.save()


@on()
async def close_new_h2o3_steam_engine(q: Q):
    q.page['meta'].dialog = None
    if (
        isinstance(q.app.callbacks, dict)
        and 'new_h2o3_steam_engine_cancel' in q.app.callbacks
    ):
        callback_func = q.app.callbacks['new_h2o3_steam_engine_cancel']
        del q.app.callbacks['new_h2o3_steam_engine_cancel']
        if 'new_h2o3_steam_engine' in q.app.callbacks:
            del q.app.callbacks['new_h2o3_steam_engine']
    else:
        callback_func = None

    if callback_func:
        await callback_func(q)
    else:
        await q.page.save()
