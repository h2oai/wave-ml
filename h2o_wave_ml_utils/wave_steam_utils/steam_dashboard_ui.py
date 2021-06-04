from dataclasses import asdict, dataclass
from h2o_wave import Q, ui
from .steam_libs import DAIEngineConfig, H2oEngineConfig, get_steam_engines, add_user


def time_human_readable(seconds):
    days, rem = divmod(seconds, 60 * 60 * 24)
    hours, rem = divmod(rem, 60 * 60)
    minutes, seconds = divmod(rem, 60)
    hours = int(hours)
    seconds = int(seconds)
    minutes = int(minutes)
    time_str = f'{seconds}s'
    if minutes > 0:
        time_str = f'{minutes}m ' + time_str
    if hours > 0:
        time_str = f'{hours}h ' + time_str
    if days > 0:
        time_str = f'{days}d ' + time_str
    return time_str


@dataclass(init=False, frozen=True)
class WaveColors:
    gray: str = '#9E9E9E'
    red: str = '#F44336'
    tangerine: str = '#FF5722'
    brown: str = '#795548'
    orange: str = '#FF9800'
    amber: str = '#FFC107'
    yellow: str = '#FFEB3B'
    lime: str = '#CDDC39'
    green: str = '#8BC34A'
    mint: str = '#4CAF50'
    teal: str = '#009688'
    cyan: str = '#00BCD4'
    azure: str = '#03A9F4'
    blue: str = '#2196F3'
    indigo: str = '#3F51B5'
    violet: str = '#673AB7'
    purple: str = '#9C27B0'
    pink: str = '#E91E63'


# Dashboard
def get_ai_engine_dashboard_header_items(q: Q, is_dai, status):
    if not is_dai:
        if status == 'running':
            h2o_url = q.user.user.h2o3_engines[q.client.selected_steam_engine][
                'engine'
            ].url
            return [
                ui.buttons(
                    items=[
                        ui.button(
                            name='manage_ai_engines', label='Back', primary=False
                        ),
                        ui.button(
                            name='h2o3_steam_engine_terminate',
                            label='Terminate',
                            primary=True,
                        ),
                    ],
                    justify='end',
                ),
                ui.link(label='Go to H2O-3', path=h2o_url, target='_blank', button=True),
            ]
        elif status == 'stopped':
            return [
                ui.buttons(
                    items=[
                        ui.button(
                            name='manage_ai_engines', label='Back', primary=False
                        ),
                        ui.button(
                            name='h2o3_steam_engine_terminate',
                            label='Terminate',
                            primary=True,
                        ),
                    ],
                    justify='end',
                )
            ]
        else:
            return [
                ui.buttons(
                    items=[
                        ui.button(
                            name='manage_ai_engines', label='Back', primary=False
                        ),
                    ],
                    justify='end',
                )
            ]
    if status == 'running':
        dai_url = q.user.user.dai_engines[q.client.selected_steam_engine][
            'engine'
        ].openid_login_url
        return [
            ui.buttons(
                items=[
                    ui.button(name='manage_ai_engines', label='Back', primary=False),
                    ui.button(
                        name='dai_steam_engine_terminate',
                        label='Terminate',
                        primary=False,
                    ),
                    ui.button(
                        name='dai_steam_engine_stop',
                        label='Stop',
                        primary=False,
                    ),
                    ui.button(
                        name='dai_steam_engine_connect',
                        label='Connect',
                        primary=True,
                    ),
                ],
                justify='end',
            ),
            ui.link(
                label='Go to Driverless AI', path=dai_url, target='_blank', button=True
            ),
        ]
    elif status == 'stopped':
        return [
            ui.buttons(
                items=[
                    ui.button(name='manage_ai_engines', label='Back', primary=False),
                    ui.button(
                        name='dai_steam_engine_terminate',
                        label='Terminate',
                        primary=False,
                    ),
                    ui.button(
                        name='dai_steam_engine_start',
                        label='Start',
                        primary=True,
                    ),
                ],
                justify='end',
            )
        ]
    else:
        return [
            ui.buttons(
                items=[
                    ui.button(name='manage_ai_engines', label='Back', primary=True),
                ],
                justify='end',
            )
        ]


def get_ai_engine_dashboard_header(q: Q):
    box_name = q.app.steam_box_name
    return ui.section_card(
        box=ui.box(zone=box_name, height='45px'),
        title='AI Engine Dashboard',
        subtitle='A control panel to monitor, start, stop, and terminate your instances in Steam',
        items=[
            ui.buttons(
                items=[
                    ui.button(
                        name='new_dai_steam_engine_dialog',
                        label='New Driverless AI Engine',
                        primary=True,
                    ),
                    ui.button(
                        name='new_h2o3_steam_engine_dialog',
                        label='New H2O-3 Engine',
                        primary=True,
                    ),
                    ui.button(
                        name='refresh_steam_engine_table', label='Refresh', primary=True
                    ),
                ]
            )
        ],
    )


def get_ai_engine_dashboard(q: Q):
    box_name = q.app.steam_box_name
    status_colors_dai = dict(
        running=WaveColors.blue,
        stopped=WaveColors.amber,
        stopping=WaveColors.yellow,
        failed=WaveColors.red,
        terminating=WaveColors.tangerine,
        submitting=WaveColors.lime,
        starting=WaveColors.azure,
        unreachable=WaveColors.orange,
    )
    status_colors_h2o3 = dict(
        running=WaveColors.blue,
        stopped=WaveColors.red,
        stopping=WaveColors.tangerine,
        failed=WaveColors.red,
        terminating=WaveColors.tangerine,
        submitting=WaveColors.lime,
        starting=WaveColors.azure,
        unreachable=WaveColors.orange,
    )
    dai_steam_engines = []
    for k, v in q.user.user.dai_engines.items():
        if v['type'] != 'steam':
            continue
        engine = v['engine']
        if engine.details.created_by != q.user.user.email:
            continue
        dai_steam_engines.append(
            ui.stat_table_item(
                name=f'dai_steam_engine_{k}',
                label=k,
                caption=engine.status.title(),
                values=[
                    'Driverless AI',
                    engine.details.version,
                    time_human_readable(engine.details.current_uptime_seconds),
                    time_human_readable(engine.details.current_idle_seconds),
                    engine.details.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                ],
                icon='Server',
                icon_color=status_colors_dai.get(engine.status, '#000000'),
            )
        )
    h2o3_steam_engines = []
    for k, v in q.user.user.h2o3_engines.items():
        if v['type'] != 'steam':
            continue
        if v['engine'].details.created_by != q.user.user.email:
            continue
        engine = v['engine']
        status = engine.status
        version = engine.details.version
        uptime = time_human_readable(engine.details.current_uptime_millis / 1000.0)
        idle_time = time_human_readable(engine.details.current_idle_millis / 1000.0)
        created_at = engine.details.created_at.strftime('%Y-%m-%d %H:%M:%S')
        h2o3_steam_engines.append(
            ui.stat_table_item(
                name=f'h2o3_steam_engine_{k}',
                label=k,
                caption=status.title(),
                values=[
                    'H2O-3',
                    version,
                    uptime,
                    idle_time,
                    created_at,
                ],
                icon='Server',
                icon_color=status_colors_h2o3.get(status, '#000000'),
            )
        )
    return ui.stat_table_card(
        name='steam_engine_table',
        box=ui.box(zone=box_name, order=2),
        title='Updated at',
        subtitle=q.user.last_steam_refresh.strftime('%Y-%m-%d %H:%M:%S'),
        columns=[
            'Name',
            '  Type',
            '  Version',
            '    Up Time',
            '    Idle Time',
            '      Created At',
        ],
        items=dai_steam_engines + h2o3_steam_engines,
    )


# New Driverless AI


def get_new_dai_steam_engine_options(
    q: Q,
    dai_engine_config: DAIEngineConfig = None,
    error_msg=None,
):
    dai_versions_in_steam = ['1.9.0.6', '1.9.1.1', '1.9.1.3', '1.9.2.1']
    disabled = ['1.9.0.6', '1.9.1.3', '1.9.2.1']
    dai_choices = [
        ui.choice(name=x, label=x, disabled=x in disabled)
        for x in dai_versions_in_steam
    ]
    if dai_engine_config is None:
        dai_engine_config = DAIEngineConfig(  # type: ignore
            name='',
            version='1.9.1.1',
            cpu_count=31 if q.app.max_steam else 8,
            gpu_count=2 if q.app.max_steam else 0,
            memory_gb=232 if q.app.max_steam else 30,
            storage_gb=1024 if q.app.max_steam else 64,
            max_idle_h=24 if q.app.max_steam else 2,
            max_uptime_h=24 if q.app.max_steam else 8,
            timeout_s=15 * 60,
        )
    return ui.dialog(
        title='Create new Driverless AI Steam Engine',
        items=[
            ui.textbox(
                name='dai_steam_instance_name',
                label='Instance name',
                required=False,
                value=dai_engine_config.name,
                error=error_msg,
            ),
            ui.text_s(
                'A valid name Starts with a letter, At least 3 characters long, Allowed characters: a-z, A-Z, 0-9, _, -'
            ),
            ui.dropdown(
                name='dai_steam_instance_version',
                label='Version',
                choices=dai_choices,
                value=dai_engine_config.version,
            ),
            ui.text_s(
                'Disabled versions are also available in Steam but not compatible with this app at the moment.'
            ),
            ui.slider(
                name='dai_steam_instance_cpu_count',
                label="Number of CPUs",
                min=1,
                max=31,
                step=1,
                value=dai_engine_config.cpu_count,
            ),
            ui.text_s('min: 1, max: 31'),
            ui.slider(
                name='dai_steam_instance_gpu_count',
                label="Number of GPUs",
                min=0,
                max=2,
                step=1,
                value=dai_engine_config.gpu_count,
            ),
            ui.text_s('min: 0, max: 2'),
            ui.spinbox(
                name='dai_steam_instance_mem',
                label='Memory (GB)',
                min=8,
                max=232,
                step=1,
                value=dai_engine_config.memory_gb,
            ),
            ui.text_s('min: 8 GB, max: 232 GB'),
            ui.spinbox(
                name='dai_steam_instance_storage',
                label='Storage (GB)',
                min=10,
                max=1024,
                step=4,
                value=dai_engine_config.storage_gb,
            ),
            ui.text_s('min: 10 GB, max: 1024 GB'),
            ui.spinbox(
                name='dai_steam_instance_idle_time',
                label='Maximum Idle Time (Hours)',
                min=1,
                max=24,
                step=1,
                value=dai_engine_config.max_idle_h,
            ),
            ui.text_s('min: 1 Hour, max: 24 Hours'),
            ui.spinbox(
                name='dai_steam_instance_uptime',
                label='Maximum Uptime (Hours)',
                min=1,
                max=24,
                step=1,
                value=dai_engine_config.max_uptime_h,
            ),
            ui.text_s('min: 1 Hour, max: 24 Hours'),
            ui.spinbox(
                name='dai_steam_instance_timeout',
                label='Timeout (Minutes)',
                min=10,
                max=30,
                step=5,
                value=dai_engine_config.timeout_s / 60,
            ),
            ui.text_s('min: 10 Minutes, max: 60 Minutes'),
            ui.spinbox(
                name='new_dai_instance_running_timeout_min',
                label='Wait for Running Status (Minutes)',
                min=1,
                max=10,
                step=1,
                value=q.user.new_dai_instance_running_timeout_min,
            ),
            ui.text_s('min: 1 Minute, max: 10 Minutes'),
            ui.buttons(
                items=[
                    ui.button(
                        name='close_new_dai_steam_engine',
                        label='Cancel',
                        primary=False,
                    ),
                    ui.button(
                        name='launch_new_dai_steam_engine',
                        label='Launch Instance',
                        primary=True,
                    ),
                ],
                justify='end',
            ),
        ],
    )


async def make_new_dai_steam_engine_dialog(q: Q):
    q.page['meta'].dialog = get_new_dai_steam_engine_options(q)
    await q.page.save()


# New H2O-3


def get_new_h2o3_steam_engine_options(
    q: Q, h2o3_engine_config: H2oEngineConfig = None, error_msg=None
):
    h2o3_versions = ['3.32.0.3', '3.32.0.5']
    disabled = ['3.32.0.3']
    h2o3_choices = [
        ui.choice(name=x, label=x, disabled=x in disabled) for x in h2o3_versions
    ]
    if h2o3_engine_config is None:
        h2o3_engine_config = H2oEngineConfig(  # type: ignore
            name='',
            version='3.32.0.5',
            node_count=8 if q.app.max_steam else 1,
            cpu_count=8 if q.app.max_steam else 1,
            memory_gb=48 if q.app.max_steam else 4,
            max_idle_h=24 if q.app.max_steam else 8,
            max_uptime_h=24 if q.app.max_steam else 12,
            timeout_s=10 * 60,
        )
    items = [
        ui.textbox(
            name='h2o3_steam_name',
            label='Cluster name',
            required=False,
            value=h2o3_engine_config.name,
            error=error_msg,
        ),
        ui.text_s(
            'A valid name Starts with a letter, At least 3 characters long, Allowed characters: a-z, A-Z, 0-9, _, -'
        ),
        ui.dropdown(
            name='h2o3_steam_version',
            label='Version',
            choices=h2o3_choices,
            value=h2o3_engine_config.version,
        ),
        ui.text_s(
            'Disabled versions are also available in Steam but not compatible with this app at the moment.'
        ),
        ui.slider(
            name='h2o3_steam_node_count',
            label='Number of Nodes',
            min=1,
            max=8,
            step=1,
            value=h2o3_engine_config.node_count,
        ),
        ui.text_s('min: 1, max: 8'),
        ui.slider(
            name='h2o3_steam_cpu_count',
            label='Number of CPUs',
            min=1,
            max=8,
            step=1,
            value=h2o3_engine_config.cpu_count,
        ),
        ui.text_s('min: 1, max: 8'),
        ui.spinbox(
            name='h2o3_steam_memory',
            label='Memory per Node (GB)',
            min=4,
            max=48,
            step=1,
            value=h2o3_engine_config.memory_gb,
        ),
        ui.text_s('min: 4 GB, max: 48 GB'),
        ui.spinbox(
            name='h2o3_steam_max_idle_time',
            label='Maximum Idle Time (Hours)',
            min=1,
            max=24,
            step=1,
            value=h2o3_engine_config.max_idle_h,
        ),
        ui.text_s('min: 1 Hour, max: 24 Hours'),
        ui.spinbox(
            name='h2o3_steam_max_uptime',
            label='Maximum Uptime (Hours)',
            min=1,
            max=24,
            step=1,
            value=h2o3_engine_config.max_uptime_h,
        ),
        ui.text_s('min: 1 Hour, max: 24 Hours'),
        ui.spinbox(
            name='h2o3_steam_timeout',
            label='Timeout (Minutes)',
            min=10,
            max=30,
            step=5,
            value=h2o3_engine_config.timeout_s / 60,
        ),
        ui.text_s('min: 10 Minutes, max: 60 Minutes'),
        ui.buttons(
            items=[
                ui.button(
                    name='close_new_h2o3_steam_engine',
                    label='Cancel',
                    primary=False,
                ),
                ui.button(
                    name='launch_new_h2o3_steam_engine',
                    label='Launch Cluster',
                    primary=True,
                ),
            ],
            justify='end',
        ),
    ]

    return ui.dialog(
        title='Create new H2O-3 Steam Engine',
        items=items,
    )


async def make_new_h2o3_steam_engine_dialog(q: Q):
    q.page['meta'].dialog = get_new_h2o3_steam_engine_options(q)
    await q.page.save()


# Details


def get_dai_engine_details_card(q: Q, engine_name):
    box_name = q.app.steam_box_name
    valid_fields = [
        'profile_name',
        'name',
        'status',
        'version',
        'instance_type',
        'cpu_count',
        'gpu_count',
        'memory_gb',
        'storage_gb',
        'max_idle_seconds',
        'max_uptime_seconds',
        # 'address',
        'created_at',
        'created_by',
        'current_uptime_seconds',
        'current_idle_seconds',
    ]
    return ui.stat_list_card(
        box=ui.box(zone=box_name, order=2),
        title=engine_name,
        items=[
            ui.stat_list_item(label=k, value=str(v))
            for k, v in asdict(
                q.user.user.dai_engines[engine_name]['engine'].details
            ).items()
            if k in valid_fields
        ],
    )


def get_h2o3_engine_details_card(q: Q, engine_name):
    engine = q.user.user.h2o3_engines[engine_name]['engine']
    box_name = q.app.steam_box_name
    # TODO: use a loop
    items = [
        # ui.stat_list_item(label='id', value=str(engine.id)),
        ui.stat_list_item(label='Name', value=engine.name),
        ui.stat_list_item(label='Profile Name', value=engine.details.profile_name),
        ui.stat_list_item(label='Status', value=engine.status),
        ui.stat_list_item(label='Version', value=engine.details.version),
        ui.stat_list_item(label='Node Count', value=str(engine.details.node_count)),
        ui.stat_list_item(label='Cpu Count', value=str(engine.details.cpu_count)),
        ui.stat_list_item(label='Gpu Count', value=str(engine.details.gpu_count)),
        ui.stat_list_item(label='Memory GB', value=str(engine.details.memory_gb)),
        ui.stat_list_item(
            label='Max Idle Time (hours)', value=str(engine.details.max_idle_h)
        ),
        ui.stat_list_item(
            label='Max Uptime (hours)', value=str(engine.details.max_uptime_h)
        ),
        ui.stat_list_item(
            label='Created At',
            value=engine.details.created_at.strftime('%Y-%m-%d %H:%M:%S'),
        ),
        ui.stat_list_item(label='Created by', value=engine.details.created_by),
    ]
    return ui.stat_list_card(
        box=ui.box(zone=box_name, order=2),
        title=engine_name,
        items=items,
    )


# Actions: Start, Stop, Terminate, Wait


def get_wait_for_terminate_dialog(q: Q, done=False):
    if done:
        return ui.dialog(
            title='Steam Engine is now terminated',
            items=[
                ui.progress(label='', caption='done!', value=1.0),
                ui.buttons(
                    items=[
                        ui.button(
                            name='close_dialog',
                            label='Close',
                            primary=True,
                            disabled=False,
                        )
                    ],
                    justify='end',
                ),
            ],
        )

    return ui.dialog(
        title='Terminating Driverless AI Steam Engine',
        items=[
            ui.progress(
                label='', caption='This will take a minute or two, please standby ...'
            ),
            ui.buttons(
                items=[
                    ui.button(
                        name='close_dialog',
                        label='Close',
                        primary=True,
                        disabled=True,
                    )
                ],
                justify='end',
            ),
        ],
    )


def get_wait_for_connect_dialog(q: Q, done=False):
    if done:
        return ui.dialog(
                title='DAI Connection',
                items=[
                    ui.message_bar('success', 'Connected Successfully!'),
                    ui.buttons(
                        items=[
                            ui.button(
                                name='dai_engine_connected',
                                label='Next',
                                primary=True,
                                disabled=False,
                            )
                        ],
                        justify='end',
                    )
                ],
        )
    else:
        return ui.dialog(
                title='DAI Connection',
                items=[
                    ui.progress(label='Connecting to DAI instance', caption='')
                ],
        )

def get_wait_for_h2o3_terminate_dialog(q: Q, done=False):
    if done:
        return ui.dialog(
            title='Steam Engine is now terminated',
            items=[
                ui.progress(label='', caption='done!', value=1.0),
                ui.buttons(
                    items=[
                        ui.button(
                            name='close_dialog',
                            label='Close',
                            primary=True,
                            disabled=False,
                        )
                    ],
                    justify='end',
                ),
            ],
        )

    return ui.dialog(
        title='Terminating H2O-3 Steam Engine',
        items=[
            ui.progress(
                label='', caption='This will take a minute or two, please standby ...'
            ),
            ui.buttons(
                items=[
                    ui.button(
                        name='close_dialog',
                        label='Close',
                        primary=True,
                        disabled=True,
                    )
                ],
                justify='end',
            ),
        ],
    )


def wait_for_steam_refresh_dialog(error_msg=None):
    items = [
        ui.message_bar(
            type='error', text=error_msg, visible=True if error_msg else False
        ),
        ui.progress(
            label='', caption='This will take a second or two, please standby ...'
        ),
    ]
    return ui.dialog(
        title='Refreshing Steam Engines',
        items=items,
    )


def get_wait_for_steam_dialog(q: Q, done=False, error_msg=None):
    if done:
        return ui.dialog(
            title='New Steam Engine is now available'
            if not error_msg
            else 'Unable to start the new Steam Engine!',
            items=[
                ui.progress(
                    label='', caption='done!' if not error_msg else '', value=1.0
                ),
                ui.message_bar(
                    type='error', text=error_msg, visible=True if error_msg else False
                ),
                ui.buttons(
                    items=[
                        ui.button(
                            name='close_dialog',
                            label='Close',
                            primary=True,
                            disabled=False,
                        )
                    ],
                    justify='end',
                ),
            ],
        )

    return ui.dialog(
        title='Launching new AI Engine in Steam',
        items=[
            ui.progress(
                label='', caption='This will take a minute or two, please standby ...'
            ),
            ui.buttons(
                items=[
                    ui.button(
                        name='close_dialog',
                        label='Close',
                        primary=True,
                        disabled=True,
                    )
                ],
                justify='end',
            ),
        ],
    )


def get_wait_for_restart_dialog(q: Q, done=False):
    if done:
        return ui.dialog(
            title='New Steam Engine is now running',
            items=[
                ui.progress(label='', caption='done!', value=1.0),
                ui.buttons(
                    items=[
                        ui.button(
                            name='close_dialog', #'close_wait_for_restart',
                            label='Close',
                            primary=True,
                            disabled=False,
                        )
                    ],
                    justify='end',
                ),
            ],
        )

    return ui.dialog(
        title='Restarting Driverless AI Steam Engine',
        items=[
            ui.progress(
                label='', caption='This will take a minute or two, please standby ...'
            ),
            ui.buttons(
                items=[
                    ui.button(
                        name='close_dialog', #'close_wait_for_restart',
                        label='Close',
                        primary=True,
                        disabled=True,
                    )
                ],
                justify='end',
            ),
        ],
    )


def get_wait_for_stop_dialog(q: Q, done=False):
    if done:
        return ui.dialog(
            title='Steam Engine is now stopped',
            items=[
                ui.progress(label='', caption='done!', value=1.0),
                ui.buttons(
                    items=[
                        ui.button(
                            name='close_dialog',
                            label='Close',
                            primary=True,
                            disabled=False,
                        )
                    ],
                    justify='end',
                ),
            ],
        )

    return ui.dialog(
        title='Stopping Driverless AI Steam Engine',
        items=[
            ui.progress(
                label='', caption='This will take a minute or two, please standby ...'
            ),
            ui.buttons(
                items=[
                    ui.button(
                        name='close_dialog',
                        label='Close',
                        primary=True,
                        disabled=True,
                    )
                ],
                justify='end',
            ),
        ],
    )


# Main menu for Steam engines
async def steam_menu(q: Q):
    add_user(q)
    q.page['meta'].dialog = wait_for_steam_refresh_dialog()
    await q.page.save()
    # Once connected to steam - This will return true
    connected = await get_steam_engines(q)
    # q.app.users will be a dict of user of AppUser class (see steam_utils.py)
    q.page['engine_dashboard_header'] = get_ai_engine_dashboard_header(q)
    q.page['engine_dashboard'] = get_ai_engine_dashboard(q)
    q.page['meta'].dialog = None
    q.client.selected_steam_engine = None
    await q.page.save()
