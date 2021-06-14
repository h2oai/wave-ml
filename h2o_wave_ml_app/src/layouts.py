from h2o_wave import ui


def small() -> ui.Layout:
    """
    Layout for small screen sizes.
    """

    layout = ui.layout(
        breakpoint='0px',
        zones=[
            ui.zone(name='header'),
            ui.zone(
                name='main',
                zones=[
                    ui.zone(
                        name='navbar',
                        zones=[
                            ui.zone(name='misc'),
                            ui.zone(name='tabs')
                        ]
                    ),
                    ui.zone(name='home'),
                    ui.zone(
                        name='demo',
                        zones=[
                            ui.zone(name='demo_h2oaml_local'),
                            ui.zone(name='demo_h2oaml_cloud'),
                            ui.zone(name='demo_dai_cloud'),
                            ui.zone(name='inputs_h2oaml_local'),
                            ui.zone(name='outputs_h2oaml_local'),
                            ui.zone(name='inputs_h2oaml_cloud'),
                            ui.zone(name='outputs_h2oaml_cloud'),
                            ui.zone(name='inputs_dai_cloud'),
                            ui.zone(name='outputs_dai_cloud')
                        ]
                    ),
                    ui.zone(
                        name='resources',
                        zones=[
                            ui.zone(name='code_examples_heading'),
                            ui.zone(name='code_examples'),
                            ui.zone(name='wave_examples_heading'),
                            ui.zone(
                                name='wave_examples',
                                direction='row',
                                zones=[
                                    ui.zone(name='h2oaml_examples', size='50%'),
                                    ui.zone(name='dai_examples', size='50%')
                                ]
                            )
                        ]
                    )
                ]
            ),
            ui.zone(name='error'),
            ui.zone(name='footer')
        ]
    )

    return layout


def large() -> ui.Layout:
    """
    Layout for large screen sizes.
    """

    layout = ui.layout(
        breakpoint='835px',
        zones=[
            ui.zone(name='header'),
            ui.zone(
                name='main',
                zones=[
                    ui.zone(
                        name='navbar',
                        direction='row',
                        zones=[
                            ui.zone(name='tabs', size='50%'),
                            ui.zone(name='misc', size='50%')
                        ]
                    ),
                    ui.zone(name='home'),
                    ui.zone(
                        name='demo',
                        zones=[
                            ui.zone(
                                name='demos',
                                direction='row',
                                zones=[
                                    ui.zone(name='demo_h2oaml_local', size='33%'),
                                    ui.zone(name='demo_h2oaml_cloud', size='34%'),
                                    ui.zone(name='demo_dai_cloud', size='33%')
                                ]
                            ),
                            ui.zone(
                                name='h2oaml_local',
                                direction='row',
                                zones=[
                                    ui.zone(name='inputs_h2oaml_local', size='50%'),
                                    ui.zone(name='outputs_h2oaml_local', size='50%')
                                ]
                            ),
                            ui.zone(
                                name='h2oaml_cloud',
                                direction='row',
                                zones=[
                                    ui.zone(name='inputs_h2oaml_cloud', size='50%'),
                                    ui.zone(name='outputs_h2oaml_cloud', size='50%')
                                ]
                            ),
                            ui.zone(
                                name='dai_cloud',
                                direction='row',
                                zones=[
                                    ui.zone(name='inputs_dai_cloud', size='50%'),
                                    ui.zone(name='outputs_dai_cloud', size='50%')
                                ]
                            )
                        ]
                    ),
                    ui.zone(
                        name='resources',
                        direction='row',
                        zones=[
                            ui.zone(
                                name='code_examples_section',
                                size='50%',
                                zones=[
                                    ui.zone(name='code_examples_heading'),
                                    ui.zone(name='code_examples')
                                ]
                            ),
                            ui.zone(
                                name='wave_examples_section',
                                size='50%',
                                zones=[
                                    ui.zone(name='wave_examples_heading'),
                                    ui.zone(
                                        name='wave_examples',
                                        direction='row',
                                        zones=[
                                            ui.zone(name='h2oaml_examples', size='50%'),
                                            ui.zone(name='dai_examples', size='50%')
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            ui.zone(name='error'),
            ui.zone(name='footer')
        ]
    )

    return layout
