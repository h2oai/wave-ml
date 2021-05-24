import sys
import traceback
from pathlib import Path

from h2o_wave import ui
from pygments import highlight
from pygments.formatters.html import HtmlFormatter
from pygments.lexers.python import PythonLexer
from pygments.styles.default import DefaultStyle
from pygments.token import Name, Punctuation


FUNCTION_CHOICES = [ui.choice(name=str(x), label=str(x)) for x in [
    'Train Model', 'Score Model', 'Save Model', 'Load Model', 'Get Model',
    'List DAI Single Instances', 'List DAI Multinode Clusters'
]]
FUNCTION_DESCRIPTIONS = {
    'Train Model': 'Code snippet for training a model',
    'Score Model': 'Code snippet for scoring a model',
    'Save Model': 'Code snippet for saving a model to a local path',
    'Load Model': 'Code snippet for loading a model from a local path',
    'Get Model': 'Code snippet for fetching a model from H2O MLOps',
    'List DAI Single Instances': 'Code snippet for listing available Driverless AI single instances with Steam',
    'List DAI Multinode Clusters': 'Code snippet for listing available Driverless AI multinode clusters with Steam'
}
FUNCTION_DOCUMENTATIONS = {
    'Train Model': ui.link(label='build_model', path='https://wave.h2o.ai/docs/api/h2o_wave_ml/ml#build_model'),
    'Score Model': ui.link(label='predict', path='https://wave.h2o.ai/docs/api/h2o_wave_ml/types#predict'),
    'Save Model': ui.link(label='save_model', path='https://wave.h2o.ai/docs/api/h2o_wave_ml/ml#save_model'),
    'Load Model': ui.link(label='load_model', path='https://wave.h2o.ai/docs/api/h2o_wave_ml/ml#load_model'),
    'Get Model': ui.link(label='get_model', path='https://wave.h2o.ai/docs/api/h2o_wave_ml/ml#get_model'),
    'List DAI Single Instances': ui.link(
        label='list_dai_instances',
        path='https://wave.h2o.ai/docs/api/h2o_wave_ml/utils#list_dai_instances'
    ),
    'List DAI Multinode Clusters': ui.link(
        label='list_dai_multinodes',
        path='https://wave.h2o.ai/docs/api/h2o_wave_ml/utils#list_dai_multinodes'
    ),
}

PYTHON_LEXER = PythonLexer()


class LightStyle(DefaultStyle):
    styles = DefaultStyle().styles
    styles[Name.Namespace] = 'bold #CDD015'


class NeonStyle(DefaultStyle):
    background_color = '#202020'
    styles = DefaultStyle().styles
    styles[Name] = '#FFFFFF'
    styles[Punctuation] = '#FFFFFF'
    styles[Name.Namespace] = 'bold #CDD015'


class Utils:
    """
    Utility variables and functions for app.
    """

    def __init__(self):
        self.droppable_cards = [
            'home',
            'demo_h2o3',
            'demo_dai_standalone',
            'demo_dai_cloud',
            'wave_examples_heading',
            'h2o3_examples',
            'dai_examples',
            'code_examples_heading',
            'code_examples',
            'error'
        ]

    @staticmethod
    def card_dummy() -> ui.FormCard:
        """
        Card for dummy use.
        """

        card = ui.form_card(
            box='dummy',
            items=[]
        )

        return card

    @staticmethod
    def card_meta() -> ui.MetaCard:
        """
        Card for meta information.
        """

        card = ui.meta_card(
            box='',
            title='H2O Wave ML',
            redirect='#home',
            layouts=[
                ui.layout(
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
                                        ui.zone(name='demo_h2o3'),
                                        ui.zone(name='demo_dai_cloud'),
                                        ui.zone(name='demo_dai_standalone')
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
                                                ui.zone(name='h2o3_examples', size='50%'),
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
                ),
                ui.layout(
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
                                    direction='row',
                                    zones=[
                                        ui.zone(name='demo_h2o3', size='33%'),
                                        ui.zone(name='demo_dai_cloud', size='34%'),
                                        ui.zone(name='demo_dai_standalone', size='33%')
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
                                                        ui.zone(name='h2o3_examples', size='50%'),
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
            ],
            theme='neon'
        )

        return card

    @staticmethod
    def card_error(q_app: dict, q_user: dict, q_args: dict, q_client: dict) -> ui.FormCard:
        """
        Card for handling crash.
        """

        error_msg_items = [
            ui.text_xl('The app encountered an error'),
            ui.text_l(
                'Apologies for the inconvenience.'
                'Please refresh your browser to restart H2O Wave ML.'
            ),
        ]
        error_report_items = [
            ui.text('To report this crash, please send an email to [cloud-feedback@h2o.ai](cloud-feedback@h2o.ai) '
                    'with the following information:'),
            ui.separator('Crash Report'),
            ui.text_l('H2O Wave ML')
        ]

        q_app = [f'{k}: {v}' for k, v in q_app.items()]
        q_app_str = '#### q.app\n```\n' + '\n'.join(q_app) + '\n```'
        q_app_items = [ui.text_m(q_app_str)]

        q_user = [f'{k}: {v}' for k, v in q_user.items()]
        q_user_str = '#### q.user\n```\n' + '\n'.join(q_user) + '\n```'
        q_user_items = [ui.text_m(q_user_str)]

        q_client = [f'{k}: {v}' for k, v in q_client.items()]
        q_client_str = '#### q.client\n```\n' + '\n'.join(q_client) + '\n```'
        q_client_items = [ui.text_m(q_client_str)]

        q_args = [f'{k}: {v}' for k, v in q_args.items()]
        q_args_str = '#### q.args\n```\n' + '\n'.join(q_args) + '\n```'
        q_args_items = [ui.text_m(q_args_str)]

        type_, value_, traceback_ = sys.exc_info()
        stack_trace = traceback.format_exception(type_, value_, traceback_)
        stack_trace_items = [ui.text('**Stack Trace**')] + [
            ui.text(f'`{x}`') for x in stack_trace
        ]

        error_report_items.extend(q_args_items + q_app_items + q_user_items + q_client_items + stack_trace_items)

        card = ui.form_card(
            box='error',
            items=error_msg_items + [
                ui.expander(name='error_report', label='Report this error', expanded=False, items=error_report_items)
            ]
        )

        return card

    @staticmethod
    def card_header() -> ui.HeaderCard:
        """
        Card for header.
        """

        card = ui.header_card(
            box='header',
            title='H2O Wave ML',
            subtitle='Simplifying Machine Learning for Wave Applications',
            icon='ProductVariant',
            icon_color='black'
        )

        return card

    @staticmethod
    def card_tabs() -> ui.TabCard:
        """
        Card for tabs.
        """

        card = ui.tab_card(
            box='tabs',
            items=[
                ui.tab(name='#home', label='Home', icon='HomeSolid'),
                ui.tab(name='#demo', label='Demo', icon='ShowGrid'),
                ui.tab(name='#resources', label='Resources', icon='ReadingModeSolid')
            ],
            link=True
        )

        return card

    @staticmethod
    def card_misc(theme_dark: bool) -> ui.SectionCard:
        """
        Card for links and theme.
        """

        card = ui.section_card(
            box='misc',
            title='',
            subtitle='',
            items=[
                ui.link(label='Documentation', path='https://wave.h2o.ai/docs/api/h2o_wave_ml/index', target=''),
                ui.link(label='GitHub', path='https://github.com/h2oai/wave-ml', target=''),
                ui.text(content=''),
                ui.toggle(name='theme_dark', label='Dark Mode', value=theme_dark, trigger=True)
            ]
        )

        return card

    @staticmethod
    def card_footer() -> ui.FooterCard:
        """
        Card for footer.
        """

        card = ui.footer_card(
            box='footer',
            caption='Made with 💛 using <a href="https://wave.h2o.ai" target="_blank">H2O Wave</a>'
        )

        return card

    @staticmethod
    def card_home() -> ui.FormCard:
        """
        Card for home page.
        """

        card = ui.form_card(
            box='home',
            items=[
                ui.stats(
                    items=[
                        ui.stat(
                            label='',
                            value='H2O Wave ML',
                            icon='ProductVariant',
                            icon_color='#CDDD38'
                        )
                    ],
                    justify='center'
                ),
                ui.text_l(content='''<center>A simple, high-level API
                    for powering Machine Learning in Wave Applications</center>'''),
                ui.text_s(content='<center>TO DO: Architecture Diagram to be updated here</center>'),
                ui.separator(),
                ui.stats(
                    items=[
                        ui.stat(
                            label='',
                            value='AutoML',
                            caption='Runs H2O-3 AutoML and Driverless AI',
                            icon='Processing',
                            icon_color='#CDDD38'
                        ),
                        ui.stat(
                            label='',
                            value='H2O AI Hybrid Cloud',
                            caption='Integrates AI Engines using Steam',
                            icon='Cloud',
                            icon_color='#CDDD38'
                        ),
                        ui.stat(
                            label='',
                            value='MLOps',
                            caption='Manages Models using H2O MLOps',
                            icon='OfflineStorage',
                            icon_color='#CDDD38'
                        )
                    ],
                    justify='around'
                )
            ]
        )

        return card

    @staticmethod
    def card_demo_h2o3() -> ui.FormCard:
        """
        Card for H2O-3 AutoML demo.
        """

        card = ui.form_card(
            box='demo_h2o3',
            items=[
                ui.stats(
                    items=[
                        ui.stat(
                            label='',
                            value='H2O-3 AutoML',
                            caption='Using a local H2O-3 cluster',
                            icon='Product',
                            icon_color='#CDDD38'
                        )
                    ],
                    justify='center'
                ),
                ui.separator(),
                ui.text(content='* Manages the backend H2O-3 cluster under the hood for running H2O AutoML'),
                ui.text(content='* Supports training, scoring, saving, loading and interpretability of models'),
                ui.text(content='* Supports hyperparameter configuration and custom settings for model'),
                ui.text(content='* Provides the final H2O-3 model object for downstream tasks'),
                ui.buttons(
                    items=[ui.button(name='demo_h2o3', label='Try Demo', primary=True)],
                    justify='center'
                )
            ]
        )

        return card

    @staticmethod
    def card_demo_dai_cloud() -> ui.FormCard:
        """
        Card for H2O AI Hybrid Cloud Driverless AI demo.
        """

        card = ui.form_card(
            box='demo_dai_cloud',
            items=[
                ui.stats(
                    items=[
                        ui.stat(
                            label='',
                            value='Driverless AI (Cloud)',
                            caption='Using DAI in H2O AI Hybrid Cloud',
                            icon='Product',
                            icon_color='#CDDD38'
                        )
                    ],
                    justify='center'
                ),
                ui.separator(),
                ui.text(content='* Manages the backend Driverless AI connectivity & Steam integration under the hood'),
                ui.text(content='* Supports training, scoring, saving and loading of models'),
                ui.text(content='* Supports hyperparameter configuration and custom settings for model'),
                ui.text(content='* Seamlessly integrates with H2O MLOps for automatic deployment and retrieval'),
                ui.buttons(
                    items=[ui.button(name='demo_dai_cloud', label='Coming Soon!', primary=True, disabled=True)],
                    justify='center'
                )
            ]
        )

        return card

    @staticmethod
    def card_demo_dai_standalone() -> ui.FormCard:
        """
        Card for standalone Driverless AI demo.
        """

        card = ui.form_card(
            box='demo_dai_standalone',
            items=[
                ui.stats(
                    items=[
                        ui.stat(
                            label='',
                            value='Driverless AI (Standalone)',
                            caption='Using a publicly hosted DAI instance',
                            icon='Product',
                            icon_color='#CDDD38'
                        )
                    ],
                    justify='center'
                ),
                ui.separator(),
                ui.text(content='* Manages the backend Driverless AI connectivity under the hood'),
                ui.text(content='* Supports training, scoring, saving and loading of models'),
                ui.text(content='* Supports hyperparameter configuration and custom settings for model'),
                ui.text(content='* Seamlessly integrates with H2O MLOps for automatic deployment and retrieval'),
                ui.buttons(
                    items=[ui.button(name='demo_dai_cloud', label='Coming Soon!', primary=True, disabled=True)],
                    justify='center'
                )
            ]
        )

        return card

    @staticmethod
    def card_code_examples_heading() -> ui.FormCard:
        """
        Card for code examples heading.
        """

        card = ui.form_card(
            box='code_examples_heading',
            items=[
                ui.stats(
                    items=[
                        ui.stat(
                            label='',
                            value='Sample Code Snippets',
                            caption='Core functionalities of H2O Wave ML',
                            icon='ReadingMode',
                            icon_color='#CDDD38'
                        )
                    ],
                    justify='center'
                )
            ]
        )

        return card

    @staticmethod
    def card_code_examples(code_function: str, theme_dark: bool) -> ui.FormCard:
        """
        Card for code examples.
        """

        style = NeonStyle if theme_dark else LightStyle
        html_formatter = HtmlFormatter(full=True, style=style)

        path_snippet = Path('snippets') / f'{"_".join(code_function.lower().split())}.py'
        with open(path_snippet, encoding='utf-8') as f:
            code_snippet = f.read()

        card = ui.form_card(
            box=ui.box(zone='code_examples', height='565px'),
            items=[
                ui.separator(label='Functionalities'),
                ui.dropdown(
                    name='code_function',
                    label='Select function:',
                    choices=FUNCTION_CHOICES,
                    value=code_function,
                    trigger=True
                ),
                ui.label(label=f'{FUNCTION_DESCRIPTIONS[code_function]}:'),
                ui.inline(
                    items=[ui.text(content=highlight(code_snippet, PYTHON_LEXER, html_formatter))],
                    inset=True
                ),
                ui.inline(
                    items=[
                        ui.label(label='Read more: '),
                        FUNCTION_DOCUMENTATIONS[code_function]
                    ]
                )
            ]
        )

        return card

    @staticmethod
    def card_wave_examples_heading() -> ui.FormCard:
        """
        Card for Wave examples heading.
        """

        card = ui.form_card(
            box='wave_examples_heading',
            items=[
                ui.stats(
                    items=[
                        ui.stat(
                            label='',
                            value='Wave App Examples',
                            caption='Using H2O Wave ML in apps',
                            icon='ReadingMode',
                            icon_color='#CDDD38'
                        )
                    ],
                    justify='center'
                )
            ]
        )

        return card

    @staticmethod
    def card_h2o3_examples() -> ui.FormCard:
        """
        Card for H2O-3 examples.
        """

        card = ui.form_card(
            box=ui.box(zone='h2o3_examples', height='565px'),
            items=[
                ui.separator(label='H2O-3 AutoML'),
                ui.text(content='''<center>
                    <a href="https://wave.h2o.ai/docs/api/h2o_wave_ml/index" target="_blank">
                    Training & Prediction</a>
                    <br /><a href="https://wave.h2o.ai/docs/api/h2o_wave_ml/index" target="_blank">
                    Saving & Loading</a>
                    <br /><a href="https://wave.h2o.ai/docs/api/h2o_wave_ml/index" target="_blank">
                    Setting Categorical Columns</a>
                    <br /><a href="https://wave.h2o.ai/docs/api/h2o_wave_ml/index" target="_blank">
                    Configuring Hyperparameters</a>
                    <br /><a href="https://wave.h2o.ai/docs/api/h2o_wave_ml/index" target="_blank">
                    Extracting SHAP Values</a>
                    <br />
                    <br /><a href="https://github.com/h2oai/wave-apps/tree/main/credit-risk" target="_blank">
                    Credit Card Default Prediction</a>
                    <br /><a href="https://github.com/h2oai/wave-apps/tree/main/insurance-churn-risk" target="_blank">
                    Home Insurance Customer Churn Prediction</a>
                    <br /><a href="https://github.com/h2oai/wave-apps/tree/main/churn-risk" target="_blank">
                    Telecom Customer Churn Prediction</a>
                    </center>''')
            ]
        )

        return card

    @staticmethod
    def card_dai_examples() -> ui.FormCard:
        """
        Card for DAI examples.
        """

        card = ui.form_card(
            box=ui.box(zone='dai_examples', height='565px'),
            items=[
                ui.separator(label='Driverless AI'),
                ui.text(content='''<center>
                    Train & Predict
                    <br />Save & Load
                    <br />Categorical
                    <br />Hyperparameters
                    <br />AutoDoc
                    <br />Instances
                    </center>''')
            ]
        )

        return card
