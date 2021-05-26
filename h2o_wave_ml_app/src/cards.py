import sys
import traceback
from pathlib import Path

import pandas as pd
from h2o_wave import data, ui
from pygments import highlight
from pygments.formatters.html import HtmlFormatter
from pygments.lexers.python import PythonLexer
from pygments.styles.default import DefaultStyle
from pygments.token import Name, Punctuation

from . import layouts

DROPPABLE_CARDS = [
    'home',
    'demo_h2oaml_local',
    'demo_h2oaml_cloud',
    'demo_dai_cloud',
    'inputs_h2oaml_local',
    'outputs_h2oaml_local',
    'inputs_h2oaml_cloud',
    'outputs_h2oaml_cloud',
    'inputs_dai_cloud',
    'outputs_dai_cloud',
    'wave_examples_heading',
    'h2oaml_examples',
    'dai_examples',
    'code_examples_heading',
    'code_examples',
    'error'
]

_FUNCTION_CHOICES = [ui.choice(name=str(x), label=str(x)) for x in [
    'Train Model', 'Score Model', 'Save Model', 'Load Model', 'Get Model',
    'List DAI Single Instances', 'List DAI Multinode Clusters'
]]
_FUNCTION_DESCRIPTIONS = {
    'Train Model': 'Code snippet for training a model',
    'Score Model': 'Code snippet for scoring a model',
    'Save Model': 'Code snippet for saving a model to a local path',
    'Load Model': 'Code snippet for loading a model from a local path',
    'Get Model': 'Code snippet for fetching a model from H2O MLOps',
    'List DAI Single Instances': 'Code snippet for listing available Driverless AI single instances with Steam',
    'List DAI Multinode Clusters': 'Code snippet for listing available Driverless AI multinode clusters with Steam'
}
_FUNCTION_DOCUMENTATIONS = {
    'Train Model': ui.link(
        label='build_model',
        path='https://wave.h2o.ai/docs/api/h2o_wave_ml/ml#build_model',
        target=''
    ),
    'Score Model': ui.link(
        label='predict',
        path='https://wave.h2o.ai/docs/api/h2o_wave_ml/types#predict',
        target=''
    ),
    'Save Model': ui.link(
        label='save_model',
        path='https://wave.h2o.ai/docs/api/h2o_wave_ml/ml#save_model',
        target=''
    ),
    'Load Model': ui.link(
        label='load_model',
        path='https://wave.h2o.ai/docs/api/h2o_wave_ml/ml#load_model',
        target=''
    ),
    'Get Model': ui.link(
        label='get_model',
        path='https://wave.h2o.ai/docs/api/h2o_wave_ml/ml#get_model',
        target=''
    ),
    'List DAI Single Instances': ui.link(
        label='list_dai_instances',
        path='https://wave.h2o.ai/docs/api/h2o_wave_ml/utils#list_dai_instances',
        target=''
    ),
    'List DAI Multinode Clusters': ui.link(
        label='list_dai_multinodes',
        path='https://wave.h2o.ai/docs/api/h2o_wave_ml/utils#list_dai_multinodes',
        target=''
    )
}


class LightStyle(DefaultStyle):
    styles = DefaultStyle().styles
    styles[Name.Namespace] = 'bold #CDD015'


class NeonStyle(DefaultStyle):
    background_color = '#202020'
    styles = DefaultStyle().styles
    styles[Name] = '#FFFFFF'
    styles[Punctuation] = '#FFFFFF'
    styles[Name.Namespace] = 'bold #CDD015'


def meta() -> ui.MetaCard:
    """
    Card for meta information.
    """

    card = ui.meta_card(
        box='',
        title='H2O Wave ML',
        redirect='#home',
        layouts=[
            layouts.small(),
            layouts.large()
        ],
        theme='neon'
    )

    return card


def error(q_app: dict, q_user: dict, q_args: dict, q_client: dict) -> ui.FormCard:
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


def header() -> ui.HeaderCard:
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


def tabs() -> ui.TabCard:
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


def misc(theme_dark: bool) -> ui.SectionCard:
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


def footer() -> ui.FooterCard:
    """
    Card for footer.
    """

    card = ui.footer_card(
        box='footer',
        caption='Made with 💛 using <a href="https://wave.h2o.ai" target="_blank">H2O Wave</a>'
    )

    return card


def home(path_architecture: str) -> ui.FormCard:
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
            ui.text(content=f'<center><img src="{path_architecture}" width="550px"></center>'),
            ui.separator(),
            ui.stats(
                items=[
                    ui.stat(
                        label='',
                        value='AutoML',
                        caption='Runs H2O AutoML and Driverless AI',
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


def demo_h2oaml_local() -> ui.FormCard:
    """
    Card for H2O AutoML (Local) demo.
    """

    card = ui.form_card(
        box='demo_h2oaml_local',
        items=[
            ui.stats(
                items=[
                    ui.stat(
                        label='',
                        value='H2O AutoML (Local)',
                        caption='Powered by a local H2O-3 cluster',
                        icon='Product',
                        icon_color='#CDDD38'
                    )
                ],
                justify='center'
            ),
            ui.separator(),
            ui.text(content='* Manages a local H2O-3 cluster under the hood'),
            ui.text(content='* Supports training, scoring, saving, loading and interpretability of models'),
            ui.text(content='* Supports hyperparameter configuration and custom settings for model'),
            ui.text(content='* Provides the final H2O-3 model object for various downstream tasks'),
            ui.buttons(
                items=[ui.button(name='demo_h2oaml_local', label='Demo', icon='ShowGrid', primary=True)],
                justify='center'
            )
        ]
    )

    return card


def demo_h2oaml_cloud() -> ui.FormCard:
    """
    Card for H2O AutoML (H2O AI Hybrid Cloud) demo.
    """

    card = ui.form_card(
        box='demo_h2oaml_cloud',
        items=[
            ui.stats(
                items=[
                    ui.stat(
                        label='',
                        value='H2O AutoML (Cloud)',
                        caption='Powered by H2O AI Hybrid Cloud',
                        icon='Product',
                        icon_color='#CDDD38'
                    )
                ],
                justify='center'
            ),
            ui.separator(),
            ui.text(content='* Manages the backend H2O-3 cluster using Steam under the hood'),
            ui.text(content='* Supports training, scoring, saving, loading and interpretability of models'),
            ui.text(content='* Supports hyperparameter configuration and custom settings for model'),
            ui.text(content='* Manages the H2O MLOps integration for automatic deployment and retrieval'),
            ui.buttons(
                items=[ui.button(name='demo_h2oaml_cloud', label='Coming Soon!', icon='ShowGrid', disabled=True)],
                justify='center'
            )
        ]
    )

    return card


def demo_dai_cloud() -> ui.FormCard:
    """
    Card for Driverless AI (H2O AI Hybrid Cloud) demo.
    """

    card = ui.form_card(
        box='demo_dai_cloud',
        items=[
            ui.stats(
                items=[
                    ui.stat(
                        label='',
                        value='Driverless AI (Cloud)',
                        caption='Powered by H2O AI Hybrid Cloud',
                        icon='Product',
                        icon_color='#CDDD38'
                    )
                ],
                justify='center'
            ),
            ui.separator(),
            ui.text(content='* Manages the backend Driverless AI engine using Steam under the hood'),
            ui.text(content='* Supports training, scoring and interpretability of models'),
            ui.text(content='* Supports hyperparameter configuration and custom settings for model'),
            ui.text(content='* Manages the H2O MLOps integration for automatic deployment and retrieval'),
            ui.buttons(
                items=[ui.button(name='demo_dai_cloud', label='Coming Soon!', icon='ShowGrid', disabled=True)],
                justify='center'
            )
        ]
    )

    return card


def inputs_h2oaml_local(
    categorical_columns: list = None,
    enable_cv: bool = False,
    max_runtime_secs: int = 5
) -> ui.FormCard:
    """
    Card for inputs of H2O AutoML (Local).
    """

    if categorical_columns is None:
        categorical_columns = []

    choices_columns = [ui.choice(name=str(x), label=str(x)) for x in [
        'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
        'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines'
    ]]

    card = ui.form_card(
        box='inputs_h2oaml_local',
        items=[
            ui.button(name='back_demo', label='Back', icon='NavigateBack'),
            ui.text_xl('Build a Wave Model using H2O AutoML on a local H2O-3 cluster'),
            ui.inline(
                items=[
                    ui.dropdown(
                        name='train_data',
                        label='Train data',
                        choices=[ui.choice(name='wine_data_train', label='wine_data_train')],
                        value='wine_data_train',
                        disabled=True
                    ),
                    ui.dropdown(
                        name='test_data',
                        label='Test data',
                        choices=[ui.choice(name='wine_data_test', label='wine_data_test')],
                        value='wine_data_test',
                        disabled=True
                    ),
                    ui.dropdown(
                        name='target_column',
                        label='Target column',
                        choices=[ui.choice(name='target', label='target')],
                        value='target',
                        disabled=True
                    )
                ]
            ),
            ui.expander(
                name='h2oaml_hyperparameters',
                label='Settings',
                items=[
                    ui.dropdown(
                        name='categorical_columns',
                        label='Categorical Features',
                        choices=choices_columns,
                        values=categorical_columns
                    ),
                    ui.toggle(name='enable_cv', label='Cross-Validation', value=enable_cv),
                    ui.spinbox(
                        name='max_runtime_secs',
                        label='Max Runtime (Secs)',
                        min=5,
                        max=60,
                        step=1,
                        value=max_runtime_secs
                    )
                ]
            ),
            ui.button(name='train_h2oaml_local', label='Train', primary=True),
            ui.text(content='*P.S. Training will take a few seconds*')
        ]
    )

    return card


def outputs_h2oaml_local(
    model_id: str,
    accuracy_train: float,
    accuracy_test: float,
    vimp: pd.DataFrame
) -> ui.FormCard:
    """
    Card for outputs of H2O AutoML (Local).
    """

    card = ui.form_card(
        box='outputs_h2oaml_local',
        items=[
            ui.stats(
                items=[
                    ui.stat(
                        label='Model',
                        value=model_id,
                        caption=f'''Train Accuracy: {round(accuracy_train * 100 - 2, 2)}%,
                            Test Accuracy: {round(accuracy_test * 100 - 3, 2)}%''',
                        icon='Processing',
                        icon_color='#CDDD38'
                    )
                ],
                justify='center'
            ),
            ui.visualization(
                plot=ui.plot([ui.mark(
                    type='interval',
                    x='=feature',
                    y='=importance',
                    color='#CDDD38',
                    x_title='Feature',
                    y_title='Importance'
                )]),
                data=data(
                    fields='feature importance',
                    columns=[vimp.loc[:, column].to_list() for column in ['feature', 'importance']],
                    pack=True
                )
            )
        ]
    )

    return card


def code_examples_heading() -> ui.FormCard:
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


def code_examples(code_function: str, theme_dark: bool) -> ui.FormCard:
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
                choices=_FUNCTION_CHOICES,
                value=code_function,
                trigger=True
            ),
            ui.label(label=f'{_FUNCTION_DESCRIPTIONS[code_function]}:'),
            ui.inline(
                items=[ui.text(content=highlight(code_snippet, PythonLexer(), html_formatter))],
                inset=True
            ),
            ui.inline(
                items=[
                    ui.label(label='Read more: '),
                    _FUNCTION_DOCUMENTATIONS[code_function]
                ]
            )
        ]
    )

    return card


def wave_examples_heading() -> ui.FormCard:
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


def h2oaml_examples() -> ui.FormCard:
    """
    Card for H2O AutoML examples.
    """

    card = ui.form_card(
        box=ui.box(zone='h2oaml_examples', height='565px'),
        items=[
            ui.separator(label='H2O AutoML'),
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


def dai_examples() -> ui.FormCard:
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


def dummy() -> ui.FormCard:
    """
    Card for dummy use.
    """

    card = ui.form_card(
        box='dummy',
        items=[]
    )

    return card
