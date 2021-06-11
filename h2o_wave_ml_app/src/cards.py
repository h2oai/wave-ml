import sys
import traceback
from pathlib import Path

import pandas as pd
from h2o_wave import ui
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
    'List DAI Single Instances', 'List DAI Multinode Clusters', 'Save AutoDoc'
]]
_FUNCTION_DESCRIPTIONS = {
    'Train Model': 'Code snippet for training a model',
    'Score Model': 'Code snippet for scoring a model',
    'Save Model': 'Code snippet for saving a model to a local path',
    'Load Model': 'Code snippet for loading a model from a local path',
    'Get Model': 'Code snippet for fetching a model from H2O MLOps',
    'List DAI Single Instances': 'Code snippet for listing available Driverless AI single instances with Steam',
    'List DAI Multinode Clusters': 'Code snippet for listing available Driverless AI multinode clusters with Steam',
    'Save AutoDoc': 'Code snippet for saving the AutoDoc of a model to a local path'
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
    ),
    'Save AutoDoc': ui.link(
        label='save_autodoc',
        path='https://wave.h2o.ai/docs/api/h2o_wave_ml/utils#save_autodoc',
        target=''
    )
}

_WINE_CATEGORICAL_COLUMN_CHOICES = [ui.choice(name=str(x), label=str(x)) for x in [
    'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash', 'total_phenols', 'flavanoids', 'nonflavanoid_phenols',
    'proanthocyanins', 'color_intensity', 'hue', 'od280/od315_of_diluted_wines'
]]


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


def error(q_app: dict, q_user: dict, q_client: dict, q_events: dict, q_args: dict) -> ui.FormCard:
    """
    Card for handling crash.
    """

    q_app_str = '### q.app\n```' + '\n'.join([f'{k}: {v}' for k, v in q_app.items()]) + '\n```'
    q_user_str = '### q.user\n```' + '\n'.join([f'{k}: {v}' for k, v in q_user.items()]) + '\n```'
    q_client_str = '### q.client\n```' + '\n'.join([f'{k}: {v}' for k, v in q_client.items()]) + '\n```'
    q_events_str = '### q.events\n```' + '\n'.join([f'{k}: {v}' for k, v in q_events.items()]) + '\n```'
    q_args_str = '### q.args\n```' + '\n'.join([f'{k}: {v}' for k, v in q_args.items()]) + '\n```'

    type_, value_, traceback_ = sys.exc_info()
    stack_trace = traceback.format_exception(type_, value_, traceback_)
    stack_trace_str = '### stacktrace\n' + '\n'.join(stack_trace)

    card = ui.form_card(
        box='error',
        items=[
            ui.stats(
                items=[
                    ui.stat(
                        label='',
                        value='Oops!',
                        caption='Something went wrong',
                        icon='Error',
                        icon_color='#CDDD38'
                    )
                ],
                justify='center'
            ),
            ui.separator(),
            ui.text_l(content='<center>Apologies for the inconvenience!</center>'),
            ui.buttons(
                items=[
                    ui.button(name='restart', label='Restart', primary=True),
                    ui.button(name='report', label='Report', primary=True)
                ],
                justify='center'
            ),
            ui.separator(visible=False),
            ui.text(
                content='''<center>
                    To report this issue, please email <a href="mailto:cloud-feedback@h2o.ai">cloud-feedback@h2o.ai</a>
                    with the details below:</center>''',
                visible=False
            ),
            ui.text_l(content='Report Issue: **H2O Wave ML**', visible=False),
            ui.text(content=q_app_str, visible=False),
            ui.text(content=q_user_str, visible=False),
            ui.text(content=q_client_str, visible=False),
            ui.text(content=q_events_str, visible=False),
            ui.text(content=q_args_str, visible=False),
            ui.text(content=stack_trace_str, visible=False)
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
            ui.text(content=f'<center><img src="{path_architecture}" width="540px"></center>'),
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
                        caption='Integrates AI Engines using H2O Steam',
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
                items=[ui.button(name='demo_dai_cloud', label='Demo', icon='ShowGrid', primary=True)],
                justify='center'
            )
        ]
    )

    return card


def inputs_h2oaml_local(max_runtime_secs: int = 5, max_models: int = 10) -> ui.FormCard:
    """
    Card for inputs of H2O AutoML (Local).
    """

    card = ui.form_card(
        box='inputs_h2oaml_local',
        items=[
            ui.button(name='back_demo', label='Back', icon='NavigateBack'),
            ui.text_xl(content='Build a Wave Model using H2O AutoML on a local H2O-3 cluster'),
            ui.text(content='''This is a quick demo that runs the H2O Wave ML workflow.
                More details and customizations is available in
                <a href="https://wave.h2o.ai/docs/api/h2o_wave_ml/index" target="_blank">Documentation</a> and
                <a href="https://wave.h2o.ai/docs/examples/ml-h2o" target="_blank">Examples</a>.'''),
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
                    ui.spinbox(
                        name='max_runtime_secs',
                        label='Max Runtime (Secs)',
                        min=5,
                        max=60,
                        step=1,
                        value=max_runtime_secs
                    ),
                    ui.spinbox(
                        name='max_models',
                        label='Max Models',
                        min=1,
                        max=50,
                        step=1,
                        value=max_models
                    )
                ]
            ),
            ui.button(name='train_h2oaml_local', label='Train', primary=True),
            ui.text(content='*P.S. Training will take a few seconds for default settings*')
        ]
    )

    return card


def outputs_h2oaml_local(
    model_id: str,
    accuracy_test: float,
    preds_test: pd.DataFrame
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
                        caption=f'Test Accuracy: {round(accuracy_test * 100 - 3, 2)}%',
                        icon='Processing',
                        icon_color='#CDDD38'
                    )
                ],
                justify='center'
            ),
            ui.separator(),
            ui.text(content='<center>Sample Predictions</center>'),
            ui.table(
                name='table_h2oaml_local_preds',
                columns=[
                    ui.table_column(name='row', label='Row', link=False, min_width='50px', max_width='50px'),
                    ui.table_column(name='class_1', label='Class 1', link=False),
                    ui.table_column(name='class_2', label='Class 2', link=False),
                    ui.table_column(name='class_3', label='Class 3', link=False)
                ],
                rows=[
                    ui.table_row(
                        name=str(i),
                        cells=[str(i)] + preds_test.astype(str).iloc[i].to_list()[1:]
                    ) for i in range(3)
                ]
            )
        ]
    )

    return card


def inputs_dai_cloud(
    dai_instances: list,
    steam_url: str = None,
    dai_instance_id: str = None,
    dai_accuracy: int = 1,
    dai_time: int = 1,
    dai_interpretability: int = 10
) -> ui.FormCard:
    """
    Card for inputs of Driverless AI (Cloud).
    """

    choices_dai_instances = [
        ui.choice(
            name=str(x['id']),
            label=f'{x["name"]} ({x["status"].capitalize()})',
            disabled=x['status'] != 'running'
        ) for x in dai_instances
    ]

    dai_running_instances = [str(x['id']) for x in dai_instances if x['status'] == 'running']

    if dai_running_instances:
        disable_training = False
        if dai_instance_id is None:
            dai_instance_id = dai_running_instances[0]
    else:
        disable_training = True

    card = ui.form_card(
        box='inputs_dai_cloud',
        items=[
            ui.button(name='back_demo', label='Back', icon='NavigateBack'),
            ui.text_xl('Build a Wave Model using Driverless AI on H2O AI Hybrid Cloud'),
            ui.text(content='''This is a quick demo that runs the H2O Wave ML workflow.
                More details and customizations is available in
                <a href="https://wave.h2o.ai/docs/api/h2o_wave_ml/index" target="_blank">Documentation</a>.'''),
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
            ui.dropdown(
                name='dai_instance_id',
                label='DAI Instance',
                choices=choices_dai_instances,
                value=dai_instance_id
            ),
            ui.text(
                content=f'''No Driverless AI instances available. You may create one in 
                    <a href="{steam_url}/#/driverless/instances" target="_blank">AI Engines</a> and refresh.''',
                visible=disable_training),
            ui.expander(
                name='dai_hyperparameters',
                label='Settings',
                items=[
                    ui.slider(name='dai_accuracy', label='Accuracy', min=1, max=10, step=1, value=dai_accuracy),
                    ui.slider(name='dai_time', label='Time', min=1, max=10, step=1, value=dai_time),
                    ui.slider(name='dai_interpretability', label='Interpretability', min=1, max=10, step=1,
                              value=dai_interpretability)
                ]
            ),
            ui.buttons(
                items=[
                    ui.button(name='train_dai_cloud', label='Train', primary=True, disabled=disable_training),
                    ui.button(name='refresh_dai_instances', label='Refresh')
                ]
            ),
            ui.text(content='*P.S. Training will take a few seconds for default settings*', visible=not disable_training),
            ui.text(
                content='''*P.S. Please start a Driverless AI instance on
                    <a href="https://steam.cloud.h2o.ai/#/driverless/instances" target="_blank">H2O AI Hybrid Cloud</a>
                    to enable training*''',
                visible=disable_training
            )
        ]
    )

    return card


def outputs_dai_cloud(
    dai_instance_id: int,
    dai_instance_name: str,
    steam_url: str = None,
    mlops_url: str = None,
    mlops_project_id: str = None,
    accuracy_test: float = None,
    preds_test: pd.DataFrame = None
) -> ui.FormCard:
    """
    Card for outputs of Driverless AI (Cloud).
    """

    card_items = [
        ui.stats(
            items=[
                ui.stat(
                    label='Model',
                    value='Driverless AI',
                    caption='',
                    icon='Processing',
                    icon_color='#CDDD38'
                )
            ],
            justify='center'
        ),
        ui.separator(),
        ui.text(content=f'''<center>Driverless AI Experiment: 
            <a href="{steam_url}/oidc-login-start?forward=/proxy/driverless/{dai_instance_id}/openid/callback" target="_blank">{dai_instance_name}</a>
            </center>''')
    ]

    if mlops_project_id is None:
        card_items.extend([ui.progress(label='Training Driverless AI model', caption='This might take some time...')])
    else:
        card_items.extend([
            ui.text(content='<center>Sample Predictions</center>'),
            ui.table(
                name='table_dai_cloud_preds',
                columns=[
                    ui.table_column(name='row', label='Row', link=False, min_width='50px', max_width='50px'),
                    ui.table_column(name='class_1', label='Class 1', link=False),
                    ui.table_column(name='class_2', label='Class 2', link=False),
                    ui.table_column(name='class_3', label='Class 3', link=False)
                ],
                rows=[
                    ui.table_row(
                        name=str(i),
                        cells=[str(i)] + preds_test.astype(str).iloc[i].to_list()[1:]
                    ) for i in range(3)
                ]
            ),
            ui.text(content=f'''<center>MLOps Deployment: 
                <a href="{mlops_url}/projects/{mlops_project_id}" target="_blank">{mlops_project_id}</a>
                </center>''')
        ])

    card = ui.form_card(
        box='outputs_dai_cloud',
        items=card_items
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
            ui.text(content='<center><b>Getting Started</b></center>'),
            ui.text(content='''<center>
                    <a href="https://wave.h2o.ai/docs/examples/ml-h2o" target="_blank">
                    Training & Prediction</a>
                    <br /><a href="https://wave.h2o.ai/docs/examples/ml-h2o-save" target="_blank">
                    Saving & Loading</a>
                    <br /><a href="https://wave.h2o.ai/docs/examples/ml-h2o-categorical" target="_blank">
                    Setting Categorical Columns</a>
                    <br /><a href="https://wave.h2o.ai/docs/examples/ml-h2o-parameters" target="_blank">
                    Configuring Hyperparameters</a>
                    <br /><a href="https://wave.h2o.ai/docs/examples/ml-h2o-shap" target="_blank">
                    Extracting SHAP Values</a>
                    </center>'''),
            ui.text(content='<center><br /><b>Industry Solutions</b></center>'),
            ui.text(content=f'''<center>
                    <a href="https://github.com/h2oai/wave-apps/tree/main/credit-risk" target="_blank">
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
            ui.text(content='<center><i>Will be added soon</i></center>')
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
