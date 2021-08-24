import argparse
import fileinput
from pathlib import Path
import shutil
import sys
from typing import List, Dict, Any, Callable, Tuple, NamedTuple

import h2o


class ColumnMeta(NamedTuple):
    """A column description."""
    id: str
    name: str
    type: str
    rnd_value: str
    values: List[str] = []
    nvalues: int = 0
    min: float = 0
    max: float = 0


class _Buffer:

    _INDENT_SPACES = 4

    def __init__(self):
        self._buffer = ''

    @classmethod
    def _indent(cls, n: int):
        return ' ' * n * cls._INDENT_SPACES

    def p(self, text: str = '', tab: int = 0):
        self._buffer += self._indent(tab)
        self._buffer += f'{text}\n'

    def __str__(self):
        return self._buffer


def _make_new_id() -> Callable[[str], str]:
    """Provide a safe way to generate new id based on a given-existing name.
    Returns:
        Function responsible for a name generation.
    Examples:
        >>> new_id = make_new_id()
        >>> new_id('My name')
        'my_name'
        >>> new_id('My other name')
        'my_other_name'
        >>> new_id('My name')
        'my_name_1'
    """
    ids = []

    def prepare_id(id_: str) -> str:
        return id_.replace(' ', '_').lower()

    def new_id(id_: str) -> str:
        id_prepared = prepare_id(id_)
        candidate = id_prepared
        suffix = 1
        while True:
            if candidate not in ids:
                ids.append(candidate)
                return candidate
            candidate = f'{id_prepared}_{suffix}'
            suffix += 1

    return new_id


def generate_utility_file(output_dir: str, target: str, columns: List[ColumnMeta]):
    b = _Buffer()
    file_path = Path(output_dir).joinpath('utility.py')
    with open(file_path, 'w') as f:
        b.p('from h2o_wave import ui')
        b.p()

        features = [c.name for c in columns if c.name != target]
        b.p(f'features = {features}')
        b.p()

        b.p('choices = {')
        for c in columns:
            if c.type == 'enum':
                key = c.name
                values = c.values
                b.p(f'\'{key}\': [ui.choice(str(item)) for item in {values}],', tab=1)
        b.p('}')
        b.p()

        b.p('default_value = {')
        for c in columns:
            key = c.name
            value = c.rnd_value
            value = f'\'{value}\'' if c.type in ('enum', 'string') else value
            b.p(f'\'{key}\': {value},', tab=1)
        b.p('}')
        b.p()

        f.write(str(b))


def _prepare_defaults(target: str, columns: List[ColumnMeta]) -> str:
    b = _Buffer()
    for c in columns:
        if c.name == target:
            continue
        if c.type in ('int', 'real'):
            b.p(f'{c.id} = q.args.{c.id} if q.args.{c.id} else default_value[\'{c.id}\']', tab=1)
        else:
            b.p(f'{c.id} = q.args.{c.id} or default_value[\'{c.id}\']', tab=1)
    return str(b)


def _prepare_inputs(target: str, columns: List[ColumnMeta]) -> str:
    names = [c.name for c in columns if c.name != target]
    names = ', '.join(names)
    return f'    input_data = [features, [{names}]]\n'


def _prepare_form_items(target: str, columns: List[ColumnMeta]) -> str:
    b = _Buffer()
    for c in columns:
        if c.name == target:
            continue
        if c.type == 'enum':
            # See `choice_group` for another component.
            b.p(f'ui.dropdown(name=\'{c.id}\', label=\'{c.name}\', value={c.id}, trigger=True, choices=choices[\'{c.id}\']),', tab=3)
        elif c.type == 'int':
            # See `spinbox` when trigger option is ready.
            b.p(f'ui.slider(name=\'{c.id}\', label=\'{c.name}\', min={c.min}, max={c.max}, step=1, value=int({c.id}), trigger=True),', tab=3)
        elif c.type == 'real':
            # See `spinbox` when trigger option is ready.
            b.p(f'ui.slider(name=\'{c.id}\', label=\'{c.name}\', min={c.min}, max={c.max}, step=0.2, value=int({c.id}), trigger=True),', tab=3)
        elif c.type == 'string':
            b.p(f'ui.textbox(name=\'{c.id}\', label=\'{c.name}\', value=str({c.id}), trigger=True)')
        elif c.type == 'time':
            b.p(f'ui.datepicker(name=\'{c.id}\', label=\'{c.name}\', value=str({c.id}, trigger=True)')
        else:
            print('unknown type')
    return str(b)


def _prepare_output(target: str, columns: List[ColumnMeta]) -> Tuple[str, str]:
    # String, Enum - text, markdown, Info / Wide
    b = _Buffer()
    u = _Buffer()
    for c in columns:
        if c.name == target:
            if c.type in ('int', 'real'):
                b.p('q.page[\'result\'] = ui.tall_gauge_stat_card(', tab=2)
                b.p('box=ui.box(\'body\', height=\'180px\'),', tab=3)
                b.p('value=str(score),', tab=3)
                b.p(f'aux_value=\'{target}\',', tab=3)
                b.p('title=\'Result\',', tab=3)
                b.p(f'progress=float(score)/{c.max},', tab=3)
                b.p(')', tab=2)
                u.p('q.page[\'result\'].value = str(score)', tab=2)
                u.p(f'q.page[\'result\'].progress = float(score)/{c.max}', tab=2)
            elif c.type in ('string', 'enum'):
                b.p('page[\'result\'] = ui.form_card(', tab=2)
                b.p('box=ui.box(\'body\', height=\'180px\'),', tab=3)
                b.p('items=[ui.text_l(score)],', tab=3)
                b.p(')', tab=2)
            else:
                print('unknown type')
            break
    return str(b), str(u)


def prepare_templates(template_dir: str, output_dir: str, dataset_src_path: str, target: str,
                      title: str, columns: List[ColumnMeta]):

    makefile_src_path = Path(template_dir).joinpath('Makefile')
    makefile_dst_path = Path(output_dir).joinpath('Makefile')
    index_src_path = Path(template_dir).joinpath('from_dataset.py')
    index_dst_path = Path(output_dir).joinpath('index.py')
    dataset_filename = Path(dataset_src_path).name

    shutil.copy2(makefile_src_path, makefile_dst_path)
    shutil.copy2(index_src_path, index_dst_path)
    shutil.copy2(dataset_src_path, output_dir)

    defaults = _prepare_defaults(target, columns)
    inputs = _prepare_inputs(target, columns)
    form_items = _prepare_form_items(target, columns)
    output_card, score_update = _prepare_output(target, columns)  # Choose better name.

    for line in fileinput.input([index_dst_path], inplace=True):
        print(line.replace('__DATASET__', str(dataset_filename))
                  .replace('__TARGET__', target)
                  .replace('__DEFAULT_VALUES__\n', defaults)
                  .replace('__INPUTS__\n', inputs)
                  .replace('__TITLE__', title)
                  .replace('__OUTPUT_CARD__\n', output_card)
                  .replace('__FORM_ITEMS__\n', form_items)
                  .replace('__SCORE_UPDATE__\n', score_update), end='')


def _get_type_specifics(frame: h2o.H2OFrame, type_: str) -> Dict[str, Any]:
    f = frame.na_omit()
    if type_ == 'enum':
        return {
            'nvalues': f.unique().nrows,
            'values': f.categories(),
            'rnd_value': f[0, 0],
        }
    elif type_ in ('int', 'real'):
        return {
            'nvalues': f.unique().nrows,
            'max': f.max(),
            'min': f.min(),
            'rnd_value': f[0, 0],
        }
    elif type_ == 'string':
        return {
            'rnd_value': f[0, 0],
        }
    elif type_ == 'time':
        return {
            'max': f.max(),
            'min': f.min(),
            'rnd_value': f[0, 0],
        }

    print(f'new type value {type_}')
    return {}


def _get_columns_info(frame: h2o.H2OFrame) -> List[ColumnMeta]:
    make_id = _make_new_id()
    return [
        ColumnMeta(**{'id': make_id(name), 'name': name, 'type': type_, **_get_type_specifics(frame[name], type_)})
        for name, type_ in frame.types.items()
    ]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate a Wave app based on a dataset', epilog='Have fun!')
    parser.add_argument('target_column', type=str, help='a column to be predicted')
    parser.add_argument('--h2o3-url', type=str, help='an H2O-3 instance to use (if any)')
    parser.add_argument('--from-dataset', type=str, help='a dataset to examine')
    parser.add_argument('--output-dir', type=str, help='a directory to save the generated project to', default='./')
    parser.add_argument('--template-dir', type=str, help='a directory containing the templates for generator', default='./templates')
    parser.add_argument('--drop', type=str, metavar='column', nargs='*', help='a list of column names to skip')
    parser.add_argument('--title', type=str, help='a title for the application', default='My app')
    args = parser.parse_args()

    if args.from_dataset is None:
        print('no dataset to inspect')
        sys.exit(1)

    if not Path(args.output_dir).exists():
        print('output directory does not exist')
        sys.exit(1)

    h2o.init(url=args.h2o3_url)
    train_frame = h2o.import_file(args.from_dataset)

    cols = _get_columns_info(train_frame)
    if args.drop is not None:
        cols = [c for c in cols if c.name not in args.drop]

    prepare_templates(args.template_dir, args.output_dir, args.from_dataset, args.target_column,
                      args.title, cols)
    generate_utility_file(args.output_dir, args.target_column, cols)
