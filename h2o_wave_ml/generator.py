import argparse
import fileinput
from pathlib import Path
import shutil
from typing import List, Dict, Any, Callable

import h2o

# Column name, type and attributes.
# A proper name for the column respecting the redundancy (python name, name for gui, ...).
# Picking a right component for column, input/output component.
# Default row - a row without missing values.


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


def generate_utility_file(output_dir: str, column_info: List[Dict[str, Any]]):
    file_path = Path(output_dir).joinpath('utility.py')
    with open(file_path, 'w') as f:
        f.write('from h2o_wave import ui\n\n')

        features = [c['name'] for c in column_info]
        f.write(f'features = {features}\n\n')

        f.write(f'choices = {{\n')
        for c in column_info:
            if c['type'] == 'enum':
                key = c['name']
                values = c['values']
                f.write(f'\t\'{key}\': [ui.choice(str(item)) for item in {values}],\n')
        f.write(f'}}\n\n')

        f.write(f'default_value = {{\n')
        for c in column_info:
            key = c['name']
            value = c['first_value']
            value = f'\'{value}\'' if c['type'] in ('enum', 'string') else value
            f.write(f'\t\'{key}\': {value},\n')
        f.write('}}\n\n')


def prepare_templates(template_dir: str, output_dir: str, dataset_src_path: str, target: str,
                      title: str, column_info: List[Dict[str, Any]]):

    index_src_path = Path(template_dir).joinpath('from_dataset.py')
    index_dst_path = Path(output_dir).joinpath('index.py')
    dataset_filename = Path(dataset_src_path).name

    shutil.copy2(index_src_path, index_dst_path)
    shutil.copy2(dataset_src_path, output_dir)

    t = lambda n: ' ' * n * 4  # Space generator.

    defaults = ''
    for c in column_info:
        if c['name'] == target:
            continue
        id_ = c['id']
        if c['type'] in ('int', 'real'):
            defaults += f'{t(1)}{id_} = q.args.{id_} if q.args.{id_} else default_value[\'{id_}\']\n'
        else:
            defaults += f'{t(1)}{id_} = q.args.{id_} or default_value[\'{id_}\']\n'

    names = [c['name'] for c in column_info if c['name'] != target]
    names = ', '.join(names)
    prediction = f'    input_data = [features, [{names}]]'

    form_items = ''
    for c in column_info:
        if c['name'] == target:
            continue
        id_ = c['id']
        name = c['name']
        if c['type'] == 'enum':
            form_items += f'{t(3)}ui.dropdown(name=\'{id_}\', label=\'{name}\', value={id_}, trigger= True, choices=choices[\'{id_}\']),\n'
        elif c['type'] == 'int':
            max_ = c['max']
            min_ = c['min']
            form_items += f'{t(3)}ui.slider(name=\'{id_}\', label=\'{name}\', min={min_}, max={max_}, step=1, value=float({id_}), trigger=True),\n'
        else:
            print('unknown type')

    for line in fileinput.input([index_dst_path], inplace=True):
        print(line.replace('__DATASET__', str(dataset_filename))
                  .replace('__TARGET__', target)
                  .replace('__DEFAULT_VALUES__', defaults)
                  .replace('__PREDICTION__', prediction)
                  .replace('__TITLE__', title)
                  .replace('__FORM_ITEMS__', form_items), end='')


def _get_type_specifics(frame: h2o.H2OFrame, type_: str) -> Dict[str, Any]:
    f = frame.na_omit()
    if type_ == 'enum':
        return {
            'nvalues': f.unique().nrows,
            'values': f.categories(),
            'first_value': f[0, 0],
        }
    elif type_ in ('int', 'real'):
        return {
            'nvalues': f.unique().nrows,
            'max': f.max(),
            'min': f.min(),
            'first_value': f[0, 0],
        }
    elif type_ == 'string':
        return {
            'first_value': f[0, 0],
        }
    elif type_ == 'time':
        return {
            'max': f.max(),
            'min': f.min(),
            'first_value': f[0, 0],
        }

    print(f'new type value {type_}')
    return {}


def _get_columns_info(frame: h2o.H2OFrame) -> List[Dict[str, Any]]:
    make_id = _make_new_id()
    return [
        {'id': make_id(name), 'name': name, 'type': type_, **_get_type_specifics(frame[name], type_)}
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

    h2o.init(url=args.h2o3_url)
    train_frame = h2o.import_file(args.from_dataset)

    cols = _get_columns_info(train_frame)
    if args.drop is not None:
        cols = [col for col in cols if col["name"] not in args.drop]

    prepare_templates(args.template_dir, args.output_dir, args.from_dataset, args.target_column,
                      args.title, cols)
    generate_utility_file(args.output_dir, cols)
