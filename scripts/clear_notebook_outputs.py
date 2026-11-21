"""Remove generated notebook output before committing; keep source cells intact."""

import argparse
import json
from pathlib import Path


def clear_outputs(path):
    notebook = json.loads(path.read_text())
    for cell in notebook.get('cells', []):
        if cell.get('attachments'):
            raise ValueError(f'{path}: embedded attachments must be moved out of the tracked notebook')
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
    notebook.get('metadata', {}).pop('widgets', None)
    path.write_text(json.dumps(notebook, indent=1) + '\n')
    print(f'Cleared generated outputs: {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebooks', type=Path, nargs='+')
    for path in parser.parse_args().notebooks:
        clear_outputs(path)
