import ast
import json
from pathlib import Path
import unittest


class NotebookSourceTests(unittest.TestCase):
    def test_tracked_notebooks_contain_valid_source_and_no_generated_images(self):
        root = Path(__file__).resolve().parents[1]
        notebooks = list((root / 'notebooks').glob('*.ipynb'))
        self.assertTrue(notebooks)
        for path in notebooks:
            notebook = json.loads(path.read_text())
            self.assertNotIn('widgets', notebook.get('metadata', {}))
            for cell in notebook['cells']:
                self.assertFalse(cell.get('attachments'), str(path))
                if cell['cell_type'] == 'code':
                    self.assertFalse(cell.get('outputs'), str(path))
                    self.assertIsNone(cell.get('execution_count'))
                    ast.parse(''.join(cell['source']), filename=str(path))


if __name__ == '__main__':
    unittest.main()
