"""Frozen task manifests remain valid after relocating the project checkout."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.toolmaze_agent.prepare import load_task


class FrozenManifestPortabilityTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.relative = 'data/toolmaze/perturbed_tasks/task.json'
        task = self.root/self.relative
        task.parent.mkdir(parents=True)
        task.write_text(json.dumps({'task_id': 'portable-test'}))
        self.sha = hashlib.sha256(task.read_bytes()).hexdigest()

    def test_old_absolute_path_resolves_without_rewriting_manifest(self):
        row = {'path': '/nonexistent-old-server/program/'+self.relative, 'sha256': self.sha}
        before = dict(row)
        with patch('scripts.toolmaze_agent.common.ROOT', self.root):
            self.assertEqual(load_task(row), {'task_id': 'portable-test'})
        self.assertEqual(row, before)

    def test_relocation_still_checks_original_task_hash(self):
        row = {'path': '/nonexistent-old-server/program/'+self.relative, 'sha256': '0'*64}
        with patch('scripts.toolmaze_agent.common.ROOT', self.root):
            with self.assertRaisesRegex(ValueError, 'Task changed'):
                load_task(row)

    def test_relative_path_works(self):
        with patch('scripts.toolmaze_agent.common.ROOT', self.root):
            self.assertEqual(load_task({'path': self.relative, 'sha256': self.sha}),
                             {'task_id': 'portable-test'})


if __name__ == '__main__':
    unittest.main()
