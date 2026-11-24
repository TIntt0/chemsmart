"""Plugin installation command construction and failures."""

import subprocess
import sys
import unittest
from unittest.mock import patch

from click.testing import CliRunner

from chemsmart.cli.install import install


class TestInstall(unittest.TestCase):
    def test_default_install(self):
        with patch("chemsmart.cli.install.subprocess.run") as run:
            result = CliRunner().invoke(install, ["ml"])
        self.assertEqual(result.exit_code, 0, result.output)
        run.assert_called_once_with(
            [sys.executable, "-m", "pip", "install", "chemsmart-ml"],
            check=True,
        )

    def test_local_backend_upgrade(self):
        runner = CliRunner()
        with runner.isolated_filesystem():
            from pathlib import Path

            source = str(Path.cwd())
            with patch("chemsmart.cli.install.subprocess.run") as run:
                result = runner.invoke(
                    install, ["ml", "mace", "--source", source, "--upgrade"]
                )
        self.assertEqual(result.exit_code, 0, result.output)
        run.assert_called_once_with(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--upgrade",
                source + "[mace]",
            ],
            check=True,
        )

    def test_failure_is_reported(self):
        with patch("chemsmart.cli.install.subprocess.run") as run:
            run.side_effect = subprocess.CalledProcessError(1, "pip")
            result = CliRunner().invoke(install, ["ml"])
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("Installation failed", result.output)
        self.assertNotIn("Next:", result.output)

    def test_unknown_backend_does_not_install(self):
        with patch("chemsmart.cli.install.subprocess.run") as run:
            result = CliRunner().invoke(install, ["ml", "unknown"])
        self.assertNotEqual(result.exit_code, 0)
        run.assert_not_called()
