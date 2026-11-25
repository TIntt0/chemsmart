"""Executable environment values must retain embedded assignment characters."""

from io import StringIO

from chemsmart.settings.executable import XTBExecutable
from chemsmart.settings.submitters import SLURMSubmitter


def test_environment_value_with_equals_is_preserved():
    executable = XTBExecutable(
        envars='export XTB_OPTIONS="mode=fast threads=1"\nexport EMPTY=\n'
    )
    assert executable.env == {
        "XTB_OPTIONS": '"mode=fast threads=1"',
        "EMPTY": "",
    }
    stream = StringIO()
    SLURMSubmitter._write_executable_configuration(stream, executable)
    assert 'export XTB_OPTIONS="mode=fast threads=1"\n' in stream.getvalue()
    assert "export EMPTY=\n" in stream.getvalue()
