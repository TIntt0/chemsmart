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


def test_scratch_path_preserves_equals_sign():
    executable = XTBExecutable(envars="export SCRATCH=/scratch/project=abc\n")
    assert executable.scratch_dir == "/scratch/project=abc"


def test_environment_recognises_export_keyword_and_whitespace():
    executable = XTBExecutable(
        envars="  export FIRST=one\nexport\tSECOND=two=three\n"
        "exported=ignored\nexport EXISTING\nexport\n"
    )
    assert executable.env == {"FIRST": "one", "SECOND": "two=three"}
