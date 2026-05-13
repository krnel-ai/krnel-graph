# Copyright (c) 2025-2026 Krnel

import sys
import textwrap

import pytest

from krnel.graph.cli import CommonParameters, app, parse_common_parameters
from krnel.graph.op_spec import OpSpec


def _write_custom_op_module(tmp_path, module_name, class_name):
    module_file = tmp_path / f"{module_name}.py"
    module_file.write_text(
        textwrap.dedent(
            f"""
            from krnel.graph.op_spec import OpSpec

            class {class_name}(OpSpec):
                pass
            """
        )
    )
    sys.path.insert(0, str(tmp_path))
    return module_file


def test_import_flag_registers_custom_op(tmp_path, monkeypatch):
    module_name = "krnel_test_custom_op_mod"
    class_name = "MyCustomTestOpForImport"
    _write_custom_op_module(tmp_path, module_name, class_name)

    monkeypatch.delitem(sys.modules, module_name, raising=False)
    assert all(
        c.__name__ != class_name for c in OpSpec.__subclasses__()
    ), "precondition: class is not yet registered"

    parse_common_parameters(CommonParameters(import_module=[module_name]))

    assert module_name in sys.modules
    assert any(c.__name__ == class_name for c in OpSpec.__subclasses__())


def test_import_flag_missing_module_exits_cleanly(capsys):
    with pytest.raises(SystemExit) as excinfo:
        parse_common_parameters(
            CommonParameters(
                import_module=["definitely_not_a_real_module_xyz_abc"]
            )
        )
    assert excinfo.value.code == 1
    captured = capsys.readouterr()
    assert "Failed to import module" in captured.out
    assert "definitely_not_a_real_module_xyz_abc" in captured.out


def test_import_flag_wired_through_cyclopts_app(tmp_path, monkeypatch):
    module_name = "krnel_test_custom_op_mod_via_app"
    class_name = "MyCustomTestOpViaApp"
    _write_custom_op_module(tmp_path, module_name, class_name)
    monkeypatch.delitem(sys.modules, module_name, raising=False)

    # Drive the cyclopts app end-to-end. We don't care whether the subcommand
    # ultimately succeeds (it will fail to find ops / may hit the storage
    # backend); we just need to confirm that --import ran before any of that.
    try:
        app(
            [
                "print",
                "--import",
                module_name,
                "-u",
                "NoSuchUUID_0000000000000000",
            ],
            exit_on_error=False,
        )
    except BaseException:
        pass

    assert module_name in sys.modules
    assert any(c.__name__ == class_name for c in OpSpec.__subclasses__())
