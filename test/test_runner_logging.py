"""Tests for runner logging setup: setup_logging function."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest


class TestSetupLogging:
    """Verify setup_logging creates correct file and stream handlers."""

    def test_creates_file_handler(self, tmp_path: Path) -> None:
        """setup_logging should create a FileHandler at log_dir/pipeline.log."""
        # Reset root logger before test
        root = logging.getLogger()
        root.handlers.clear()

        from runners.py.runner import setup_logging

        log_dir = tmp_path / 'runs' / 'test_run'
        setup_logging(log_dir=log_dir)

        root = logging.getLogger()
        file_handlers = [
            h for h in root.handlers if isinstance(h, logging.FileHandler)
        ]
        assert len(file_handlers) == 1
        assert Path(file_handlers[0].baseFilename).name == 'pipeline.log'  # type: ignore[arg-type]

    def test_file_handler_info_level(self, tmp_path: Path) -> None:
        """FileHandler should be configured at INFO level."""
        root = logging.getLogger()
        root.handlers.clear()

        from runners.py.runner import setup_logging

        log_dir = tmp_path / 'runs' / 'test_run'
        setup_logging(log_dir=log_dir)

        root = logging.getLogger()
        file_handlers = [
            h for h in root.handlers if isinstance(h, logging.FileHandler)
        ]
        assert file_handlers[0].level == logging.INFO

    def test_stream_handler_warning_level(self, tmp_path: Path) -> None:
        """StreamHandler should be configured at WARNING level."""
        root = logging.getLogger()
        root.handlers.clear()

        from runners.py.runner import setup_logging

        log_dir = tmp_path / 'runs' / 'test_run'
        setup_logging(log_dir=log_dir)

        root = logging.getLogger()
        stream_handlers = [
            h for h in root.handlers if isinstance(h, logging.StreamHandler)
            and not isinstance(h, logging.FileHandler)
        ]
        assert len(stream_handlers) >= 1
        assert stream_handlers[0].level == logging.WARNING

    def test_log_format_includes_module_name(self, tmp_path: Path) -> None:
        """Log format should include timestamp, level, module name, message."""
        root = logging.getLogger()
        root.handlers.clear()

        from runners.py.runner import setup_logging

        log_dir = tmp_path / 'runs' / 'test_run'
        setup_logging(log_dir=log_dir)

        root = logging.getLogger()
        file_handlers = [
            h for h in root.handlers if isinstance(h, logging.FileHandler)
        ]
        formatter = file_handlers[0].formatter
        fmt_str = formatter._style._fmt  # type: ignore[attr-defined]
        assert '%(name)s' in fmt_str
        assert '%(levelname)s' in fmt_str
        assert '%(asctime)s' in fmt_str
        assert '%(message)s' in fmt_str

    def test_duplicate_guard(self, tmp_path: Path) -> None:
        """Calling setup_logging twice should not add duplicate handlers."""
        root = logging.getLogger()
        root.handlers.clear()

        from runners.py.runner import setup_logging

        log_dir = tmp_path / 'runs' / 'test_run'
        setup_logging(log_dir=log_dir)
        handler_count_before = len(root.handlers)

        setup_logging(log_dir=log_dir)
        handler_count_after = len(root.handlers)

        assert handler_count_before == handler_count_after

    def test_creates_log_directory(self, tmp_path: Path) -> None:
        """setup_logging should create the log directory if it doesn't exist."""
        root = logging.getLogger()
        root.handlers.clear()

        from runners.py.runner import setup_logging

        log_dir = tmp_path / 'nested' / 'dir' / 'that' / 'doesnt' / 'exist'
        assert not log_dir.exists()

        setup_logging(log_dir=log_dir)

        assert log_dir.exists()
