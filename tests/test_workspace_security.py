"""Tests for Workspace.resolve_file() path traversal and absolute path protection."""
from __future__ import annotations

import pytest

from groundmemory.core.workspace import Workspace


@pytest.fixture
def workspace(tmp_path):
    return Workspace(tmp_path / "test_ws")


class TestResolveFileValidPaths:
    def test_bare_filename(self, workspace):
        result = workspace.resolve_file("MEMORY.md")
        assert result == (workspace.path / "MEMORY.md").resolve()

    def test_daily_subpath(self, workspace):
        result = workspace.resolve_file("daily/2026-03-23.md")
        assert result == (workspace.path / "daily" / "2026-03-23.md").resolve()

    def test_nested_subpath(self, workspace):
        result = workspace.resolve_file("daily/2026-01-01.md")
        assert result.parent == (workspace.path / "daily").resolve()


class TestResolveFileAbsolutePaths:
    def test_absolute_path_raises(self, workspace):
        # On Windows, "/etc/passwd" is drive-relative (not absolute), so Path.is_absolute()
        # returns False and the traversal check fires instead.
        import sys
        if sys.platform == "win32":
            with pytest.raises(ValueError, match="Access denied"):
                workspace.resolve_file("/etc/passwd")
        else:
            with pytest.raises(ValueError, match="absolute paths are not allowed"):
                workspace.resolve_file("/etc/passwd")

    def test_absolute_path_windows_style_raises(self, workspace):
        # On Windows this is a real absolute path; on Linux it just starts with C:\
        # Either way resolve_file must reject it if Path() considers it absolute.
        import sys
        if sys.platform == "win32":
            with pytest.raises(ValueError, match="absolute paths are not allowed"):
                workspace.resolve_file("C:\\Windows\\System32\\drivers\\etc\\hosts")
        else:
            # On Linux "C:\..." is not absolute, skip the windows-specific check
            pytest.skip("Windows-only test")

    def test_absolute_workspace_path_itself_raises(self, workspace):
        # Even supplying the workspace root itself as an absolute path should be rejected
        with pytest.raises(ValueError, match="absolute paths are not allowed"):
            workspace.resolve_file(str(workspace.path))


class TestResolveFilePathTraversal:
    def test_simple_traversal_raises(self, workspace):
        with pytest.raises(ValueError, match="resolves outside the workspace"):
            workspace.resolve_file("../outside.txt")

    def test_deep_traversal_raises(self, workspace):
        with pytest.raises(ValueError, match="resolves outside the workspace"):
            workspace.resolve_file("../../etc/passwd")

    def test_traversal_after_valid_prefix_raises(self, workspace):
        with pytest.raises(ValueError, match="resolves outside the workspace"):
            workspace.resolve_file("daily/../../outside.txt")

    def test_traversal_to_parent_of_workspace_raises(self, workspace):
        # Navigate up to the tmp_path parent dir
        with pytest.raises(ValueError, match="resolves outside the workspace"):
            workspace.resolve_file("../")

    def test_dotdot_in_middle_raises(self, workspace):
        with pytest.raises(ValueError, match="resolves outside the workspace"):
            workspace.resolve_file("daily/../../../secret")