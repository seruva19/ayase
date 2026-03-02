"""Tests for the Ayase TUI (Textual app)."""

import asyncio
from pathlib import Path
from typing import Dict, List, Optional
from unittest.mock import MagicMock, patch, AsyncMock

import pytest

from ayase.models import Sample, QualityMetrics, ValidationIssue, ValidationSeverity
from ayase.tui import (
    AyaseApp,
    WelcomeScreen,
    ConfigScreen,
    ExecutionScreen,
    ResultsScreen,
    FolderSelectionScreen,
    ReadinessScreen,
    ExportDialog,
    ModuleConfigWidget,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_sample(name: str, tech: float = 75.0, issues: int = 0) -> Sample:
    s = Sample(path=Path(f"/data/{name}"), is_video=name.endswith(".mp4"))
    s.quality_metrics = QualityMetrics(technical_score=tech)
    for i in range(issues):
        s.validation_issues.append(
            ValidationIssue(
                severity=ValidationSeverity.WARNING,
                message=f"issue {i}",
            )
        )
    return s


FAKE_MODULES = {
    "metadata": "Extract metadata",
    "basic_quality": "Basic quality metrics",
    "motion": "Motion analysis",
}


class FakePipelineModule:
    name = "metadata"
    description = "Extract metadata"
    default_config = {"threshold": 0.5, "enabled": True}

    def __init__(self, config=None):
        self.config = config or self.default_config.copy()
        self._mounted = False

    def on_mount(self):
        self._mounted = True

    def process(self, sample):
        return sample


def _make_fake_module_class(name: str, desc: str = "fake"):
    class Mod(FakePipelineModule):
        pass
    Mod.name = name
    Mod.description = desc
    Mod.default_config = {"threshold": 0.5}
    return Mod


def _patch_registry_and_config():
    """Return a stack of patches that isolate the TUI from real modules."""
    fake_classes = {n: _make_fake_module_class(n, d) for n, d in FAKE_MODULES.items()}

    def fake_list_modules():
        return dict(FAKE_MODULES)

    def fake_get_module(name):
        return fake_classes.get(name)

    def fake_discover(*a, **kw):
        pass

    def fake_readiness():
        return {}

    return [
        patch("ayase.tui.ModuleRegistry.discover_modules", side_effect=fake_discover),
        patch("ayase.tui.ModuleRegistry.list_modules", side_effect=fake_list_modules),
        patch("ayase.tui.ModuleRegistry.get_module", side_effect=fake_get_module),
        patch("ayase.tui.ModuleRegistry.readiness_report", side_effect=fake_readiness),
        patch("ayase.config.AyaseConfig.load", return_value=_make_default_config()),
    ]


def _make_default_config():
    from ayase.config import AyaseConfig
    return AyaseConfig()


# ---------------------------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------------------------

class TestAppLifecycle:
    @pytest.mark.asyncio
    async def test_app_starts_on_welcome_screen(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                assert isinstance(app.screen, WelcomeScreen)
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_app_has_four_modes(self):
        assert set(AyaseApp.MODES.keys()) == {"welcome", "config", "execution", "results"}

    @pytest.mark.asyncio
    async def test_app_initial_state(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                assert app.selected_path is None
                assert app.selected_modules == []
                assert app.pipeline is None
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_app_theme_is_monokai(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                assert app.theme == "monokai"
        finally:
            for p in patches:
                p.stop()


# ---------------------------------------------------------------------------
# Welcome screen
# ---------------------------------------------------------------------------

class TestWelcomeScreen:
    @pytest.mark.asyncio
    async def test_welcome_has_open_folder_button(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                btn = app.screen.query_one("#btn_folder")
                assert btn is not None
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_welcome_has_load_config_button_disabled(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                btn = app.screen.query_one("#btn_config")
                assert btn.disabled is True
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_open_folder_pushes_modal(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                await pilot.click("#btn_folder")
                await pilot.pause()
                # The top of the screen stack should be FolderSelectionScreen
                assert any(
                    isinstance(s, FolderSelectionScreen) for s in app.screen_stack
                )
        finally:
            for p in patches:
                p.stop()


# ---------------------------------------------------------------------------
# Config screen
# ---------------------------------------------------------------------------

class TestConfigScreen:
    @pytest.mark.asyncio
    async def test_config_screen_shows_modules(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                # Navigate to config screen by setting a path
                app.selected_path = Path("/tmp/fake")
                app.switch_mode("config")
                await pilot.pause()
                module_list = app.screen.query_one("#module_list")
                assert module_list is not None
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_config_screen_has_start_button(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                app.selected_path = Path("/tmp/fake")
                app.switch_mode("config")
                await pilot.pause()
                btn = app.screen.query_one("#btn_start")
                assert btn is not None
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_config_screen_shows_path_label(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                test_path = Path("/tmp/my_dataset")
                app.selected_path = test_path
                app.switch_mode("config")
                await pilot.pause()
                label = app.screen.query_one("#path_label")
                assert label is not None
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_start_without_modules_shows_error(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test(notifications=True) as pilot:
                app.selected_path = Path("/tmp/fake")
                app.switch_mode("config")
                await pilot.pause()
                app.selected_modules = []
                await pilot.click("#btn_start")
                await pilot.pause()
                # Should still be on config screen (not switched to execution)
                assert isinstance(app.screen, ConfigScreen)
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_start_with_modules_switches_to_execution(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                app.selected_path = Path("/tmp/fake")
                app.switch_mode("config")
                await pilot.pause()
                app.selected_modules = ["metadata"]
                await pilot.click("#btn_start")
                await pilot.pause()
                assert isinstance(app.screen, ExecutionScreen)
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_module_order_reorder_up(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                app.selected_path = Path("/tmp/fake")
                app.switch_mode("config")
                await pilot.pause()
                original_order = list(app.module_order)
                assert len(original_order) >= 2
                # Select the second item and move it up
                module_list = app.screen.query_one("#module_list")
                module_list.index = 1
                await pilot.press("u")
                await pilot.pause()
                # The second item should now be first
                assert app.module_order[0] == original_order[1]
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_module_order_reorder_down(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                app.selected_path = Path("/tmp/fake")
                app.switch_mode("config")
                await pilot.pause()
                original_order = list(app.module_order)
                module_list = app.screen.query_one("#module_list")
                module_list.index = 0
                await pilot.press("d")
                await pilot.pause()
                assert app.module_order[1] == original_order[0]
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_module_order_move_to_top(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                app.selected_path = Path("/tmp/fake")
                app.switch_mode("config")
                await pilot.pause()
                original_order = list(app.module_order)
                last_name = original_order[-1]
                module_list = app.screen.query_one("#module_list")
                module_list.index = len(original_order) - 1
                await pilot.press("t")
                await pilot.pause()
                assert app.module_order[0] == last_name
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_module_order_move_to_bottom(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                app.selected_path = Path("/tmp/fake")
                app.switch_mode("config")
                await pilot.pause()
                original_order = list(app.module_order)
                first_name = original_order[0]
                module_list = app.screen.query_one("#module_list")
                module_list.index = 0
                await pilot.press("b")
                await pilot.pause()
                assert app.module_order[-1] == first_name
        finally:
            for p in patches:
                p.stop()


# ---------------------------------------------------------------------------
# Module config widget
# ---------------------------------------------------------------------------

class TestModuleConfigWidget:
    @pytest.mark.asyncio
    async def test_config_widget_renders_options(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                app.selected_path = Path("/tmp/fake")
                app.switch_mode("config")
                await pilot.pause()
                # Highlight a module to show its config
                module_list = app.screen.query_one("#module_list")
                module_list.index = 0
                await pilot.pause()
                panel = app.screen.query_one("#config_panel")
                widgets = panel.query(ModuleConfigWidget)
                assert len(widgets) > 0
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_empty_config_shows_no_options(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                app.selected_path = Path("/tmp/fake")
                app.switch_mode("config")
                await pilot.pause()
                # Mount a widget with empty config
                panel = app.screen.query_one("#config_panel")
                panel.remove_children()
                widget = ModuleConfigWidget("empty_mod", {})
                await panel.mount(widget)
                await pilot.pause()
                labels = widget.query(".no-config")
                assert len(labels) > 0
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_update_module_config(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                app.update_module_config("test_mod", {"key": "val"})
                assert app.module_configs["test_mod"] == {"key": "val"}
        finally:
            for p in patches:
                p.stop()


# ---------------------------------------------------------------------------
# Execution screen
# ---------------------------------------------------------------------------

class TestExecutionScreen:
    @pytest.mark.asyncio
    async def test_execution_screen_has_progress_bar(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                app.selected_path = Path("/tmp/fake")
                app.selected_modules = ["metadata"]
                app.switch_mode("execution")
                await pilot.pause()
                progress = app.screen.query_one("#progress")
                assert progress is not None
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_execution_screen_has_log(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                app.selected_path = Path("/tmp/fake")
                app.selected_modules = ["metadata"]
                app.switch_mode("execution")
                await pilot.pause()
                log = app.screen.query_one("#log")
                assert log is not None
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_execution_no_files_shows_error(self, tmp_path):
        """Empty directory -> no media files found."""
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                app.selected_path = tmp_path  # empty dir
                app.selected_modules = ["metadata"]
                app.switch_mode("execution")
                await pilot.pause(delay=0.5)
                title = app.screen.query_one("#status_title")
                assert "FAILED" in str(title.render())
                # Results button should be enabled to allow navigating away
                btn = app.screen.query_one("#btn_results")
                assert btn.disabled is False
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_execution_processes_files(self, tmp_path):
        """Directory with a media file -> pipeline processes it."""
        (tmp_path / "test.mp4").write_bytes(b"\x00" * 100)
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                app.selected_path = tmp_path
                app.selected_modules = ["metadata"]
                app.switch_mode("execution")
                await pilot.pause(delay=1.0)
                title = app.screen.query_one("#status_title")
                assert "COMPLETE" in str(title.render())
                btn = app.screen.query_one("#btn_results")
                assert btn.disabled is False
        finally:
            for p in patches:
                p.stop()


# ---------------------------------------------------------------------------
# Results screen
# ---------------------------------------------------------------------------

class TestResultsScreen:
    @pytest.mark.asyncio
    async def test_results_screen_has_table(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                # Set up fake pipeline results
                fake_pipeline = MagicMock()
                fake_pipeline.results = {
                    "/data/a.mp4": _make_sample("a.mp4", tech=80.0, issues=1),
                    "/data/b.mp4": _make_sample("b.mp4", tech=60.0, issues=0),
                }
                app.pipeline = fake_pipeline
                app.switch_mode("results")
                await pilot.pause()
                table = app.screen.query_one("#results_table")
                assert table.row_count == 2
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_results_table_columns(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                fake_pipeline = MagicMock()
                fake_pipeline.results = {
                    "/data/a.mp4": _make_sample("a.mp4"),
                }
                app.pipeline = fake_pipeline
                app.switch_mode("results")
                await pilot.pause()
                table = app.screen.query_one("#results_table")
                col_labels = [str(col.label) for col in table.columns.values()]
                assert "FILE" in col_labels
                assert "SCORE" in col_labels
                assert "ISSUES" in col_labels
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_results_back_button_goes_to_config(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                fake_pipeline = MagicMock()
                fake_pipeline.results = {}
                app.pipeline = fake_pipeline
                app.selected_path = Path("/tmp/fake")
                app.switch_mode("results")
                await pilot.pause()
                await pilot.click("#btn_back")
                await pilot.pause()
                assert isinstance(app.screen, ConfigScreen)
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_results_escape_goes_to_config(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                fake_pipeline = MagicMock()
                fake_pipeline.results = {}
                app.pipeline = fake_pipeline
                app.selected_path = Path("/tmp/fake")
                app.switch_mode("results")
                await pilot.pause()
                await pilot.press("escape")
                await pilot.pause()
                assert isinstance(app.screen, ConfigScreen)
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_export_button_opens_dialog(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                fake_pipeline = MagicMock()
                fake_pipeline.results = {
                    "/data/a.mp4": _make_sample("a.mp4"),
                }
                app.pipeline = fake_pipeline
                app.selected_path = Path("/tmp/fake")
                app.switch_mode("results")
                await pilot.pause()
                await pilot.click("#btn_export")
                await pilot.pause()
                assert any(isinstance(s, ExportDialog) for s in app.screen_stack)
        finally:
            for p in patches:
                p.stop()


# ---------------------------------------------------------------------------
# Export dialog
# ---------------------------------------------------------------------------

class TestExportDialog:
    @pytest.mark.asyncio
    async def test_export_dialog_has_three_format_buttons(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                fake_pipeline = MagicMock()
                fake_pipeline.results = {"/data/a.mp4": _make_sample("a.mp4")}
                app.pipeline = fake_pipeline
                app.selected_path = Path("/tmp/fake")
                app.switch_mode("results")
                await pilot.pause()
                await pilot.click("#btn_export")
                await pilot.pause()
                dialog = [s for s in app.screen_stack if isinstance(s, ExportDialog)][0]
                assert dialog.query_one("#btn_json") is not None
                assert dialog.query_one("#btn_csv") is not None
                assert dialog.query_one("#btn_html") is not None
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_export_escape_dismisses(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                fake_pipeline = MagicMock()
                fake_pipeline.results = {"/data/a.mp4": _make_sample("a.mp4")}
                app.pipeline = fake_pipeline
                app.selected_path = Path("/tmp/fake")
                app.switch_mode("results")
                await pilot.pause()
                await pilot.click("#btn_export")
                await pilot.pause()
                await pilot.press("escape")
                await pilot.pause(delay=0.3)
                assert not any(isinstance(s, ExportDialog) for s in app.screen_stack)
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_export_json_calls_export_report(self, tmp_path):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test(notifications=True) as pilot:
                fake_pipeline = MagicMock()
                fake_pipeline.results = {"/data/a.mp4": _make_sample("a.mp4")}
                fake_pipeline.export_report = MagicMock()
                app.pipeline = fake_pipeline
                app.selected_path = tmp_path
                app.switch_mode("results")
                await pilot.pause()
                await pilot.click("#btn_export")
                await pilot.pause()
                await pilot.click("#btn_json")
                await pilot.pause(delay=0.3)
                fake_pipeline.export_report.assert_called_once()
                call_args = fake_pipeline.export_report.call_args
                assert str(call_args[0][0]).endswith(".json")
                assert call_args[1]["format"] == "json"
        finally:
            for p in patches:
                p.stop()


# ---------------------------------------------------------------------------
# Readiness screen
# ---------------------------------------------------------------------------

class TestReadinessScreen:
    @pytest.mark.asyncio
    async def test_readiness_screen_shown_when_data_present(self):
        app = AyaseApp()
        readiness_data = {
            "metadata": {"status": "ready", "error": None},
            "broken_mod": {"status": "missing", "error": "No such module"},
        }
        patches = _patch_registry_and_config()
        # Override readiness to return data
        patches[3] = patch(
            "ayase.tui.ModuleRegistry.readiness_report",
            return_value=readiness_data,
        )
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                await pilot.pause()
                assert any(isinstance(s, ReadinessScreen) for s in app.screen_stack)
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_readiness_not_shown_when_all_ready(self):
        """Modal should NOT appear when every module is ready."""
        app = AyaseApp()
        readiness_data = {
            "metadata": {"status": "ready", "error": None},
        }
        patches = _patch_registry_and_config()
        patches[3] = patch(
            "ayase.tui.ModuleRegistry.readiness_report",
            return_value=readiness_data,
        )
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                await pilot.pause()
                assert not any(isinstance(s, ReadinessScreen) for s in app.screen_stack)
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_readiness_continue_dismisses(self):
        app = AyaseApp()
        readiness_data = {
            "metadata": {"status": "ready", "error": None},
            "broken": {"status": "missing", "error": "not installed"},
        }
        patches = _patch_registry_and_config()
        patches[3] = patch(
            "ayase.tui.ModuleRegistry.readiness_report",
            return_value=readiness_data,
        )
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                await pilot.pause()
                await pilot.click("#btn_continue")
                await pilot.pause()
                assert not any(isinstance(s, ReadinessScreen) for s in app.screen_stack)
        finally:
            for p in patches:
                p.stop()


# ---------------------------------------------------------------------------
# Folder selection modal
# ---------------------------------------------------------------------------

class TestFolderSelectionScreen:
    @pytest.mark.asyncio
    async def test_folder_modal_has_tree_and_buttons(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                await pilot.click("#btn_folder")
                await pilot.pause()
                modal = [s for s in app.screen_stack if isinstance(s, FolderSelectionScreen)]
                assert len(modal) == 1
                assert modal[0].query_one("#tree") is not None
                assert modal[0].query_one("#select_btn") is not None
                assert modal[0].query_one("#cancel_btn") is not None
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_folder_cancel_dismisses(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                await pilot.click("#btn_folder")
                await pilot.pause()
                await pilot.click("#cancel_btn")
                await pilot.pause()
                assert not any(
                    isinstance(s, FolderSelectionScreen) for s in app.screen_stack
                )
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_folder_escape_dismisses(self):
        app = AyaseApp()
        patches = _patch_registry_and_config()
        for p in patches:
            p.start()
        try:
            async with app.run_test() as pilot:
                await pilot.click("#btn_folder")
                await pilot.pause()
                await pilot.press("escape")
                await pilot.pause()
                assert not any(
                    isinstance(s, FolderSelectionScreen) for s in app.screen_stack
                )
        finally:
            for p in patches:
                p.stop()


# ---------------------------------------------------------------------------
# Config preloading from ayase.toml
# ---------------------------------------------------------------------------

class TestConfigPreloading:
    @pytest.mark.asyncio
    async def test_config_with_dataset_path_skips_welcome(self):
        from ayase.config import AyaseConfig

        cfg = AyaseConfig()
        cfg.pipeline.dataset_path = Path("/tmp/preloaded")

        patches = _patch_registry_and_config()
        patches[4] = patch("ayase.config.AyaseConfig.load", return_value=cfg)
        for p in patches:
            p.start()
        try:
            app = AyaseApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                assert isinstance(app.screen, ConfigScreen)
                assert app.selected_path == Path("/tmp/preloaded")
        finally:
            for p in patches:
                p.stop()

    @pytest.mark.asyncio
    async def test_config_with_preselected_modules(self):
        from ayase.config import AyaseConfig

        cfg = AyaseConfig()
        cfg.pipeline.modules = ["metadata", "motion"]

        patches = _patch_registry_and_config()
        patches[4] = patch("ayase.config.AyaseConfig.load", return_value=cfg)
        for p in patches:
            p.start()
        try:
            app = AyaseApp()
            async with app.run_test() as pilot:
                await pilot.pause()
                assert "metadata" in app.selected_modules
                assert "motion" in app.selected_modules
        finally:
            for p in patches:
                p.stop()
