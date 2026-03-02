import asyncio
import platform
import string
from pathlib import Path
from typing import Dict, List, Any, Optional

from textual.app import App, ComposeResult
from textual.containers import Container, Vertical, Horizontal, ScrollableContainer
from textual.screen import Screen, ModalScreen
from textual.widgets import (
    Header,
    Footer,
    Button,
    Label,
    DirectoryTree,
    Checkbox,
    ProgressBar,
    RichLog,
    DataTable,
    Input,
    Static,
    ListView,
    ListItem,
    Select,
    TabbedContent,
    TabPane,
)
from textual.reactive import reactive
from textual.binding import Binding

from ayase.pipeline import Pipeline, ModuleRegistry
from ayase.models import Sample, ValidationSeverity

import ayase


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _available_drives() -> List[str]:
    """Return available Windows drive letters."""
    drives = []
    for letter in string.ascii_uppercase:
        try:
            if Path(f"{letter}:\\").exists():
                drives.append(letter)
        except OSError:
            pass
    return drives


# ---------------------------------------------------------------------------
# FolderSelectionScreen  (modal — kept for test compat)
# ---------------------------------------------------------------------------


class FolderSelectionScreen(ModalScreen[Path]):
    """Modal for selecting a folder."""

    BINDINGS = [("escape", "dismiss(None)", "Cancel")]

    def compose(self) -> ComposeResult:
        is_windows = platform.system() == "Windows"
        tree_root = Path.home()

        children: list = [Label("SELECT INPUT DIRECTORY", classes="modal-header")]

        if is_windows:
            drives = _available_drives()
            options = [(f"{d}:\\", d) for d in drives]
            children.append(
                Horizontal(
                    Label("Drive:", classes="drive-label"),
                    Select(
                        options,
                        value=tree_root.drive[0] if tree_root.drive else "C",
                        id="drive_select",
                    ),
                    classes="drive-row",
                )
            )

        children.append(
            Input(placeholder="Type path and press Enter…", id="path_input"),
        )
        children.append(DirectoryTree(str(tree_root), id="tree"))
        children.append(
            Horizontal(
                Button("SELECT", variant="primary", id="select_btn"),
                Button("CANCEL", variant="error", id="cancel_btn"),
                classes="modal-buttons",
            )
        )

        yield Container(*children, classes="modal-window")

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "drive_select" and event.value is not Select.BLANK:
            new_root = Path(f"{event.value}:\\")
            tree = self.query_one("#tree", DirectoryTree)
            if Path(str(tree.path)).drive.upper() != new_root.drive.upper():
                tree.path = str(new_root)
                tree.reload()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "path_input":
            target = Path(event.value)
            if target.is_dir():
                tree = self.query_one("#tree", DirectoryTree)
                tree.path = str(target)
                tree.reload()

    def on_directory_tree_directory_selected(
        self, event: DirectoryTree.DirectorySelected
    ) -> None:
        if event.path.is_dir():
            self.dismiss(event.path)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "select_btn":
            tree = self.query_one(DirectoryTree)
            if tree.cursor_node and tree.cursor_node.data:
                path = tree.cursor_node.data.path
                if path.is_dir():
                    self.dismiss(path)
        elif event.button.id == "cancel_btn":
            self.dismiss(None)


# ---------------------------------------------------------------------------
# ReadinessScreen
# ---------------------------------------------------------------------------


class ReadinessScreen(ModalScreen[None]):
    BINDINGS = [("escape", "dismiss(None)", "Close")]

    def compose(self) -> ComposeResult:
        yield Container(
            Label("MODULE READINESS", classes="modal-header"),
            DataTable(id="readiness_table"),
            Button("CONTINUE", id="btn_continue", variant="primary"),
            classes="modal-window",
        )

    def on_mount(self) -> None:
        table = self.query_one("#readiness_table", DataTable)
        table.add_columns("MODULE", "STATUS", "ERROR")
        readiness = getattr(self.app, "readiness_data", {}) or {}
        for name, info in sorted(readiness.items()):
            status = "READY" if info.get("status") == "ready" else "MISSING"
            error = info.get("error") or ""
            table.add_row(name, status, error)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_continue":
            self.dismiss(None)


# ---------------------------------------------------------------------------
# WelcomeScreen
# ---------------------------------------------------------------------------


class WelcomeScreen(Screen):
    def compose(self) -> ComposeResult:
        yield Vertical(
            Static("A Y A S E", id="logo"),
            Static(
                f"Video Quality Validation  ·  v{ayase.__version__}",
                id="tagline",
            ),
            Vertical(
                Button("OPEN FOLDER", variant="primary", id="btn_folder"),
                Button("LOAD CONFIG", variant="default", id="btn_config", disabled=True),
                classes="menu",
            ),
            id="welcome-center",
        )
        yield Footer()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_folder":
            self.app.push_screen(
                FolderSelectionScreen(), self.on_folder_selected
            )

    def on_folder_selected(self, path: Path | None) -> None:
        if path:
            self.app.selected_path = path
            self.app.switch_mode("config")


# ---------------------------------------------------------------------------
# ModuleConfigWidget
# ---------------------------------------------------------------------------


class ModuleConfigWidget(Static):
    """Compact widget to configure a specific module."""

    def __init__(self, module_name: str, config: Dict[str, Any]):
        super().__init__()
        self.module_name = module_name
        self.current_config = config.copy()

    def compose(self) -> ComposeResult:
        yield Label(f"CONFIG: {self.module_name}", classes="config-title")

        # Filter out internal/ambiguous config keys
        display_config = {
            k: v for k, v in self.current_config.items()
            if k not in ("models_dir",)
        }

        if not display_config:
            yield Label("No configurable options.", classes="no-config")
            return

        for key, value in display_config.items():
            yield Horizontal(
                Label(key, classes="config-key"),
                self._create_input(key, value),
                classes="config-row",
            )

    def _create_input(self, key: str, value: Any):
        if isinstance(value, bool):
            return Checkbox(value=value, id=f"cfg_{key}")
        elif isinstance(value, (int, float)):
            return Input(
                value=str(value),
                id=f"cfg_{key}",
                type="number",
                classes="config-input",
            )
        else:
            return Input(value=str(value), id=f"cfg_{key}", classes="config-input")

    def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if not event.checkbox.id.startswith("cfg_"):
            return
        key = event.checkbox.id[4:]
        self.current_config[key] = event.value
        self.app.update_module_config(self.module_name, self.current_config)

    def on_input_changed(self, event: Input.Changed) -> None:
        if not event.input.id.startswith("cfg_"):
            return
        key = event.input.id[4:]
        val = event.value
        orig = self.app.module_configs[self.module_name].get(key)

        try:
            if isinstance(orig, int):
                self.current_config[key] = int(val)
            elif isinstance(orig, float):
                self.current_config[key] = float(val)
            else:
                self.current_config[key] = val
        except ValueError:
            return

        self.app.update_module_config(self.module_name, self.current_config)


# ---------------------------------------------------------------------------
# ConfigScreen  — Norton Commander two-panel layout
# ---------------------------------------------------------------------------


class ConfigScreen(Screen):
    BINDINGS = [
        ("u", "move_up", "Move Up"),
        ("d", "move_down", "Move Down"),
        ("t", "move_top", "Move Top"),
        ("b", "move_bottom", "Move Bottom"),
    ]

    def compose(self) -> ComposeResult:
        is_windows = platform.system() == "Windows"
        tree_root = (
            str(self.app.selected_path)
            if self.app.selected_path and self.app.selected_path.is_dir()
            else str(Path.home())
        )

        # --- Right panel children (tree browser) ---
        right_children: list = []
        if is_windows:
            drives = _available_drives()
            options = [(f"{d}:\\", d) for d in drives]
            current_drive = (
                Path(tree_root).drive[0] if Path(tree_root).drive else "C"
            )
            right_children.append(
                Horizontal(
                    Select(options, value=current_drive, id="config_drive"),
                    Input(
                        placeholder="Type path…",
                        id="config_path",
                        classes="tree-path-input",
                    ),
                    classes="tree-controls",
                )
            )
        else:
            right_children.append(
                Input(
                    placeholder="Type path…",
                    id="config_path",
                    classes="tree-path-input",
                ),
            )
        right_children.append(
            DirectoryTree(tree_root, id="config_tree"),
        )
        right_children.append(
            Button(
                "SELECT FOLDER",
                variant="primary",
                id="btn_select_folder",
                classes="select-folder-btn",
            ),
        )

        yield Header()
        yield Horizontal(
            # ---- LEFT PANEL: modules + config + start ----
            Vertical(
                Horizontal(
                    Label(" MODULES", classes="panel-title"),
                    Label("0 selected", id="selected_count", classes="count-badge"),
                    classes="panel-bar",
                ),
                ListView(id="module_list"),
                ScrollableContainer(
                    Container(id="config_panel"),
                    classes="config-scroll",
                ),
                Button(
                    "\u25b6 START ANALYSIS",
                    variant="success",
                    id="btn_start",
                    classes="start-btn",
                ),
                classes="left-panel",
            ),
            # ---- RIGHT PANEL: directory tree ----
            Vertical(
                Horizontal(
                    Label(" BROWSE", classes="panel-title"),
                    Label(
                        str(self.app.selected_path or "No folder selected"),
                        id="path_label",
                        classes="path-display",
                    ),
                    classes="panel-bar",
                ),
                *right_children,
                classes="right-panel",
            ),
            classes="commander",
        )
        yield Footer()

    # --- mount ---

    async def on_mount(self) -> None:
        available_modules = ModuleRegistry.list_modules()
        sorted_modules = sorted(available_modules.keys())
        if not self.app.module_order:
            self.app.module_order = sorted_modules
        else:
            existing = [
                n for n in self.app.module_order if n in available_modules
            ]
            missing = [n for n in sorted_modules if n not in existing]
            self.app.module_order = existing + missing
        await self._rebuild_module_list()
        self._update_count()
        # Show config for initially highlighted module
        if self.app.module_order:
            await self.show_config(self.app.module_order[0])

    # --- helpers ---

    def _update_count(self) -> None:
        try:
            count = len(self.app.selected_modules)
            self.query_one("#selected_count", Label).update(f"{count} selected")
        except Exception:
            pass

    async def _rebuild_module_list(
        self, selected_index: Optional[int] = None
    ) -> None:
        module_list = self.query_one("#module_list", ListView)
        available_modules = ModuleRegistry.list_modules()
        await module_list.clear()
        for idx, name in enumerate(self.app.module_order):
            if name not in available_modules:
                continue
            desc = available_modules.get(name, "")
            item = ListItem(
                Horizontal(
                    Checkbox(
                        value=name in self.app.selected_modules,
                        id=f"chk_{name}",
                        classes="module-checkbox",
                    ),
                    Label(f"{idx + 1}.", classes="module-index"),
                    Label(name, classes="list-label", id=f"lbl_{name}"),
                    Label(f"\u00b7 {desc}", classes="module-desc") if desc else Static(""),
                    classes="module-row",
                ),
                id=f"item_{name}",
                classes="module-item",
            )
            await module_list.append(item)

            module_cls = ModuleRegistry.get_module(name)
            if module_cls and hasattr(module_cls, "default_config"):
                if name not in self.app.module_configs:
                    self.app.module_configs[name] = module_cls.default_config.copy()
        if selected_index is not None:
            module_list.index = selected_index

    # --- events: module list ---

    async def on_checkbox_changed(self, event: Checkbox.Changed) -> None:
        if not event.checkbox.id.startswith("chk_"):
            return
        module_name = event.checkbox.id[4:]
        if event.value:
            if module_name not in self.app.selected_modules:
                self.app.selected_modules.append(module_name)
        else:
            if module_name in self.app.selected_modules:
                self.app.selected_modules.remove(module_name)

        self._update_count()

        if event.value:
            await self.show_config(module_name)

    async def on_list_view_highlighted(
        self, event: ListView.Highlighted
    ) -> None:
        if event.item:
            module_name = event.item.id.replace("item_", "")
            await self.show_config(module_name)

    def on_list_view_selected(self, event: ListView.Selected) -> None:
        if event.item:
            module_name = event.item.id.replace("item_", "")
            try:
                chk = event.item.query_one(f"#chk_{module_name}", Checkbox)
                chk.value = not chk.value
            except Exception:
                pass

    async def show_config(self, module_name: str) -> None:
        panel = self.query_one("#config_panel", Container)
        await panel.remove_children()
        config = self.app.module_configs.get(module_name, {})
        await panel.mount(ModuleConfigWidget(module_name, config))

    # --- events: directory tree (right panel) ---

    def on_directory_tree_directory_selected(
        self, event: DirectoryTree.DirectorySelected
    ) -> None:
        if event.path.is_dir():
            self.app.selected_path = event.path
            try:
                self.query_one("#path_label", Label).update(str(event.path))
            except Exception:
                pass

    def on_select_changed(self, event: Select.Changed) -> None:
        if event.select.id == "config_drive" and event.value is not Select.BLANK:
            new_root = Path(f"{event.value}:\\")
            try:
                tree = self.query_one("#config_tree", DirectoryTree)
                if Path(str(tree.path)).drive.upper() != new_root.drive.upper():
                    tree.path = str(new_root)
                    tree.reload()
            except Exception:
                pass

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "config_path":
            target = Path(event.value)
            if target.is_dir():
                try:
                    tree = self.query_one("#config_tree", DirectoryTree)
                    tree.path = str(target)
                    tree.reload()
                    self.app.selected_path = target
                    self.query_one("#path_label", Label).update(str(target))
                except Exception:
                    pass

    # --- actions ---

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_start":
            self.action_start_analysis()
        elif event.button.id == "btn_select_folder":
            self._select_highlighted_folder()

    def action_start_analysis(self) -> None:
        if not self.app.selected_modules:
            self.app.notify("Select at least one module", severity="error")
            return
        if not self.app.selected_path:
            self.app.notify("Select a folder first", severity="error")
            return
        self.app.switch_mode("execution")

    def _select_highlighted_folder(self) -> None:
        """Set the currently highlighted tree node as selected_path."""
        try:
            tree = self.query_one("#config_tree", DirectoryTree)
            if tree.cursor_node and tree.cursor_node.data:
                path = tree.cursor_node.data.path
                if path.is_dir():
                    self.app.selected_path = path
                    self.query_one("#path_label", Label).update(str(path))
                    self.app.notify(f"Selected: {path}", severity="information")
        except Exception:
            pass

    async def action_move_up(self) -> None:
        await self._move_item(-1)

    async def action_move_down(self) -> None:
        await self._move_item(1)

    async def action_move_top(self) -> None:
        await self._move_item(-999)

    async def action_move_bottom(self) -> None:
        await self._move_item(999)

    async def _move_item(self, delta: int) -> None:
        list_view = self.query_one("#module_list", ListView)
        idx = list_view.index
        if idx is None:
            return
        order = self.app.module_order
        if not order:
            return
        if delta <= -999:
            new_idx = 0
        elif delta >= 999:
            new_idx = len(order) - 1
        else:
            new_idx = max(0, min(len(order) - 1, idx + delta))
        if new_idx == idx:
            return
        name = order.pop(idx)
        order.insert(new_idx, name)
        self.app.module_order = order
        self.app.selected_modules = [
            n for n in order if n in self.app.selected_modules
        ]
        await self._rebuild_module_list(selected_index=new_idx)


# ---------------------------------------------------------------------------
# ExecutionScreen
# ---------------------------------------------------------------------------


class ExecutionScreen(Screen):
    _abort: bool = False

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Label("ANALYZING…", id="status_title"),
            Horizontal(
                Container(
                    Label("0", id="stat_processed", classes="stat-value"),
                    Label("Processed", classes="stat-label"),
                    classes="stat-card stat-blue",
                ),
                Container(
                    Label("0", id="stat_failed", classes="stat-value"),
                    Label("Failed", classes="stat-label"),
                    classes="stat-card stat-red",
                ),
                Container(
                    Label("0", id="stat_total", classes="stat-value"),
                    Label("Total", classes="stat-label"),
                    classes="stat-card stat-green",
                ),
                classes="stats-row",
            ),
            ProgressBar(total=100, show_eta=True, id="progress"),
            RichLog(id="log", wrap=True, highlight=True, markup=True),
            Horizontal(
                Button("ABORT", id="btn_abort", variant="error"),
                Button(
                    "VIEW RESULTS",
                    id="btn_results",
                    disabled=True,
                    variant="primary",
                ),
                classes="exec-buttons",
            ),
            classes="exec-box",
        )

    async def on_mount(self) -> None:
        self._abort = False
        await self.run_pipeline()

    async def run_pipeline(self):
        log = self.query_one(RichLog)
        progress = self.query_one(ProgressBar)

        log.write(f"[bold cyan]TARGET:[/bold cyan] {self.app.selected_path}")

        processed = 0
        failed = 0

        modules = []
        for name in self.app.selected_modules:
            try:
                cls = ModuleRegistry.get_module(name)
                config = self.app.module_configs.get(name, {}).copy()

                if self.app.ayase_config:
                    config["models_dir"] = str(
                        self.app.ayase_config.general.models_dir
                    )

                log.write(f"  Init [bold]{name}[/bold]")
                module = cls(config)
                module.on_mount()
                module._mounted = True
                modules.append(module)
            except Exception as e:
                log.write(f"[bold red]FAILED[/bold red] {name}: {e}")

        pipeline = Pipeline(modules)
        self.app.pipeline = pipeline

        path = self.app.selected_path
        files: list = []
        if path.is_file():
            files = [path]
        else:
            files = [
                f
                for f in path.rglob("*")
                if f.suffix.lower()
                in {
                    ".mp4", ".mkv", ".mov", ".avi",
                    ".jpg", ".png", ".jpeg", ".webp", ".gif",
                }
            ]

        if not files:
            log.write(f"[bold red]NO MEDIA FILES FOUND IN:[/bold red] {path}")
            log.write(
                "Please select a different folder or check file extensions."
            )
            self.query_one("#btn_results").disabled = False
            self.query_one("#status_title").update("ANALYSIS FAILED")
            return

        total = len(files)
        self.query_one("#stat_total", Label).update(str(total))
        progress.update(total=total)

        pipeline.start()

        for i, file_path in enumerate(files):
            if self._abort:
                log.write("[bold yellow]ABORTED BY USER[/bold yellow]")
                break

            log.write(f"[{i + 1}/{total}] {file_path.name}")
            VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".gif"}
            sample = Sample(
                path=file_path,
                is_video=file_path.suffix.lower() in VIDEO_EXTS,
            )

            try:
                result = await pipeline.process_sample(sample)
                processed += 1
                issues = len(result.validation_issues)
                if issues > 0:
                    log.write(f"  -> [yellow]{issues} issues[/yellow]")
            except Exception as e:
                failed += 1
                log.write(f"  -> [bold red]ERROR: {e}[/bold red]")

            self.query_one("#stat_processed", Label).update(str(processed))
            self.query_one("#stat_failed", Label).update(str(failed))
            progress.advance(1)

        pipeline.stop()

        try:
            if self.app.ayase_config:
                output_dir = self.app.ayase_config.output.artifacts_dir
                format = (
                    self.app.ayase_config.output.artifacts_format or "json"
                )
                if output_dir:
                    output_dir = Path(output_dir)
                    output_dir.mkdir(parents=True, exist_ok=True)
                    import datetime

                    timestamp = datetime.datetime.now().strftime(
                        "%Y%m%d_%H%M%S"
                    )
                    output_path = (
                        output_dir / f"ayase_tui_{timestamp}.{format}"
                    )
                    pipeline.export_report(output_path, format=format)
                    log.write(f"[green]Report saved:[/green] {output_path}")
        except Exception as e:
            log.write(f"[yellow]Artifact export failed: {e}[/yellow]")

        if self._abort:
            log.write("[bold yellow]ANALYSIS ABORTED[/bold yellow]")
            self.query_one("#status_title").update("ANALYSIS ABORTED")
        else:
            log.write("[bold green]COMPLETE[/bold green]")
            self.query_one("#status_title").update("ANALYSIS COMPLETE")
        self.query_one("#btn_results").disabled = False
        self.query_one("#btn_abort").disabled = True

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_results":
            if not self.app.pipeline or not self.app.pipeline.results:
                self.app.switch_mode("config")
            else:
                self.app.switch_mode("results")
        elif event.button.id == "btn_abort":
            self._abort = True
            self.query_one("#btn_abort").disabled = True


# ---------------------------------------------------------------------------
# ExportDialog
# ---------------------------------------------------------------------------


class ExportDialog(ModalScreen[str]):
    """Modal for selecting export format."""

    BINDINGS = [("escape", "dismiss(None)", "Cancel")]

    def compose(self) -> ComposeResult:
        yield Container(
            Label("EXPORT REPORT", classes="modal-header"),
            Label("Select format:", classes="dialog-label"),
            Vertical(
                Button(
                    "JSON (Full Data)",
                    id="btn_json",
                    variant="primary",
                    classes="dialog-btn",
                ),
                Button(
                    "CSV (Summary)",
                    id="btn_csv",
                    variant="default",
                    classes="dialog-btn",
                ),
                Button(
                    "HTML (Readable)",
                    id="btn_html",
                    variant="default",
                    classes="dialog-btn",
                ),
                classes="dialog-content",
            ),
            Button(
                "CANCEL",
                id="btn_cancel",
                variant="error",
                classes="dialog-footer-btn",
            ),
            classes="modal-window-small",
        )

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_json":
            self.dismiss("json")
        elif event.button.id == "btn_csv":
            self.dismiss("csv")
        elif event.button.id == "btn_html":
            self.dismiss("html")
        elif event.button.id == "btn_cancel":
            self.dismiss(None)


# ---------------------------------------------------------------------------
# ResultsScreen
# ---------------------------------------------------------------------------


class ResultsScreen(Screen):
    BINDINGS = [("escape", "back", "Back")]

    def compose(self) -> ComposeResult:
        yield Header()
        yield Container(
            Horizontal(
                Container(
                    Label("0", id="summary_total", classes="stat-value"),
                    Label("Total", classes="stat-label"),
                    classes="stat-card stat-blue",
                ),
                Container(
                    Label("0", id="summary_passed", classes="stat-value"),
                    Label("Passed", classes="stat-label"),
                    classes="stat-card stat-green",
                ),
                Container(
                    Label("0", id="summary_failed", classes="stat-value"),
                    Label("Failed", classes="stat-label"),
                    classes="stat-card stat-red",
                ),
                Container(
                    Label("-", id="summary_avg", classes="stat-value"),
                    Label("Avg Score", classes="stat-label"),
                    classes="stat-card stat-blue",
                ),
                classes="stats-row",
            ),
            DataTable(id="results_table", zebra_stripes=True),
            Horizontal(
                Button("EXPORT REPORT", id="btn_export", variant="warning"),
                Button("BACK", id="btn_back", variant="default"),
                classes="action-bar",
            ),
            classes="results-box",
        )
        yield Footer()

    def on_mount(self) -> None:
        table = self.query_one(DataTable)
        table.cursor_type = "row"
        table.add_columns("FILE", "TYPE", "SCORE", "STATUS", "ISSUES")

        pipeline = self.app.pipeline
        if not pipeline:
            return

        total = 0
        passed = 0
        failed = 0
        scores: list = []

        for path_str, sample in pipeline.results.items():
            path = Path(path_str)
            total += 1

            ftype = "video" if sample.is_video else "image"

            score = "-"
            score_val = None
            if sample.quality_metrics:
                if sample.quality_metrics.fast_vqa_score:
                    score_val = sample.quality_metrics.fast_vqa_score
                elif sample.quality_metrics.technical_score:
                    score_val = sample.quality_metrics.technical_score
            if score_val is not None:
                score = f"{score_val:.1f}"
                scores.append(score_val)

            issue_count = len(sample.validation_issues)
            issues = str(issue_count)
            status = "FAIL" if issue_count > 0 else "PASS"
            if status == "PASS":
                passed += 1
            else:
                failed += 1

            table.add_row(path.name, ftype, score, status, issues)

        self.query_one("#summary_total", Label).update(str(total))
        self.query_one("#summary_passed", Label).update(str(passed))
        self.query_one("#summary_failed", Label).update(str(failed))
        if scores:
            avg = sum(scores) / len(scores)
            self.query_one("#summary_avg", Label).update(f"{avg:.1f}")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn_back":
            self.app.switch_mode("config")
        elif event.button.id == "btn_export":
            self.app.push_screen(
                ExportDialog(), self.on_export_format_selected
            )

    def on_export_format_selected(self, format: str | None) -> None:
        if not format:
            return
        import datetime

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ayase_report_{timestamp}.{format}"
        output_path = (
            self.app.selected_path.parent / filename
            if self.app.selected_path
            else Path(filename)
        )

        try:
            self.app.pipeline.export_report(output_path, format=format)
            self.app.notify(
                f"Report saved to: {output_path}", severity="information"
            )
        except Exception as e:
            self.app.notify(f"Export failed: {e}", severity="error")

    def action_back(self):
        self.app.switch_mode("config")


# ---------------------------------------------------------------------------
# AyaseApp
# ---------------------------------------------------------------------------


class AyaseApp(App):
    CSS = """
    /* =============================================================
       AYASE TUI — Norton-Commander-style theme
       ============================================================= */

    /* ---------- Global ---------- */

    Screen {
        background: $surface;
        color: $text;
    }

    Button {
        min-width: 16;
        height: 3;
        border: none;
        background: $surface-lighten-1;
    }
    Button:hover {
        background: $accent;
        color: $surface;
    }
    Button:disabled {
        color: $text-muted;
        background: transparent;
        text-style: dim;
    }

    /* ---------- Welcome Screen ---------- */

    WelcomeScreen {
        align: center middle;
    }
    #welcome-center {
        width: auto;
        height: auto;
        padding: 1 4;
        border: heavy $accent;
        background: $surface-darken-1;
    }
    #logo {
        width: 42;
        text-align: center;
        text-style: bold;
        color: $accent;
        padding: 1 0;
    }
    #tagline {
        width: 42;
        text-align: center;
        color: $text-muted;
        padding-bottom: 2;
    }
    .menu {
        width: 42;
        height: auto;
    }
    .menu Button {
        width: 100%;
        margin-bottom: 1;
    }

    /* ---------- Modals ---------- */

    FolderSelectionScreen {
        align: center middle;
        background: rgba(0,0,0,0.7);
    }
    ReadinessScreen {
        align: center middle;
    }
    ExportDialog {
        align: center middle;
    }

    .modal-window {
        width: 70%;
        height: 75%;
        background: $surface;
        border: thick $accent;
        padding: 1 2;
    }
    .modal-header {
        text-align: center;
        text-style: bold;
        background: $accent-darken-2;
        color: $text;
        padding: 0 2;
        margin-bottom: 1;
    }
    .modal-buttons {
        align: center middle;
        height: auto;
        dock: bottom;
        padding-top: 1;
    }
    .modal-buttons Button {
        margin: 0 1;
    }
    .drive-row {
        height: auto;
        margin-bottom: 1;
        align: left middle;
    }
    .drive-label {
        width: 8;
        color: $text-muted;
    }
    #drive_select {
        width: 20;
    }

    .modal-window-small {
        width: 44%;
        height: auto;
        max-height: 80%;
        background: $surface;
        border: thick $accent;
        padding: 1 2;
    }
    .dialog-label {
        margin-bottom: 1;
        color: $text;
    }
    .dialog-content {
        height: auto;
        margin-bottom: 1;
    }
    .dialog-btn {
        width: 100%;
        margin-bottom: 1;
    }
    .dialog-footer-btn {
        width: 100%;
    }

    /* ---------- Commander layout (ConfigScreen) ---------- */

    .commander {
        width: 100%;
        height: 1fr;
    }
    .left-panel {
        width: 1fr;
        border: solid $accent;
    }
    .right-panel {
        width: 1fr;
        border: solid $accent;
    }
    .panel-bar {
        width: 100%;
        height: auto;
        background: $accent-darken-2;
    }
    .panel-title {
        text-style: bold;
        color: $text;
    }
    .count-badge {
        color: $text-muted;
        text-style: italic;
        margin-left: 1;
    }

    /* module list */
    #module_list {
        height: 1fr;
    }
    .module-item {
        height: 2;
        padding: 0;
    }
    .module-item:hover {
        background: $surface-lighten-1;
    }
    .module-row {
        height: 2;
        align: left middle;
        width: 100%;
    }
    .module-checkbox {
        border: none;
        background: transparent;
        color: $accent;
        height: 1;
        width: 5;
        min-width: 5;
    }
    .module-checkbox:focus {
        background: $accent-darken-1;
    }
    .list-label {
        text-style: bold;
        width: auto;
    }
    .module-desc {
        color: $text-muted;
        text-style: italic;
        width: 1fr;
    }

    /* config sub-panel (bottom of left) */
    .config-scroll {
        height: auto;
        max-height: 40%;
        border-top: solid $accent;
        background: $surface-darken-1;
        padding: 0;
    }
    .config-title {
        background: $surface-lighten-1;
        color: $accent;
        text-style: bold;
        padding: 0 1;
    }
    .config-row {
        height: 3;
        align: left middle;
        padding: 0 1;
        border-bottom: dashed $surface-lighten-1;
    }
    .config-key {
        width: 20;
        min-width: 16;
        color: $text-muted;
    }
    .config-input {
        width: 1fr;
        border: none;
        background: $surface-lighten-1;
    }
    Checkbox.config-input {
        width: auto;
    }
    .no-config {
        color: $text-muted;
        text-style: italic;
        padding: 1;
    }

    /* tree browser (right panel) */
    .tree-controls {
        height: auto;
        padding: 0 1;
    }
    .tree-path-input {
        width: 1fr;
    }
    #config_drive {
        width: 16;
        margin-right: 1;
    }
    #config_tree {
        height: 1fr;
        margin: 0 1;
    }
    DirectoryTree {
        background: $surface-lighten-1;
    }

    /* select folder button */
    .select-folder-btn {
        dock: bottom;
        width: 100%;
        margin: 0 1;
    }

    /* status bar */
    .status-bar {
        dock: bottom;
        height: 3;
        background: $accent-darken-2;
        border-top: solid $accent;
        align: right middle;
        padding: 0 2;
    }
    .status-bar Button {
        min-width: 20;
    }
    #path_label {
        width: 1fr;
        color: $text;
    }

    /* ---------- Stats cards (Execution + Results) ---------- */

    .stats-row {
        height: auto;
        margin: 1 0;
    }
    .stat-card {
        width: 1fr;
        height: auto;
        align: center middle;
        padding: 0 2;
        margin: 0 1;
        border: solid $surface-lighten-2;
    }
    .stat-blue  { border: solid #458588; }
    .stat-green { border: solid #98971a; }
    .stat-red   { border: solid #cc241d; }
    .stat-value {
        text-style: bold;
        text-align: center;
        width: 100%;
    }
    .stat-label {
        color: $text-muted;
        text-align: center;
        width: 100%;
    }

    /* ---------- Execution Screen ---------- */

    .exec-box {
        padding: 1 2;
    }
    #status_title {
        text-style: bold;
        text-align: center;
        background: $accent-darken-2;
        margin-bottom: 1;
    }
    .exec-buttons {
        height: auto;
        margin-top: 1;
        align: right middle;
    }
    .exec-buttons Button {
        margin: 0 1;
    }
    RichLog {
        border: solid $surface-lighten-1;
        background: $surface-darken-1;
        margin: 1 0;
        min-height: 8;
    }

    /* ---------- Results Screen ---------- */

    .results-box {
        padding: 1 2;
    }
    .action-bar {
        height: auto;
        margin-top: 1;
        align: right middle;
    }
    .action-bar Button {
        margin: 0 1;
    }
    DataTable {
        border: solid $surface-lighten-1;
    }
    """

    MODES = {
        "welcome": WelcomeScreen,
        "config": ConfigScreen,
        "execution": ExecutionScreen,
        "results": ResultsScreen,
    }

    selected_path: reactive[Path | None] = reactive(None)
    selected_modules: reactive[List[str]] = reactive([])
    module_configs: Dict[str, Dict[str, Any]] = {}
    module_order: List[str] = []
    pipeline: Pipeline | None = None
    ayase_config: Any = None
    readiness_data: Dict[str, Dict[str, Optional[str]]] = {}

    def on_mount(self) -> None:
        from ayase.config import AyaseConfig

        self.ayase_config = AyaseConfig.load()
        ModuleRegistry.discover_modules(
            plugin_paths=self.ayase_config.pipeline.plugin_folders
        )
        self.readiness_data = ModuleRegistry.readiness_report()
        if self.ayase_config.pipeline.modules:
            self.selected_modules = list(self.ayase_config.pipeline.modules)
        if self.ayase_config.pipeline.dataset_path:
            self.selected_path = self.ayase_config.pipeline.dataset_path
            self.switch_mode("config")
        else:
            self.switch_mode("welcome")
        has_issues = any(
            info.get("status") != "ready"
            for info in self.readiness_data.values()
        )
        if has_issues:
            self.push_screen(ReadinessScreen())
        self.theme = "monokai"

    def update_module_config(
        self, module: str, config: Dict[str, Any]
    ) -> None:
        self.module_configs[module] = config


if __name__ == "__main__":
    app = AyaseApp()
    app.run()
