from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


SCRIPT_DIR = Path(__file__).resolve().parent
VENDOR_DIR = SCRIPT_DIR / "vendor"
if VENDOR_DIR.is_dir():
    sys.path.insert(0, str(VENDOR_DIR))

from tkinterdnd2 import COPY, DND_FILES, TkinterDnD


APP_NAME = "FolderDropRouter"
WINDOW_TITLE = "文件拖放分流器"
_config_override = os.environ.get("FOLDER_DROP_ROUTER_CONFIG")
if _config_override:
    CONFIG_FILE = Path(_config_override).expanduser()
    CONFIG_DIR = CONFIG_FILE.parent
else:
    CONFIG_DIR = Path(os.environ.get("APPDATA", Path.home())) / APP_NAME
    CONFIG_FILE = CONFIG_DIR / "config.json"
FOLDER_COLORS = [
    ("#F5B83D", "#FFD36A"),
    ("#4A9FE8", "#7EC8FF"),
    ("#44A67A", "#7BD4AA"),
    ("#8B6DD1", "#B69BE8"),
    ("#E16F5C", "#F39A8C"),
    ("#4AA7A5", "#7BD2D0"),
]


def default_folders() -> list[dict[str, str]]:
    return [
        {"id": "folder-a", "name": "文件夹 A", "path": ""},
        {"id": "folder-b", "name": "文件夹 B", "path": ""},
    ]


def load_config() -> list[dict[str, str]]:
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return default_folders()

    # Migrate the original two-path configuration without losing user settings.
    if isinstance(data, dict) and "folders" not in data:
        left = data.get("left", "")
        right = data.get("right", "")
        folders = default_folders()
        folders[0]["path"] = left if isinstance(left, str) else ""
        folders[1]["path"] = right if isinstance(right, str) else ""
        return folders

    raw_folders = data.get("folders", []) if isinstance(data, dict) else []
    folders: list[dict[str, str]] = []
    used_ids: set[str] = set()
    for index, item in enumerate(raw_folders):
        if not isinstance(item, dict):
            continue
        folder_id = item.get("id")
        name = item.get("name")
        path = item.get("path", "")
        if not isinstance(folder_id, str) or not folder_id or folder_id in used_ids:
            folder_id = f"folder-{index + 1}"
        if not isinstance(name, str) or not name:
            name = f"文件夹 {column_label(index + 1)}"
        folders.append(
            {
                "id": folder_id,
                "name": name,
                "path": path if isinstance(path, str) else "",
            }
        )
        used_ids.add(folder_id)

    return folders if len(folders) >= 2 else default_folders()


def save_config(folders: list[dict[str, str]]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    temporary = CONFIG_FILE.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {"schema_version": 2, "folders": folders},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    temporary.replace(CONFIG_FILE)


def column_label(number: int) -> str:
    label = ""
    while number > 0:
        number, remainder = divmod(number - 1, 26)
        label = chr(65 + remainder) + label
    return label


def next_folder_identity(folders: list[dict[str, str]]) -> tuple[str, str]:
    used_names = {item["name"] for item in folders}
    used_ids = {item["id"] for item in folders}
    number = 1
    while f"文件夹 {column_label(number)}" in used_names:
        number += 1
    folder_id = f"folder-{column_label(number).lower()}"
    suffix = 2
    while folder_id in used_ids:
        folder_id = f"folder-{column_label(number).lower()}-{suffix}"
        suffix += 1
    return folder_id, f"文件夹 {column_label(number)}"


def unique_destination(target_directory: Path, filename: str) -> Path:
    candidate = target_directory / filename
    if not candidate.exists():
        return candidate

    source_name = Path(filename)
    stem = source_name.stem
    suffix = source_name.suffix
    counter = 1
    while True:
        candidate = target_directory / f"{stem} ({counter}){suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def move_files(paths: list[str], target_directory: str) -> tuple[list[tuple[Path, Path]], list[str]]:
    target = Path(target_directory).expanduser()
    moved: list[tuple[Path, Path]] = []
    errors: list[str] = []

    if not target.is_dir():
        return moved, [f"目标文件夹不存在：{target}"]

    try:
        resolved_target = target.resolve()
    except OSError:
        resolved_target = target.absolute()

    for raw_path in paths:
        source = Path(raw_path)
        if not source.exists():
            errors.append(f"文件不存在：{source}")
            continue
        if not source.is_file():
            errors.append(f"已跳过文件夹：{source.name}")
            continue

        try:
            resolved_parent = source.parent.resolve()
        except OSError:
            resolved_parent = source.parent.absolute()
        if resolved_parent == resolved_target:
            errors.append(f"文件已经在目标文件夹中：{source.name}")
            continue

        destination = unique_destination(target, source.name)
        try:
            shutil.move(str(source), str(destination))
            moved.append((source, destination))
        except OSError as exc:
            errors.append(f"移动失败 {source.name}：{exc}")

    return moved, errors


class FolderCard(tk.Frame):
    def __init__(
        self,
        master: tk.Misc,
        key: str,
        title: str,
        accent: str,
        front: str,
        set_callback,
        open_callback,
        remove_callback=None,
    ) -> None:
        super().__init__(
            master,
            background="#FFFFFF",
            highlightbackground="#D9DEE8",
            highlightthickness=1,
            bd=0,
            cursor="hand2",
        )
        self.key = key
        self.accent = accent
        self.front = front
        self._normal_border = "#D9DEE8"

        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        body = tk.Frame(self, background="#FFFFFF")
        body.grid(row=0, column=0, sticky="nsew", padx=22, pady=(18, 14))
        body.columnconfigure(0, weight=1)

        self.icon = tk.Canvas(
            body,
            width=150,
            height=112,
            background="#FFFFFF",
            highlightthickness=0,
            bd=0,
        )
        self.icon.grid(row=0, column=0, pady=(0, 6))
        self._draw_folder(False)

        self.title_label = tk.Label(
            body,
            text=title,
            font=("Microsoft YaHei UI", 15, "bold"),
            foreground="#1D2433",
            background="#FFFFFF",
        )
        self.title_label.grid(row=1, column=0, pady=(0, 5))

        self.path_label = tk.Label(
            body,
            text="尚未设置目标路径",
            font=("Microsoft YaHei UI", 9),
            foreground="#788197",
            background="#FFFFFF",
            wraplength=270,
            justify="center",
        )
        self.path_label.grid(row=2, column=0, sticky="ew", pady=(0, 12))

        button_row = tk.Frame(body, background="#FFFFFF")
        button_row.grid(row=3, column=0)
        self.set_button = ttk.Button(
            button_row, text="设置路径", command=lambda: set_callback(self.key)
        )
        self.set_button.pack(side="left", padx=(0, 7))
        self.open_button = ttk.Button(
            button_row,
            text="打开文件夹",
            command=lambda: open_callback(self.key),
            state="disabled",
        )
        self.open_button.pack(side="left")
        if remove_callback is not None:
            self.remove_button = ttk.Button(
                button_row,
                text="移除",
                command=lambda: remove_callback(self.key),
            )
            self.remove_button.pack(side="left", padx=(7, 0))

        for widget in (self, body, self.icon, self.title_label, self.path_label):
            widget.bind("<Enter>", lambda _event: self.set_hover(True))
            widget.bind("<Leave>", lambda _event: self.set_hover(False))
        self.icon.bind("<Double-Button-1>", lambda _event: open_callback(self.key))

    def _draw_folder(self, highlighted: bool) -> None:
        self.icon.delete("all")
        shadow = "#CAD0DB" if not highlighted else "#B7C5E4"
        folder = self.accent
        front = self.front
        self.icon.create_rectangle(24, 35, 91, 56, fill=folder, outline=folder)
        self.icon.create_rectangle(18, 47, 136, 99, fill=shadow, outline=shadow)
        self.icon.create_polygon(
            18,
            51,
            136,
            51,
            126,
            101,
            28,
            101,
            fill=front,
            outline=folder,
            width=2,
        )

    def set_path(self, path: str) -> None:
        if path:
            self.path_label.configure(text=path, foreground="#4F596D")
            self.open_button.configure(state="normal")
        else:
            self.path_label.configure(
                text="尚未设置目标路径", foreground="#788197"
            )
            self.open_button.configure(state="disabled")

    def set_hover(self, active: bool) -> None:
        self.configure(
            highlightbackground=self.accent if active else self._normal_border,
            highlightthickness=2 if active else 1,
        )
        self._draw_folder(active)

    def flash(self, success: bool) -> None:
        color = "#37A866" if success else "#D94A4A"
        self.configure(highlightbackground=color, highlightthickness=3)
        self.after(700, lambda: self.set_hover(False))


class FolderDropRouterApp:
    def __init__(self) -> None:
        self.root = TkinterDnD.Tk()
        self.root.title(WINDOW_TITLE)
        self.root.geometry("820x650")
        self.root.minsize(720, 500)
        self.root.configure(background="#F3F5F9")
        self.folders = load_config()
        self.cards: dict[str, FolderCard] = {}
        self._drop_hover_card: FolderCard | None = None

        self._configure_style()
        self._build_ui()
        self.root.update_idletasks()
        self._center_window()

        self.root.drop_target_register(DND_FILES)
        self.root.dnd_bind("<<DropEnter>>", self._on_drop_position)
        self.root.dnd_bind("<<DropPosition>>", self._on_drop_position)
        self.root.dnd_bind("<<DropLeave>>", self._on_drop_leave)
        self.root.dnd_bind("<<Drop>>", self._on_drop)

        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _configure_style(self) -> None:
        style = ttk.Style(self.root)
        if "vista" in style.theme_names():
            style.theme_use("vista")
        style.configure("TButton", font=("Microsoft YaHei UI", 9), padding=(11, 6))

    def _build_ui(self) -> None:
        header = tk.Frame(self.root, background="#F3F5F9")
        header.pack(fill="x", padx=32, pady=(24, 14))
        header.columnconfigure(0, weight=1)

        heading = tk.Frame(header, background="#F3F5F9")
        heading.grid(row=0, column=0, sticky="w")
        tk.Label(
            heading,
            text="拖放文件到目标文件夹",
            font=("Microsoft YaHei UI", 21, "bold"),
            foreground="#182033",
            background="#F3F5F9",
        ).pack(anchor="w")
        tk.Label(
            heading,
            text="文件会被移动；同名文件会自动改名，不会覆盖。",
            font=("Microsoft YaHei UI", 10),
            foreground="#657086",
            background="#F3F5F9",
        ).pack(anchor="w", pady=(5, 0))

        ttk.Button(
            header,
            text="＋ 新增文件夹",
            command=self._add_folder,
        ).grid(row=0, column=1, sticky="e", padx=(16, 0))

        cards_shell = tk.Frame(self.root, background="#F3F5F9")
        cards_shell.pack(fill="both", expand=True, padx=(32, 21))
        cards_shell.rowconfigure(0, weight=1)
        cards_shell.columnconfigure(0, weight=1)

        self.cards_canvas = tk.Canvas(
            cards_shell,
            background="#F3F5F9",
            highlightthickness=0,
            bd=0,
        )
        self.cards_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(
            cards_shell, orient="vertical", command=self.cards_canvas.yview
        )
        scrollbar.grid(row=0, column=1, sticky="ns", padx=(8, 0))
        self.cards_canvas.configure(yscrollcommand=scrollbar.set)

        self.cards_frame = tk.Frame(self.cards_canvas, background="#F3F5F9")
        self.cards_frame.columnconfigure(0, weight=1, uniform="cards")
        self.cards_frame.columnconfigure(1, weight=1, uniform="cards")
        self.cards_window = self.cards_canvas.create_window(
            (0, 0), window=self.cards_frame, anchor="nw"
        )
        self.cards_frame.bind(
            "<Configure>",
            lambda _event: self.cards_canvas.configure(
                scrollregion=self.cards_canvas.bbox("all")
            ),
        )
        self.cards_canvas.bind(
            "<Configure>",
            lambda event: self.cards_canvas.itemconfigure(
                self.cards_window, width=event.width
            ),
        )
        self.root.bind_all("<MouseWheel>", self._on_mousewheel)
        self._render_cards()

        self.status = tk.Label(
            self.root,
            text="设置目标路径，然后从桌面或资源管理器拖入文件。",
            font=("Microsoft YaHei UI", 9),
            foreground="#657086",
            background="#F3F5F9",
            anchor="w",
        )
        self.status.pack(fill="x", padx=34, pady=(14, 20))

    def _on_mousewheel(self, event) -> None:
        if self.cards_canvas.winfo_exists():
            self.cards_canvas.yview_scroll(int(-event.delta / 120), "units")

    def _render_cards(self) -> None:
        for child in self.cards_frame.winfo_children():
            child.destroy()
        self.cards.clear()

        for index, folder in enumerate(self.folders):
            accent, front = FOLDER_COLORS[index % len(FOLDER_COLORS)]
            card = FolderCard(
                self.cards_frame,
                folder["id"],
                folder["name"],
                accent,
                front,
                self._choose_path,
                self._open_path,
                self._remove_folder if index >= 2 else None,
            )
            row, column = divmod(index, 2)
            left_pad = 0 if column == 0 else 9
            right_pad = 9 if column == 0 else 0
            card.grid(
                row=row,
                column=column,
                sticky="nsew",
                padx=(left_pad, right_pad),
                pady=(0 if row == 0 else 9, 9),
            )
            card.set_path(folder["path"])
            self.cards[folder["id"]] = card

        self.root.after_idle(
            lambda: self.cards_canvas.configure(
                scrollregion=self.cards_canvas.bbox("all")
            )
        )

    def _folder_data(self, key: str) -> dict[str, str] | None:
        return next((item for item in self.folders if item["id"] == key), None)

    def _add_folder(self) -> None:
        folder_id, name = next_folder_identity(self.folders)
        self.folders.append({"id": folder_id, "name": name, "path": ""})
        try:
            save_config(self.folders)
        except OSError as exc:
            self.folders.pop()
            messagebox.showerror("保存设置失败", str(exc))
            return
        self._render_cards()
        self.status.configure(
            text=f"已新增 {name}，请为它设置目标路径。",
            foreground="#287A4B",
        )
        self.root.after_idle(lambda: self.cards_canvas.yview_moveto(1.0))
        self.root.after(120, lambda: self._choose_path(folder_id))

    def _remove_folder(self, key: str) -> None:
        folder = self._folder_data(key)
        if folder is None:
            return
        confirmed = messagebox.askyesno(
            "移除文件夹",
            f"要从程序中移除“{folder['name']}”吗？\n\n"
            "这不会删除真实目标文件夹或里面的文件。",
        )
        if not confirmed:
            return
        original = list(self.folders)
        self.folders = [item for item in self.folders if item["id"] != key]
        try:
            save_config(self.folders)
        except OSError as exc:
            self.folders = original
            messagebox.showerror("保存设置失败", str(exc))
            return
        self._render_cards()
        self.status.configure(
            text=f"已从程序移除 {folder['name']}；真实文件夹和文件没有被删除。",
            foreground="#657086",
        )

    def _center_window(self) -> None:
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = max(0, (self.root.winfo_screenwidth() - width) // 2)
        y = max(0, (self.root.winfo_screenheight() - height) // 2 - 25)
        self.root.geometry(f"{width}x{height}+{x}+{y}")

    def _choose_path(self, key: str) -> None:
        folder = self._folder_data(key)
        if folder is None:
            return
        current = folder["path"]
        initial = current if current and Path(current).is_dir() else str(Path.home())
        selected = filedialog.askdirectory(
            title=f"设置{self.cards[key].title_label.cget('text')}的目标路径",
            initialdir=initial,
            mustexist=True,
        )
        if not selected:
            return
        folder["path"] = selected
        try:
            save_config(self.folders)
        except OSError as exc:
            folder["path"] = current
            messagebox.showerror("保存设置失败", str(exc))
            return
        self.cards[key].set_path(selected)
        self.status.configure(
            text=f"已设置 {self.cards[key].title_label.cget('text')}：{selected}",
            foreground="#287A4B",
        )

    def _open_path(self, key: str) -> None:
        folder = self._folder_data(key)
        path = folder["path"] if folder is not None else ""
        if not path or not Path(path).is_dir():
            self._choose_path(key)
            return
        try:
            os.startfile(path)
        except OSError as exc:
            messagebox.showerror("无法打开文件夹", str(exc))

    def _card_from_widget(self, widget: tk.Misc | None) -> FolderCard | None:
        current = widget
        while current is not None:
            for card in self.cards.values():
                if current is card:
                    return card
            current = getattr(current, "master", None)
        return None

    def _card_at_screen(self, screen_x: int, screen_y: int) -> FolderCard | None:
        widget = self.root.winfo_containing(screen_x, screen_y)
        return self._card_from_widget(widget)

    def _set_drop_hover(self, card: FolderCard | None) -> None:
        if self._drop_hover_card is card:
            return
        if self._drop_hover_card is not None:
            self._drop_hover_card.set_hover(False)
        self._drop_hover_card = card
        if card is not None:
            card.set_hover(True)

    def _on_drop_position(self, event) -> str:
        try:
            self._set_drop_hover(
                self._card_at_screen(int(event.x_root), int(event.y_root))
            )
        except (AttributeError, tk.TclError, ValueError):
            self._set_drop_hover(None)
        return COPY

    def _on_drop_leave(self, _event) -> str:
        self._set_drop_hover(None)
        return COPY

    def _on_drop(self, event) -> str:
        try:
            paths = list(self.root.tk.splitlist(event.data))
            screen_x = int(event.x_root)
            screen_y = int(event.y_root)
            self._set_drop_hover(None)
            self._handle_drop(paths, screen_x, screen_y)
        except Exception as exc:
            self._set_drop_hover(None)
            self.status.configure(
                text=f"处理拖放时出错：{exc}",
                foreground="#C83B3B",
            )
        # The app performs the filesystem move itself. Reporting COPY keeps
        # Explorer from trying to delete the source a second time.
        return COPY

    def _handle_drop(self, paths: list[str], screen_x: int, screen_y: int) -> None:
        card = self._card_at_screen(screen_x, screen_y)

        if card is None:
            self.status.configure(
                text="请把文件准确拖到某个文件夹卡片中。",
                foreground="#C07818",
            )
            return

        folder = self._folder_data(card.key)
        target = folder["path"] if folder is not None else ""
        if not target:
            card.flash(False)
            self.status.configure(
                text=f"请先设置 {card.title_label.cget('text')} 的目标路径。",
                foreground="#C83B3B",
            )
            self._choose_path(card.key)
            return

        moved, errors = move_files(paths, target)
        if moved:
            card.flash(True)
            if len(moved) == 1:
                message = f"已移动：{moved[0][0].name} → {moved[0][1].parent}"
            else:
                message = f"已移动 {len(moved)} 个文件 → {target}"
            self.status.configure(text=message, foreground="#287A4B")
        else:
            card.flash(False)

        if errors:
            summary = "\n".join(errors[:8])
            if len(errors) > 8:
                summary += f"\n……另有 {len(errors) - 8} 项"
            if not moved:
                self.status.configure(text=errors[0], foreground="#C83B3B")
            messagebox.showwarning("部分文件未移动", summary)

    def _close(self) -> None:
        self.root.unbind_all("<MouseWheel>")
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()


def self_test() -> None:
    with tempfile.TemporaryDirectory(prefix="folder_drop_router_") as temporary:
        base = Path(temporary)
        source_dir = base / "source"
        target_dir = base / "target"
        source_dir.mkdir()
        target_dir.mkdir()

        first = source_dir / "sample.wav"
        second = source_dir / "another.wav"
        first.write_bytes(b"first")
        second.write_bytes(b"second")
        (target_dir / "sample.wav").write_bytes(b"existing")

        moved, errors = move_files([str(first), str(second)], str(target_dir))
        assert not errors, errors
        assert len(moved) == 2
        assert (target_dir / "sample (1).wav").read_bytes() == b"first"
        assert (target_dir / "another.wav").read_bytes() == b"second"
        assert not first.exists() and not second.exists()
        assert unique_destination(target_dir, "sample.wav").name == "sample (2).wav"
    print("FolderDropRouter self-test passed")


def gui_self_test() -> None:
    global CONFIG_DIR, CONFIG_FILE
    original_config_dir = CONFIG_DIR
    original_config_file = CONFIG_FILE
    with tempfile.TemporaryDirectory(prefix="folder_drop_router_gui_") as temporary:
        base = Path(temporary)
        CONFIG_DIR = base / "config"
        CONFIG_FILE = CONFIG_DIR / "config.json"
        source_dir = base / "source"
        target_dir = base / "target-a"
        source_dir.mkdir()
        target_dir.mkdir()
        source = source_dir / "third target test.wav"
        source.write_bytes(b"third target")

        app = FolderDropRouterApp()
        try:
            app.root.update_idletasks()
            # Adding a card normally opens its path picker after a short
            # delay. Suppress that modal only inside this automated test.
            app._choose_path = lambda _key: None
            app._add_folder()
            app.root.update_idletasks()
            assert len(app.folders) == 3
            folder_c = app._folder_data("folder-c")
            assert folder_c is not None and folder_c["name"] == "文件夹 C"
            folder_a = app._folder_data("folder-a")
            assert folder_a is not None
            folder_a["path"] = str(target_dir)
            save_config(app.folders)
            app.cards["folder-a"].set_path(str(target_dir))

            card = app.cards["folder-a"]
            # winfo_containing() only resolves screen coordinates after the
            # native window has actually been mapped, not just laid out.
            app.root.update()
            screen_x = card.winfo_rootx() + card.winfo_width() // 2
            screen_y = card.winfo_rooty() + card.winfo_height() // 2
            assert app._card_at_screen(screen_x, screen_y) is card

            class DropEvent:
                data = f"{{{source}}}"
                x_root = screen_x
                y_root = screen_y

            assert list(app.root.tk.splitlist(DropEvent.data)) == [str(source)]
            assert app._on_drop(DropEvent()) == COPY
            destination = target_dir / source.name
            assert destination.exists(), app.status.cget("text")
            assert destination.read_bytes() == b"third target"
            assert not source.exists()
            saved = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
            assert saved["schema_version"] == 2
            assert len(saved["folders"]) == 3
        finally:
            app._close()
            CONFIG_DIR = original_config_dir
            CONFIG_FILE = original_config_file
    print("FolderDropRouter GUI integration test passed")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        self_test()
    elif "--gui-self-test" in sys.argv:
        gui_self_test()
    else:
        FolderDropRouterApp().run()
