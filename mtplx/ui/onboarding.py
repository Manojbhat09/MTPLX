"""Interactive onboarding flow for ``mtplx start``.

Three screens for first-time users (model -> mode -> interface), and a
"same as last time?" prompt for returning users. Choices persist to
``~/.mtplx/quickstart.json`` so the next run can offer the same defaults.

The module gracefully degrades to plain stdlib ``print`` and ``input`` when
``rich`` is not available, so the onboarding works even in minimal venvs.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any

from mtplx.profiles import DEFAULT_HF_MODEL_ID

DEFAULT_HF_MODEL = DEFAULT_HF_MODEL_ID
STATE_PATH = Path("~/.mtplx/quickstart.json").expanduser()


# ---------- state file ------------------------------------------------------
def _state_path() -> Path:
    env = os.environ.get("MTPLX_QUICKSTART_STATE")
    return Path(env).expanduser() if env else STATE_PATH


def load_state() -> dict | None:
    """Return the last saved quickstart state, or ``None`` if absent / invalid."""

    path = _state_path()
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def save_state(state: dict) -> None:
    """Persist the chosen configuration. Adds a UTC timestamp."""

    path = _state_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(state)
    payload["saved_at"] = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


# ---------- rich helpers (with stdlib fallback) -----------------------------
def _console() -> Any | None:
    try:
        from rich.console import Console
    except ImportError:
        return None
    try:
        return Console()
    except Exception:
        return None


def _step_panel(
    *,
    step: int,
    total: int,
    title: str,
    options: list[tuple[str, str, str]],
) -> None:
    """Render a numbered onboarding step (Step N of M)."""

    _choice_panel(
        heading=f"Step {step} of {total}  ·  {title}",
        options=options,
        border_style="cyan",
    )


def _choice_panel(
    *,
    heading: str,
    options: list[tuple[str, str, str]],
    intro: str | None = None,
    border_style: str = "cyan",
) -> None:
    """Render a panel of numbered choices; falls back to plain text."""

    try:
        from rich.panel import Panel
        from rich.text import Text
    except ImportError:
        print()
        print(f"  {heading}")
        print()
        if intro:
            print(f"  {intro}")
            print()
        for key, headline, subline in options:
            print(f"  {key}.  {headline}")
            if subline:
                print(f"      {subline}")
            print()
        return

    console = _console()
    if console is None:
        print()
        print(f"  {heading}")
        print()
        if intro:
            print(f"  {intro}")
            print()
        for key, headline, subline in options:
            print(f"  {key}.  {headline}")
            if subline:
                print(f"      {subline}")
            print()
        return

    body = Text()
    if intro:
        body.append(intro, style="")
        body.append("\n\n")
    for index, (key, headline, subline) in enumerate(options):
        if index > 0:
            body.append("\n\n")
        body.append(f"{key}.  ", style="bold cyan")
        body.append(headline, style="bold")
        if subline:
            body.append("\n    ")
            body.append(subline, style="dim")

    panel = Panel(
        body,
        title=Text(heading, style="bold"),
        title_align="left",
        border_style=border_style,
        padding=(1, 2),
        expand=False,
    )
    console.print()
    console.print(panel)
    console.print()


def _prompt_choice(prompt: str, choices: list[str], default: str | None = None) -> str:
    """Read a digit choice from stdin, looping until valid. Raises on Ctrl-C.

    The prompt is rendered as ``Type 1-N and press Enter [1]:`` so users with
    no prior CLI familiarity know exactly what action to take.
    """

    if choices:
        try:
            low = min(int(c) for c in choices)
            high = max(int(c) for c in choices)
            if low == high:
                hint = f"Type {low} and press Enter"
            else:
                hint = f"Type {low}-{high} and press Enter"
        except ValueError:
            hint = f"Type one of {', '.join(choices)} and press Enter"
    else:
        hint = "Type your choice and press Enter"

    suffix = f" [default {default}]" if default else ""
    while True:
        answer = input(f"  {hint}{suffix}: ").strip()
        if not answer and default:
            return default
        if answer in choices:
            return answer
        print(f"  please type one of: {', '.join(choices)}")


def _prompt_text(prompt: str, *, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    answer = input(f"  {prompt}{suffix}: ").strip()
    if not answer and default:
        return default
    return answer


def _print_welcome() -> None:
    try:
        from rich.panel import Panel
        from rich.text import Text
    except ImportError:
        print()
        print("Welcome to MTPLX. Three quick questions to get you set up.")
        print("For each step, type the number of your choice and press Enter.")
        print()
        return
    console = _console()
    if console is None:
        print()
        print("Welcome to MTPLX. Three quick questions to get you set up.")
        print("For each step, type the number of your choice and press Enter.")
        print()
        return
    body = Text()
    body.append("Welcome to MTPLX.\n\n", style="bold")
    body.append("Three quick questions to get you set up:\n", style="")
    body.append("  1. Which model?\n", style="dim")
    body.append("  2. Which runtime mode?\n", style="dim")
    body.append("  3. Browser chat or terminal chat?\n\n", style="dim")
    body.append("For each step, type the number of your choice and press Enter.", style="italic")
    panel = Panel(
        body,
        title=Text("First-time setup", style="bold cyan"),
        title_align="left",
        border_style="cyan",
        padding=(1, 2),
        expand=False,
    )
    console.print()
    console.print(panel)
    console.print()


def _print_summary(state: dict) -> None:
    try:
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        print()
        print("  Your quickstart configuration:")
        print(f"    Model:     {state.get('model', '?')}")
        print(f"    Mode:      {mode_label(state)}")
        print(f"    Interface: {interface_label(state.get('target'))}")
        print()
        return
    console = _console()
    if console is None:
        return
    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right", no_wrap=True)
    table.add_column(no_wrap=False)
    table.add_row("Model", str(state.get("model", "?")))
    table.add_row("Mode", mode_label(state))
    table.add_row("Interface", interface_label(state.get("target")))
    panel = Panel(
        table,
        title=Text("Ready to go", style="bold green"),
        title_align="left",
        border_style="green",
        padding=(1, 2),
        expand=False,
    )
    console.print()
    console.print(panel)
    console.print()


# ---------- screens ---------------------------------------------------------
def screen_model(*, configured: str | None = None) -> str:
    """Render screen 1 and return the user's choice.

    If ``configured`` is set and differs from the canonical default, it is
    surfaced as the first option (so accepting the default reuses the user's
    already-resolved local path instead of forcing a re-download).
    """

    show_configured = bool(configured) and configured != DEFAULT_HF_MODEL

    options: list[tuple[str, str, str]] = []
    if show_configured:
        options.append(
            (
                "1",
                "Use your configured model",
                str(configured),
            )
        )
        options.append(
            (
                "2",
                "Verified default",
                f"{DEFAULT_HF_MODEL} · cold-speed champion",
            )
        )
        options.append(
            (
                "3",
                "Custom Hugging Face repo",
                "e.g. Qwen/Qwen3-Next-80B-A3B-Instruct",
            )
        )
        options.append(
            (
                "4",
                "Local folder",
                "e.g. /Users/you/models/your-model",
            )
        )
    else:
        options.append(
            (
                "1",
                "Verified default",
                f"{DEFAULT_HF_MODEL} · cold-speed champion",
            )
        )
        options.append(
            (
                "2",
                "Custom Hugging Face repo",
                "e.g. Qwen/Qwen3-Next-80B-A3B-Instruct",
            )
        )
        options.append(
            (
                "3",
                "Local folder",
                "e.g. /Users/you/models/your-model",
            )
        )

    _step_panel(step=1, total=3, title="Choose your model", options=options)
    valid_choices = [opt[0] for opt in options]
    choice = _prompt_choice("Select", valid_choices, default="1")

    if show_configured:
        if choice == "1":
            return str(configured)
        if choice == "2":
            return DEFAULT_HF_MODEL
        if choice == "3":
            entered = _prompt_text("Hugging Face repo id (namespace/name)", default=DEFAULT_HF_MODEL)
            return entered or DEFAULT_HF_MODEL
        # choice == "4"
        entered = _prompt_text("Local folder path", default=str(configured) or DEFAULT_HF_MODEL)
        return entered or str(configured) or DEFAULT_HF_MODEL

    if choice == "1":
        return DEFAULT_HF_MODEL
    if choice == "2":
        entered = _prompt_text("Hugging Face repo id (namespace/name)", default=DEFAULT_HF_MODEL)
        return entered or DEFAULT_HF_MODEL
    # choice == "3"
    entered = _prompt_text("Local folder path", default=DEFAULT_HF_MODEL)
    return entered or DEFAULT_HF_MODEL


def screen_mode() -> tuple[str, bool]:
    """Return (profile_name, max_mode_flag).

    The consumer onboarding is speed-first:

      Medium : native-MTP speed path, Apple fan curve, burst not sustained
      Max    : same path, ThermalForge pins fans at 100% for sustained speed

    Stable remains available through ``--profile stable`` / ``--profile safe``.
    """

    _step_panel(
        step=2,
        total=3,
        title="Choose a runtime mode",
        options=[
            (
                "1",
                "Medium  ·  native-MTP speed path, about 2.2x burst (not sustained)",
                "Fast speculative path with Apple's default fan curve; snappy on short replies, slower on long hot runs.",
            ),
            (
                "2",
                "Max  ·  Medium path plus fans pinned at 100%, about 2.24x (loud)",
                "Same runtime path as Medium, plus ThermalForge fan control to reduce long-reply slowdown.",
            ),
        ],
    )
    choice = _prompt_choice("Select", ["1", "2"], default="1")
    if choice == "1":
        return "performance-cold", False
    return "performance-cold", True


def screen_interface() -> str:
    """Return the target string (``openwebui`` or ``terminal``)."""

    _step_panel(
        step=3,
        total=3,
        title="Where do you want to chat?",
        options=[
            (
                "1",
                "Web UI [browser at http://127.0.0.1:8000/]",
                "Markdown rendering · live tokens-per-second · inference settings sidebar.",
            ),
            (
                "2",
                "CLI [this terminal]",
                "Streamed answers with rich styling and a stats footer.",
            ),
        ],
    )
    choice = _prompt_choice("Select", ["1", "2"], default="1")
    return "openwebui" if choice == "1" else "terminal"


def run_onboarding_screens(*, configured_model: str | None = None) -> dict:
    """Walk all three screens and return the chosen state dict.

    When the user picks Max mode but no fan controller is detected, this
    function offers to install ThermalForge automatically. If install is
    declined or fails, ``max`` is left as ``False`` so the rest of the
    pipeline doesn't promise fan-boosted speeds it can't deliver.
    """

    model = screen_model(configured=configured_model)
    profile, max_mode = screen_mode()
    if max_mode:
        max_mode = ensure_thermal_control_installed()
    target = screen_interface()
    return {
        "model": model,
        "profile": profile,
        "max": max_mode,
        "target": target,
    }


def ensure_thermal_control_installed() -> bool:
    """Detect a fan controller; offer to install ThermalForge if absent.

    Returns ``True`` when fan control is available after this call (so Max
    mode is honest), ``False`` otherwise. The caller is expected to drop
    ``args.max`` when this returns ``False``.

    The chooser screen mirrors the rest of the onboarding (numbered options,
    ``Type 1-N and press Enter`` prompt) so the UX is consistent — no more
    Y/N prompt sandwiched between numbered screens.
    """

    from mtplx.thermal import detect_thermal_control, install_thermal_control

    detection = detect_thermal_control()
    if detection.get("available"):
        return True

    _choice_panel(
        heading="Max mode setup  ·  ThermalForge",
        intro=(
            "Max mode needs a fan controller. ThermalForge (free, open "
            "source) is not installed on this Mac yet."
        ),
        options=[
            (
                "1",
                "Install ThermalForge now (recommended)",
                "Builds from source via git + Xcode CLI tools. One sudo password prompt.",
            ),
            (
                "2",
                "Skip — Max falls back to Medium",
                "No fan boost, but everything else still works.",
            ),
        ],
        border_style="yellow",
    )
    choice = _prompt_choice("Select", ["1", "2"], default="1")
    if choice == "2":
        _print_install_skipped()
        return False

    try:
        from rich.console import Console

        console = Console()
        console.rule("Installing ThermalForge", style="cyan")
        console.print(
            "[dim]Streaming git + swift + sudo output. "
            "Enter your password if prompted.[/dim]"
        )
    except Exception:
        print("\n--- Installing ThermalForge ---")
        print("Streaming git + swift + sudo output. Enter your password if prompted.")

    result = install_thermal_control()
    _print_install_result(result)
    return bool(result.get("ok") and result.get("daemon_ok") is not False)


def _print_install_skipped() -> None:
    try:
        from rich.console import Console

        console = Console()
        console.print(
            "[yellow]  Skipped. Continuing without fan boost — "
            "Max behaves like Medium.[/yellow]\n"
        )
    except Exception:
        print("  Skipped. Continuing without fan boost.\n")


def _print_install_result(result: dict) -> None:
    ok = bool(result.get("ok"))
    daemon_ok = result.get("daemon_ok")
    message = result.get("message") or ("ThermalForge installed." if ok else "Install failed.")

    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.text import Text
    except ImportError:
        prefix = "[ok]" if ok and daemon_ok is not False else "[partial]" if ok else "[fail]"
        print(f"\n  {prefix} {message}\n")
        return

    console = _console()
    if console is None:
        prefix = "[ok]" if ok and daemon_ok is not False else "[partial]" if ok else "[fail]"
        print(f"\n  {prefix} {message}\n")
        return

    if ok and daemon_ok is not False:
        title = Text("ThermalForge ready", style="bold green")
        border = "green"
    elif ok:
        title = Text("ThermalForge installed (daemon pending)", style="bold yellow")
        border = "yellow"
    else:
        title = Text("ThermalForge install did not complete", style="bold red")
        border = "red"
    panel = Panel(
        Text(message, style=""),
        title=title,
        title_align="left",
        border_style=border,
        padding=(1, 2),
        expand=False,
    )
    console.print()
    console.print(panel)
    console.print()


# ---------- top-level flow --------------------------------------------------
def confirm_same_as_last(last: dict) -> bool:
    """Ask the user whether to reuse the last configuration."""

    try:
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text
    except ImportError:
        print()
        print("  Last time you used:")
        print(f"    Model:     {last.get('model', '?')}")
        print(f"    Mode:      {mode_label(last)}")
        print(f"    Interface: {interface_label(last.get('target'))}")
        print()
        answer = input("  Use the same configuration? [Y/n] ").strip().lower()
        return answer in {"", "y", "yes", "same"}

    console = _console()
    if console is None:
        print()
        print("  Last time you used:")
        print(f"    Model:     {last.get('model', '?')}")
        print(f"    Mode:      {mode_label(last)}")
        print(f"    Interface: {interface_label(last.get('target'))}")
        print()
        answer = input("  Use the same configuration? [Y/n] ").strip().lower()
        return answer in {"", "y", "yes", "same"}

    table = Table.grid(padding=(0, 2))
    table.add_column(style="dim", justify="right", no_wrap=True)
    table.add_column(no_wrap=False)
    table.add_row("Model", str(last.get("model", "?")))
    table.add_row("Mode", mode_label(last))
    table.add_row("Interface", interface_label(last.get("target")))
    panel = Panel(
        table,
        title=Text("Welcome back", style="bold cyan"),
        title_align="left",
        border_style="cyan",
        padding=(1, 2),
        expand=False,
    )
    console.print()
    console.print(panel)
    console.print()
    answer = input("  Use the same configuration? [Y/n] ").strip().lower()
    return answer in {"", "y", "yes", "same"}


def _quickstart_state_is_reusable(last: dict | None) -> bool:
    if not isinstance(last, dict):
        return False
    # Stable/safe remains available by explicit flag, but old saved states
    # should not keep showing it in the speed-first onboarding path.
    return str(last.get("profile") or "") == "performance-cold"


def run_quickstart_flow(
    *,
    fresh: bool = False,
    configured_model: str | None = None,
) -> dict | None:
    """Decide between 'same as last time' or fresh onboarding.

    Returns the chosen config dict (with keys ``model``, ``profile``, ``max``,
    ``target``), or ``None`` if the user aborted (Ctrl-C / EOF).

    ``configured_model`` is the model already resolved from
    ``~/.mtplx/config.toml`` (if any). When set and not the canonical default,
    screen 1 surfaces it as the top choice so accepting the default does not
    force a re-download of an HF mirror that's already on disk elsewhere.
    """

    last = None if fresh else load_state()
    if not _quickstart_state_is_reusable(last):
        last = None
    try:
        if last and not fresh:
            if confirm_same_as_last(last):
                # If the saved state says Max but ThermalForge has gone away
                # (uninstalled, new machine) we must re-offer the install
                # rather than silently dump a JSON warning at runtime.
                if last.get("max") and not ensure_thermal_control_installed():
                    last = dict(last)
                    last["max"] = False
                    save_state(last)
                return last
        else:
            _print_welcome()
        choice = run_onboarding_screens(configured_model=configured_model)
        save_state(choice)
        _print_summary(choice)
        return choice
    except (KeyboardInterrupt, EOFError):
        try:
            print()
        except Exception:  # pragma: no cover - stdout is closed
            pass
        return None


# ---------- label helpers ---------------------------------------------------
def mode_label(state: dict) -> str:
    profile = state.get("profile", "performance-cold")
    if state.get("max"):
        return "Max  ·  Medium path plus fans pinned at 100%, ~2.24x"
    if profile == "performance-cold":
        return "Medium  ·  native-MTP speed path, ~2.2x burst (not sustained)"
    if profile == "stable":
        return "Stable  ·  exact/staged long-reply path"
    return str(profile)


def interface_label(target: str | None) -> str:
    if target in ("openwebui", "open-webui", "web"):
        return "Web UI  ·  browser"
    if target in ("cli", "terminal"):
        return "CLI  ·  this terminal"
    return target or "?"
