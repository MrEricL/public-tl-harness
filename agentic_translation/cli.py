"""Command-line adapter for source-grounded book translation."""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from .env_config import load_cli_env
from .profiles import BOOK_STRATEGIES, CONTEXTUAL_PROFILE, PORTFOLIO_REVIEW_PROFILE, resolve_book_profile


app = typer.Typer(help="Translate a book with source memory and an optional translator glossary.")
harness_app = typer.Typer(help="Run a source-grounded book translation.")
app.add_typer(harness_app, name="harness")
console = Console()


@app.callback()
def main(
    env_file: Path | None = typer.Option(None, "--env-file", help="Load provider settings from a dotenv-style file."),
) -> None:
    try:
        load_cli_env(env_file)
    except OSError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc


@harness_app.command("run")
def harness_run(
    story: Path = typer.Option(Path("samples/showcase/story.yaml"), "--story"),
    out: Path = typer.Option(..., "--out", "--output-dir"),
    provider_mode: str = typer.Option("live", "--provider-mode", help="Book translation uses live generation."),
    profile: str = typer.Option("deepseek", "--profile", help="Book provider: deepseek."),
    model: str | None = typer.Option(None, "--model"),
    strategy: str = typer.Option("contextual", "--strategy", help="contextual or portfolio-review; exact versioned profiles and legacy adaptive/always-review also work."),
    interaction: str = typer.Option("interactive", "--interaction", help="Use unattended for a book run."),
    memory: str | None = typer.Option(None, "--memory", help="Use source-grounded for a book run."),
    glossary_policy: str | None = typer.Option(None, "--glossary-policy", help="Use run-local-auto for a book run."),
    decision_provider: str | None = typer.Option(None, "--decision-provider", help="Only legacy adaptive screening uses jev."),
    decision_model: str = typer.Option("jev-1.13.0", "--decision-model"),
    max_usd: float = typer.Option(10.0, "--max-usd", min=0.01, help="Estimated API expense ceiling."),
) -> None:
    """Translate the chapters in a story configuration and write a book and manifest."""
    try:
        if strategy not in BOOK_STRATEGIES:
            raise ValueError("Choose --strategy contextual or portfolio-review; adaptive and always-review retain legacy behavior")
        book_profile = resolve_book_profile(strategy)
        if interaction != "unattended" or memory != "source-grounded" or glossary_policy != "run-local-auto":
            raise ValueError("Book runs require --interaction unattended --memory source-grounded --glossary-policy run-local-auto")
        if provider_mode != "live" or profile != "deepseek":
            raise ValueError("Book runs require --provider-mode live --profile deepseek")
        if book_profile in {CONTEXTUAL_PROFILE, PORTFOLIO_REVIEW_PROFILE}:
            if decision_provider not in {None, "none"}:
                raise ValueError("Contextual and portfolio-review profiles use no Jev router; remove --decision-provider")
        elif decision_provider not in {None, "jev"}:
            raise ValueError("Legacy adaptive screening uses --decision-provider jev")

        from .autonomous_book import run_unattended_book

        manifest = run_unattended_book(
            story, out, model=model or "deepseek-flash", decision_model=decision_model,
            max_usd=max_usd, strategy=strategy,
        )
        console.print(f"Profile: {manifest.get('profile', book_profile)}")
        console.print(f"Status: {manifest['status']}")
        console.print(f"Manifest: {out / 'manifest.json'}")
        delivery_name = manifest.get("delivery")
        delivery_path = out / delivery_name if isinstance(delivery_name, str) and delivery_name else None
        if delivery_path is not None and delivery_path.is_file():
            console.print(f"Delivery: {delivery_path}")
        if manifest["status"] == "failed_no_output":
            cause = (manifest.get("error") or {}).get("type") or "no chapter produced output"
            console.print(f"[red]Error:[/red] the book run produced no output ({cause}). "
                          f"Check the provider key (DEEPSEEK_API_KEY) and {out / 'manifest.json'}.")
            raise typer.Exit(1)
    except typer.Exit:
        raise
    except (OSError, ValueError, RuntimeError, TypeError, KeyError) as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc
