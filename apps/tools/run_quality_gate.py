"""Quality gate CLI — validates artifacts, JSONL, smoke checks."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sentinel.quality.artifacts import find_latest_run_dir, validate_pipeline_run_dir
from sentinel.quality.smoke import run_basic_smoke_checks

app = typer.Typer(help="Run quality gate checks.")
console = Console()


@app.command()
def main(
    run_dir: Path | None = typer.Option(None, help="Specific run dir to validate"),
    latest: bool = typer.Option(True, help="Validate latest run dir"),
    base_dir: Path = typer.Option(Path("runs"), help="Base runs directory"),
    project_root: Path = typer.Option(Path("."), help="Project root"),
    strict: bool = typer.Option(True, help="Exit with code 1 on failures"),
) -> None:
    console.print(Panel("[bold green]ONS Sentinel Core — Quality Gate[/bold green]"))

    has_errors = False

    # Smoke checks
    smoke = run_basic_smoke_checks(project_root)
    status = "[green]PASS[/green]" if smoke["valid"] else "[red]FAIL[/red]"
    console.print(f"  Smoke checks: {status}")
    if not smoke["valid"]:
        for err in smoke["errors"]:
            console.print(f"    [red]- {err}[/red]")
        has_errors = True

    # Artifact validation
    target_dir = run_dir
    if target_dir is None and latest:
        target_dir = find_latest_run_dir(base_dir)

    if target_dir is not None:
        console.print(f"  Run dir: {target_dir}")
        result = validate_pipeline_run_dir(target_dir)
        status = "[green]PASS[/green]" if result["valid"] else "[red]FAIL[/red]"
        console.print(f"  Artifacts: {status}")

        if result["missing"]:
            console.print("    [yellow]Missing:[/yellow]")
            for m in result["missing"]:
                console.print(f"      - {m}")

        if result["json_errors"]:
            console.print("    [red]JSON errors:[/red]")
            for e in result["json_errors"]:
                console.print(f"      - {e}")

        if result["errors"]:
            console.print("    [red]Errors:[/red]")
            for e in result["errors"]:
                console.print(f"      - {e}")

        # JSONL summary table
        if result["jsonl_results"]:
            table = Table(title="JSONL Files")
            table.add_column("File")
            table.add_column("Lines")
            table.add_column("Valid")
            for r in result["jsonl_results"]:
                name = Path(r["path"]).name
                valid = "[green]ok[/green]" if r["valid"] else f"[red]{r['error']}[/red]"
                table.add_row(name, str(r["line_count"]), valid)
            console.print(table)

        if not result["valid"]:
            has_errors = True
    else:
        console.print("  [dim]No run dir specified or found. Skipping artifact validation.[/dim]")

    console.print()
    if has_errors:
        console.print(Panel("[bold red]Quality gate: FAILED[/bold red]"))
        if strict:
            raise typer.Exit(code=1)
    else:
        console.print(Panel("[bold green]Quality gate: PASSED[/bold green]"))


if __name__ == "__main__":
    app()
