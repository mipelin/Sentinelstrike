"""ONS Sentinel Dashboard — CLI entry point."""

from __future__ import annotations

from pathlib import Path

import typer
import uvicorn

app = typer.Typer(help="ONS Sentinel Tactical Replay Dashboard")


@app.command()
def run(
    host: str = typer.Option("0.0.0.0", help="Bind host"),
    port: int = typer.Option(8787, help="Bind port"),
    base_dir: str = typer.Option("runs", help="Base directory for run artifacts"),
) -> None:
    """Launch the tactical replay dashboard."""
    from sentinel.dashboard.app import create_app

    app_instance = create_app(base_dir=Path(base_dir))
    typer.echo(f"ONS Sentinel Dashboard starting on http://{host}:{port}")
    typer.echo(f"Base directory: {base_dir}")
    uvicorn.run(app_instance, host=host, port=port, log_level="info")


if __name__ == "__main__":
    app()
