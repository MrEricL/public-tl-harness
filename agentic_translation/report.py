from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape


PACKAGE_TEMPLATE_DIR = Path(__file__).resolve().parent / "templates"


def render_report(
    *,
    output_path: Path,
    context: dict[str, object],
    template_dir: Path | None = None,
) -> Path:
    env = Environment(
        loader=FileSystemLoader(str(template_dir or PACKAGE_TEMPLATE_DIR)),
        # The template is report.html.j2: ".html" alone would leave autoescape off.
        autoescape=select_autoescape(["html", "xml", "html.j2"]),
    )
    template = env.get_template("report.html.j2")
    output_path.write_text(template.render(**context), encoding="utf-8")
    return output_path
