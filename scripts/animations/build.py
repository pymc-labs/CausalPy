"""Build the documentation animation from committed source and fitted data."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "scripts" / "animations"
OUTPUT = ROOT / "docs" / "source" / "_static" / "animations"


def main():
    html = (SOURCE / "templates" / "interrupted-time-series.html").read_text()
    for marker, path in {
        "__STYLE__": SOURCE / "player.css",
        "__PLAYER__": SOURCE / "player.js",
        "__DATA__": SOURCE / "data" / "interrupted-time-series.json",
    }.items():
        html = html.replace(marker, path.read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    target = OUTPUT / "interrupted-time-series.html"
    target.write_text(html)
    print(target.relative_to(ROOT))


if __name__ == "__main__":
    main()
