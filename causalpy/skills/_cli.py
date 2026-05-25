#   Copyright 2026 - 2026 The PyMC Labs Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""CLI entry point: ``causalpy skills install|uninstall|list|check``."""

from __future__ import annotations

import argparse
import sys

from causalpy.version import __version__


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--project-dir`` and ``--platform`` arguments to *parser*."""
    parser.add_argument(
        "--project-dir",
        default=".",
        help="Path to the project directory (default: current directory).",
    )
    parser.add_argument(
        "--platform",
        choices=["cursor", "claude", "copilot", "windsurf", "generic"],
        default=None,
        help="Target AI platform. Auto-detected if omitted.",
    )


def _cmd_install(args: argparse.Namespace) -> None:
    """Handle ``causalpy skills install``."""
    from causalpy.skills._installer import install

    result = install(project_dir=args.project_dir, platform=args.platform)
    for plat, paths in result.items():
        print(f"\n  [{plat}] Installed {len(paths)} file(s):")
        for p in paths:
            print(f"    {p}")
    print(f"\n  Skills version: {__version__}")


def _cmd_uninstall(args: argparse.Namespace) -> None:
    """Handle ``causalpy skills uninstall``."""
    from causalpy.skills._installer import uninstall

    result = uninstall(project_dir=args.project_dir, platform=args.platform)
    any_removed = False
    for plat, paths in result.items():
        if paths:
            any_removed = True
            print(f"\n  [{plat}] Removed {len(paths)} item(s):")
            for p in paths:
                print(f"    {p}")
    if not any_removed:
        print("  No installed skills found to remove.")


def _cmd_list(args: argparse.Namespace) -> None:
    """Handle ``causalpy skills list``."""
    from causalpy.skills._installer import list_skills

    skills = list_skills()
    print(f"\n  CausalPy v{__version__} bundles {len(skills)} user skill(s):\n")
    for name in skills:
        print(f"    - {name}")
    print()


def _cmd_check(args: argparse.Namespace) -> None:
    """Handle ``causalpy skills check``."""
    from causalpy.skills._installer import check_version

    installed = check_version(project_dir=args.project_dir)
    if installed is None:
        print("  No installed CausalPy skills found in this project.")
    elif installed == __version__:
        print(f"  Skills are up to date (v{__version__}).")
    else:
        print(
            f"  Installed skills are v{installed}, "
            f"but CausalPy is v{__version__}.\n"
            f"  Run `causalpy skills install` to update."
        )


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``causalpy``."""
    parser = argparse.ArgumentParser(
        prog="causalpy",
        description="CausalPy command-line tools.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"CausalPy {__version__}",
    )
    sub = parser.add_subparsers(dest="command")

    skills_parser = sub.add_parser("skills", help="Manage AI agent skills.")
    skills_sub = skills_parser.add_subparsers(dest="skills_command")

    install_p = skills_sub.add_parser(
        "install",
        help="Install CausalPy skills into your project.",
    )
    _add_common_args(install_p)

    uninstall_p = skills_sub.add_parser(
        "uninstall",
        help="Remove previously installed CausalPy skills.",
    )
    _add_common_args(uninstall_p)

    skills_sub.add_parser(
        "list",
        help="List bundled skills.",
    )

    check_p = skills_sub.add_parser(
        "check",
        help="Check if installed skills are up to date.",
    )
    check_p.add_argument(
        "--project-dir",
        default=".",
        help="Path to the project directory (default: current directory).",
    )

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "skills":
        if args.skills_command is None:
            skills_parser.print_help()
            sys.exit(0)
        dispatch = {
            "install": _cmd_install,
            "uninstall": _cmd_uninstall,
            "list": _cmd_list,
            "check": _cmd_check,
        }
        dispatch[args.skills_command](args)


if __name__ == "__main__":
    main()
