"""``crtm-py fetch`` command."""

from __future__ import annotations

import argparse
import sys

from crtm_py.coeffs import coefficient_releases, default_cache_dir, fetch_coefficients


def main(argv: list[str] | None = None) -> int:
    """Run the coefficient helper. Returns a process status code."""
    parser = argparse.ArgumentParser(prog="crtm-py", description="CRTM coefficient cache for crtm-py")
    sub = parser.add_subparsers(dest="cmd", required=True)
    fetch = sub.add_parser("fetch", help="list or unpack official CRTM coefficient tarballs")
    fetch.add_argument("--list", action="store_true", help="print known tarball URLs and exit")
    fetch.add_argument("--release", default="2.4.0", help="release key (default: 2.4.0)")
    fetch.add_argument("--url", default=None, help="tarball URL instead of --release")
    fetch.add_argument("--dest", default=None, help="coefficient directory (default: ~/.cache/crtm-py/coefficients)")
    fetch.add_argument(
        "--sensors",
        default="abi_g16,abi_g17,abi_gr",
        help="comma-separated sensor ids to keep",
    )
    fetch.add_argument("--local-tarball", default=None, help="unpack this .tgz instead of downloading")
    fetch.add_argument(
        "--yes",
        action="store_true",
        help="actually download the multi-gigabyte fix tarball",
    )
    fetch.add_argument("--force", action="store_true", help="download again even if the tarball is cached")
    args = parser.parse_args(argv)
    if args.cmd == "fetch":
        return _fetch(args)
    parser.error(f"unknown command {args.cmd}")
    return 2


def _fetch(args) -> int:
    if args.list:
        for name, url in coefficient_releases().items():
            print(f"{name}\t{url}")
        print(f"cache\t{default_cache_dir()}")
        return 0
    sensors = tuple(part.strip() for part in args.sensors.split(",") if part.strip())
    try:
        dest = fetch_coefficients(
            args.dest,
            release=args.release,
            url=args.url,
            sensors=sensors,
            local_tarball=args.local_tarball,
            yes=args.yes,
            force=args.force,
        )
    except (RuntimeError, FileNotFoundError, KeyError, OSError) as exc:
        print(f"crtm-py: {exc}", file=sys.stderr)
        return 2
    print(dest)
    return 0
