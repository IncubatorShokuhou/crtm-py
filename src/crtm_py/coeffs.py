"""Locate or unpack official CRTM coefficient tarballs.

The binaries are NOAA / JCSDA data and are not part of this repository.
This module only selects and unpacks files. SpcCoeff, ODPS TauCoeff,
CloudCoeff and the infrared emissivity tables are read by
``crtm_py.spc``, ``crtm_py.odps``, ``crtm_py.cloudlut`` and ``crtm_py.emis``
once :func:`crtm_py.forward.ForwardModel` finds them.

Default tarball (CRTM REL-2.4.0 fix, about 3.4 GB)::

    https://bin.ssec.wisc.edu/pub/s4/CRTM/fix_REL-2.4.0.tgz

``ftp://ftp.ssec.wisc.edu/pub/s4/CRTM/`` redirects there. The same
directory also holds ``fix_REL-2.3.0_emc.tgz``, which is closer to the
CRTM vintage linked by UPP v4.1. UPP looks for ``abi_g16.SpcCoeff.bin``
in the run directory and falls back to ``abi_gr`` when that file is
absent (``link_crtm_fix.sh`` in ``dtc_post_v4.1.0`` only links ``abi_gr``).

A download requires ``yes=True``. Pass a local tarball to unpack without
touching the network.
"""

from __future__ import annotations

import json
import os
import tarfile
from pathlib import Path
from urllib.request import urlretrieve

RELEASES = {
    "2.4.0-jedi": "https://bin.ssec.wisc.edu/pub/s4/CRTM/file/crtm_coefficients_jedi_release_2.4.0.tar.gz",
    "2.4.0": "https://bin.ssec.wisc.edu/pub/s4/CRTM/fix_REL-2.4.0.tgz",
    "2.4.0-emc": "https://bin.ssec.wisc.edu/pub/s4/CRTM/fix_REL-2.4.0_emc.tgz",
    "2.3.0-emc": "https://bin.ssec.wisc.edu/pub/s4/CRTM/fix_REL-2.3.0_emc.tgz",
}

_PATH: Path | None = None


def coefficient_releases() -> dict[str, str]:
    """Map of release name to the tarball URL this package knows about."""
    return dict(RELEASES)


def default_cache_dir() -> Path:
    """``~/.cache/crtm-py/coefficients``."""
    return Path.home() / ".cache" / "crtm-py" / "coefficients"


def set_coeff_path(path) -> Path:
    """Remember a coefficient directory for this process."""
    global _PATH
    _PATH = Path(path).expanduser().resolve()
    return _PATH


def coeff_path() -> Path | None:
    """Configured directory, else ``CRTM_PY_COEFF_DIR`` / ``CRTM_COEFF_DIR`` / ``FIXCRTM``.

    Returns ``None`` when nothing is configured and the default cache
    directory does not exist yet.
    """
    if _PATH is not None:
        return _PATH
    for key in ("CRTM_PY_COEFF_DIR", "CRTM_COEFF_DIR", "FIXCRTM"):
        env = os.environ.get(key)
        if env:
            return Path(env).expanduser()
    default = default_cache_dir()
    if default.is_dir():
        return default
    return None


def _wanted(name: str, sensors: set[str]) -> bool:
    lower = name.lower()
    if "emis" in lower and lower.endswith((".bin", ".nc", ".nc4", ".h5")):
        return True
    if name in ("CloudCoeff.bin", "AerosolCoeff.bin"):
        return True
    for sensor in sensors:
        if name in (f"{sensor}.SpcCoeff.bin", f"{sensor}.TauCoeff.bin"):
            return True
        if name.lower() in (
            f"{sensor}.spccoeff.nc".lower(),
            f"{sensor}.taucoeff.nc".lower(),
        ):
            return True
    return False


def _rank(member_name: str) -> int:
    low = member_name.lower()
    if "little_endian" in low:
        return 0
    if "big_endian" in low:
        return 2
    return 1


def select_members(names: list[str], sensors: tuple[str, ...]) -> list[str]:
    """Pick archive member names, preferring little-endian copies.

    Parameters
    ----------
    names : list of str
        Paths inside the tarball.
    sensors : tuple of str
        Sensor ids such as ``abi_g16``.

    Returns
    -------
    list of str
        One member path per output basename.
    """
    wanted = {s.strip().lower() for s in sensors}
    best: dict[str, tuple[int, str]] = {}
    for member in names:
        base = Path(member).name
        if not _wanted(base, wanted):
            continue
        rank = _rank(member)
        current = best.get(base)
        if current is None or rank < current[0]:
            best[base] = (rank, member)
    return [pair[1] for pair in best.values()]


def unpack_coefficients(tarball, dest, sensors: tuple[str, ...] = ("abi_g16", "abi_g17", "abi_gr")) -> Path:
    """Extract the ABI / cloud / emissivity members of a CRTM fix tarball.

    Files are written under ``dest`` by basename, so a
    ``Little_Endian/abi_g16.SpcCoeff.bin`` entry becomes
    ``dest/abi_g16.SpcCoeff.bin``. Big-endian copies are skipped when a
    little-endian member with the same basename exists. A
    ``crtm_py_manifest.json`` records what was written.

    Parameters
    ----------
    tarball : path-like
        ``.tgz`` / ``.tar.gz`` file already on disk.
    dest : path-like
        Coefficient directory.
    sensors : tuple of str
        Sensor ids to keep.

    Returns
    -------
    Path
        ``dest``.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    tarball = Path(tarball)
    with tarfile.open(tarball, "r:*") as tar:
        names = [m.name for m in tar.getmembers() if m.isfile()]
        chosen = select_members(names, sensors)
        if not chosen:
            raise FileNotFoundError(
                f"no ABI SpcCoeff/TauCoeff, CloudCoeff or emissivity members in {tarball}"
            )
        written = []
        for member_name in chosen:
            member = tar.getmember(member_name)
            target = dest / Path(member_name).name
            src = tar.extractfile(member)
            if src is None:
                continue
            with src, target.open("wb") as out:
                while True:
                    block = src.read(1024 * 1024)
                    if not block:
                        break
                    out.write(block)
            written.append(target.name)
    manifest = {
        "tarball": str(tarball),
        "sensors": list(sensors),
        "files": sorted(written),
    }
    (dest / "crtm_py_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    set_coeff_path(dest)
    return dest


def fetch_coefficients(
    dest=None,
    *,
    release: str = "2.4.0",
    url: str | None = None,
    sensors: tuple[str, ...] = ("abi_g16", "abi_g17", "abi_gr"),
    local_tarball=None,
    yes: bool = False,
    force: bool = False,
) -> Path:
    """Download, if asked, and unpack CRTM fix files.

    Parameters
    ----------
    dest : path-like, optional
        Defaults to :func:`default_cache_dir`.
    release : str
        Key of :func:`coefficient_releases` when ``url`` is omitted.
    url : str, optional
        Override the tarball URL.
    sensors : tuple of str
        Sensor ids passed to :func:`unpack_coefficients`.
    local_tarball : path-like, optional
        Unpack this file instead of downloading.
    yes : bool
        Required for a network download. The REL-2.4.0 fix file is about
        3.4 GB.
    force : bool
        Download again even if a tarball is already cached.

    Returns
    -------
    Path
        Coefficient directory.
    """
    dest = Path(dest) if dest is not None else default_cache_dir()
    if local_tarball is not None:
        return unpack_coefficients(local_tarball, dest, sensors)
    if url is None:
        try:
            url = RELEASES[release]
        except KeyError as exc:
            known = ", ".join(sorted(RELEASES))
            raise KeyError(f"unknown release {release!r}; known releases: {known}") from exc
    if not yes:
        raise RuntimeError(
            f"refusing to download {url} (multi-gigabyte CRTM fix tarball); "
            "pass yes=True, or unpack a local file with local_tarball="
        )
    cache = dest.parent / "tarballs"
    cache.mkdir(parents=True, exist_ok=True)
    filename = url.rstrip("/").rsplit("/", 1)[-1]
    target = cache / filename
    if force or not target.is_file():
        urlretrieve(url, target)  # noqa: S310 — URL is the documented CRTM host or a caller override
    return unpack_coefficients(target, dest, sensors)
