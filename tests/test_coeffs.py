import json
import tarfile
from pathlib import Path

import pytest

from crtm_py.cli import main
from crtm_py.coeffs import coefficient_releases, fetch_coefficients, select_members, unpack_coefficients


def test_known_release_urls_point_at_ssec():
    releases = coefficient_releases()
    assert releases["2.4.0"] == "https://bin.ssec.wisc.edu/pub/s4/CRTM/fix_REL-2.4.0.tgz"
    assert "2.3.0-emc" in releases


def test_little_endian_member_wins():
    names = [
        "fix/SpcCoeff/Big_Endian/abi_g16.SpcCoeff.bin",
        "fix/SpcCoeff/Little_Endian/abi_g16.SpcCoeff.bin",
        "fix/SpcCoeff/Little_Endian/amsua_n15.SpcCoeff.bin",
        "fix/TauCoeff/ODPS/Little_Endian/abi_g17.TauCoeff.bin",
        "CloudCoeff.bin",
        "IR.EmisCoeff.bin",
    ]
    chosen = select_members(names, ("abi_g16", "abi_g17"))
    bases = {Path(name).name for name in chosen}
    assert "abi_g16.SpcCoeff.bin" in bases
    assert "amsua_n15.SpcCoeff.bin" not in bases
    assert any("Little_Endian" in name and name.endswith("abi_g16.SpcCoeff.bin") for name in chosen)
    assert "CloudCoeff.bin" in bases
    assert "IR.EmisCoeff.bin" in bases


def test_unpack_flattens_and_skips_other_sensors(tmp_path):
    tarball = tmp_path / "fix.tgz"
    with tarfile.open(tarball, "w:gz") as tar:
        payload = {
            "fix/SpcCoeff/Big_Endian/abi_g16.SpcCoeff.bin": b"BIG",
            "fix/SpcCoeff/Little_Endian/abi_g16.SpcCoeff.bin": b"LITTLE",
            "fix/SpcCoeff/Little_Endian/amsua_n15.SpcCoeff.bin": b"SKIP",
            "CloudCoeff.bin": b"CLOUD",
            "nesdis/AerosolCoeff.bin": b"AERO",
        }
        for name, data in payload.items():
            source = tmp_path / Path(name).name
            # Disambiguate the two abi files that share a basename on disk.
            source = tmp_path / name.replace("/", "_")
            source.write_bytes(data)
            info = tarfile.TarInfo(name)
            info.size = len(data)
            with source.open("rb") as handle:
                tar.addfile(info, handle)
    dest = tmp_path / "coeff"
    unpack_coefficients(tarball, dest, sensors=("abi_g16",))
    assert (dest / "abi_g16.SpcCoeff.bin").read_bytes() == b"LITTLE"
    assert (dest / "CloudCoeff.bin").read_bytes() == b"CLOUD"
    assert (dest / "AerosolCoeff.bin").read_bytes() == b"AERO"
    assert not (dest / "amsua_n15.SpcCoeff.bin").exists()
    manifest = json.loads((dest / "crtm_py_manifest.json").read_text())
    assert "abi_g16.SpcCoeff.bin" in manifest["files"]


def test_fetch_refuses_a_network_download(tmp_path):
    with pytest.raises(RuntimeError, match="yes=True"):
        fetch_coefficients(tmp_path, yes=False)


def test_cli_list_and_refusal(capsys):
    assert main(["fetch", "--list"]) == 0
    out = capsys.readouterr().out
    assert "fix_REL-2.4.0.tgz" in out
    assert main(["fetch", "--dest", "/tmp/crtm-py-should-not-download"]) == 2
    err = capsys.readouterr().err
    assert "yes" in err


def test_repository_does_not_vendor_coefficient_binaries():
    root = Path(__file__).resolve().parents[1]
    binaries = [path for path in root.rglob("*") if path.suffix in {".bin", ".tgz"} and ".git" not in path.parts]
    assert binaries == []
