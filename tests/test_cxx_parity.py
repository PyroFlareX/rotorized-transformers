"""Bit-exact topology and numeric cascade parity between ``runtime/interfaces/rotor_topology.hpp`` and Python."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from rotor.math import CascadeSpec, RotorCascade, RotorLinear, Topology, stage_pairs

pytestmark = pytest.mark.cxx

INTERFACES = Path(__file__).resolve().parents[1] / "runtime" / "interfaces"

DRIVER = r"""
#include <cstdio>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "rotor_topology.hpp"

static std::vector<double> read_vec(size_t n) {
    std::vector<double> v(n);
    for (auto& e : v) {
        if (!(std::cin >> e)) throw std::runtime_error("short stdin");
    }
    return v;
}

int main() {
    std::string cmd;
    while (std::cin >> cmd) {
        if (cmd == "P") {
            std::string kind;
            unsigned n;
            unsigned long long stage, seed;
            std::cin >> kind >> n >> stage >> seed;
            for (const auto& p : rotor::stage_pairs(kind, n, stage, seed)) std::printf("%u %u ", p.a, p.b);
            std::printf("\n");
        } else if (cmd == "C") {
            std::string kind;
            unsigned in, out, k, m, r, has_bias, rows;
            unsigned long long seed;
            std::cin >> in >> out >> k >> m >> kind >> seed >> r >> has_bias >> rows;
            rotor_cascade_spec spec{};
            spec.in_features = in;
            spec.out_features = out;
            spec.num_rotors = k;
            spec.stages_per_rotor = m;
            spec.topology_kind = rotor::topology_kind_from_string(kind);
            spec.adapter_kind = ROTOR_ADAPTER_PAD_TRUNCATE;
            spec.residual_rank = r;
            spec.has_bias = has_bias;
            spec.topology_seed = seed;
            const unsigned n = rotor_cascade_width(&spec);
            const size_t per_rotor =
                rotor::angle_count(static_cast<rotor::TopologyKind>(spec.topology_kind), n, m, seed);
            std::vector<double> angles = read_vec(per_rotor * k);
            std::vector<double> d_in = read_vec(in);
            std::vector<double> d_mid = read_vec(static_cast<size_t>(k - 1) * n);
            std::vector<double> d_out = read_vec(out);
            std::vector<double> U = read_vec(static_cast<size_t>(out) * r);
            std::vector<double> V = read_vec(static_cast<size_t>(r) * in);
            std::vector<double> bias = read_vec(has_bias ? out : 0);
            rotor::CascadeParams<double> w;
            w.angles = angles.data();
            w.d_in = d_in.data();
            w.d_mid = d_mid.data();
            w.d_out = d_out.data();
            w.res_u = U.data();
            w.res_v = V.data();
            w.bias = bias.data();
            std::vector<double> y(out);
            for (unsigned row = 0; row < rows; ++row) {
                std::vector<double> x = read_vec(in);
                rotor::apply_cascade<double>(spec, w, x.data(), y.data());
                for (unsigned o = 0; o < out; ++o) std::printf("%.17g ", y[o]);
                std::printf("\n");
            }
        } else {
            throw std::runtime_error("unknown command " + cmd);
        }
    }
    return 0;
}
"""

PAIR_CASES = [
    # (kind, n, stage, seed)
    ("brick", 2, 0, 0),
    ("brick", 7, 0, 0),
    ("brick", 7, 1, 0),
    ("brick", 8, 3, 0),
    ("butterfly", 2, 5, 0),
    ("butterfly", 5, 0, 0),
    ("butterfly", 5, 2, 0),
    ("butterfly", 5, 3, 0),  # wraps: 3 % ceil(log2 5)
    ("butterfly", 6, 1, 0),
    ("butterfly", 9, 3, 0),
    ("butterfly", 16, 2, 0),
    ("butterfly", 100, 6, 0),
    ("butterfly", 129, 7, 0),
    ("seeded", 2, 0, 0),
    ("seeded", 7, 0, 0),
    ("seeded", 7, 4, 3),
    ("seeded", 10, 1, 1),
    ("seeded", 33, 5, 12345),
    ("seeded", 100, 2, 7),
    ("seeded", 100, 9, 0xDEADBEEFCAFEF00D),
    ("seeded", 257, 11, 2**63 + 5),
    ("brick", 1, 0, 0),
    ("butterfly", 1, 0, 0),
    ("seeded", 1, 0, 0),
]

CASCADE_CASES = [
    # (in, out, rotors, stages_per_rotor, kind, seed, residual_rank, bias)
    (6, 6, 2, 3, "butterfly", 0, 0, False),
    (5, 9, 3, 4, "seeded", 7, 2, True),
    (9, 5, 2, 3, "brick", 0, 0, True),
    (7, 7, 1, 7, "brick", 0, 0, False),
    (10, 13, 3, 4, "seeded", 0xDEADBEEFCAFEF00D, 3, True),
    (12, 7, 2, 4, "butterfly", 0, 1, False),
    (11, 11, 2, 1, "butterfly", 0, 0, False),
]


@pytest.fixture(scope="module")
def driver(tmp_path_factory) -> Path:
    cxx = shutil.which("c++") or shutil.which("g++")
    if cxx is None:
        pytest.skip("no C++ compiler (c++/g++) on PATH")
    work = tmp_path_factory.mktemp("cxx_parity")
    src = work / "driver.cpp"
    exe = work / "driver"
    src.write_text(DRIVER)
    proc = subprocess.run(
        [cxx, "-std=c++17", "-O1", f"-I{INTERFACES}", str(src), "-o", str(exe)],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, f"C++ compile failed:\n{proc.stderr}"
    return exe


def _run(exe: Path, stdin: str) -> list[str]:
    proc = subprocess.run([str(exe)], input=stdin, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, f"driver failed:\n{proc.stderr}"
    return proc.stdout.splitlines()


def test_stage_pairs_match_python_exactly(driver: Path) -> None:
    stdin = "".join(f"P {kind} {n} {stage} {seed}\n" for kind, n, stage, seed in PAIR_CASES)
    lines = _run(driver, stdin)
    assert len(lines) == len(PAIR_CASES)
    for (kind, n, stage, seed), line in zip(PAIR_CASES, lines, strict=True):
        flat = [int(t) for t in line.split()]
        got = list(zip(flat[0::2], flat[1::2], strict=True))
        expected = list(stage_pairs(kind, n, stage, seed))
        assert got == expected, f"pair mismatch for {(kind, n, stage, seed)}"


def _fmt(t: torch.Tensor) -> str:
    return " ".join(repr(float(v)) for v in t.reshape(-1).tolist())


@pytest.mark.parametrize("case", CASCADE_CASES, ids=lambda c: f"{c[4]}-{c[0]}x{c[1]}-k{c[2]}m{c[3]}-r{c[6]}")
def test_cascade_forward_matches_python(driver: Path, case) -> None:
    in_f, out_f, k, m, kind, seed, rank, has_bias = case
    spec = CascadeSpec(
        in_features=in_f,
        out_features=out_f,
        num_rotors=k,
        stages_per_rotor=m,
        topology=Topology(kind=kind, seed=seed),
        residual_rank=rank,
    )
    gen = torch.Generator().manual_seed(1234 + in_f * 31 + out_f)
    cascade = RotorCascade(spec, dtype=torch.float64)
    with torch.no_grad():
        cascade.angles.copy_(torch.randn(cascade.angles.shape, generator=gen, dtype=torch.float64))
        cascade.d_in.copy_(torch.rand(cascade.d_in.shape, generator=gen, dtype=torch.float64) + 0.5)
        cascade.d_mid.copy_(torch.rand(cascade.d_mid.shape, generator=gen, dtype=torch.float64) + 0.5)
        cascade.d_out.copy_(torch.rand(cascade.d_out.shape, generator=gen, dtype=torch.float64) + 0.5)
        if cascade.residual is not None:
            cascade.residual.U.copy_(torch.randn(cascade.residual.U.shape, generator=gen, dtype=torch.float64))
            cascade.residual.V.copy_(torch.randn(cascade.residual.V.shape, generator=gen, dtype=torch.float64))
    bias = torch.randn(out_f, generator=gen, dtype=torch.float64) if has_bias else None
    layer = RotorLinear(cascade, bias=bias)
    rows = 3
    x = torch.randn(rows, in_f, generator=gen, dtype=torch.float64)
    with torch.no_grad():
        expected = layer(x)

    parts = [f"C {in_f} {out_f} {k} {m} {kind} {seed} {rank} {int(has_bias)} {rows}"]
    parts.append(_fmt(cascade.angles.detach()))
    parts.append(_fmt(cascade.d_in.detach()))
    parts.append(_fmt(cascade.d_mid.detach()))
    parts.append(_fmt(cascade.d_out.detach()))
    if cascade.residual is not None:
        parts.append(_fmt(cascade.residual.U.detach()))
        parts.append(_fmt(cascade.residual.V.detach()))
    if bias is not None:
        parts.append(_fmt(bias))
    parts.append(_fmt(x))
    lines = _run(driver, "\n".join(parts) + "\n")

    assert len(lines) == rows
    got = torch.tensor([[float(t) for t in line.split()] for line in lines], dtype=torch.float64)
    assert got.shape == expected.shape
    torch.testing.assert_close(got, expected, atol=1e-10, rtol=0.0)
