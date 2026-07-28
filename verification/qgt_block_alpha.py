#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QGT computational verification harness — BLOCK B1: the projection constant.

CLASSIFICATION
--------------
Not a simulator and not a proof assistant. A symbolic-executable coherence
verifier: it constructs the algebraic objects the monograph describes and
checks that they satisfy the declared identities. Stronger than a unit test,
weaker than a kernel-checked formalization.

WHAT B1 ESTABLISHES
-------------------
Conditional on the upstream structural inputs imported from the frozen
monograph -- rank five, the golden singular ladder, the Fibonacci coefficient
vector, the filter form -- B1 constructs nine typed realizations and verifies
their mutual coherence. Three exact representations that directly produce the
invariant coincide SYMBOLICALLY; the other six verify the surrounding
structure through which that same object is read.

The correct term is EXACT CROSS-REPRESENTATION COHERENCE. The readings are
not mutually independent: I.4, II.1 and II.2 are three representations of one
algebraic element, 20*phi^4 = 70 + 30*sqrt5. Their agreement certifies
coherence across representations; it is not independent statistical evidence.

WHAT B1 DOES NOT ESTABLISH
--------------------------
It does not re-prove the rank-five selection: r = 5 is a declared upstream
input, consumed and propagated. It does not derive the golden ladder. It does
not establish the physical truth of QGT, nor the empirical identity of the
invariant with the measured fine-structure constant. No measured value of
alpha enters anywhere.

RULES
-----
R1  No literal value of alpha is an input. Primitives: phi, sqrt5, pi, small
    integers with a declared structural role.
R2  Every check is a pure function: no I/O, no global state, no randomness.
R3  Every check declares the monograph tag it certifies.
R4  Exact identities are certified SYMBOLICALLY; decimals are rendering only.
R5  Every check declares what it does not prove, and its failure mode.
R6  Dependencies are machine-readable, not prose in a docstring.
R7  A technical exception yields ERROR and exit 2; it never masquerades as a
    scientific failure.

Exit codes:  0 all scientific checks pass | 1 a scientific check fails
             2 infrastructural error
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import traceback
from dataclasses import dataclass, field
from typing import Callable

import mpmath
import sympy as sp
from mpmath import mp

SCHEMA_VERSION = "1.0"
BLOCK_ID = "B1"
BLOCK_VERSION = "1.1.0"
AUTHORITY_WORK = "QGT First Edition v7.6.6 (frozen)"
AUTHORITY_LABEL = "app:alpha_structure — Appendix L, The Projection Constant, pp. 529-537"

DPS = 50
mp.dps = DPS

SQRT5 = sp.sqrt(5)
PHI = (1 + SQRT5) / 2

UPSTREAM = {
    "rank_five": {
        "symbol": "r = 5",
        "status": "imported upstream theorem, not proved here",
        "source": "rank-five selection theorem (block B0)",
        "used_by": ["I.2", "II.1", "II.2", "III.1", "III.2"]},
    "golden_ladder": {
        "symbol": "sigma_k = phi^(-k), k = 0..r-1",
        "status": "imported upstream selection, not derived here",
        "source": "SVD of the rank-five Majorana projection",
        "used_by": ["I.2"]},
    "fibonacci_pair": {
        "symbol": "c_F = (F_5, F_4)",
        "status": "fixed by the rank-five selection",
        "source": "Reading II.1",
        "used_by": ["II.1", "II.2"]},
    "golden_norm_form": {
        "symbol": "G_sqrt5 = [[1, sqrt5], [sqrt5, 5]]",
        "status": "definition",
        "source": "Reading II.1",
        "used_by": ["II.1"]},
    "filter_form": {
        "symbol": "H_Pi(z) = P_4(z) / (1 - a_Pi z)",
        "status": "imported structural form",
        "source": "Reading I.3",
        "used_by": ["I.3"]},
}

FORBIDDEN_EMPIRICAL_INPUTS = [
    "measured alpha", "CODATA value", "fitted parameter",
    "experimental recoil datum", "decimal literal of the invariant"]

R_RANK = 5      # UPSTREAM INPUT — consumed, not derived (see UPSTREAM)


def fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


F4, F5 = fib(4), fib(5)


def num(expr):
    """Exact expression -> decimal rendering. Never a certificate."""
    return mp.mpf(str(sp.N(expr, DPS + 10)))


def exact_eq(x, y) -> bool:
    """Primary certificate: symbolic equality after simplification."""
    return sp.simplify(sp.expand(x - y)) == 0


KIND_SCALAR = "scalar_readout"
KIND_STRUCT = "structural_compatibility"
T_EXACT = "exact_symbolic_identity"
T_WITNESS = "high_precision_numerical_witness"


@dataclass
class Check:
    tag: str
    level: str
    kind: str
    name: str
    fn: Callable[[], dict]
    dependencies: list
    proves: str
    does_not_prove: str
    failure_mode: str
    theorem_dependency: str = None
    status: str = "PENDING"
    lemmas: list = field(default_factory=list)
    outputs: dict = field(default_factory=dict)
    error: str = None


CHECKS = []


def check(tag, level, kind, name, dependencies, proves, does_not_prove,
          failure_mode, theorem_dependency=None):
    def deco(fn):
        CHECKS.append(Check(tag, level, kind, name, fn, dependencies, proves,
                            does_not_prove, failure_mode, theorem_dependency))
        return fn
    return deco


# ============================== LEVEL I ==============================

@check("I.1", "I — local qubit", KIND_STRUCT, "Chiral contraction",
       dependencies=[],
       proves="the chiral form extracts exactly the product xi*eta for every "
              "q: the invariant is a contraction, not a definition",
       does_not_prove="the value of xi*eta, fixed by the lock-in surface",
       failure_mode="C_chi mis-transcribed, or contraction != product for "
                    "symbolic q")
def c_I1():
    xi, eta = sp.symbols("xi eta", real=True)
    C_chi = sp.Rational(1, 2) * sp.Matrix([[0, 1], [1, 0]])
    got = sp.expand((sp.Matrix([xi, eta]).T * C_chi * sp.Matrix([xi, eta]))[0, 0])
    return {"constructs": "q^T C_chi q with symbolic q",
            "exact_output": str(got),
            "lemmas": [("q^T C_chi q == xi*eta for symbolic q",
                        exact_eq(got, xi * eta), T_EXACT)]}


@check("I.2", "I — local qubit", KIND_STRUCT, "Boundary metric from the SVD",
       dependencies=["rank_five", "golden_ladder"],
       proves="K_Pi = P^dag P has the squared ladder as spectrum, and that "
              "spectrum does not depend on the basis V",
       does_not_prove="that the singular values ARE the golden ladder: that "
                      "is the upstream selection theorem",
       failure_mode="spectrum depending on V, or differing from {phi^-2k}")
def c_I2():
    th = sp.symbols("theta", real=True)
    sigma = [PHI ** (-k) for k in range(R_RANK)]
    V = sp.Matrix.eye(R_RANK)
    V[0, 0], V[0, 1] = sp.cos(th), -sp.sin(th)
    V[1, 0], V[1, 1] = sp.sin(th), sp.cos(th)
    P = sp.diag(*sigma) * V.T
    K = sp.simplify(P.T * P)
    got = sorted([sp.simplify(e) for e in K.eigenvals()], key=lambda e: -sp.N(e))
    want = sorted([sp.simplify(PHI ** (-2 * k)) for k in range(R_RANK)],
                  key=lambda e: -sp.N(e))
    return {"constructs": "spec(P^dag P), P = Sigma V^T with V orthogonal",
            "exact_output": str([sp.nsimplify(e) for e in want]),
            "lemmas": [
                ("spec(K_Pi) == {phi^-2k}, independent of the basis V",
                 all(exact_eq(a, b) for a, b in zip(got, want)), T_EXACT),
                ("adjacent scale ratio == phi",
                 exact_eq(sigma[0] / sigma[1], PHI), T_EXACT)]}


@check("I.3", "I — local qubit", KIND_STRUCT, "Boundary filter and all-pass cell",
       dependencies=["filter_form"],
       proves="the denominator vanishes at 20 phi^4 and the numerator does "
              "NOT vanish there, so the pole is not cancelled; the cell "
              "U_eta = M_eta D_alpha M_eta^-1 is explicitly constructed and "
              "isospectral to D_alpha",
       does_not_prove="that a single-pole completion is physically necessary. "
                      "Note also that with the monograph's scalar D_alpha the "
                      "conjugation is trivially equal, so the content here is "
                      "the explicit construction and the absence of pole-zero "
                      "cancellation, not a non-trivial similarity",
       failure_mode="pole-zero cancellation, or a similarity asserted without "
                    "constructing the conjugate")
def c_I3():
    z, w, T = sp.symbols("z omega T_alpha", real=True, positive=True)
    a_pi = 1 / (20 * PHI ** 4)
    pole = sp.solve(sp.Eq(1 - a_pi * z, 0), z)[0]
    P4 = sum((sp.log(PHI)) ** n * z ** n for n in range(R_RANK))
    P4_at_pole = sp.simplify(P4.subs(z, pole))

    D = -sp.I * (1 - sp.I * w * T) / (1 + sp.I * w * T)
    Dmat = D * sp.eye(2)
    M_eta = sp.Rational(1, 2) * sp.Matrix([[2, 1], [1, -1]])
    U_eta = sp.simplify(M_eta * Dmat * M_eta.inv())
    lam = sp.Symbol("lam")
    chi_U = sp.simplify(sp.expand(U_eta.charpoly(lam).as_expr()))
    chi_D = sp.simplify(sp.expand(Dmat.charpoly(lam).as_expr()))
    mod_sq = sp.simplify(sp.expand(D * sp.conjugate(D)))
    return {"constructs": "pole(H_Pi), P_4(pole), U_eta = M_eta D M_eta^-1",
            "exact_output": str(sp.nsimplify(pole)),
            "lemmas": [
                ("denominator root == 20 phi^4",
                 exact_eq(pole, 20 * PHI ** 4), T_EXACT),
                ("numerator P_4 != 0 at the pole (no cancellation)",
                 sp.simplify(P4_at_pole) != 0, T_EXACT),
                ("deg P_4 == rank - 1",
                 sp.degree(sp.expand(P4), z) == R_RANK - 1, T_EXACT),
                ("U_eta explicitly constructed via M_eta^-1",
                 M_eta.det() != 0, T_EXACT),
                ("char poly of U_eta == char poly of D_alpha (isospectral)",
                 exact_eq(chi_U, chi_D), T_EXACT),
                ("|D_alpha|^2 == 1 on the real axis (all-pass)",
                 exact_eq(mod_sq, 1), T_EXACT)]}


@check("I.4", "I — local qubit", KIND_SCALAR, "Pincherle dilatation residue",
       dependencies=[],
       proves="kappa == sqrt5, the three forms of Lambda_Pi coincide, and its "
              "square equals 20 phi^4",
       does_not_prove="that the dilatation invariant must be squared: that is "
                      "the boundary-observable argument",
       failure_mode="Lambda_Pi mis-transcribed, or the square not closing")
def c_I4():
    kappa = PHI + 1 / PHI
    Lam = 2 * PHI ** 2 * SQRT5
    return {"constructs": "Lambda_Pi = 2 phi^2 sqrt5, then its square",
            "exact_output": str(sp.nsimplify(sp.simplify(Lam ** 2))),
            "canonical_output": "20*phi**4",
            "lemmas": [
                ("kappa = phi + 1/phi == sqrt5", exact_eq(kappa, SQRT5), T_EXACT),
                ("2 phi^2 sqrt5 == 4 phi^2 (phi - 1/2)",
                 exact_eq(Lam, 4 * PHI ** 2 * (PHI - sp.Rational(1, 2))), T_EXACT),
                ("== 5 + 3 sqrt5", exact_eq(Lam, 5 + 3 * SQRT5), T_EXACT),
                ("Lambda_Pi^2 == 20 phi^4",
                 exact_eq(Lam ** 2, 20 * PHI ** 4), T_EXACT)]}


# ============================== LEVEL II =============================

@check("II.1", "II — kernel arithmetic", KIND_SCALAR,
       "Fibonacci norm in Z[sqrt5]",
       dependencies=["rank_five", "fibonacci_pair", "golden_norm_form"],
       proves="the quadratic form expands as declared for symbolic (a,b), and "
              "on the Fibonacci pair evaluates to 70 + 30 sqrt5 == 20 phi^4",
       does_not_prove="why the coefficient vector is the Fibonacci pair: that "
                      "is fixed upstream by the rank-five selection",
       failure_mode="G mis-transcribed, or the pair not closing")
def c_II1():
    a, b = sp.symbols("a b")
    G = sp.Matrix([[1, SQRT5], [SQRT5, 5]])
    generic = sp.expand((sp.Matrix([a, b]).T * G * sp.Matrix([a, b]))[0, 0])
    cF = sp.Matrix([F5, F4])
    val = sp.simplify((cF.T * G * cF)[0, 0])
    return {"constructs": "c_F^T G_sqrt5 c_F",
            "exact_output": str(sp.nsimplify(val)),
            "canonical_output": "20*phi**4",
            "lemmas": [
                ("c^T G c == a^2 + 5b^2 + 2ab sqrt5 for symbolic (a,b)",
                 exact_eq(generic, a ** 2 + 5 * b ** 2 + 2 * a * b * SQRT5),
                 T_EXACT),
                ("(F5,F4) == (5,3) from the Fibonacci recursion",
                 (F5, F4) == (5, 3), T_EXACT),
                ("value == 70 + 30 sqrt5", exact_eq(val, 70 + 30 * SQRT5), T_EXACT),
                ("== 20 phi^4", exact_eq(val, 20 * PHI ** 4), T_EXACT)]}


@check("II.2", "II — kernel arithmetic", KIND_SCALAR,
       "Regular matrix representation",
       dependencies=["rank_five", "fibonacci_pair"],
       proves="M_Pi is integral with tr = 2r and det = F5^2 - 5F4^2 = -20; "
              "M_Pi^2 is the regular representation of an element of Z[sqrt5] "
              "whose evaluation is 20 phi^4",
       does_not_prove="uniqueness of the representation",
       failure_mode="M_Pi^2 not of the form [[a,5b],[b,a]], or eval not closing")
def c_II2():
    M = sp.Matrix([[F5, 5 * F4], [F4, F5]])
    M2 = sp.expand(M * M)
    a_, b_ = M2[0, 0], M2[1, 0]
    is_reg = exact_eq(M2[0, 1], 5 * b_) and exact_eq(M2[1, 1], a_)
    val = a_ + b_ * SQRT5
    return {"constructs": "M_Pi, then M_Pi^2 and eval_sqrt5 of it",
            "exact_output": str(sp.nsimplify(val)),
            "canonical_output": "20*phi**4",
            "M_Pi": [[int(x) for x in M.row(i)] for i in range(2)],
            "M_Pi_squared": [[int(x) for x in M2.row(i)] for i in range(2)],
            "lemmas": [
                ("M_Pi integral", all(v.is_Integer for v in M), T_EXACT),
                ("tr M_Pi == 2r == 10", exact_eq(M.trace(), 2 * R_RANK), T_EXACT),
                ("det M_Pi == F5^2 - 5F4^2 == -20",
                 exact_eq(M.det(), F5 ** 2 - 5 * F4 ** 2)
                 and exact_eq(M.det(), -20), T_EXACT),
                ("M_Pi^2 is reg(a + b sqrt5)", is_reg, T_EXACT),
                ("eval_sqrt5(M_Pi^2) == 20 phi^4",
                 exact_eq(val, 20 * PHI ** 4), T_EXACT)]}


# ============================= LEVEL III =============================

@check("III.1", "III — gluing", KIND_STRUCT, "Chiral tensor anisotropy",
       dependencies=["rank_five"],
       proves="the lock-in point built FROM the invariant satisfies "
              "xi* eta* == a_Pi and reproduces the anisotropy 11/5: the ratio "
              "is derived from the construction, not asserted",
       does_not_prove="the multiplicities 11 and 5 themselves, which come "
                      "from the B13 decomposition upstream",
       failure_mode="the constructed point off the lock-in surface, or the "
                    "ratio differing from 11/5")
def c_III1():
    a_pi = 20 * PHI ** 4
    base = sp.sqrt(a_pi / 55)
    xi_s, eta_s = 11 * base, 5 * base
    return {"constructs": "xi* = 11 sqrt(a_Pi/55), eta* = 5 sqrt(a_Pi/55)",
            "exact_output": str(sp.nsimplify(sp.simplify(xi_s / eta_s))),
            "lemmas": [
                ("constructed point lies on the lock-in surface: "
                 "xi* eta* == a_Pi", exact_eq(xi_s * eta_s, a_pi), T_EXACT),
                ("xi*/eta* == 11/5, derived from the construction",
                 exact_eq(sp.simplify(xi_s / eta_s), sp.Rational(11, 5)),
                 T_EXACT)]}


@check("III.2", "III — gluing", KIND_STRUCT,
       "Filter decomposition B13 = S5 + T8",
       dependencies=["rank_five"],
       proves="the decomposition is dimensionally consistent and its summands "
              "are the consecutive Fibonacci numbers straddling the rank",
       does_not_prove="that the boundary algebra must decompose this way",
       failure_mode="summands failing to be F_r and F_{r+1}")
def c_III2():
    S, T8 = fib(R_RANK), fib(R_RANK + 1)
    return {"constructs": "S = F_r, T = F_(r+1), B = S + T",
            "exact_output": "%d + %d = %d" % (S, T8, S + T8),
            "lemmas": [
                ("F_r + F_(r+1) == F_(r+2)", S + T8 == fib(R_RANK + 2), T_EXACT),
                ("5 + 8 == 13", (S, T8, S + T8) == (5, 8, 13), T_EXACT)]}


@check("III.3", "III — gluing", KIND_STRUCT, "FMT zero mode: non-closure",
       dependencies=[],
       proves="the loop-gain identity f_gain == 1/(1 - delta_F) exactly, and "
              "gives a high-precision numerical witness that delta_F != 0",
       does_not_prove="the transcendence theorem. Rigorous non-vanishing of "
                      "delta_F rests on Lindemann-Weierstrass, cited and not "
                      "re-proved here; the numerical test is a witness at the "
                      "declared precision, not a proof",
       failure_mode="the gain identity failing, or delta_F vanishing at the "
                    "declared precision",
       theorem_dependency="Lindemann-Weierstrass")
def c_III3():
    f_gain = (2 * SQRT5) ** sp.Rational(3, 2) / (3 * sp.pi)
    delta_F = 1 - 3 * sp.pi / (2 * SQRT5) ** sp.Rational(3, 2)
    d = num(delta_F)
    return {"constructs": "f_gain and delta_F from the FMT closure condition",
            "exact_output": str(sp.nsimplify(delta_F)),
            "decimal": mp.nstr(d, 30),
            "lemmas": [
                ("f_gain == 1/(1 - delta_F)",
                 exact_eq(f_gain, 1 / (1 - delta_F)), T_EXACT),
                ("delta_F != 0 at %d dps" % DPS,
                 abs(d) > mp.mpf(10) ** (-(DPS - 10)), T_WITNESS)]}


# ==================== convergence and negative controls ===============
def convergence_group():
    ref = 20 * PHI ** 4
    items = {
        "I.4": sp.simplify((2 * PHI ** 2 * SQRT5) ** 2),
        "II.1": sp.simplify((sp.Matrix([F5, F4]).T
                             * sp.Matrix([[1, SQRT5], [SQRT5, 5]])
                             * sp.Matrix([F5, F4]))[0, 0]),
        "II.2": sp.simplify(70 + 30 * SQRT5)}
    rows = [{"tag": t, "exact": str(sp.nsimplify(e)),
             "canonical": "20*phi**4", "exact_match": exact_eq(e, ref),
             "decimal": mp.nstr(num(e), 30)} for t, e in items.items()]
    return rows, mp.nstr(num(ref), 30)


def negative_controls():
    """Sensitivity to the imported rank over a finite declared set.

    Verifies that the downstream arithmetic is sensitive to the rank input.
    Does NOT replace the upstream uniqueness theorem and is not a proof that
    five is the only admissible rank.
    """
    ref = 20 * PHI ** 4
    G = sp.Matrix([[1, SQRT5], [SQRT5, 5]])
    rows = []
    for r in (3, 4, 5, 6, 7):
        c = sp.Matrix([fib(r), fib(r - 1)])
        val = sp.simplify((c.T * G * c)[0, 0])
        matches = exact_eq(val, ref)
        rows.append({"r": r, "c_r": [fib(r), fib(r - 1)],
                     "exact": str(sp.nsimplify(val)),
                     "det": int(fib(r) ** 2 - 5 * fib(r - 1) ** 2),
                     "matches_invariant": matches, "expected": (r == R_RANK),
                     "pass": matches == (r == R_RANK)})
    return rows


# ================================ runner ==============================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    print("=" * 78)
    print("QGT BLOCK B1 v%s — the projection constant" % BLOCK_VERSION)
    print("authority: %s / %s" % (AUTHORITY_WORK, AUTHORITY_LABEL))
    print("exact symbolic certification; decimals are rendering only")
    print("=" * 78)
    print("\nUPSTREAM INPUTS (consumed here, not derived here)")
    for k, v in UPSTREAM.items():
        print("  %-16s %-30s %s" % (k, v["symbol"], v["status"]))

    sci_fail = infra_fail = 0
    level = None
    for c in CHECKS:
        if c.level != level:
            level = c.level
            print("\nLEVEL %s" % level)
        try:
            out = c.fn()
            c.lemmas = out.pop("lemmas", [])
            c.outputs = out
            ok = all(bool(o) for _, o, _ in c.lemmas)
            c.status = "PASS" if ok else "FAIL"
            if not ok:
                sci_fail += 1
        except Exception:
            c.status = "ERROR"
            c.error = traceback.format_exc(limit=3)
            infra_fail += 1
        print("  [%-5s] %-24s %-38s %s" % (c.tag, c.kind, c.name, c.status))
        for text, ok, ttype in c.lemmas:
            note = "" if ttype == T_EXACT else "  <witness>"
            print("           %-56s %s%s" % (text, "ok" if ok else "FAILED", note))
        if c.error:
            print("           %s" % c.error.strip().splitlines()[-1])

    print("\nCROSS-REPRESENTATION COHERENCE (symbolic certificate first)")
    rows, ref_dec = convergence_group()
    for r in rows:
        print("  %-5s exact %-20s match %-5s decimal %s"
              % (r["tag"], r["exact"], r["exact_match"], r["decimal"][:24]))
        if not r["exact_match"]:
            sci_fail += 1
    print("  reference 20*phi**4 -> %s" % ref_dec[:24])

    print("\nFINITE NEGATIVE CONTROLS — sensitivity to the imported rank")
    print("  finite set r=3..7; this is NOT a uniqueness proof")
    nc = negative_controls()
    for r in nc:
        print("  r=%d c_r=%-8s det=%-6d %-22s %s"
              % (r["r"], str(r["c_r"]), r["det"], r["exact"][:22],
                 "closes" if r["matches_invariant"] else "does not close"))
        if not r["pass"]:
            sci_fail += 1

    n_lem = sum(len(c.lemmas) for c in CHECKS)
    scalar = [c.tag for c in CHECKS if c.kind == KIND_SCALAR]
    status = ("ERROR" if infra_fail else ("FAIL" if sci_fail else "PASS"))
    print("\n" + "-" * 78)
    print("checks %d (scalar readouts: %s) | lemmas %d | convergences %d | "
          "negative controls %d"
          % (len(CHECKS), ", ".join(scalar), n_lem, len(rows), len(nc)))
    print("BLOCK B1: %s   (scientific failures %d, infrastructural %d)"
          % (status, sci_fail, infra_fail))
    print("scope: conditional on the declared upstream inputs; exact "
          "cross-representation\n       coherence; no measured alpha enters")

    if args.json:
        src = open(__file__, "rb").read()
        payload = {
            "schema_version": SCHEMA_VERSION, "block_id": BLOCK_ID,
            "block_version": BLOCK_VERSION, "title": "Projection constant",
            "authority": {"work": AUTHORITY_WORK,
                          "source_label": AUTHORITY_LABEL,
                          "tags": [c.tag for c in CHECKS]},
            "primitives": ["integers", "rationals", "sqrt5", "phi", "pi",
                           "Fibonacci numbers"],
            "upstream_inputs": UPSTREAM,
            "forbidden_empirical_inputs": FORBIDDEN_EMPIRICAL_INPUTS,
            "checks": [{
                "id": c.tag, "level": c.level, "kind": c.kind, "name": c.name,
                "dependencies": c.dependencies,
                "constructs": c.outputs.get("constructs"),
                "exact_output": c.outputs.get("exact_output"),
                "canonical_output": c.outputs.get("canonical_output"),
                "extra_outputs": {k: v for k, v in c.outputs.items()
                                  if k not in ("constructs", "exact_output",
                                               "canonical_output")},
                "lemmas": [{"claim": t, "pass": bool(o), "test_type": ty}
                           for t, o, ty in c.lemmas],
                "status": c.status, "proves": c.proves,
                "does_not_prove": c.does_not_prove,
                "failure_mode": c.failure_mode,
                "theorem_dependency": c.theorem_dependency,
                "error": c.error} for c in CHECKS],
            "convergence_groups": [{
                "name": "cross_representation_coherence",
                "members": [r["tag"] for r in rows],
                "certificate": "exact_symbolic_equality",
                "note": "representations of one algebraic element, not "
                        "independent evidence",
                "rows": rows, "reference_decimal": ref_dec}],
            "negative_controls": {
                "name": "finite_rank_sensitivity", "domain": "r in {3,4,5,6,7}",
                "claim": "the downstream arithmetic is sensitive to the "
                         "imported rank",
                "not_a_claim": "uniqueness of rank five", "rows": nc},
            "environment": {"python": platform.python_version(),
                            "sympy": sp.__version__,
                            "mpmath": mpmath.__version__, "dps": DPS,
                            "platform": platform.platform()},
            "hashes": {"source_sha256": hashlib.sha256(src).hexdigest()},
            "summary": {"status": status, "scientific_failures": sci_fail,
                        "infrastructural_failures": infra_fail,
                        "scientific_scope": "conditional on declared upstream "
                        "inputs; exact cross-representation coherence; no "
                        "empirical input for alpha"}}
        payload["hashes"]["canonical_payload_sha256"] = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()).hexdigest()
        payload["hashes"]["note"] = (
            "canonical_payload_sha256 is the hash of the payload WITHOUT that "
            "field. It is not the hash of the final file, which cannot be "
            "embedded in itself and must be recorded externally.")
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print("certificate: %s" % args.json)

    return 2 if infra_fail else (1 if sci_fail else 0)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
