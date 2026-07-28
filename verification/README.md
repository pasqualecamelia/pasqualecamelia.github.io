# QGT verification certificates

Two machine-checkable counterparts to Appendix L of the frozen First Edition.

| artefact | what it is | assurance |
|---|---|---|
| `qgt_block_alpha.py` | symbolic-executable coherence verifier (SymPy, exact) | stronger than a unit test |
| `QGT_Formal_Pilot_B0_B1_rel1.zip` | Lean 4 formalization of the arithmetic core | kernel-checked proof terms |

## Run

```bash
python3 qgt_block_alpha.py --json b1.json     # exit 0 on pass

unzip QGT_Formal_Pilot_B0_B1_rel1.zip && cd QGTFormal
lake build && bash scripts/audit.sh
```

The Lean project needs only a Lean 4.22.0 toolchain: it has no dependency on
mathlib or anything else.

## Direction of authority

The monograph is the authority; these artefacts are derived from it. Each
declares the frozen version it was built against. The monograph, in turn, cites
them by name, version and URL — never by hash, so that adding the citation
cannot invalidate the artefact that points back at it.

## Scope

No measured value of alpha, no CODATA figure, no fitted parameter enters either
artefact. Rank five is an imported upstream input in both. The negative controls
are finite and are not a uniqueness proof. Neither artefact bears on the
physical truth of QGT.
