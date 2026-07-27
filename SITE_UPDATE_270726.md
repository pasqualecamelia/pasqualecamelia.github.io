# Site update — 27 July 2026

## Corrections to the existing site

1. **Duplicated copy of the whole site removed.** The repository contained
   `pasqualecamelia.github.io-main/` — a nested, complete second copy of itself
   (17 MB of the 32 MB total, an extracted zip committed by accident). It served
   duplicate URLs for every page, which is bad for indexing and confusing for
   anyone browsing the repository. Deleted; the site is now 18 MB.
2. **Broken PDF link fixed.** `papers.html` pointed at
   `papers/Camelia_2026_QGT_GaugeAlgebra_MajoranaProjection.pdf`, which does not
   exist. The file that is there is `papers/Camelia_2026_QGT_GaugeAlgebra_RG.pdf`;
   its title page reads *Observable Gauge Algebra from Finite-Rank Majorana
   Projection*, i.e. the same paper, so the link was repointed rather than removed.
   It was the only broken local link on the site — the whole tree is now checked
   and clean.
3. **`__pycache__/qgt_simulator.cpython-312.pyc` removed**, and a `.gitignore`
   added so compiled artefacts do not come back. `.nojekyll` added so GitHub Pages
   serves every path verbatim.
4. **SVD paper entry updated**: 28 pages → revision r270726, 31 pages, and the
   abstract now qualifies the rational representative *H*(z) by the declared
   conditions (H1)–(H4), matching the paper.
5. **FMT paper added.** It was missing from `papers.html` altogether. The entry
   carries the current abstract, the certified condition number, and links to both
   the PDF and the new environment. Note that `papers/Camelia_2026_QGT_DSP_FMT.pdf`
   is an earlier version of that work (*Vandermonde Structure, Non-Isometry, and
   Finite-Field Orthogonality*); it is left in place but no longer linked from the
   paper list.
6. **Both current revisions published locally**: `papers/Camelia_2026_FMT_r270726.pdf`
   (25 pp) and `papers/Camelia_2026_ProjectionSVD_BoundaryMetric_r270726.pdf` (31 pp).
7. **`sitemap.xml`** extended with the three new URLs.

## The QFMT environment (new `qfmt/` section)

* `qfmt/index.html` — landing page in the site's own style and colour scheme:
  what the transform is and what it is not (Mellin is unitary; the deficit belongs
  to the finite construction), the certified record, the section on what double
  precision cannot do, and the local-install instructions.
* `qfmt/console.html` and `qfmt/dashboard.html` — the two interactive tools from
  the research package, self-contained, nothing uploaded anywhere.

Both tools received the site's GA4 tag and a footer link back to the environment
and to the repository.

**The Console's certification panel was out of date and is now corrected.** It
displayed:

    condition_number (float64) = 1.4829343e17
    numerical rank             = 8

The first is a machine-dependent float64 artefact quoted as if it were a value of
the frame; the second is a numerical rank quoted without its tolerance. Both are
replaced by the certified record — σ_max, σ_min, κ = 4.70161841441e18,
‖FF*−I‖_F = 9.42795, exact rank 10, effective rank 4/5/6/7 at absolute tolerance
1e−4/1e−6/1e−8/1e−12 — followed by an explicit float64 block stating that the
polar factor is not determined in double precision (0.5 ≤ ‖ΔU‖_F ≤ 3, varying
with the linear-algebra library), that the retained rank-five block still agrees
to ≤1e−10, and that the coisometry identities hold at any precision.

## Checks run

* every local `href`/`src` on every page resolves — 0 broken links;
* all pages parse;
* the certified tokens appear on the new pages and the stale ones are gone;
* GA4 present on all three new pages.

## Not changed

Nothing else on the site was touched: the companion simulator, the animations,
the α page, the CMB and camH folders, the videolibro, the covers and the footer
are byte-identical to what you uploaded.
