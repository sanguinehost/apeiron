# RFC 0001 Review Package

**Frozen historical proposal and review artifact.** This package records a proposal for review; it is
not the current approved programme and it is not dispatch authority for programme work. The sole
programme authority is [Apeiron programme direction](../../direction.md). That document's current
approved decision retains canonical Lean with reusable Rust and their refinements converging in
Apeiron, keeps Song an adjacent independent assembly-up implementation with no Rust or Aeneas runtime
linkage, and requires a checked cross-repo primitive correspondence. This package's ownership and
review status is authoritative only for RFC 0001's own scope; it does not assign programme roles.

**Short form posted to GitHub `Ptyktos/fold#15` on 2026-09-11. Canonical GitLab publication (project 211) is not authorized; the full draft and evidence package are below.**

All code discussed here was written by AI under the user's direction. Commit authorship and git
trailers do not infer direction, inventorship, labour or contribution. Role assignments in
[ownership.md](ownership.md) are proposed; this documentation pass records no new acceptance of
proposed task allocations.

- Read [RFC.md](RFC.md) for the complete issue draft.
- Open [index.html](index.html) for the current/proposed SVG diagrams.
- [Current SVG](diagrams/current.svg) and [proposed SVG](diagrams/proposed.svg) are scalable downloads.
- [ownership.md](ownership.md) records the basis for the proposed work split, not work-hour estimates.
- [publication.json](publication.json) identifies the eventual Fold issue and verified reviewer `ctown`. Its approval flag is false.

Edit `source.md` for RFC prose, `topology.json` for detailed dependency semantics,
and `views.json` for the simplified ring map. Mermaid and SVG use the same view
definitions. Every view references and accounts for the detailed topology's nodes.
Generated files MUST NOT be edited independently.

From the Apeiron root, using Python's standard library:

```sh
python docs/rfcs/0001-core-architecture/render.py
python docs/rfcs/0001-core-architecture/render.py --check
```

The renderer checks the detailed dependency policy, ring membership, view coverage,
text capacity, diagram markers, SVG XML/viewBox validity, attachment existence,
and generated-file freshness. Fixed SVG geometry gives each ring the same aligned
band and each peer the same rank. Only three layer-direction arrows are drawn;
direct R2-to-R0 access is stated explicitly. Detailed relationships remain in the
RFC text rather than crossing the diagram.

The renderer has no publishing code. Mermaid syntax is checked separately with
Mermaid 11.12.0; the SVGs are rasterized for visual inspection.

After Lucas approves publication, approved attachments must be uploaded to GitLab
and local links replaced. The issue will request Clay's review; it will not claim
that joint architecture acceptance has already occurred.
