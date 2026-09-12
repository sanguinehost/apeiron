# RFC 0001 Review Package

**Short form posted to GitHub `Ptyktos/fold#15` on 2026-09-11. Canonical GitLab publication (project 211) is not authorized; the full draft and evidence package are below.**

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
