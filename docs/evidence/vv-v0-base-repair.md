# VV-V0 base-repair — receipt (Scan / Search / Representation)

Row: `APEIRON-VV-V0` (`apeiron-vv-v0-repair-the-Gbp966ehv0fzykUb3hLXWC`).

Installs the three repaired modules onto the live tree at `0d3df58`:

| file | sha256 |
|---|---|
| `Growable/Scan.lean` | `d31ce116981c0de69630d0162583ebe6c9289b479895bdab3c3825c620ce4352` |
| `Growable/Search.lean` | `6bc2b38caba267de120b665a5a21dccf276726f74ec3b5d25772f5353eb2f291` |
| `Growable/Representation.lean` | `8f892d5816a06503327a0c9cdde96290d7abbbd6aa032a51e34f28aebc1665a6` |

What changed: the generated `publish_live` interleaves `Region.coord_of` and an
`before.coords.index_mut_usize` **after** `table.tag`, so the two-bind `cases` chain is inserted and
the `vis`/`div` cases use `simp only [bind_tc_vis]/[bind_tc_div]` + `(vis_not_ok hpub).elim`; the
`TableLocator` literal gains `admitted := 0#usize` (the value `table.TableLocator.new` produces);
four unreferenced `hpublish`/`hlocator` binders are renamed `_…`. **No theorem statement changed;
no `sorry`/`admit`/`axiom`.**

Gate (live tree, `.olean` deleted first): `cd proof/corpus && lake build Growable.Scan
Growable.Search Growable.Representation` under `build-slot --mem 12 --timeout 570 --
systemd-run --user --scope -p MemoryMax=12G -p MemorySwapMax=0` — **Build completed successfully
(3349 jobs)**, exit 0.

Control: revert the inserted `cases` block (VV-V0/VV-V0b runs) -> `unsolved goals` at
`Scan.lean:248/249` returns; restore -> green.

Open: `lake build AxiomAudit` is still red on `Growable/ReadLoop.lean` plus the 20
`Growable/ReadState.lean` audit rows (`BASE-CORPUS-COVERAGE.md`: route (b), re-derive). This
receipt closes the first step only; the base-corpus barrier is not closed.

Grade: `proved` for these three modules building under the closed cone; barrier open.
