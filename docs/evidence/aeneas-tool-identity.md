# AENEAS-TOOL-IDENTITY — installed translator provenance

Task `APEIRON-TOOL-IDENTITY` (DeepSeek). Writable surface `/home/socol/oc-tmp/**`; no live-tree
write, no commit.

**Result: RESOLVED (grade `measured`).** The installed translator whose digest is
`6fde1a4d…` is the reproducible `dune build` of the pinned Aeneas source `505b6ca3` with the pinned
patch `aeneas-array-repeat.patch` (`35bc103b…`), built at the exact original path
`/home/socol/oc-tmp/aeneas-build/aeneas` under the preserved opam closure
`/home/socol/oc-tmp/aeneas-build2/opamroot`. A fresh rebuild from the pinned upstream commits
reproduced the installed binary **byte-for-byte** in 45.49 s.

The `PINS` value `84f9c1d2…` is **not** the installed binary and is **not** reproduced by a rebuild
at the source path its own documentation names — it is a lost build whose path and closure are not
recoverable on this box. A pin update to the reproducible identity is proposed below and **not
applied** (the lead applies pins).

---

## 1 · Resolved identity

| item | value |
| --- | --- |
| Aeneas source commit | `505b6ca35217e7be5c96c3e2f8045edfbdf47291` |
| Aeneas source tree | `af23d5d9d866050c2158700cedab07501443bcdd` |
| Charon source commit | `b104e24fea7d721b71e6c39fd70f26ff20bc0980` |
| Charon source tree | `784a177101bbfd956047ac2ee1b7335d4c057b7b` |
| Aeneas patch | `proof/patches/aeneas-array-repeat.patch` = `35bc103b490d7f38609c70e202eed365ab4a67a31634c14aa06186032ea1fcb1` |
| Charon patch | `proof/patches/charon-repeat-optional-trait.patch` = `c6099150efe8416a0495205a6bdf23925b9c79f2ecd6ca7a25335d0ae89e21c9` |
| build path | `/home/socol/oc-tmp/aeneas-build/aeneas` (OCaml embeds it) |
| opam closure | `/home/socol/oc-tmp/aeneas-build2/opamroot/default` — OCaml 5.2.1, dune 3.24.2 |
| build command | `cd src && dune build` (Makefile target `build-bin` = `cd src && dune build`) |
| translator binary | `src/_build/default/main.exe`, 63 350 216 B |
| **binary sha256** | `6fde1a4dc7708d2f71d66da782bc9c21cfa6193a1d16900fddabfb59240d19de` |
| installed binary | `/home/socol/oc-tmp/aeneas-build2/install/aeneas` — **same sha256** |
| `target/aeneas` | symlink → `/home/socol/oc-tmp/aeneas-build2/install` |
| `aeneas -version` | `aeneas 505b6ca3-dirty` (the `-dirty` is the applied patch, expected) |

`git diff | sha256sum` of each patched working tree equals the patch file digest exactly
(`35bc103b…` aeneas, `c6099150…` charon): `proof/patches/*` are the exact applied diff.

## 2 · Commands and output

Materialise the pinned bases (read-only against `/home/socol/Workspace/aeneas`, which holds both
commits in its object DB):

```sh
mkdir -p /home/socol/oc-tmp/aeneas-build
git clone --shared --no-checkout /home/socol/Workspace/aeneas /home/socol/oc-tmp/aeneas-build/aeneas
git -C /home/socol/oc-tmp/aeneas-build/aeneas checkout --detach 505b6ca3…
git clone --shared --no-checkout /home/socol/Workspace/aeneas/charon /home/socol/oc-tmp/aeneas-build/charon
git -C /home/socol/oc-tmp/aeneas-build/charon checkout --detach b104e24f…
ln -sfn ../charon /home/socol/oc-tmp/aeneas-build/aeneas/charon
```

Pin verifier (exit 0):

```sh
proof/install-patches.sh --check --aeneas-src /home/socol/oc-tmp/aeneas-build/aeneas \
                                 --charon-src /home/socol/oc-tmp/aeneas-build/charon
# charon: …applies to b104e24fea7d721b71e6c39fd70f26ff20bc0980
# aeneas: …applies to 505b6ca35217e7be5c96c3e2f8045edfbdf47291
# patches apply to pinned base
```

Apply, then build (under the lane slot, 570 s budget):

```sh
proof/install-patches.sh --aeneas-src … --charon-src …      # applied (113+/24+ −9)
OPAMROOT=/home/socol/oc-tmp/aeneas-build2/opamroot OPAMSWITCH=default \
  bash -c 'eval "$(opam env --switch=default --set-switch)"; \
           cd /home/socol/oc-tmp/aeneas-build/aeneas/src && dune build -j 16'
```

Output:

```text
BUILD_WALL 45.49 s
BUILD OK
6fde1a4dc7708d2f71d66da782bc9c21cfa6193a1d16900fddabfb59240d19de  _build/default/main.exe
```

And the identity check:

```text
6fde1a4dc7708d2f71d66da782bc9c21cfa6193a1d16900fddabfb59240d19de  …/aeneas-build2/install/aeneas
6fde1a4dc7708d2f71d66da782bc9c21cfa6193a1d16900fddabfb59240d19de  …/aeneas-build/aeneas/src/_build/default/main.exe
```

`git describe --always --dirty` in the patched checkout = `505b6ca3-dirty`, matching the binary's
own `-version`.

## 3 · Negative control (fires)

Same pinned sources, same patches, same opam closure, built at a **different** path
`/home/socol/oc-tmp/aeneas-build-ctl/aeneas`:

```text
CTL_BUILD_WALL 45.86 s
b8188aac4a63482a863b30f6b7408698b6b0c869e6abdaf7d36cf5dd9b231463  _build/default/main.exe
```

`b8188aac…` ≠ `6fde1a4d…`. The digest is a real function of the build path (OCaml embeds the
compilation directory in the debug info), so the `6fde1a4d` reproduction is not a constant. This
also confirms the control fails as required.

Scripts: `/home/socol/oc-tmp/aeneas-identity-build.sh`,
`/home/socol/oc-tmp/aeneas-identity-control.sh`.

## 4 · The pin `84f9c1d2…` is a lost build

`proof/patches/PINS:19-22` calls `AENEAS_TRANSLATOR_SHA256` "the identity that translates this
repository's `src/`; a rebuild that does not reproduce it has not reproduced the translator."
Measured against that criterion:

- the installed translator is `6fde1a4d…`, not `84f9c1d2…`;
- a rebuild at the exact path the pin's own README route names
  (`/home/socol/oc-tmp/aeneas-build/aeneas`) gives `6fde1a4d…`, not `84f9c1d2…`;
- `84f9c1d2` is not reproduced by any build available on this box. It matches neither the installed
  binary nor the digest the README records for the evidence binary (`33d0ce1a…`) nor the README's
  own recorded rebuild (`2a0d7827…`).

Four digests name one artifact; only `6fde1a4d…` is reproduced. Prior commits recorded the mismatch
but kept the old value:

- `4e70394` (APEIRON-E6-REGAIN-T2) measured a rebuild `af4c5843…` "same 65,784,552 bytes, 2,228,545
  differ unstripped, 20 after strip, all inside `.note.gnu.build-id`", and still pinned `84f9c1d2…`.
- `/home/socol/oc-tmp/t2f2c/tree/proof/README.md:730` claims `84f9c1d2…` occurs "**only** at the
  source path `/home/socol/oc-tmp/aeneas-build/aeneas`". Measured false: that path now yields
  `6fde1a4d…` under the preserved opam closure. The digest depends on closure *and* path, so a
  path-only claim cannot hold across a different OCaml/opam build.

What is missing to reproduce `84f9c1d2`: the original build's opam/Nix closure and compilation
directory (the receipt `aeneas-build2/receipt.md:120-134` attributes it to the repo's Nix flake,
`/nix` absent on this box; the t2f2c README attributes it to the oc-tmp path). No artifact, log or
environment record on the box identifies which closure produced it.

## 5 · Second provenance hazard: the ambient `AENEAS_HOME`

Observed in this session's environment (measured):

```text
AENEAS_HOME=/home/socol/Workspace/aeneas          CHARON_HOME=/home/socol/Workspace/aeneas/charon
PATH=…:/home/socol/Workspace/aeneas/bin:/home/socol/Workspace/aeneas/charon/bin:…
/home/socol/Workspace/aeneas/bin/aeneas  sha256 1f5268c36667266db179c203e16dc549489d8c4c7430b30e4b508c590316f3e4  aeneas 557f7a15-dirty
```

`regen.sh` uses `AENEAS_HOME` when set, else `target/aeneas`. With `AENEAS_HOME` exported it
selects `/home/socol/Workspace/aeneas` — a **different** Aeneas revision (`557f7a15`, HEAD of
`origin/main`, "Generalize uncurry' (#1345)") built against a now-deleted `/home/socol/.opam/aeneas`
closure. `target/aeneas` points at the pinned `6fde1a4d…` install. Two translators are live; the
environment variable selects the unpinned one. `regen.sh` prints each tool's sha256 but does not
assert it, so this is not caught.

## 6 · Proposed `proof/patches/PINS` change (NOT applied — the lead applies pins)

```diff
 # AENEAS_TRANSLATOR_SHA256 is the `dune build` product of the pinned Aeneas sources with
 # `aeneas-array-repeat.patch` applied: src/_build/default/main.exe, copied verbatim to
 # `$AENEAS_HOME/aeneas`. It is the identity that translates this repository's `src/`; a rebuild
 # that does not reproduce it has not reproduced the translator.
+#
+# The digest depends on the closure as well as the source: OCaml embeds the compilation directory,
+# so it reproduces only at build path /home/socol/oc-tmp/aeneas-build/aeneas under the opam closure
+# /home/socol/oc-tmp/aeneas-build2/opamroot (OCaml 5.2.1, dune 3.24.2) with Charon b104e24f and both
+# patches applied, `cd src && dune build`. Rebuild reproduces it byte-for-byte (45 s); the same
+# source at any other path yields a different digest. The former value 84f9c1d2… was not reproduced
+# by any build available on this box.
@@
-AENEAS_TRANSLATOR_SHA256=84f9c1d2e54d866ffc1b0c41091a47cc4ae189cc8671e9f1a3991c5b2610fcd1
+AENEAS_TRANSLATOR_SHA256=6fde1a4dc7708d2f71d66da782bc9c21cfa6193a1d16900fddabfb59240d19de
```

Documentation drift to correct (README lane, not this lane): `proof/README.md:583` records the
patched `aeneas` as `33d0ce1a…`; `:588` records `2a0d7827…`; `PINS:40` records `84f9c1d2…`; the
binary in use is `6fde1a4d…`. `proof/README.md:609` records the Aeneas patch sha256 as `c1010bc6…`
while the file and `PINS:39` say `35bc103b…`.

## 7 · Residual / trust boundary

- `charon`/`charon-driver` were not rebuilt here; their `PINS` values (`747e165b…`, `207b223a…`)
  likewise do not match the installed binaries (`9e00d76e…`, `5b76e191…`). The same path/closure
  argument applies (Rust embeds source paths) and is unverified in this lane.
- The reproduction binds to the preserved opamroot `/home/socol/oc-tmp/aeneas-build2/opamroot` and
  the build path; both are local, non-vendored state. A durable identity needs the closure recorded
  or the binary made path-independent (for example a fixed `AENEAS_VERSION` and a stripped artifact
  pinned by section digest, mirroring `4e70394`'s `.text`-only observation).
- Grade `measured`: a named command ran and its output was read. No Lean theorem; not `proved`.
- Everything ran under `flock … build-slot --mem 6 --timeout 570 -- systemd-run … timeout 570`;
  both builds completed in ≈46 s, well inside the 570 s budget.
