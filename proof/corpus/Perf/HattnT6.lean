import Mathlib.Data.Nat.Basic

/-!
# HATTN-T6: the cheapest deployable decode-tail selector at whole-model KLD 0.05

`docs/evidence/hattn-decode-tail.md` (T6 section): the HATTN-T5 setup with 64 tail positions before each
`t ∈ {4096, 8192, 16384, 32768}` (`KV_TAIL_SWEEP=1`, output `/home/socol/oc-tmp/hattn-long/sweep.out`). Every
figure is a `model` input copied from that output; no theorem here produces a measurement. Mean whole-model KLD
over each `t`'s 64 positions and mean key reads per `t`, both in ten-thousandths; rel-L2 in units of `1e-8`.
rustc, LLVM and the instrument are outside every statement.
-/

namespace Perf

namespace HattnT6

/-- Dense-arm tail control, max rel-L2 over the four `t` and 24 layers, units of `1e-8`. -/
def ctlMax : ℕ := 1815

/-- Mean KLD at `t = 4096, 8192, 16384, 32768`: Quest-64 with fixed far budgets 2048, 1024, 512. -/
def fix2048Kld : List ℕ := [36, 92, 158, 53]

def fix1024Kld : List ℕ := [165, 368, 293, 89]

def fix512Kld : List ℕ := [377, 1296, 449, 129]

/-- Mean KLD of the near set alone at 8192, and of the oracle at per-head KL 1.0 at the four `t`. -/
def nearKld8 : ℕ := 6145

def oracle1Kld : List ℕ := [3443, 6019, 3530, 4271]

/-- Oracle at per-head KL 0.25, `t = 8192`: whole-model KLD and mean per-head KL. -/
def oracleQKld8 : ℕ := 876

def oracleQHeadKl8 : ℕ := 2191

/-- Mean key reads per `t` at 32768: fixed budgets 1024 and 512 (measured); page bounds alone, derived from
the A64 layout: 2 reads per 64-key page over the `32768 − 260` far keys, `0.0310 t`. -/
def fix1024Reads32 : ℕ := 712

def fix512Reads32 : ℕ := 555

def boundReads : ℕ := 310

/-! ## Verdicts on `PREDICTIONS-T6.md` (sha256 `c2a6b29f…`) -/

/-- **PS1 held.** The dense tail reproduces the dump within `1e-4` at every `t`. -/
theorem ps1_held : ctlMax ≤ 10000 := by decide

/-- **PS2 held.** A 2048-token far budget stays under mean KLD 0.05 at every `t`. -/
theorem ps2_held : ∀ k ∈ fix2048Kld, k ≤ 500 := by decide

/-- **PS3 refuted.** A 512-token budget misses at 8192 (0.130). -/
theorem ps3_refuted : ¬ ∀ k ∈ fix512Kld.drop 1, k ≤ 500 := by decide

/-- **PS4 held.** The near set alone costs 0.61 nats at 8192: the far keys matter there. -/
theorem ps4_held : 500 < nearKld8 := by decide

/-- **PS5 refuted.** Per-head KL 1.0 gives whole-model KLD 0.34–0.60 at every `t`. -/
theorem ps5_refuted : ∀ k ∈ oracle1Kld, ¬ k ≤ 500 := by decide

/-- **PS6 refuted.** The cheapest passing deployable arm, fix1024, reads `0.071 t` at 32k, not `≤ 0.06 t`. -/
theorem ps6_refuted : ¬ fix1024Reads32 ≤ 600 := by decide

/-! ## The operating point -/

/-- **fix1024 is the cheapest measured deployable arm at KLD ≤ 0.05 at all four `t`**: it passes everywhere,
and the next cheaper budget fails at 8192. -/
theorem fix1024_cheapest : (∀ k ∈ fix1024Kld, k ≤ 500) ∧ ¬ (∀ k ∈ fix512Kld, k ≤ 500) := by decide

/-- **14× fewer key reads at 32k**, as a count, not a time. -/
theorem fix1024_fourteen_x : 14 * fix1024Reads32 < 10000 := by decide

/-- **The page bounds are 44 % of fix1024's reads at 32k and grow linearly in `t`**: the selector is `O(t/32)`,
not sublinear. -/
theorem bounds_dominate : 10 * boundReads > 4 * fix1024Reads32 := by decide

/-- **Per-head error attenuates only 2.5× at per-head KL 0.25** (8192), against 25–70× at 0.05 in T5. -/
theorem attenuation_weak_at_quarter : 2 * oracleQKld8 < oracleQHeadKl8 ∧ ¬ 3 * oracleQKld8 < oracleQHeadKl8 := by
  decide

end HattnT6

end Perf
