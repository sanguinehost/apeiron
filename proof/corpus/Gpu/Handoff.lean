import ApeironWord

/-!
# Gpu.Handoff — the Vulkan handoff contract

`src/gpu/runtime.rs` records host copies and compute dispatches into one Vulkan command buffer
and submits it once. The bodies bottom out in FFI calls on a dlopen'd `Vk`
(`src/gpu/vulkan.rs`), which are neither extracted nor proved. This file separates the two
halves of that layer.

* The **handoff**: the pure arithmetic and list structure of the runtime — binding order,
  staging offsets and 256-byte alignment, uploads before dispatches, one submit per command
  buffer, a memory barrier between consecutive dispatches, and region containment for `scatter`.
  These are proved from definitions that mirror the Rust bodies. The Rust wiring that calls the
  extracted `word_gpu_runtime_*` roots is not installed yet: it waits on a completed
  regeneration, so this file is a standalone model of the current `src/gpu/runtime.rs` and
  claims no extraction link until that patch lands.
* The **Vulkan operating contract** (`VulkanContract`): four obligations a conforming driver
  supplies. They are hypotheses over an abstract `System`, not axioms; the handoff theorems do
  not depend on them, and the one composition theorem below consumes them explicitly.

Nothing here claims the driver, the SPIR-V kernels, rustc/LLVM codegen or the `unsafe`
container. The arithmetic is graded `proved` for its Lean statements only; the current Rust
bodies are graded `tested` by the crate's tests, and no refinement theorem connects the two
until the extraction patch lands. `import ApeironWord` is the corpus reachability edge the
structure gate requires; no statement below uses an extracted definition yet.

Board row: `apeiron-gpu-handoff-w1-E07GEZ6NatfOs4dGjhJl3c`.
-/

namespace Gpu.Handoff

/-! ## Commands and submissions -/

/-- One recorded command: a host→device copy into `dst` over device bytes `[lo, hi)`, a compute
dispatch of `pipe` over its bound buffer ids, a memory barrier, or a timestamp write. -/
inductive Cmd where
  | copy (dst : Nat) (lo hi : Nat)
  | dispatch (pipe : Nat) (bound : List Nat)
  | barrier
  | timestamp (slot : Nat)
  deriving DecidableEq, Repr

/-- A recorded command buffer, in recording order. -/
abbrev Cmds := List Cmd

/-- A copy is the only command that writes host bytes. -/
def Cmd.isCopy : Cmd → Bool
  | .copy _ _ _ => true
  | _ => false

/-- No copy occurs anywhere in the list. -/
def noCopyAnywhere : Cmds → Bool
  | [] => true
  | x :: rest => (!x.isCopy) && noCopyAnywhere rest

/-- A recorded copy precedes every recorded dispatch: the copies are a prefix. -/
def uploadsFirst : Cmds → Bool
  | [] => true
  | .copy _ _ _ :: rest => uploadsFirst rest
  | _ :: rest => noCopyAnywhere rest

/-- Every pair of consecutive dispatches has a barrier between them. -/
def barriersBetween : Cmds → Bool
  | [] => true
  | .dispatch _ _ :: rest =>
      match rest with
      | [] => true
      | .barrier :: rest' => barriersBetween rest'
      | _ => false
  | _ :: rest => barriersBetween rest

/-- The barrier `Gpu::record` emits for `PREFILL_BAR = mode`. Mode 9 records none
(`src/gpu/runtime.rs`, `step_barrier`); every other mode records one. -/
def barrierAt (mode : Nat) : Cmds := if mode = 9 then [] else [.barrier]

/-- `Gpu::record`: bind, dispatch, and a barrier between each adjacent pair
(`src/gpu/runtime.rs:407`). `ds` carries each dispatch's pipeline id and bound buffer ids. -/
def recordCmds (mode : Nat) : List (Nat × List Nat) → Cmds
  | [] => []
  | d :: rest =>
      Cmd.dispatch d.1 d.2 ::
        (if rest.isEmpty then [] else barrierAt mode ++ recordCmds mode rest)

/-- `Gpu::run`: one command buffer, one submit. -/
def runCmds (mode : Nat) (ds : List (Nat × List Nat)) : Cmds := recordCmds mode ds

/-- `Gpu::time`: the same recording, bracketed by the two device timestamps
(`src/gpu/runtime.rs:349`). -/
def timeCmds (mode : Nat) (ds : List (Nat × List Nat)) : Cmds :=
  Cmd.timestamp 0 :: (recordCmds mode ds ++ [Cmd.timestamp 1])

/-- `Gpu::staged_dispatch`: every staged copy, one transfer→compute barrier, then the dispatches
(`src/gpu/runtime.rs:375`). -/
def stagedCmds (mode : Nat) (uploads : List (Nat × Nat))
    (ds : List (Nat × List Nat)) : Cmds :=
  uploads.map (fun u => Cmd.copy u.1 0 u.2) ++ Cmd.barrier :: recordCmds mode ds

/-- A submission: the recorded commands and the number of queue submissions. -/
structure Submission where
  cmds : Cmds
  submits : Nat

/-- The submission `Gpu::staged_dispatch` issues. -/
def stagedSubmission (mode : Nat) (uploads : List (Nat × Nat))
    (ds : List (Nat × List Nat)) : Submission :=
  ⟨stagedCmds mode uploads ds, 1⟩

/-- The dispatch subsequence of a recorded buffer. -/
def dispatchSeq (c : Cmds) : List (Nat × List Nat) :=
  c.filterMap (fun x =>
    match x with
    | .dispatch p b => some (p, b)
    | _ => none)

theorem dispatchSeq_cons (x : Cmd) (rest : Cmds) :
    dispatchSeq (x :: rest) =
      (match x with | .dispatch p b => [(p, b)] | _ => []) ++ dispatchSeq rest := by
  cases x <;> rfl

theorem dispatchSeq_append (a b : Cmds) :
    dispatchSeq (a ++ b) = dispatchSeq a ++ dispatchSeq b := by
  simp [dispatchSeq, List.filterMap_append]

/-! ## Handoff theorems: submission shape -/

theorem noCopyAnywhere_append (a b : Cmds) :
    noCopyAnywhere (a ++ b) = (noCopyAnywhere a && noCopyAnywhere b) := by
  induction a with
  | nil => rfl
  | cons x rest ih =>
      cases x <;> simp [noCopyAnywhere, ih, Bool.and_assoc]

/-- The equation `recordCmds` takes on a one-element list. -/
theorem recordCmds_single (mode : Nat) (d : Nat × List Nat) :
    recordCmds mode [d] = [Cmd.dispatch d.1 d.2] := by
  simp [recordCmds, List.isEmpty]

/-- The equation `recordCmds` takes on two or more dispatches. -/
theorem recordCmds_step (mode : Nat) (d e : Nat × List Nat) (es : List (Nat × List Nat)) :
    recordCmds mode (d :: e :: es) =
      [Cmd.dispatch d.1 d.2] ++ barrierAt mode ++ recordCmds mode (e :: es) := by
  simp [recordCmds, List.isEmpty]

theorem noCopyAnywhere_record (mode : Nat) (ds : List (Nat × List Nat)) :
    noCopyAnywhere (recordCmds mode ds) = true := by
  induction ds with
  | nil => rfl
  | cons d rest ih =>
      cases rest with
      | nil =>
          rw [recordCmds_single]
          simp [noCopyAnywhere, Cmd.isCopy]
      | cons e es =>
          rw [recordCmds_step]
          by_cases h : mode = 9
          · subst h
            simpa [noCopyAnywhere_append, noCopyAnywhere, Cmd.isCopy, barrierAt] using ih
          · simpa [noCopyAnywhere_append, noCopyAnywhere, Cmd.isCopy, barrierAt, h] using ih

theorem uploadsFirst_record (mode : Nat) (ds : List (Nat × List Nat)) :
    uploadsFirst (recordCmds mode ds) = true := by
  induction ds with
  | nil => rfl
  | cons d rest ih =>
      cases rest with
      | nil =>
          rw [recordCmds_single]
          simp [uploadsFirst, noCopyAnywhere]
      | cons e es =>
          rw [recordCmds_step]
          by_cases h : mode = 9
          · subst h
            simpa [uploadsFirst, noCopyAnywhere_append, noCopyAnywhere, Cmd.isCopy, barrierAt]
              using noCopyAnywhere_record 9 (e :: es)
          · simpa [uploadsFirst, noCopyAnywhere_append, noCopyAnywhere, Cmd.isCopy, barrierAt, h]
              using noCopyAnywhere_record mode (e :: es)

theorem uploadsFirst_prepend_copies (u : List (Nat × Nat)) (tail : Cmds)
    (h : uploadsFirst tail = true) :
    uploadsFirst (u.map (fun x => Cmd.copy x.1 0 x.2) ++ tail) = true := by
  induction u with
  | nil => simpa using h
  | cons x rest ih =>
      simp [uploadsFirst, ih]

/-- `staged_dispatch` records every upload before any dispatch. -/
theorem staged_uploads_first (mode : Nat) (uploads : List (Nat × Nat))
    (ds : List (Nat × List Nat)) :
    uploadsFirst (stagedCmds mode uploads ds) = true := by
  unfold stagedCmds
  apply uploadsFirst_prepend_copies
  simp [uploadsFirst, noCopyAnywhere_record]

/-- `staged_dispatch` submits once. This is the model's construction, not a driver fact: the
Rust body calls `submit_wait` exactly once (`src/gpu/runtime.rs:403`). -/
theorem staged_one_submit (mode : Nat) (uploads : List (Nat × Nat))
    (ds : List (Nat × List Nat)) :
    (stagedSubmission mode uploads ds).submits = 1 := rfl

theorem record_dispatchSeq (mode : Nat) (ds : List (Nat × List Nat)) :
    dispatchSeq (recordCmds mode ds) = ds := by
  induction ds with
  | nil => rfl
  | cons d rest ih =>
      cases rest with
      | nil =>
          rw [recordCmds_single]
          simp [dispatchSeq]
      | cons e es =>
          rw [recordCmds_step]
          by_cases h : mode = 9
          · subst h
            simp [dispatchSeq_cons, barrierAt, ih]
          · simp [dispatchSeq_cons, barrierAt, h, ih]

/-- `run` and `time` cover the same command sequence; `time` adds only the two timestamps. -/
theorem run_time_same (mode : Nat) (ds : List (Nat × List Nat)) :
    dispatchSeq (timeCmds mode ds) = dispatchSeq (runCmds mode ds) := by
  simp [timeCmds, runCmds, dispatchSeq]

/-- `record` places a memory barrier between consecutive dispatches whenever the barrier mode
emits one (`PREFILL_BAR ≠ 9`). -/
theorem record_barriers (mode : Nat) (ds : List (Nat × List Nat)) (h : mode ≠ 9) :
    barriersBetween (recordCmds mode ds) = true := by
  induction ds with
  | nil => rfl
  | cons d rest ih =>
      cases rest with
      | nil =>
          rw [recordCmds_single]
          simp [barriersBetween]
      | cons e es =>
          rw [recordCmds_step]
          have hb : barrierAt mode = [Cmd.barrier] := by simp [barrierAt, h]
          rw [hb]
          simpa [barriersBetween] using ih

/-- Non-vacuity: mode 9 records no barrier, so two consecutive dispatches fail the predicate. -/
example : barriersBetween (recordCmds 0 [(3, [7]), (3, [7])]) = true := by decide
example : barriersBetween (recordCmds 9 [(3, [7]), (3, [7])]) = false := by decide
example : uploadsFirst (stagedCmds 0 [(7, 8)] [(3, [7])]) = true := by decide
example : uploadsFirst [Cmd.dispatch 3 [7], Cmd.copy 7 0 8] = false := by decide

/-! ## Handoff arithmetic (the extracted `gpu.runtime` bodies)

Each definition below is the Lean statement of a Rust function in `src/gpu/runtime.rs`. The
extraction roots them through `proof/extract/ports/gpu_runtime_port.rs` as
`gpu.runtime.stage_fit`, `stage_align`, `stage_region`, `scatter_src`, `binds_exactly`,
`write_chunk` and `bar_emits`. -/

/-- 256-byte alignment of the next staged block (`(off + bytes + 255) & !255`). -/
def align256 (n : Nat) : Nat := (n + 255) / 256 * 256

theorem le_align256 (n : Nat) : n ≤ align256 n := by
  unfold align256
  have h := Nat.div_add_mod (n + 255) 256
  have hr := Nat.mod_lt (n + 255) (by decide : 0 < 256)
  omega

/-- The staging-window fit test (`off + bytes ≤ stage_cap`). -/
def stageFit (off len cap : Nat) : Bool := off + len ≤ cap

/-- The next staging offset after a block of `len` bytes at `off`. -/
def stageNext (off len : Nat) : Nat := align256 (off + len)

/-- The `BufferCopy` `staged_dispatch` records: `(src, dst, size)`. -/
def stageRegion (off len : Nat) : Nat × Nat × Nat := (off, 0, len)

/-- The staging source offset of a `scatter` region. -/
def scatterSrc (off s : Nat) : Nat := off + s

/-- The `bind` count check (`buffers.len() == pipe.bindings`). -/
def bindsExactly (n bindings : Nat) : Bool := n = bindings

/-- The next chunk of `write`: `min (bytes - done) cap`. -/
def writeChunk (done bytes cap : Nat) : Nat := (bytes - done).min cap

/-- Does `step_barrier` emit a barrier for this mode? Mode 9 does not. -/
def barEmits (mode : Nat) : Bool := mode != 9

theorem align256_ge_self (n : Nat) : n ≤ align256 n := le_align256 n

theorem stage_next_ge (off len : Nat) : off ≤ stageNext off len := by
  unfold stageNext
  exact Nat.le_trans (Nat.le_add_right off len) (le_align256 (off + len))

theorem stage_region_src (off len : Nat) : (stageRegion off len).1 = off := rfl
theorem stage_region_dst (off len : Nat) : (stageRegion off len).2.1 = 0 := rfl
theorem stage_region_len (off len : Nat) : (stageRegion off len).2.2 = len := rfl

theorem scatter_src_ge (off s : Nat) : off ≤ scatterSrc off s := by
  unfold scatterSrc
  exact Nat.le_add_right off s

theorem bindsExactly_iff (n bindings : Nat) : bindsExactly n bindings = true ↔ n = bindings := by
  simp [bindsExactly]

theorem write_chunk_ge (done bytes cap : Nat) (h : done < bytes) (hc : 0 < cap) :
    done < done + writeChunk done bytes cap := by
  show done < done + (bytes - done).min cap
  have hb : 0 < bytes - done := Nat.sub_pos_of_lt h
  have : 0 < (bytes - done).min cap := Nat.lt_min.mpr ⟨hb, hc⟩
  omega

theorem write_chunk_le (done bytes cap : Nat) :
    done + writeChunk done bytes cap ≤ done + (bytes - done) :=
  Nat.add_le_add_left (Nat.min_le_left (bytes - done) cap) done

theorem write_chunk_le_bytes (done bytes cap : Nat) (h : done ≤ bytes) :
    done + writeChunk done bytes cap ≤ bytes := by
  have h1 := write_chunk_le done bytes cap
  have h2 : done + (bytes - done) = bytes := Nat.add_sub_cancel' h
  omega

theorem barEmits_true (mode : Nat) (h : mode ≠ 9) : barEmits mode = true := by
  simp [barEmits, h]

theorem barEmits_false (mode : Nat) (h : mode = 9) : barEmits mode = false := by
  simp [barEmits, h]

/-! ## Handoff theorems: descriptor binding order -/

/-- A compiled compute pipeline: its identity and its storage-buffer binding count. -/
structure Pipe where
  id : Nat
  nBindings : Nat

/-- The descriptor writes `descriptor_set` records starting at binding `k`: binding `k + j`
holds buffer `j`. This is the `(0..n).map (dst_binding := i, infos[i])` in the Rust body. -/
def writesFrom (k : Nat) : List Nat → List (Nat × Nat)
  | [] => []
  | b :: bs => (k, b) :: writesFrom (k + 1) bs

/-- `Gpu::bind`: one storage buffer per binding, in order, or a refusal when the buffer count
disagrees with the pipeline's layout (`src/gpu/runtime.rs:323`). -/
def bindSet (p : Pipe) (bufs : List Nat) : Option (List (Nat × Nat)) :=
  if bindsExactly bufs.length p.nBindings then some (writesFrom 0 bufs) else none

theorem writesFrom_length (k : Nat) (bufs : List Nat) :
    (writesFrom k bufs).length = bufs.length := by
  induction bufs generalizing k with
  | nil => rfl
  | cons b bs ih => simp [writesFrom, ih]

theorem writesFrom_fst (k : Nat) (bufs : List Nat) (j : Nat) (hj : j < bufs.length) :
    ((writesFrom k bufs)[j]?).map Prod.fst = some (k + j) := by
  induction bufs generalizing k j with
  | nil => simp at hj
  | cons b bs ih =>
      cases j with
      | zero => simp [writesFrom]
      | succ j =>
          have hj' : j < bs.length := by simpa using hj
          rw [writesFrom, List.getElem?_cons_succ, ih (k + 1) j hj']
          simp only [Option.some.injEq]
          omega

theorem writesFrom_snd (k : Nat) (bufs : List Nat) (j : Nat) :
    ((writesFrom k bufs)[j]?).map Prod.snd = bufs[j]? := by
  induction bufs generalizing k j with
  | nil => simp [writesFrom]
  | cons b bs ih =>
      cases j with
      | zero => simp [writesFrom]
      | succ j => simp [writesFrom, ih]

/-- `bind` yields exactly `n_bindings` descriptors. -/
theorem bind_length (p : Pipe) (bufs : List Nat) (ds : List (Nat × Nat))
    (h : bindSet p bufs = some ds) : ds.length = p.nBindings := by
  unfold bindSet at h
  split at h
  · rename_i hcount
    simp at h
    subst h
    rw [writesFrom_length]
    simpa [bindsExactly] using hcount
  · simp at h

/-- `bind` places buffer `j` at binding `j`: the binding order is the buffer order. -/
theorem bind_order (p : Pipe) (bufs : List Nat) (ds : List (Nat × Nat))
    (h : bindSet p bufs = some ds) (j : Nat) (hj : j < bufs.length) :
    (ds[j]?).map Prod.fst = some j := by
  unfold bindSet at h
  split at h
  · simp at h
    subst h
    simpa using writesFrom_fst 0 bufs j hj
  · simp at h

/-- The descriptor written at binding `j` holds the `j`-th buffer. -/
theorem bind_buffer (p : Pipe) (bufs : List Nat) (ds : List (Nat × Nat))
    (h : bindSet p bufs = some ds) (j : Nat) :
    (ds[j]?).map Prod.snd = bufs[j]? := by
  unfold bindSet at h
  split at h
  · simp at h
    subst h
    simpa using writesFrom_snd 0 bufs j
  · simp at h

/-- The address table buffer's identity: the device buffer `bind_addrs` uploads. -/
def addressTableId (addrs : List Nat) : Nat := addrs.length

/-- The table's contents: one GPU address per buffer, in order. -/
def addressTable (addrs : List Nat) : List Nat := addrs

/-- `Gpu::bind_addrs`: one binding (the address table) or a refusal (`src/gpu/runtime.rs:306`). -/
def bindAddrs (p : Pipe) (addrs : List Nat) : Option (List (Nat × Nat)) :=
  if p.nBindings = 1 then bindSet p [addressTableId addrs] else none

/-- `bind_addrs` yields exactly one descriptor. -/
theorem bindAddrs_one (p : Pipe) (addrs : List Nat) (ds : List (Nat × Nat))
    (h : bindAddrs p addrs = some ds) : ds.length = 1 := by
  unfold bindAddrs at h
  split at h
  · rename_i h1
    have := bind_length p [addressTableId addrs] ds h
    simpa [addressTableId, h1] using this
  · simp at h

/-- The one descriptor is binding 0 holding the address table, whose entries are the buffers in
order. -/
theorem bindAddrs_table (p : Pipe) (addrs : List Nat) (ds : List (Nat × Nat))
    (h : bindAddrs p addrs = some ds) :
    ds = [(0, addressTableId addrs)] ∧ addressTable addrs = addrs := by
  unfold bindAddrs at h
  split at h
  · unfold bindSet at h
    simp at h
    rcases h with ⟨_, h⟩
    subst h
    exact ⟨rfl, rfl⟩
  · simp at h

/-! ## Handoff theorems: staging layout, `scatter` containment -/

/-- A staged block: its offset in the staging buffer and its length. -/
structure Block where
  off : Nat
  len : Nat
  deriving DecidableEq, Repr

/-- The first byte past the block. -/
def Block.stop (b : Block) : Nat := b.off + b.len

/-- The staging layout the Rust loop builds from offset 0. -/
def layout (off : Nat) : List Nat → List Block
  | [] => []
  | n :: ns => ⟨off, n⟩ :: layout (stageNext off n) ns

/-- The per-block fit asserts of `staged_dispatch` / `scatter`, in loop order. -/
def fitsDuring (cap : Nat) : Nat → List Nat → Prop
  | _, [] => True
  | off, n :: ns => off + n ≤ cap ∧ fitsDuring cap (stageNext off n) ns

/-- Every block lies within the staging window. -/
def within (cap : Nat) (bs : List Block) : Prop := ∀ b ∈ bs, b.stop ≤ cap

/-- Every later block starts at or after the first block's end. -/
def disjointFrom (b : Block) (bs : List Block) : Prop := ∀ c ∈ bs, b.stop ≤ c.off

/-- The layout's blocks are pairwise disjoint. -/
def pairwiseDisjoint : List Block → Prop
  | [] => True
  | b :: bs => disjointFrom b bs ∧ pairwiseDisjoint bs

theorem layout_offsets_ge (off : Nat) (lens : List Nat) :
    ∀ b ∈ layout off lens, off ≤ b.off := by
  induction lens generalizing off with
  | nil => simp [layout]
  | cons n ns ih =>
      intro b hb
      simp [layout] at hb
      rcases hb with rfl | hb
      · exact Nat.le_refl off
      · exact Nat.le_trans (stage_next_ge off n) (ih (stageNext off n) b hb)

theorem layout_disjoint (off : Nat) (lens : List Nat) : pairwiseDisjoint (layout off lens) := by
  induction lens generalizing off with
  | nil => simp [layout, pairwiseDisjoint]
  | cons n ns ih =>
      simp only [layout, pairwiseDisjoint]
      constructor
      · intro c hc
        exact Nat.le_trans (le_align256 (off + n)) (layout_offsets_ge (stageNext off n) ns c hc)
      · exact ih (stageNext off n)

theorem layout_within (cap off : Nat) (lens : List Nat) (h : fitsDuring cap off lens) :
    within cap (layout off lens) := by
  induction lens generalizing off with
  | nil => simp [layout, within]
  | cons n ns ih =>
      rcases h with ⟨h1, h2⟩
      intro b hb
      simp [layout] at hb
      rcases hb with rfl | hb
      · simpa [within, Block.stop] using h1
      · exact ih (stageNext off n) h2 b hb

/-- A `scatter` region: source offset in its group's staged block, destination offset, length. -/
structure Region where
  src : Nat
  dst : Nat
  len : Nat
  deriving DecidableEq, Repr

def Region.dstStop (r : Region) : Nat := r.dst + r.len

/-- The regions as staged at a group offset: the source shifts, the destination is untouched. -/
def stagedRegions (off : Nat) (rs : List Region) : List Region :=
  rs.map (fun r => ⟨off + r.src, r.dst, r.len⟩)

/-- The caller's destination ranges lie inside the destination buffer. -/
def regionsWithin (dstLen : Nat) (rs : List Region) : Prop :=
  ∀ r ∈ rs, r.dst + r.len ≤ dstLen

/-- The caller's destination ranges are pairwise disjoint. -/
def regionsDisjoint : List Region → Prop
  | [] => True
  | r :: rs => (∀ s ∈ rs, r.dstStop ≤ s.dst ∨ s.dstStop ≤ r.dst) ∧ regionsDisjoint rs

/-- A region whose source lies inside its staged block records a copy source inside the staging
window. -/
theorem scatter_src_within (cap : Nat) (b : Block) (r : Region)
    (hb : b.stop ≤ cap) (hr : r.src + r.len ≤ b.len) :
    b.off + r.src + r.len ≤ cap := by
  unfold Block.stop at hb
  omega

/-- Staging disjointness is a property of the layout, independent of the caller's regions. -/
theorem scatter_stage_disjoint (cap : Nat) (lens : List Nat) (h : fitsDuring cap 0 lens) :
    pairwiseDisjoint (layout 0 lens) ∧ within cap (layout 0 lens) :=
  ⟨layout_disjoint 0 lens, layout_within cap 0 lens h⟩

/-- Shifting region sources to their group's staging offset preserves destination disjointness:
the destination offsets are unchanged. -/
theorem stagedRegions_disjoint (off : Nat) (rs : List Region) (h : regionsDisjoint rs) :
    regionsDisjoint (stagedRegions off rs) := by
  induction rs with
  | nil => simp [regionsDisjoint, stagedRegions]
  | cons r rs ih =>
      rcases h with ⟨h1, h2⟩
      simp only [stagedRegions, List.map_cons, regionsDisjoint]
      constructor
      · intro s hs
        rcases List.mem_map.mp hs with ⟨s', hs', rfl⟩
        simpa [Region.dstStop] using h1 s' hs'
      · exact ih h2

/-- Destination containment is preserved by the staging shift. -/
theorem stagedRegions_within (dstLen off : Nat) (rs : List Region)
    (h : regionsWithin dstLen rs) : regionsWithin dstLen (stagedRegions off rs) := by
  intro r hr
  rcases List.mem_map.mp hr with ⟨r', hr', rfl⟩
  simpa [Region.dstStop] using h r' hr'

/-- Non-vacuity: two 8-byte blocks at 256-byte alignment fit a 1024-byte window. -/
example : fitsDuring 1024 0 [8, 8] := by
  simp [fitsDuring, stageNext, align256]
/-- Non-vacuity: an over-full staging window is a refusal. -/
example : ¬ fitsDuring 10 0 [8, 8] := by
  simp [fitsDuring, stageNext, align256]

example : (layout 0 [4, 4]).length = 2 := by decide
example : (layout 0 [4, 4])[1]?.map Block.off = some 256 := by decide
example : stageFit 0 8 1024 = true := by decide
example : stageFit 1020 8 1024 = false := by decide
example : stageNext 8 4 = 256 := by decide

/-! ## The Vulkan operating contract -/

/-- The descriptor type a layout assigns to a binding. -/
inductive DescriptorType where
  | storageBuffer
  | other
  deriving DecidableEq, Repr

/-- An abstract device. The five relations are supplied by the Vulkan implementation; they are
not defined here and no theorem about them is assumed. -/
structure System where
  /-- `before c i j`: the command at index `i` is observed to complete before index `j`. -/
  before : Cmds → Nat → Nat → Prop
  /-- The value the command at `i` reads at a device byte address. -/
  reads : Cmds → Nat → Nat → Nat
  /-- The value the command at `i` writes at a device byte address. -/
  writes : Cmds → Nat → Nat → Nat
  /-- The descriptor type a pipeline assigns to a binding. -/
  descriptor : Nat → Nat → Option DescriptorType
  /-- The device byte addresses a dispatch on `pipe` reaches through its bound buffer ids. -/
  touches : Nat → List Nat → List Nat
  /-- The device byte addresses a buffer occupies. -/
  range : Nat → List Nat

/-- The Vulkan operating contract: exactly four fields, one per obligation this repository does
not execute. Each is a hypothesis about a `System`, not an axiom about Vulkan. -/
structure VulkanContract (S : System) : Prop where
  /-- Submission order is preserved: an earlier command completes before a later one. -/
  submission_order :
    ∀ (s : Submission) (i j : Nat),
      i < j → j < s.cmds.length → S.before s.cmds i j
  /-- A host write staged into a buffer, submitted before a dispatch that binds that buffer, is
  visible to the dispatch. -/
  host_writes_visible :
    ∀ (s : Submission) (i j dw lo hi p : Nat) (bs : List Nat),
      s.cmds[i]? = some (.copy dw lo hi) → i < j →
      s.cmds[j]? = some (.dispatch p bs) → dw ∈ bs →
      ∀ a, lo ≤ a → a < hi → S.reads s.cmds j a = S.writes s.cmds i a
  /-- A pipeline's descriptor-set layout is one storage buffer per binding, in order. -/
  descriptor_layout :
    ∀ (p : Pipe) (i : Nat), i < p.nBindings →
      S.descriptor p.id i = some .storageBuffer
  /-- A dispatch reads and writes only the buffers bound to it. -/
  dispatch_confined :
    ∀ (p : Nat) (bs : List Nat) (a : Nat),
      a ∈ S.touches p bs → ∃ b ∈ bs, a ∈ S.range b

/-- The handoff composes with the contract: a copy recorded by `staged_dispatch` before a
dispatch that binds its destination is visible to that dispatch. -/
theorem staged_visible (S : System) (C : VulkanContract S) (mode : Nat)
    (uploads : List (Nat × Nat)) (ds : List (Nat × List Nat))
    (i j dw lo hi p : Nat) (bs : List Nat)
    (hcopy : (stagedCmds mode uploads ds)[i]? = some (.copy dw lo hi))
    (hij : i < j)
    (hdispatch : (stagedCmds mode uploads ds)[j]? = some (.dispatch p bs))
    (hmem : dw ∈ bs) :
    ∀ a, lo ≤ a → a < hi →
      S.reads (stagedCmds mode uploads ds) j a = S.writes (stagedCmds mode uploads ds) i a :=
  C.host_writes_visible ⟨stagedCmds mode uploads ds, 1⟩ i j dw lo hi p bs
    hcopy hij hdispatch hmem

/-- The binding-order handoff and the descriptor contract agree: a dispatched pipeline's binding
`i` is a storage buffer for every `i < n_bindings`. -/
theorem binding_is_storage (S : System) (C : VulkanContract S) (p : Pipe) (i : Nat)
    (hi : i < p.nBindings) : S.descriptor p.id i = some .storageBuffer :=
  C.descriptor_layout p i hi

end Gpu.Handoff
