# Torus and spinor sources

Literature retrieval for the Decode Atlas "lenses". Every entry below was downloaded
locally and read; a source is recorded as `retrieved` only when the local artifact
exists and its sha256 is recorded. The machine-readable index is the table in
[Retrieval index](#retrieval-index); the checker
`/home/socol/oc-tmp/dom-map-t2/check_sources.py` validates path and sha256 for every
`retrieved` row.

- Date of retrieval: 2026-09-28 (AEST).
- Downloads: `/home/socol/oc-tmp/lit/`.
- Conversion: `pdftotext -layout <file>.pdf` (PDF) or Elasticsearch tag strip (EPMC XML)
  or HTML read directly; page numbers are the PDF page unless a printed page is named.
- arXiv ids were verified by the title printed in the downloaded PDF; the one mismatch is
  recorded at item B8.
- A citation is a pointer until the artifact is in hand and read (AGENTS.md §9.3).

## Retrieval index

Status `retrieved` means the local_path exists and sha256 is the file hash. Status
`NOT RETRIEVED — anchor only` means no local artifact; the reason follows in the item.

| Item | Status | local_path | sha256 |
|---|---|---|---|
| A1 | retrieved | /home/socol/oc-tmp/lit/2104.09864_roformer.pdf | e9a481fbe1c8a20b7b1fa566b13102a1896c7829fa9a8b4c80528452a5ddaf79 |
| A2 | retrieved | /home/socol/oc-tmp/lit/2410.06205_barbero.pdf | f16f469d75f3b6b893a299d28af88e953436d5cb5b3ba10522a3630470dd5e52 |
| A3 | retrieved | /home/socol/oc-tmp/lit/2307.08621_retnet.pdf | b75fdb67fdcc760609a38d28171a785f336ece30f750d210c7379306bb483fa7 |
| A4 | retrieved | /home/socol/oc-tmp/lit/2111.00396_s4.pdf | 872b4ef302312d04ef4d55610613c0e59da32c443a2e38552d7cebb7dc3636fa |
| A4 | retrieved | /home/socol/oc-tmp/lit/2208.04933_s5.pdf | e0a2d0ed83933892bc514d4c3ec2149aa5fa89ce45f52cbf08e86c36164057d6 |
| A5 | retrieved | /home/socol/oc-tmp/lit/2406.10322_liere.pdf | b4c9a47015fe14a75631673de8e6bc904f731387f03d0c844143b5ece609e9c3 |
| B6 | retrieved | /home/socol/oc-tmp/lit/epmc_PMC8810387.xml | 655c902bfc9c791d3744ccdf1bf3d9eaacc68fb5060c140111df480761621ab0 |
| B7 | retrieved | /home/socol/oc-tmp/lit/fiete_2008.pdf | 2831591803721d477e62fe8dad51ad39f50187f1e37e1150dd194d535a1b7e72 |
| B8 | retrieved | /home/socol/oc-tmp/lit/wei_elife_2015.pdf | 4977386e36907a96b0341968516f82d7f693d76a77e820e6eea6ff0fe9476611 |
| B9 | retrieved | /home/socol/oc-tmp/lit/stensola_2012_abstract.txt | e03bb8467651fc61b6f3b32226fbcc8e597a98beae28ab5a888c88338129119f |
| B10 | retrieved | /home/socol/oc-tmp/lit/epmc_PMC4704792.xml | 27d7aa5a68e32ccf2ac267373b51e864b99339471f09632c8bcdc2d7755030a8 |
| B11 | retrieved | /home/socol/oc-tmp/lit/2112.04035_whittington.pdf | 882ceac7425501a4844dc2baf8d9ec47400fbfe2516d315f39f1bdead351bcb5 |
| B12 | retrieved | /home/socol/oc-tmp/lit/2210.02684_xu.pdf | acfef1783e5a6b5c241529fd16a1d0c58f701e7bef750d226168891d873c169c |
| C13 | retrieved | /home/socol/oc-tmp/lit/1612.04906_marklof.pdf | bcaad70ce8dd6d79ef94a0c1f54e8398e67b176cba9abd2de3364dfd9c820586 |
| C14 | NOT RETRIEVED — anchor only | — | — |
| C14b | retrieved | /home/socol/oc-tmp/lit/kuo_nuyens_qmc_guide.pdf | 9038d50311bbcaf406dd523a625547fca0b0cb3ce0c0e417ec9ac6e5843f4ac8 |
| C15 | retrieved | /home/socol/oc-tmp/lit/riefell_1981.pdf | 8d7a0747990d972ea8756437d575b6a92a5ed5bb0526b601c7e38909b9e2d647 |
| D16 | retrieved | /home/socol/oc-tmp/lit/0907.5356_clifford.pdf | da63be90496ab9dbb3d7025ccb3091bfdf83e884a3c1e1d05c1f8f626cf4b758 |
| D16 | retrieved | /home/socol/oc-tmp/lit/av8n_rotations.html | 64692a17919eb106c7b9c394c8cdae56e104f4375cfc90fe71d2cff05cb8f87f |
| E17 | retrieved | /home/socol/oc-tmp/lit/2404.00456_quarot.pdf | f611888c63ef63a5c0232e2c8416619f0c9ace08d0e05692731da82791202e3e |
| E18 | retrieved | /home/socol/oc-tmp/lit/2405.16406_spinquant.pdf | fe437770d7c981eae9e028eacaa5c772ed0add900ed9dec54c10cfce6dfd86c3 |
| F19 | retrieved | /home/socol/oc-tmp/lit/pollard_1971.pdf | 5d7d936ef418b3092bc8aeea0a1bc63f162af852bbe42f2a137077dc84d20752 |
| F20 | retrieved | /home/socol/oc-tmp/lit/andraka_cordic_survey.pdf | dd0ecddac280ea641fa931dc7cfe46200be9e788370ab4c534e9fbfc1d05e6ed |
| G21 | retrieved | /home/socol/oc-tmp/lit/barker_iota.html | 581e6547a7626c881808bd7a0fa3d37722e9e5333798b3c97f4c7126df7a5a12 |

## A. Rotary position and its recurrent twins

### A1. Su et al., RoFormer (arXiv 2104.09864)

- id verified: PDF title "ROFORMER: ENHANCED TRANSFORMER WITH ROTARY POSITION EMBEDDING".
- local_path: `/home/socol/oc-tmp/lit/2104.09864_roformer.pdf`; sha256 e9a481fb… (full value in the index).
- section/page: §3.2.2, p.3 (θ definition); §3.3 "Long-term decay", p.5.
- quote: "is the rotary matrix with pre-defined parameters Θ = {θi = 10000−2(i−1)/d , i ∈ [1, 2, ..., d/2]}"; and "Following Vaswani et al. [2017], we set θi = 10000−2i/d . One can prove that this setting provides a long-term decay property".
- relative-position property: the induced query/key inner product is a function of m−n only (Eq. 35, p.7): "q⊺[2i:2i+1] k∗[2i:2i+1] ei(m−n)θi".
- bears on the atlas: supplies the exact θ_i frequency schedule and the invariant that RoPE attention depends only on the relative offset, the anchor of the rotation lens.

### A2. Barbero et al., "Round and Round We Go!" (arXiv 2410.06205)

- id verified: PDF title "ROUND AND ROUND WE GO! WHAT MAKES ROTARY POSITIONAL ENCODINGS USEFUL?".
- local_path: `/home/socol/oc-tmp/lit/2410.06205_barbero.pdf`; sha256 f16f469d… .
- section/page: §5, p.8.
- quote: "we argue that the low frequencies are most useful to detect information related to token 'semantics' as they are the most invariant to token relative distance."
- bears on the atlas: assigns a function to the low-frequency bands — semantic channels near-invariant to relative distance — so the atlas lens distinguishes positional (high) from semantic (low) frequency use.

### A3. Sun et al., RetNet (arXiv 2307.08621)

- id verified: PDF title "Retentive Network: A Successor to Transformer for Large Language Models".
- local_path: `/home/socol/oc-tmp/lit/2307.08621_retnet.pdf`; sha256 b75fdb67… .
- section/page: §2, p.3, Eqs. (1), (3), (5).
- quote: "We diagonalize the matrix A = Λ(γeiθ )Λ−1 , where γ, θ ∈ Rd . Then we obtain An−m = Λ(γeiθ )n−m Λ−1"; parallel form "on = Σ γ^{n−m}(Qn einθ )(Km eimθ )† vm".
- bears on the atlas: the recurrence carries the decay-and-rotation factor γe^{iθ}; the parallel and recurrent forms are algebraically the same operator with decay, the temporal twin of RoPE.

### A4. Gu, Goel, Ré, S4 (arXiv 2111.00396) and Smith et al., S5 (arXiv 2208.04933)

- ids verified: S4 PDF title "Efficiently Modeling Long Sequences with Structured State Spaces"; S5 PDF title "SIMPLIFIED STATE SPACE LAYERS FOR SEQUENCE MODELING".
- local_paths: `/home/socol/oc-tmp/lit/2111.00396_s4.pdf` sha256 872b4ef3… ; `/home/socol/oc-tmp/lit/2208.04933_s5.pdf` sha256 e0a2d0ed… .
- section/page: S4 §3.1, p.5; S5 §2, p.2.
- quote (S4): "if A were diagonal, the resulting computations become much …"; quote (S5): "layer uses a parallel scan on a diagonalized linear SSM to compute the SSM outputs".
- bears on the atlas: S4 diagonalizes the state matrix and S5 makes the diagonal (complex) state matrix the implementable object; a diagonal complex state is exactly a per-channel rotation-with-decay, the discrete twin of A3.

### A5. Ostmeier et al., LieRE (arXiv 2406.10322)

- id verified: PDF title "LieRE: Lie Rotational Positional Encodings".
- local_path: `/home/socol/oc-tmp/lit/2406.10322_liere.pdf`; sha256 b4c9a470… .
- section/page: §3.1, p.2.
- quote: "Our key insight is that Lie group theory provides a natu- [ral framework]… LieRE learns a basis of skew-symmetric [generators]" and "a skew-symmetric matrix… is always a high[er-dimensional rotation generator]".
- bears on the atlas: generalizes the positional rotation from a fixed planar block to an arbitrary learned Lie-algebra element, the general-rotation lens over the same orbit picture.

## B. Tori in brains

### B6. Gardner et al., "Toroidal topology of population activity in grid cells" (Nature 2022; bioRxiv 10.1101/2021.02.25.432776)

- retrieved: Europe PMC open-access full text, PMC8810387 (the Nature version); the bioRxiv PDF endpoint returned HTTP 403 and was not used.
- local_path: `/home/socol/oc-tmp/lit/epmc_PMC8810387.xml`; sha256 655c902b… .
- section/page: Abstract.
- quote: "we show that the joint activity of grid cells from an individual module resides on a toroidal manifold, as expected in a two-dimensional CAN."
- bears on the atlas: the empirical torus; a grid-cell module's state space has the Betti signature of T², the reference the atlas lens measures other representations against.

### B7. Fiete, Burak, Brookings, "What grid cells convey about rat location" (J Neurosci 2008)

- id verified: PDF title "What Grid Cells Convey about Rat Location" (author-hosted copy; the jneurosci.org PDF endpoint was Cloudflare-blocked).
- local_path: `/home/socol/oc-tmp/lit/fiete_2008.pdf`; sha256 28315918… .
- section/page: p.3 (RNS) and Abstract.
- quote: "numbers, called the residue number system (RNS)"; the abstract states the code "is fully combinatorial in capacity".
- bears on the atlas: identifies the multiscale grid code as a residue number system; the combinatorial-capacity argument is the mathematical ground for why the module set represents position over large ranges.

### B8. Wei, Prentice, Balasubramanian, "A principle of economy predicts the functional architecture of grid cells" (eLife 2015)

- id check: the supplied arXiv 1304.0031 resolves NOT to this title but to the earlier preprint "The Sense of Place: Grid Cells in the Brain and the Transcendental Number e" (downloaded at `/home/socol/oc-tmp/lit/1304.0031_wei.pdf`, sha256 13ce882b8c0f42d0a25b18c3e9d5ddbfee5fe0a51472851f055b920d87009b2a). An arXiv API title query for "principle of economy predicts" returned no record; the eLife paper has no arXiv version, so the eLife open-access copy was retrieved instead.
- local_path: `/home/socol/oc-tmp/lit/wei_elife_2015.pdf`; sha256 4977386e… .
- section/page: p.11 (printed), §"Optimal ratio in two dimensions".
- quote: "we conclude that in two dimensions, the optimal ratio of neighboring grid periodicities is e ≈ 1.65".
- bears on the atlas: fixes the inter-module scale ratio to the transcendental e under an economy principle, the target constant for a multiscale torus lens.

### B9. Stensola et al., "The entorhinal grid map is discretized" (Nature 2012)

- paywalled: only the PubMed abstract was retrieved (journal PDF and full text not open); the numeric scale ratios (~1.4) are in the paywalled body, not in the abstract.
- local_path: `/home/socol/oc-tmp/lit/stensola_2012_abstract.txt`; sha256 e03bb846… .
- section/page: PubMed abstract, PMID 23222610.
- quote: "grid cells cluster into a small number of layer-spanning anatomically overlapping modules with distinct scale, orientation, asymmetry and theta-frequency modulation."
- bears on the atlas: the grid map is discretized into modules rather than a continuum, the empirical constraint that a scale-ratio lens must satisfy.

### B10. Seelig & Jayaraman, "Neural dynamics for landmark orientation and angular path integration" (Nature 2015)

- retrieved: Europe PMC open-access full text, PMC4704792 (abstract is enough; the full text was available and read).
- local_path: `/home/socol/oc-tmp/lit/epmc_PMC4704792.xml`; sha256 27d7aa5a… .
- section/page: Abstract; Discussion.
- quote: "Several features of the population dynamics of these neurons and their circular anatomical arrangement are suggestive of ring attractors — network structures proposed to support the function of navigational brain circuits."
- bears on the atlas: the fly head-direction ring is a one-dimensional torus (circle) attractor, the S¹ analogue of B6 in a system small enough to identify the circuit.

### B11. Whittington, Warren, Behrens, "Relating transformers to models … of the hippocampal formation" (arXiv 2112.04035)

- id verified: PDF title "RELATING TRANSFORMERS TO MODELS AND NEURAL REPRESENTATIONS OF THE HIPPOCAMPAL FORMATION".
- local_path: `/home/socol/oc-tmp/lit/2112.04035_whittington.pdf`; sha256 882ceac7… .
- section/page: Abstract.
- quote: "we show that transformers, when equipped with recurrent position encodings, replicate the precisely tuned spatial representations of the hippocampal formation; most notably place and grid cells."
- bears on the atlas: bridges the rotation lens (A) to the torus data (B6–B10): recurrent position encodings in a transformer reproduce grid/place codes.

### B12. Xu, Gao, Wu et al., "Conformal isometry of Lie group representation …" (arXiv 2210.02684)

- id verified: PDF title "Conformal Isometry of Lie Group Representation in Recurrent Network of Grid Cells".
- local_path: `/home/socol/oc-tmp/lit/2210.02684_xu.pdf`; sha256 acfef178… .
- section/page: Abstract; §2.2, p.1.
- quote: "Geometrically, we study the conformal isometry of the Lie group [representation]… the compact abelian Lie group representation automatically [gives toroidal topology]"; the abstract links conformal isometry to "hexagon periodic patterns in the grid cell responses".
- bears on the atlas: gives the group-theoretic reason a recurrent transform realizes a torus — a compact abelian Lie group is a torus — and ties conformal isometry to hexagonal tuning.

## C. Closed orbits and optimal point sets on the torus

### C13. Marklof & Strömbergsson, "The three gap theorem and the space of lattices" (arXiv 1612.04906)

- id verified: PDF title "THE THREE GAP THEOREM AND THE SPACE OF LATTICES".
- local_path: `/home/socol/oc-tmp/lit/1612.04906_marklof.pdf`; sha256 bcaad70c… .
- section/page: Abstract, p.1.
- quote: "The three gap theorem (or Steinhaus conjecture) asserts that there are at most three distinct gap lengths in the fractional parts of the sequence α, 2α, . . . , Nα, for any integer N and real number α."
- bears on the atlas: describes the discrete orbit of an irrational rotation on the circle/torus and the lattice geometry that governs its gap structure, the closed-orbit lens.

### C14. Dick, Kuo, Sloan, "High-dimensional integration: the quasi-Monte Carlo way" (Acta Numerica 2013)

- NOT RETRIEVED — anchor only: no open-access location. Unpaywall reports `is_oa=false` for DOI 10.1017/S0962492913000044 and OpenAlex reports `is_oa=false` with no OA URL; the publisher copy is paywalled.
- substitute (retrieved): Kuo & Nuyens, "A Practical Guide to Quasi-Monte Carlo Methods", the short-course notes prepared for "High-dimensional Integration: the Quasi-Monte Carlo Way" (by Kuo, a co-author of the Acta paper).
  - local_path: `/home/socol/oc-tmp/lit/kuo_nuyens_qmc_guide.pdf`; sha256 9038d503… .
  - section/page: §2.1, p.8 (generating vector and rank-1 lattice rule); Fig. 1, p.9.
  - quote: "Given an integer vector z = (z1, . . . , zs) known as the generating vector, a (rank-1) lattice rule with n points takes the form [x_k = {k z / n}]".
  - Korobov and Fibonacci lattices: not found in the retrieved substitute; that part of item C14 remains open at anchor grade.
- bears on the atlas: rank-1 lattice rules give the equal-weight optimal point sets on the unit cube, the arithmetic model of a closed orbit sampled on a torus.

### C15. Rieffel, "C*-algebras associated with irrational rotations" (Pacific J. Math. 93 (1981) 415–429)

- retrieved: MSP open-access article PDF (article p12 of vol. 93 no. 2).
- local_path: `/home/socol/oc-tmp/lit/riefell_1981.pdf`; sha256 8d7a0747… .
- section/page: Abstract, printed p.415 (PDF p.2).
- quote: "For any irrational number a let Aa be the transformation group C*-algebra for the action of the integers on the circle by powers of the rotation by angle 2πa."
- bears on the atlas: the noncommutative algebra generated by an irrational rotation is the operator-algebraic home of the rotation orbit; the trace range on projections is the orbit-density invariant.

## D. Operator algebra and spinors

### D16. Spin double cover and rotors with operation counts

Two open-access texts were retrieved (the item asks for one).

- id verified (D16a): Lundholm & Svensson, "Clifford algebra, geometric algebra, and applications", PDF title matches; local_path `/home/socol/oc-tmp/lit/0907.5356_clifford.pdf`, sha256 da63be90… ; Theorem 6.9, printed p.64 (PDF p.68).
  - quote: "f : Spin(s, t) → SO(s, t) … are surjective with kernel {±1}. In other words, Theorem 6.9 shows that the Pin and Spin groups are two-sheeted coverings of the orthogonal groups."
- D16b: Denker, "How to Represent Multi-Dimensional Rotations, Including Boosts", §7.1 "Double Coverage", §7.2 tables; local_path `/home/socol/oc-tmp/lit/av8n_rotations.html`, sha256 64692a17… .
  - quote: "Actually it is a two-to-one relationship. Any given rotation can be represented by two inequivalent rotors (r and −r)."
  - operation counts (compound rotation): dimension 3 → 16 multiplications (rotor) vs 27 (matrix); dimension 4 → 64 vs 64; dimension D → 4(D−1) vs D³.
- bears on the atlas: the rotor group is the double cover of SO(n) acting by R v R̃, and the tabulated multiplication counts state the cost of composing rotations in a Clifford algebra rather than matrices.

## E. Rotations for quantization

### E17. Ashkboos et al., QuaRot (arXiv 2404.00456)

- id verified: PDF title "QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs".
- local_path: `/home/socol/oc-tmp/lit/2404.00456_quarot.pdf`; sha256 f611888c… .
- section/page: Abstract; §3.1, p.2.
- quote: "We show that randomized Hadamard transformations can be applied to the weight matrices [without changing the network output]"; "A rotation matrix is an orthogonal matrix. A Hadamard matrix is an orthogonal matrix with entries drawing from {+1,−1}."
- bears on the atlas: shows a fixed orthogonal rotation of weights/activations is numerically invariant and reduces quantization outliers, the rotation-as-invariance lens.

### E18. Liu et al., SpinQuant (arXiv 2405.16406)

- id verified: PDF title "SPINQUANT: LLM QUANTIZATION WITH LEARNED ROTATIONS".
- local_path: `/home/socol/oc-tmp/lit/2405.16406_spinquant.pdf`; sha256 fe437770… .
- section/page: Abstract; §1.
- quote: "we propose SpinQuant, a novel approach that incorporates learned rotation matrices … SpinQuant also outperforms concurrent work QuaRot, which applies random rotations".
- bears on the atlas: picks up the learned-rotation variant, where the rotation is optimized rather than fixed, the counterpart of the learned Lie-algebra rotations of A5.

## F. Exact integer transforms and rotations

### F19. Pollard, "The fast Fourier transform in a finite field" (Math. Comp. 25 (1971) 365–374)

- retrieved: AMS open-access PDF.
- local_path: `/home/socol/oc-tmp/lit/pollard_1971.pdf`; sha256 5d7d936e… .
- section/page: Abstract and §1, p.365 (PDF p.1).
- quote: "A transform analogous to the discrete Fourier transform may be defined in a finite field, and may be calculated efficiently by the 'fast Fourier transform' algorithm."
- bears on the atlas: the number-theoretic transform realizes the rotation/DFT exactly in a finite field, the exact-integer counterpart of the floating-point rotation lens.

### F20. Volder, CORDIC (1959) — survey substitute

- Volder 1959, "The CORDIC trigonometric computing technique" (IRE Trans. Electronic Computers): NOT RETRIEVED — anchor only (paywalled).
- substitute (retrieved, permitted by the task): Andraka, "A survey of CORDIC algorithms for FPGA based computers".
  - local_path: `/home/socol/oc-tmp/lit/andraka_cordic_survey.pdf`; sha256 dd0ecddac2… .
  - section/page: §1–2, p.2.
  - quote: "[performing] vector rotations by arbitrary angles using only shifts and adds. The algorithm, credited to Volder[4], is derived from the general (Givens) rotation transform."
- bears on the atlas: CORDIC performs a rotation by a shift-add angle recurrence, the exact/fixed-width rotation lens with per-iteration angle halving.

## G. Combinators

### G21. Schönfinkel (1924) / Curry — anchor; Barker, Iota

- Schönfinkel 1924 and Curry: NOT RETRIEVED — anchor only (no open-access full text retrieved in this slice).
- Barker, "Iota: an unambiguous Turing-complete language with just two symbols", §1.
  - local_path: `/home/socol/oc-tmp/lit/barker_iota.html`; sha256 581e6547… .
  - quote: "F --> i ; Semantics: ^x.xSK"; "the symbol '^' stands for lambda … xSK = ((x(S))(K))".
- bears on the atlas: the single-combinator basis ι = λf. f S K composes S and K by application, the minimal-basis combinator lens.

## Trust boundary and residual obligations

- Retrieved means the local bytes exist and hash as recorded; it does not mean the paper's
  claims are re-derived here (grades per AGENTS.md §5).
- Not retrieved, and left as anchors: C14 (Acta Numerica paper; OA substitute at C14b), F20
  (Volder 1959; survey substitute), G21 (Schönfinkel/Curry), and the Korobov/Fibonacci-lattice
  passage of C14.
- No builds were run for this task; no repository file other than this document was written.
