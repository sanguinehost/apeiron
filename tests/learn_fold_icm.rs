//! `tests/learn_fold_icm.rs` — the `E8 ∪ O8` block-ICM refinement of the one-bit fold transfer:
//! `J` non-increasing per sweep for both `e8_icm_step` and `e8_joint_icm_2plane`, exact recovery of
//! a target representable by one or two planes, the chosen block code against a brute-force `J`
//! surface, the `r_step2` normal equations, `row_chunk` invariance, the `icm_ridge` metric, and the
//! ragged-shape refusals. Grade `tested`.

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::float_cmp,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::unreadable_literal,
    reason = "the LCG maps integers to `f64` by division, the flat row-major index arithmetic is the reference payload format, the representability and argmin checks are exact float comparisons, and the test helpers mirror the ported option surface"
)]

use apeiron::learn::code::e8::codeword;
use apeiron::learn::fold::fit::icm::{
    e8_icm_step, e8_joint_icm_2plane, j_rows, matmul, nearest_index, plane_w, r_step2,
    ridged_metric, IcmError, JointStats, Plane, PlaneRef,
};
use apeiron::learn::solve::q32;

/// Deterministic xorshift64 LCG.
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform in `[lo, hi)` from the top 53 bits.
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let u = (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0;
        lo + (hi - lo) * u
    }
}

const BLOCK: usize = 8;
const GROUP: usize = 128;
const CODE_LEN: usize = 256;

/// A fixture as the `Q32.32` words the module consumes: `round(x·2³²)` at the grid range.
fn q(x: &[f64]) -> Vec<i64> {
    x.iter()
        .map(|&v| q32::to_grid(v, q32::GRID_BOUND).expect("fixture inside the grid"))
        .collect()
}

/// A word slice read back to the reals, for the `f64` oracles and bounds.
fn unq(x: &[i64]) -> Vec<f64> {
    x.iter().copied().map(q32::from_grid).collect()
}

/// The 256 codewords as the `Q32.32` words the module stores.
fn codebook() -> Vec<[i64; BLOCK]> {
    (0..CODE_LEN)
        .map(|j| codeword(j).expect("row in range"))
        .collect()
}

fn pref(p: &Plane) -> PlaneRef<'_> {
    PlaneRef {
        s: &p.s,
        idx: &p.idx,
        r: &p.r,
        c: &p.c,
        d: &p.d,
    }
}

/// A positive-definite Toeplitz metric `Hl[i,j] = rho^|i-j|` (the autocorrelation of an AR(1) plus
/// the `j = i` term), a stand-in for the calibration Gram.
fn toeplitz(n: usize, rho: f64) -> Vec<f64> {
    let mut h = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..n {
            h[i * n + j] = rho.powi(i32::try_from(i.abs_diff(j)).expect("d < 2^31"));
        }
    }
    h
}

/// A plane with random codebook rows, random group scales in `[0.5, 1.5)` and unit `c`, `d`.
fn rand_plane(o: usize, n: usize, rng: &mut Lcg) -> Plane {
    let nb = n / BLOCK;
    let g = n / GROUP;
    let cb = codebook();
    let mut s = vec![0_i64; o * n];
    let mut idx = vec![0_u16; o * nb];
    for row in 0..o {
        for m in 0..nb {
            let j = (rng.next_u64() % 256) as usize;
            idx[row * nb + m] = j as u16;
            let at = row * n + m * BLOCK;
            s[at..at + BLOCK].copy_from_slice(&cb[j]);
        }
    }
    Plane {
        s,
        idx,
        r: q(&(0..o * g)
            .map(|_| rng.uniform(0.5, 1.5))
            .collect::<Vec<f64>>()),
        c: vec![1_i64 << 32; n],
        d: vec![1_i64 << 32; n],
    }
}

fn rand_mat(rows: usize, cols: usize, rng: &mut Lcg) -> Vec<f64> {
    (0..rows * cols).map(|_| rng.uniform(-1.0, 1.0)).collect()
}

/// Which ICM arm a merged test runs: [`e8_icm_step`] or [`e8_joint_icm_2plane`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Arm {
    Step,
    Joint,
}

/// One arm at `sweeps` sweeps from `(p1, p2)`, with `J` measured in metric `hj`: both planes, the
/// measured `J`, and the joint arm's stats (`None` for the single-plane step).
///
/// # Complexity
/// The arm's own: `O(o·n² + sweeps·o·(n/8)·CODE_LEN·…)`; spatial `O(o·n)`.
fn run_arm(
    arm: Arm,
    ws: &[i64],
    hl: &[i64],
    p1: &Plane,
    p2: &Plane,
    o: usize,
    n: usize,
    sweeps: usize,
    ridge: f64,
    hj: &[i64],
) -> (Plane, Plane, f64, Option<JointStats>) {
    match arm {
        Arm::Step => {
            let out = e8_icm_step(
                ws,
                hl,
                pref(p1),
                o,
                n,
                sweeps,
                q32::to_grid(ridge, q32::GRID_BOUND).expect("ridge inside the grid"),
            )
            .expect("step");
            let j = plane_j(ws, &out, hj, o, n);
            (out, p2.clone(), j, None)
        }
        Arm::Joint => {
            let (q1, q2, st) = joint(ws, hl, pref(p1), pref(p2), o, n, sweeps, ridge);
            let j = pair_j(ws, &q1, &q2, hj, o, n);
            (q1, q2, j, Some(st))
        }
    }
}

/// The seeded instance of one arm at `o × n`: two planes and the target. The `Step` arm draws only
/// one plane, so the draw order per arm is the one its own test used.
fn arm_case(arm: Arm, o: usize, n: usize, seed: u64) -> (Vec<i64>, Vec<i64>, Plane, Plane) {
    let mut rng = Lcg(seed ^ (o as u64));
    let hl = q(&toeplitz(n, 0.4));
    let p1 = rand_plane(o, n, &mut rng);
    let (p2, ws) = match arm {
        Arm::Step => (p1.clone(), q(&rand_mat(o, n, &mut rng))),
        Arm::Joint => (rand_plane(o, n, &mut rng), q(&rand_mat(o, n, &mut rng))),
    };
    (ws, hl, p1, p2)
}

/// The `1e-9` relative slack every "did not rise" comparison carries.
fn tol(j: f64) -> f64 {
    1e-9 * (1.0 + j.abs())
}

/// `plane_w(p1) + plane_w(p2)`: the weight of a two-plane sum.
fn pair_w(p1: &Plane, p2: &Plane, o: usize, n: usize) -> Vec<i64> {
    let mut w = plane_w(pref(p1), o, n).expect("plane shape");
    let w2 = plane_w(pref(p2), o, n).expect("plane shape");
    w.iter_mut().zip(&w2).for_each(|(a, b)| *a += *b);
    w
}

/// `J` of one plane against `ws` in metric `hl`, as the grid word read back.
fn plane_j(ws: &[i64], p: &Plane, hl: &[i64], o: usize, n: usize) -> f64 {
    let w = plane_w(pref(p), o, n).expect("plane shape");
    q32::from_grid(j_rows(ws, &w, hl, o, n, None).expect("j shape"))
}

/// `J` of a two-plane sum against `ws` in metric `hl`, as the grid word read back.
fn pair_j(ws: &[i64], p1: &Plane, p2: &Plane, hl: &[i64], o: usize, n: usize) -> f64 {
    q32::from_grid(j_rows(ws, &pair_w(p1, p2, o, n), hl, o, n, None).expect("j shape"))
}

/// The two-plane arm with the options every test but the refusal and block-subset cases uses:
/// `g_row = None`, `row_chunk = 256`, all blocks.
fn joint(
    ws: &[i64],
    hl: &[i64],
    p1: PlaneRef<'_>,
    p2: PlaneRef<'_>,
    o: usize,
    n: usize,
    sweeps: usize,
    ridge: f64,
) -> (Plane, Plane, JointStats) {
    e8_joint_icm_2plane(ws, hl, p1, p2, o, n, sweeps, ridge, None, 256, None).expect("joint")
}

/// The seeded decision-safety instance `(o, n)`: metric `hl` (`rho = 0.37`), two planes with
/// random group scales and random real `c`, `d` in `[0.5, 1.5)`, and the target `ws`.
fn seeded_case(o: usize, n: usize, seed: u64) -> (Vec<i64>, Vec<i64>, Plane, Plane) {
    let mut rng = Lcg(seed);
    let hl = q(&toeplitz(n, 0.37));
    let p1 = rand_plane(o, n, &mut rng);
    let p2 = rand_plane(o, n, &mut rng);
    let ws = q(&rand_mat(o, n, &mut rng));
    (
        ws,
        hl,
        scaled_plane(p1, n, &mut rng),
        scaled_plane(p2, n, &mut rng),
    )
}

/// The same plane with random real column and Hadamard scales in `[0.5, 1.5)`.
fn scaled_plane(mut p: Plane, n: usize, rng: &mut Lcg) -> Plane {
    p.c = q(&(0..n).map(|_| rng.uniform(0.5, 1.5)).collect::<Vec<f64>>());
    p.d = q(&(0..n).map(|_| rng.uniform(0.5, 1.5)).collect::<Vec<f64>>());
    p
}

/// A plane reconstructed from `idx` and the pass-through `r`, `c`, `d` of `base`.
fn plane_from_idx(idx: &[u8], base: &Plane, o: usize, n: usize) -> Plane {
    let cb = codebook();
    let mut s = vec![0_i64; o * n];
    for row in 0..o {
        for m in 0..n / BLOCK {
            let at = row * n + m * BLOCK;
            s[at..at + BLOCK].copy_from_slice(&cb[usize::from(idx[row * (n / BLOCK) + m])]);
        }
    }
    Plane {
        s,
        idx: idx.iter().map(|&v| u16::from(v)).collect(),
        r: base.r.clone(),
        c: base.c.clone(),
        d: base.d.clone(),
    }
}

/// The decision-safety baselines: the `idx` planes and the `r_step2` scales this module returned
/// *before* the scalar `f64` product was replaced by the shared Q32.32 [`matmul`], captured by
/// running the same seeded cases against the pre-change `src/learn/fold/fit/icm.rs` at this
/// worktree's base revision. Each `idx` baseline is one hex byte-pair per code index ([`unhex`]);
/// `#[rustfmt::skip]` keeps each fixture in its own line group and the tests below are the only
/// consumers.
#[rustfmt::skip]
const STEP_16_256_IDX: &str = "
    61b0a2aaa2a1fdfb49f4e9efc5acb3eee0801e687f003a854fb4c7ac7c14066140ffb4aa0fa3dd097986e28806fc9bc6baf29edeb1fee6d9a9a8baa2a2b0dcaee8b3d35ba15c89b7188fc7602065de61a3f29d2c840196004dfabcf381c0955bf5b02b933319644732f0b4e523b89da1e18488e4db61bebf1615b4cea70ae09ebeffc7f388b4c0f24f10ff90f4f2f2e3b5b610d900e887999bbb17004a3813cff47dd7718a9ecbc7b3f4e33afa45f1f9b5b2f384c0c67607faf99e6f8a958c94ab9a2c72894c84181b28e08ebfef5f699c649fc9e8fe9eed8339c6bf8882de90f483d7ab83b669a8cba2e0b3f5c282a5e029f1d8f861b7891fdbfbbe5ff063bf9e9ba2ffa089fcdae3c7feedf5fcb1a52637f2f4bcfe7915f2dabcbc12b2f7b3169860f3a40796b4b325e840c53fa64eeaf2811738d7361821c8b9cf67bf65c7c4b3d71af7a15da94bb2fea161acf2f39cf284ba16fef3949c6de3f6ce9aa30a21eb21b390a15eb01112e3affaf2f2f2457bf2ac4d16d4f481ebcccc486f0a86bfb10df383f6fdd7ddf3fee8af0cf2a5
    e0a6a8bba95c09129e5ee8db64abc9cecfd4a5bd8bb7fda9e6c6e0ebbcada2b6c481f2bd21fe21dcaa1de8faf2bdc8a89e9b5df282a1884aa69be235e3ecb65ed512a4ad9bfeb33fa82bcf6d8230f8cfaab3cd5589a19400b8f5bb10c635b2f2b88e821eb3eb12868649b9be815589d7";
#[rustfmt::skip]
const JOINT_16_256_IDX1: &str = "
    bca0cb831d9cddd8a69e8cc9c4751746dcd18a7775fd709e6b3f8496f095065bce49f9f15f5bca4bb8326cc9f3d2a1b2b86381c9ded5499afb752fa7d643081db091031330a9c3663aba5269c3b841977ee8da9f7a95dda0c0c1bff528fd0ccdf99487a497e144f7e30cebf83fc83d15f85a65ddc020e24e7edd41b6b9dd9ffa34c4bfaba1f256c86b08b08db38582ed5949029c20aaf5b2f7ab010a581b62fce4a98bcfb43a610612f4ba12fa758ff9ba42f05aff4476c667dfe22e4da96eed1a2604362a26f4a24cb42217b648a0dbfa08b2fdbe2ed26fe333ffb3bcb3ecafea6ab33255d449fb9ab139e77335ee2fc4098ffa44bbb789fa4058be815b63979efea217a9a132b3baaefeede9ac85a19a02b89ec0e9d10ca5daa5fef16fde811d6d16e0d991311ef720fa4f7ec04448809521d2348b8dce35d1b8736dd3acf2c4eb5abbbaa23d104a9944b1fec4f24f843157ca9bacb360b8769033bc880bd733fffff390f2fc391d12e3adce9da5b925bc5ea8f054f672490c564d79c3fd35c9c50f51cea0fdb3a419b8a84024a89c
    eef8a1e78df611f9811e4a08fd2b38a3fbf69e9c96bff788f8f236bdb538b3f5e28c26bd11feb7acbe1baa44f07d769d53a1c44eaba9c34a2c7e9ef16a490da25a771e9d8ec4b300821ba67928a7ca61930af410065056955de093678539856b3e8f3d34e5fef12c15685884997f7757";
#[rustfmt::skip]
const JOINT_16_256_IDX2: &str = "
    91d20ef02a02a7dba53a8fca990a4838fdd266080a960f9df84c878c2c1711c36836b8f00effc9f1db3f138c82f618b1be1c82a3dd523699e90ae7a4d53c77c68192fd882f5e5501fbf92d509669bc9401eb6db0659603a3c3c2bc9e57fc98cee0b0710294e2e2b6fe73e88d40b5be80f2a91aec6894e193ecde3eb5bede9ce74bc7bc2a5dae29d2a3559ef8b0f15beea1b19c4ca2259cb1d5dad8a1befd11fde757997bbb559ef091c2b9c66d0aca8cb9ea87c43d96ab17fd231e591346bd9565196f88555985d333c75d0ccaa8d2b1fb67ff96b5517a51e068d6febfadff6fff15f24d2aedb1d899b246be0c4adc5044eef8eb3b75aa87e639279eb8d3ab859dffacbbe84a4db0b9fdd74aea1086f666c0bbf2adfbec37f8dc9e994210262f02526909da920d79cddce6df830fc00783ac5ec5bbf08ecdbbdcdb14ced04683fffa25b8b9a1426fa99a3b45edc7a630824e480a54afb07ff9ae934cbf8b748a4cea01b1fb4dd14062d45eaecd9eb8ba81bf21948a92d60db880293206c0c2e2ca87702e3dd423a1a766bb0d3fb2e69f
    ace20bfc8e566ef8328d9671dc54476bcea19d9ffdcea08bdbb2d0f54f56c6f4e10a59276ee139a65164253bf1cb099e2ca2c779a8da59fc90019d8c153f7276250861848dbbb07f4764a5bac0a469ca9814bfdcf92f6b96e8e3431886caf41441f80a4be6eb84533aaefc879ab74a78";
#[rustfmt::skip]
const R_16_256_R1: &[f64] = &[
    0.0511476482440279, -0.12741000463005375, -0.08575968443736218, 0.0093718835119899, -0.032635966530201316, -0.014767012491232906, -0.14047964463817939, 0.050483716190126426, -0.023622076907397094, -0.005386127541437067, 0.03345364959890477, 0.0029903278428920045, -0.06325887023998307, -0.040727762763047, -0.012783830321332113, 0.09450519572322044,
    -0.008013489873327463, 0.08027757034219753, -0.01737416492926023, 0.012422636821798201, -0.0345889057320183, -0.1333056338014776, 0.06485053999831882, 0.09046993253584047, -0.07602058339811604, -0.06782579890987714, 0.022625090684196596, 0.05959913495093209, -0.08593297169615163, -0.01992568170353099, -0.04932587672406011, 0.00828767193417752,
];
#[rustfmt::skip]
const R_16_256_R2: &[f64] = &[
    0.000612316443107597, 0.06318978190498309, 0.020652552691880828, -0.014611859581048263, -0.03372133182555285, 0.019695285161339143, -0.043750778697326964, 0.08268515359627349, -0.06105366205889894, 0.08918528365622669, 0.038701621847146275, -0.00010707131601661429, -0.025330479181070023, 0.012252660974343311, 0.010819750428973148, 0.09335520318737098,
    -0.017037781437366152, -0.0023979885242944535, 0.09952598109652275, -0.002028360743959119, 0.006630733538060017, -0.025553027181273412, -0.10914367446229015, -0.08206662831127078, 0.047578118580948854, -0.0010362201376113648, 0.028493667276380648, 0.027493612899826183, -0.05024847063737889, -0.020209332891703475, -0.04376886766596754, -0.02458688982491713,
];
#[rustfmt::skip]
const STEP_32_512_IDX: &str = "
    fbd335baf3e6f352a782f3fa76bc24e75cdc36730e24e4de6e85173553e9e810365b72d07710d789995fe6cfed10a29ce6d6f91ff7f51bfabe716647b68ef35735a9a86f6827f2226d29bbdd9af139d2f6e80d7c2598eba2f2eea6f427cffbbc90fe38d0e97fc48466cd4fb2f5b9dfeeef46edfaa59d86aa86e36bb0b5d8aff7ebb756b8cc22f4d3b39ef337c55310e535d78728fafe954fb2afdab9ffabbcd0aad42ddfa8d0348345d1e91dc322e7d1efa6dafb35cfc7bfa172bac323cfa2abe3aae60dd4f99c97b188afa1fff5c322d40484b5cfd5eeeea2e9a9f544aa922df6e3fd7cfb21920a0df6df905ea61d65e3ccddbf8afaf4baabd8863afbf9f3de16fd2f26aaddc1afafe3f39ec693a4d34212a6abaab382bf2d2d0db1a1ccabb967feeff6a8d3b18bff9e223bc6f9f604cb1bde2b3ab8cc0c28093ab5c80fffb135fceef5a9299de521f79ea9c493d7f7937e8fecf18a33875cd023ecd39543ed3ec1e0b53900a8220b7fb15f1d5e6eade663fffac959e02691c3e0f3f1e9f26a80323ed8ad19d1aaa7d6124c4e2e2db6
    d12094cdeaebfac3dac1aa84ffd5e4a8ead4b5b5a88b9fa05f2bee95b22443fee39aeeccccf49cbea0e9b9f38bf9abf95bd3e5b9abdc34839da39e1e42bdb9564c88f3e660d4e9fc759cae834aa8c19fbe9d5562eb9ab039ead5ec237899effe65f7f989d1f4c6dba9674904891706142737fcf4a124fc7e23f7a6e40973e80b56fe8dbca9d793ed04bfdbaa57cf490a222addf6aaae75a1e6b4ed8d80ea8a26dbcaaffad3cf8545914fb7b017c9e315997a93c79f328314f9349c109ab6ac6b9beda46574acc1f07c84acf57a23a85429c4feb6fbd4c7906df7e88622d024b2e638e3e4cdcfc76e4b2cbb45d5abb3cc2c7baebcad2185863a1686a20e872bfe0b3a69f6faaae35daa7eaef4362f44d05bffc70dbd52619d0eed66fbfec6d0ede7a2fefbf7c5792381c5245aced9b0e47de6e5afa0e34bfddd0fc8a012fd2afbcabff5f696e850909f52e3d3b03b00a2e6ea58c2afee2527b9d0178f5d258a4d6282e359151ca2bc3ae93c9ce8216380b3ad97be9ea7acd0f983aeaa2e9bc0d3313f19e0ce1aa391b62cd9d2ac7ffcb1
    be04ddd3b2d7f280c18bffd8b0a5f7c654f962ceee7ea6b9f05224ddbee0c5389e57ffc1a8dd64f1a48ab9e5ff7ab2e6f7fc8ff6fa9afbebb3c0f29d7693bd14c07b3ec2a8d4a3a168018c0cd729fceee0691986eed09d3490be2a5f07daa67a1cc1e4f4d6522fc07216a7a6e9d3eee5c6d03936629b78ebf30cf385d6fac1ef32325fb4d4d08e920fc3e04cc5df236b16f9dae5f79e0a4a8ac39eb8ccc951ece8fcfcf4a97baaf22b9af2e539bfc4e4d1f617bd0acffd0e9eeab09efe4cf012dad4dccac6cbb1d4b09f7f609cbcc87ede83262a9e4a8bba81c09f1b9ff8b08f1310986306dff33597b7f9ead6f5f31adeb7f3f6fafa5917b104b2f4b02c1ca3aa09c3ca00790406ec105a6615b2df6adb8c0dfaf7f282f81e77f5a093ccf294ff057397ef7b8594cc8d5ff22fc3a371e8b7c8b190d63f9baac2e8c1ffccf4c0e6c2eefcfa7ed5a8b13894ecaa34d084e1cfdbfad093b2dabdc76dd6cd11d2ac6aeb959ec34d96e67b8a2a2eb6f4fdeb1dc1f1018f8a2893afc3407502a2579c2484e403e815d5afabb625599ff7f626
    be6161fbd2aac5c891e9f4d1b4f9c3e53dd3461521ddadac28a2a725d80719d2ddba3df680c75b1672e88386ef8cfb37a36aedc0f200b54604a4ffe2b8f7f141696925fac8f2c4b920fe9d63fa8c25483dace5c39fdf9ed1f288f28a292feecc0e4ae3b59beb24698383a6dbb3cfe8c16af5fc2f46935898ff7d10dbf26b8bffe0f7eed76b9df4da861b7badea69a0af143de69ea0049e8efd829da71926c3fc34a8639c99ad2667c23d051915bdb6c6bc3bdd7582f708f67ac3dc52b31492ff3eb9e3fad1b4c7848bfe429ccee0f2fbcbfc8a269ddeaee4dcd5635216c7ae932f4ff1a8c3128e388eb8eb145dab3efd2259a3ca81d582903e8a4be5f4d9cdb1e5f2d223a21ef4bd6b11f49fe9c8f2e49dfd3dbffaf896ac1909a1d2ffc228f7c3bbc7702d93d422aa57a8e5588d86f76361e0d3e010c9a8c3a3b1dbb1b9e56fe2a9edfab7bec5faf31ef4f2cd7002fcaacb4e3fccaf6ef0df83f2ddfed003d3dcec80a06f93fce6a8eed9719e814ab76509de9c5e975fcc37fc00b93d76531eff36a6cca3a93d4b82569e3583308291
    eabee5b8e5f9acadf2f3a7e8ff2dc4d30198a61af104fd1162acf8f465b7c5d6be66d6fcfb664cbd6cc310593266f797fcd6d4a10ec5860a86aaf5c78ba5edf87c12ac9dabf95b34aaf1f3cfb790ffcd43b72216e711d6c33d6da9f5e393fb1c7b292cf93d99b192d2bc5ae806b4c89c3a5decd07b2ef585a9759f9d54a4f290db426ce3a0e7a122fd8f44a47b90de0fe3d65e9e96ffe6eab1bea4f2fe891430cec5b62afbc7cc93ef64e60ac46295d4fdaaddbda2aa9c3acaefa79e7007afe75a75e1f5e7f958bcaca9f9dbb7bdd4e6e5d8a429909596fa226ea7c959989fd69fc9c3b79b1e26f2e8a38ce5b3c9c9b3debbecbdf61286daf2249cb3f61cb3a21b26739db1f99794de98cbac9896f922ffcf16e59190e857f2c53dc7fea54d97c6eccd74b20bbe54841056dc11f8f720e50adefdf6eaf5f9590a1ab01167a166a9d3978fb3f8f2e67ef6f3241e132d65a22c8e41dcccbc67185ea121a8a285dc9ce3c6d38746ff83fca2b0729f5ed3b269f34467f73a9a0d0f31ad288734d172b13f612ab0a4f33272a0cacbc42ee6a1
    33d08df2fa15b069f324f985a3a2719d9fc1ffd1ffccfea396f6e08fb0633aafe9cce06db4abf52580ddf4c09161f2e4";
#[rustfmt::skip]
const JOINT_32_512_IDX1: &str = "
    7f9d75f2558094716a2ff3fcf07f1462ce1f49ab1458e2f0c239042d24e21ad7beb4663166c47ed28bffead923773bbcf932e2a31790fefabced664e69d353e5dc2c6f4d6821e0edfa9ea4f0da067254d8daf87c8ccfb109e0eef29c7fa334526ae18f87c97b9878e7536891c60c6edfe9896fcdc9f40fd8b3df7b25f40791a193c295c8cc829fd3f8f90ea0c5e70120c4dfb2f7a8d65d6fb0971b1176abbcf25df235fefa9426a74d4ed58a04ce329c042b3a1429459db7ac7bf8ac1bb7b968236b8b0c60fd12f3d11b4aade435e0d7b6ed7c6fda73122550c6d9efed45dd0d1bdcef7fdaf5d03fa26977a799f45c60e58cf92c0ed8a5a1a54f831f9162fca3827df0c2b4fda970ef00257abd0da9c3f0aef1cc94cf19fef2b60d6fac0e9386b7bfef00e512a88a03e5bb8fb0eadd36da1edb82aea54823de3faaf7ee32168035069786d5969cfa96f6a6303797353f255b5b75c4f35e9f24291273e06051bc2ec5c973c4003ecc9fd68f443f4e74adbf6168b5f90bf48fbcbbf51a32999568d2b304db5dff1fbbe2961fb0fca7bbf9
    e7d098ea8374ffc20a3ef351d7993622fe7f373778999ff698c0f04cb024dab441a00a4f2298c2da88e138c68acbcf8f87fcb2b444c794d084c6b9fdb04ce22b93909aea7b8d72cf0590eedc4891fad61d20a5d1cc3f5c009ffd5fa828156df2a945fe4b64b77a446d369864bd4b9714352e85eff0f52228289553668c405700361088a63e90b3f1ee37b50564cd230803830436c9d07f4cbd190a85585dfa26931e151089245e8dff0b787caa9a14dfb12efdb14a721b8e76b31c11f084df4f35f572fcd4d5f8af483064a163dce3a8f254dfd3ddb429fa21f4d09b36ed1ffafc48ee1b48f4f288f1ee8f8c3c9f1d4b6f7afc9efda752fbd0d4023bd40280648af5d77a9f88bb8c1535ac61f1d2fba16372fd5d0b4394f687de8e52a437c930eba7dd35957bc643768114d5a7153dcaea9b43e05bcb7a0c3fc9127b34e7b89b97a1fef087a29bec4415d68dd4b85a6a75aac948dfda2f05e0bb5568c0b0314dee83dde9f7b6f541f1026ca0c42882913bbf97574bf06408909ec2e75fc4f1f20f7f87f67cf022c8eca1c8331f7d0616
    0bf1f800fb8b6060ee8633a49f0dce0bc02d9d96b95ecaa1c4247f43d0e7eafeaa16c797567de700828ab9df40c609075de43dc80983835d9bb87963eff54aa1d29a0dc34ddf03a79902e702b1f9dcef0ddc0271c1e48cfd0842fd7dac6179fe1db2b617919b2dc0b1dcf94cfcca28fec030b658704fe337b3c69e3accf4820152ba4d8718aec7aa6240a434fc3fdec8e6a7b0cbddc96aa63a0ccea5be8f11dc3f0f9658f77d87a91abceb05a0c6e001e0d45fb597eac1ff9e19389fe38845ad9f4145626e858019a1487388bb42e344312f22d5b28f57a0ebdf51c500c829cfc0588b00ed4f0536e4a5310b99bbfbf90c5016f1d1f758ceba1a0dfc786d1cb8be09ef7700fe3428fe10a76615f950c3ee842f69cf856be65e9889a524ceb0e695e4519be8696f81d1c9f3a86bc5da99ccc2e1c577ca90fa90986e587f8db75b5dbee58ba87f186ead3a94d40633cb84fb2ac1ce86d051d83332e0f0471795fdceff8162027c9f2f772e2a05c78efc24fb5d8e15db8bdf877cf789dd32ff1627cf93e1ed3e424c06a8c6d2aca2d4993e
    c2e38b19e881f034e4595ac8dd35e40a8233f0852370e4f2c07fb9d37fc4dfda8c4f8eb4fdb681d616ade6c0688c1c326763a45d3a005745148c66e32fff72d74babfe12dd32c413707876b7904680af2a203abe9f8fc06f257b260c29174224c054b9b7479386ece3bfbb284625fba51a449f08ca0b21f8b07eebbfd55fa0e5ddbaac8361b7df356c076148706ae880ea82e3025910bac894d667e3ae9a9da69f562e29c15a02ff037f17da051d09fd2450f23818061c60ca99d840765a2f63a49fede7953c3cbdf077107dda00fb64ffda2a31efe483a8df9947322f10af8a928be467c476601b85cbf3ac6b4bae92e42554dc49323a66cdd4d892eefbfbd9e97fe3cc884833fcb0318483a1ca91b085f1399f9df8aef23db1f7df11c208036dfd21314a8e3ab3a2b2c206409981a07b49d512ba328f8bdd023d91f51049661aeeaff7a387ed62c4f99ef110fb06ca77ea378477ecd2794c82e035a19d67b9fc3185cbe4b02e7bf352bafb83aa9950963e8f1eb8be5e5601b88f192aa3ff977ee3c657cbe6b89704d34234f3d97363
    78e5ee5fbcdfada98e509e15ff5fe5ef810bc04d3aae4665c842d1b8545b00f0be6549f4cf7b54f2981cd2d0a3e09cf63ef4e2ca3897e6c9cea38dc3f23c82d53d56da77efaf81469218a2d016fde55947e160e0ea8bb3bf39bafd851b1e292cfc233638f50f75eff8eb50f044c5deca0f3e97f6ff74ef314e039b0808f7898ae285e3ed92f118c56ac946b6bf8e68eae3e25e5ef1d7e4ecb1aca4a0b3cb14249ff6cccc91e64ef1b3437999324e680995378e8683e9ada5dcb5197f70539cbf5a507ec7fdde7c9e33a4f920d7b508ecf0cae2719cf8976abef01a7d4873d20faec9fe7f0006f2f707d3440a5c1cf9b3b9f9ade5b5308ec9b6e33efdbcc9a014b88431d88c478971e29b650d7e05face95f812d367f07a90ad00379b4f93b4f24578d07c780b9d54a62f53f19cd88904e9f9ec6e0ac66a5d7d5e62c6bd2f1a3791f180624bdc37fa56bbac2c53632de060f28c8600a44cfbbd5de6f0bdfdcf297fec26ee2399f5ab18df958bb6e00bb35c1ed5f6176e40f5ff85922984f68dfa13376cb47ce5b4a41d3a66da18ed1472
    c3098dcbfc771a39afb71befb27cac9dd4e1dcb4987b32ae5a39e4d4165f75ea967f8c6d43f15e52a3838d80b528fde2";
#[rustfmt::skip]
const JOINT_32_512_IDX2: &str = "
    009ec2f3e88397d41530b0ad8a8fbb1dcd893673c627f88ffd46edd653e163ea3c43b9f65bf501958898e9d35c084491f8ba86ac6893c5bdf5581dd2af1f2c75dbf732fafafae3e5ea9da78e3d790de4dbd9bb068fcc8176e34daef30038f9f116e2915dca54f157ce2cce0b1173108e910ef1b724ba70413d3ffe5abb7e92ab8c7617f7a9819c8800f8718b6be47e5fc7dcffc9b7d52210b394626e09ced99f22b2b6b1b12ee68eb938b7766acdeb0ea0ddf38d56f255b4afcfa5afb01cbae9881ef6e81ffcb0a9d295937be74a0a23b5d43310d90c6d5e2fff4aecfd3af9e92cdf1ce7d9bd3640f71e08f2c4f59a8e498f8c53ebf9f8a2caa17f301209fdb081029634bedc460fd6c05a05be7247c08378f0500acc66fff3b5c51095718d907dbcd27facd9d9896ce6364da2e9e349cfe6fa81f2a6e85cdd40a9f6ed6d691e6a795d856464f5fb62d2a54f48944240f3740c3b098233825effc2cd1184be4d5164751422d8fccf9cb16f3f52314a9fbce91787f8746d46fbd89c444d9696d76ab0a8d822d66016b398604ed1a4b8c8
    cc02ed1b6b0be0957544224ed492494dff04819107f126a10fc39a75f3aabab73e139c285d9bc1d9210bcec5603dcc90f0d3b1cb8fc434f7f5c54dedb333e154819399e9c38e07927a3ded2afaa891d562cc5661ebb223aca9ed9da7f06aefb2a63242761ab4053bd343098a9037452aa25111ecf1f45d5ee896933b8f3f9dcc812fd3a5db93b0f0ed48b67ae4ce4dfa3ca58049ca52d333be9975b927b7e659f0616a6f8a5b609be08c0703ac99bb1df351e6b2930d051009c494fef10dc0399ff40dfdd7d2f9f2374f39a2c162e0aba72b0d2ddef556915e9dd6c449ee249128c47d673db7af8b946f958f0338859a10e1b2a4a8a14d1d3016c5f0c87df11340bcd4059cf1c1f96a5a18b489d1d8f61c0dc22774f497d65ee1622de2c3dc4fd4d8434a83fd7d9b09a06ed6a46c62c9e1983cf9a9cdfff0408e889d4bfabb9557a2c796c0f28d6a2b6ad534ce43011593a954078edd10fab9dc3f125c052ef1b6b17beaa2f7b43e787d15d60513f0924ee8632e3e8f7b7738f5180397e380af708bbd76d2f1a6cbefa2ce4571b27969
    7485408c524d8e1fc5854cf33872287448da9e956ef2f6a26a5b243cfde4e9ff9c60c494290ed8b4840985dc3fc5767822bd42dec1f2802298f9241cc09e351ed1f074fc32dc7ca42ae647c805c831fe80df79879dfb8fec778aece5affa06e962b1b5cdf198cc60fefd4586d1c957d79f4fb527be30e0cdb0fff545cfbe81faf1b9ce84fead9a9d1db1b26ae540258d12acaa3cdeca153adc98cdf815918c0840707a27cd0290fa47bfe895a3ebe377d5a2012c4bd7c2e09d75899c5e2e7baa323ef135cb869bb0cd160ca1353de03e4b505e23c58c28a35bdc0f01dc56762baa25887b2630e2d6e7444274d68ecec8732b6980f7c8dcd9bdd9c5afbe12e3da54a823933d9b4acae16fa4ea6ab82e8439c65016fa1af19721358ac85b09b8463ffb63d4efbc61d0faf090a20604d938d2c1d4b0b6c98e93939b1b27008eb48e03bd75f0a35adc110feae0d28d4cc887d855f53e91d3e58ffb47e39338a996fccdeace1dd9031171b219117af68dc35baa3a3bdfe4d18ebd13d10ed2bf1a6f7a8a905be5433dc579abd5b6f389d79a41
    1efe8024eba88bd5e726e0cbde4ae7c4814c9386740ce71fc300f5ac1e6f17dc9ab19cc2ac53a5f6699a0a0c173163dc10cd41a865f4fdc1c68219b074d6f119a423076d5b4d8ccd2707097b9b7d87c3557f46fbbd5e7a1ac54c59b9a32a295bc32bf5b93894856ce0aab8d6395a32e165da7ba86abd95e2b3e6e8f48929a3e681b9af80f3f6dc4a13a0d5ce0faea9839b5de0fb2c6fb99b973a0cfe208ca2a59c82d2ddf526f8e06d8ec32dbce5e5ec5b2fab472761bb1fc99abffa0925118ef3706892f14343908f086f32f87fa81bc4d99ef8ecea804fdcf13081506faca36d88e739f5e31fa986c8b1afbde7ad91975a2b62564ddc29ce9bbb59edfae5da98009201f13798adb3b3f191f3193f5f869806809ea582a36308f6dc667a6675afc2aa4e35f945a572b1f49cd5b282a3b7363491b9bff084bb55420b3a6fc51965edacc8a05aee09c7c89df06f90db5ac9e3482e2aefd1f0e3afff4aa2591864978186c82db35104a82df4aa95809a2f954190a962bded29d829667255a0fef001e89e2e0ed30b947bf63d7bf2120cb4
    0749ffb0bf57ae1b8d2f9d6ab082e6fe8284c332bafd391a523db5b6fa247f92511a36bef646299e2e22d145a099f38bcbc37dda47ace5a3cda0e1e6f3435f57d39ad908e8ac3a39ab07ca146994e62638433be3ef88b0cc46b9e683642156b995cdcd53c584effccbe8b62af1c4ddc972e694b5e80bec4ecb7c497777cd1c89eb12d9af91916761cf8ba9a1bc90bce915f8cf2180d1e748b2b383a3b0f85b5b77b5fb9382e59684f23c061acd315676f14895f080ea941edfb6efabfc2c9ff4053705d6defdde82cddbddeb6ecd77d592f6ba0e9f8dfc92ab826302bc4c08d62489c7213f790ef6a05b3d8023631e2305ddaee6b64f8df7b5f865e64cf8d74af92fccdb8f388ae0ff981ac501dadbf696c90dfc5cf1a38eae5fd598305b6583080710aa07fad8c27150dd9401e697f9fedd421284b30522f9211f9c215065c7f189831d343adb91a01daf713c1ca3e31fa28385dca733fabea8fa8bbefc4fca00ef5aed1e9ac5a867e19688f75a74b09dde547540118af4e086e7e587101ebd6cd5132223fbb78f62451fa9673e4f0d
    c07f14f8fd08ac46acf662c2ca03f359d7c442c29bbe4dfd2546e7d269200afaf0008bc13c91212dc88015834c57e65f";
#[rustfmt::skip]
const R_32_512_R1: &[f64] = &[
    0.07848805911999562, -0.15952104911327533, -0.09418253074562377, -0.0909120036702233, -0.02307305735760893, 0.04763803024742423, 0.007692248383608649, 0.034653900932764686, 0.06017131920395385, -0.12284920636774496, 0.040664711477152664, -0.08514775981430225, -0.03813683078970495, -0.021341058973027082, -0.012019503548468407, 0.00023120646639553704,
    0.011342107476723762, -0.003389335734768465, -0.009841287365788567, 0.134135959902154, 0.04557687754774456, 0.032451333133355326, 0.023152991242574496, -0.07746348087450998, -0.07900118760979247, 0.0015573713238701988, -0.03179802059484568, 0.0426227929982589, 0.012990271085409948, 0.016194562793918045, -0.03569910555881144, -0.032140747417879124,
    -0.05284981643352273, 0.07106057146901876, -0.11787244085014091, -0.11812788994283047, 0.0163712129390241, 0.06497551758374219, -0.013857236997381945, -0.128175326417096, -0.003323450313722719, 0.07130329259196794, 0.07004033587865761, -0.005161711986281893, -0.059686289062916786, -0.0035719472842199824, -0.01607953261878329, -0.08571254652612129,
    0.06898584827073809, -0.03396033715718486, -0.0568394342079743, 0.05907367483602325, 0.09651852648816159, 0.01165402024459097, 0.009721087676506559, 0.017358648092280913, -0.03475260047597217, 0.09371979430968336, 0.12667712680756532, 0.07471942804560228, -0.10322278314465977, 0.09712667607697573, 0.015756055590391865, 0.02185252604400988,
    -0.019153129886211415, 0.08256274829605936, -0.0315814662856747, -0.01689933095555154, -0.01179794040503688, 0.0514045962585135, -0.06841936548074778, 0.031466768648461346, -0.03604626617617611, 0.017276715733669856, 0.07438158005534606, -0.05414038521174157, -0.01604776718358866, 0.07155982581184625, 0.11784181155038699, -0.020585053965243214,
    0.03469425042208642, 0.10896790372769723, -0.045792825917703284, -0.002263670318323546, 0.0559241122916425, -0.046288772674987676, 0.007653907226406747, 0.053029683590512675, 0.022805855296202923, -0.053649586450276794, -0.008555811317680436, 0.03456396318819556, 0.06529168897531636, 0.035854242618137785, 0.016301800124815587, -0.039509418536805396,
    -0.11325591935793448, 0.03752258998391129, 0.021589639036543543, 0.06629293810911047, -0.00025631527127548056, -0.059148303284552894, 0.010209835603057962, -0.07906762170121454, 0.04176598125762776, 0.04805222161243885, -0.011370423461506478, 0.017829741589821594, -0.05005283022036391, 0.03126341132430132, -0.07090666931875599, -0.012985029695498133,
    -0.09147796457132745, -0.0652566259662616, 0.020584569430551695, -0.09348796393179644, -0.05008089015534813, 0.020254858131104357, -0.024853380433829456, 0.007998540251552857, 0.04594524261583169, -0.07195177128001862, -0.03962527825061933, 0.036618499000139325, -0.17949365624648742, -0.03780313799306336, 0.007517914665405153, -0.050324774142047515,
];
#[rustfmt::skip]
const R_32_512_R2: &[f64] = &[
    -0.049182327512119366, 0.049287807257329466, 0.012607453548523008, 0.03263463256762912, -0.1222729835200716, -0.021806994819059955, 0.08304077680713837, 0.05242768612250564, -0.12744879263111428, -0.009575802396857886, 0.06356862264142243, 0.11793978918239119, -0.07236067319310643, 0.01885988413716768, -0.011519007118893324, -0.10954448802897869,
    -0.0799510025839491, 0.10589379013850854, -0.043762536515302576, 0.016590895504738832, 0.04515986649413818, 0.017546451145160274, 0.04300767405558722, 0.00757823034524026, 0.06891086441623533, -0.035928264925189586, -0.06595696123921724, -0.026276535487396464, -0.04609887166311009, -0.05897053766148767, 0.0027252448785902274, 0.08488220377662961,
    -0.11625151260186295, 0.04660277949239836, -0.031491351371752, 0.0096016998783424, -0.10171491523432372, -0.09423524165736635, 0.04736511974339139, 0.014071726585561939, -0.013375057287087528, 0.01802984994645429, -0.01446464922748989, 0.007576565725415508, 0.04217101938181653, -0.02940786022874948, 0.0020508272810192798, -0.03826999630034096,
    -0.03222238509088028, 0.11592548989573458, -0.022218843630855557, -0.02267968667265873, 0.011767116811604361, -0.017930154328297292, -0.02537713263044597, 0.010290473170916414, -0.04412657625396143, 0.04159915120434101, -0.020656155075213314, 0.06678494449984772, 0.019999020330882404, -0.015415576933536845, 0.03011372413833918, -0.04266026486200076,
    0.04523337633028763, -0.0838555731423554, 0.02330428838006084, -0.04123865100832951, -0.12654652878834546, -0.0006790100025599697, -0.023535101080269724, 0.03649490965579929, 0.10583400964050814, 0.025625756309514632, 0.10952198999317789, 0.004857473314619872, 0.0884244319923895, 0.007764444292105131, -0.10190152586903994, -0.007106568090174514,
    0.11025830828000584, 0.041978071831378354, 0.04345081431741152, 0.036582331714579024, -0.029904407614482313, -0.060756688628232275, 0.061346346865939455, -0.11843276955913906, 0.03210937131898499, 0.0008967439273570534, 0.05455027201611729, 0.041647200466059624, -0.006303972955220529, 0.0028754007063574224, 0.052456458638067485, 0.0010117201573236164,
    0.07080981438718856, 0.016073797314953175, 0.0029964247429277104, -0.09044759263401887, 0.030037231685050014, -0.043038702314943666, 0.05907102743746723, -0.09139423155387874, -0.047267984023582246, 0.11223610633393, 0.09713943122260574, 0.07825885977629242, -0.047622081582220754, 0.01776738348221445, -0.04542538029641466, -0.012076629914244442,
    -0.0707021286852132, -0.034762766623770175, 0.08957280517805781, -0.09549828966625858, 0.0389548351276302, -0.05537954709891667, 0.054129526507233766, -0.03866920858969678, 7.512546457391686e-5, 0.07576424846760009, -0.04111363002406056, -0.058194600372402434, -0.11429055094322357, 0.002258779534318201, 0.00709115737274635, 0.06538834215690126,
];

/// Decode an `idx` baseline: one `u8` code index per hex digit pair, other bytes (newlines, spaces) ignored.
///
/// # Complexity
/// Temporal `O(hex.len())`; spatial `O(hex.len())`.
fn unhex(hex: &str) -> Vec<u8> {
    let digits: Vec<u8> = hex
        .bytes()
        .filter_map(|b| (b as char).to_digit(16).map(|d| d as u8))
        .collect();
    assert!(
        digits.len().is_multiple_of(2) && !digits.is_empty(),
        "hex fixture"
    );
    digits.chunks(2).map(|p| (p[0] << 4) | p[1]).collect()
}

/// The pre-change scalar `f64` product of `src/learn/fold/fit/icm.rs` (the `ikj` loops), kept here
/// as the independent reference the shared Q32.32 [`matmul`] is measured against.
///
/// # Complexity
/// Temporal `O(rows·inner·cols)` f64 multiply-adds; spatial `O(rows·cols)`.
fn old_matmul(a: &[f64], b: &[f64], rows: usize, inner: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0_f64; rows * cols];
    for (r, k) in (0..rows).flat_map(|r| (0..inner).map(move |k| (r, k))) {
        for c in 0..cols {
            out[r * cols + c] += a[r * inner + k] * b[k * cols + c];
        }
    }
    out
}

/// The derived element-wise bound on `|matmul − old_matmul|`.
///
/// Each operand entry is rounded to the `Q32.32` grid after the exact power-of-two shift that
/// brings the panel to at most `8` in magnitude, so it carries at most `2⁻³³` (a half unit in the
/// last grid place) of error in scaled units and at most `2⁻³³·max(1, max|entry|)` in the
/// operand's own units; the contraction adds `inner` such terms against the other operand's
/// magnitude, and `gemm_nn`'s single rounding of the sum adds one more. The bound below is `4×`
/// that worst case `2⁻³³·(2·inner·max|a|·max|b| + max|a|·max|b|)`.
///
/// # Complexity
/// `O(1)`.
fn product_tolerance(inner: usize, amax: f64, bmax: f64) -> f64 {
    8.0 * 2f64.powi(-33) * inner as f64 * amax.max(1.0) * bmax.max(1.0)
}

/// The largest absolute entry of `x`.
fn max_abs(x: &[f64]) -> f64 {
    x.iter().fold(0.0_f64, |m, v| m.max(v.abs()))
}

/// One row's `r_step2` normal system `(M, t)` in plain `f64`, in the direct `(tt, j)` order:
/// `M[a][b] = Σ_{t,j} P_at·Hl[ga·128+t][gb·128+j]·P_bj`, `t[a] = Σ_j P_aj·wh[row][ga·128+j]`, where
/// `P` is the block panel `s·c` of `p1` for `a < G` and of `p2` for `a ≥ G`. The exact-arithmetic
/// reference the grid system is compared against.
///
/// # Complexity
/// Temporal `O((2G)²·GROUP² + (2G)·GROUP)`; spatial `O((2G)² + 2G)`.
fn row_normal_system(
    p1: PlaneRef<'_>,
    p2: PlaneRef<'_>,
    hl: &[f64],
    wh: &[f64],
    row: usize,
    n: usize,
) -> (Vec<f64>, Vec<f64>) {
    let g = n / GROUP;
    let two_g = 2 * g;
    let (s1, c1, s2, c2) = (unq(p1.s), unq(p1.c), unq(p2.s), unq(p2.c));
    let pv = |a_: usize, j: usize| -> f64 {
        if a_ < g {
            s1[row * n + a_ * GROUP + j] * c1[a_ * GROUP + j]
        } else {
            s2[row * n + (a_ - g) * GROUP + j] * c2[(a_ - g) * GROUP + j]
        }
    };
    let mut mmat = vec![0.0_f64; two_g * two_g];
    let mut tvec = vec![0.0_f64; two_g];
    for a_ in 0..two_g {
        let ga = a_ % g;
        for bcol in 0..two_g {
            let gb = bcol % g;
            let mut acc = 0.0;
            for tt in 0..GROUP {
                for j in 0..GROUP {
                    acc += pv(a_, tt) * hl[(ga * GROUP + tt) * n + gb * GROUP + j] * pv(bcol, j);
                }
            }
            mmat[a_ * two_g + bcol] = acc;
        }
        let at = row * n + ga * GROUP;
        tvec[a_] = (0..GROUP).fold(0.0, |acc, j| acc + pv(a_, j) * wh[at + j]);
    }
    (mmat, tvec)
}
/// `max_b Σ_j |P_bj|` over one row of `p`: the amplification of that row's inner rounding.
fn block_abs_sum(p: PlaneRef<'_>, row: usize, n: usize) -> f64 {
    let (s, c) = (unq(p.s), unq(p.c));
    (0..n / GROUP)
        .map(|b| {
            (0..GROUP).fold(0.0_f64, |m, j| {
                m + (s[row * n + b * GROUP + j] * c[b * GROUP + j]).abs()
            })
        })
        .fold(0.0_f64, f64::max)
}

// The pre-change `f64` bodies, duplicated here as the oracles the migrated `Q32.32` bodies are measured
// against (`icm.rs` at `9f440192`); each agreement bound is derived at its use site.

/// The pre-change `f64` `plane_w`: `s·d`, the normalised 128-block Hadamard transform, `·c`, `·r`.
///
/// # Complexity
/// Temporal `O(o·n·log2(GROUP))`; spatial `O(o·n)`.
fn ref_plane_w(p: PlaneRef<'_>, o: usize, n: usize) -> Vec<f64> {
    let g = n / GROUP;
    let (c, d, r) = (unq(p.c), unq(p.d), unq(p.r));
    let mut t = unq(p.s);
    for k in 0..o * n {
        t[k] *= d[k % n];
    }
    for chunk in t.chunks_mut(GROUP) {
        let mut h = 1;
        while h < GROUP {
            for i in (0..GROUP).step_by(2 * h) {
                for j in i..i + h {
                    let (a, b) = (chunk[j], chunk[j + h]);
                    chunk[j] = a + b;
                    chunk[j + h] = a - b;
                }
            }
            h *= 2;
        }
        let inv = 1.0 / (GROUP as f64).sqrt();
        for v in chunk.iter_mut() {
            *v *= inv;
        }
    }
    for k in 0..o * n {
        t[k] *= c[k % n];
    }
    for k in 0..o * n {
        t[k] *= r[k / n * g + (k % n) / GROUP];
    }
    t
}

/// The pre-change `f64` `j_rows`: `Σ_o g_row[o]·(D_o·Hl)·D_o` over the `ikj` order.
///
/// # Complexity
/// Temporal `O(o·n²)`; spatial `O(1)` beyond the inputs.
fn ref_j_rows(ws: &[f64], w: &[f64], hl: &[f64], o: usize, n: usize, g_row: Option<&[f64]>) -> f64 {
    let mut total = 0.0_f64;
    for row in 0..o {
        let mut j = 0.0_f64;
        for i in 0..n {
            let mut acc = 0.0_f64;
            for k in 0..n {
                acc += (ws[row * n + k] - w[row * n + k]) * hl[k * n + i];
            }
            j += acc * (ws[row * n + i] - w[row * n + i]);
        }
        total += g_row.map_or(j, |g| g[row] * j);
    }
    total
}

/// A plain `f64` Cholesky solve of `a x = b` (`a` is `n × n` symmetric positive-definite): the reference's `ridge::solve_spd` inlined for the oracle, `None` on a non-positive pivot.
fn ref_solve_spd(a: &[f64], b: &[f64], n: usize) -> Option<Vec<f64>> {
    let mut l = vec![0.0_f64; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 0.0 {
                    return None;
                }
                l[i * n + j] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    let mut y = vec![0.0_f64; n];
    for i in 0..n {
        let s = (0..i).fold(b[i], |s, k| s - l[i * n + k] * y[k]);
        y[i] = s / l[i * n + i];
    }
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let s = ((i + 1)..n).fold(y[i], |s, k| s - l[k * n + i] * x[k]);
        x[i] = s / l[i * n + i];
    }
    Some(x)
}

/// The pre-change `f64` `r_step2` body, with [`ref_solve_spd`] in place of `ridge::solve_spd`.
///
/// # Complexity
/// Temporal `O(o·(2G·128·n + (2G)³/3))`; spatial `O(o·n + (2G)²)`.
fn ref_r_step2(
    ws: &[f64],
    hl: &[f64],
    p1: PlaneRef<'_>,
    p2: PlaneRef<'_>,
    o: usize,
    n: usize,
) -> (Vec<f64>, Vec<f64>) {
    let g = n / GROUP;
    let two_g = 2 * g;
    let wh = old_matmul(ws, hl, o, n, n);
    let mut r1_out = vec![0.0_f64; o * g];
    let mut r2_out = vec![0.0_f64; o * g];
    let mut kvec = vec![0.0_f64; g * GROUP];
    let (s1, c1, s2, c2) = (unq(p1.s), unq(p1.c), unq(p2.s), unq(p2.c));
    for row in 0..o {
        let pv = |a_: usize, j: usize| -> f64 {
            if a_ < g {
                s1[row * n + a_ * GROUP + j] * c1[a_ * GROUP + j]
            } else {
                s2[row * n + (a_ - g) * GROUP + j] * c2[(a_ - g) * GROUP + j]
            }
        };
        let mut mmat = vec![0.0_f64; two_g * two_g];
        for a_ in 0..two_g {
            let ga = a_ % g;
            for gidx in 0..g {
                for j in 0..GROUP {
                    let mut acc = 0.0;
                    for t in 0..GROUP {
                        acc += pv(a_, t) * hl[(ga * GROUP + t) * n + gidx * GROUP + j];
                    }
                    kvec[gidx * GROUP + j] = acc;
                }
            }
            for bcol in 0..two_g {
                let gb = bcol % g;
                let mut acc = 0.0;
                for j in 0..GROUP {
                    acc += kvec[gb * GROUP + j] * pv(bcol, j);
                }
                mmat[a_ * two_g + bcol] = acc;
            }
        }
        let tvec: Vec<f64> = (0..two_g)
            .map(|a_| {
                let ga = a_ % g;
                (0..GROUP).fold(0.0, |acc, j| acc + pv(a_, j) * wh[row * n + ga * GROUP + j])
            })
            .collect();
        let mean_diag = (0..two_g).map(|k| mmat[k * two_g + k]).sum::<f64>() / two_g as f64;
        for k in 0..two_g {
            mmat[k * two_g + k] += 1.0e-12 * mean_diag;
        }
        let x = ref_solve_spd(&mmat, &tvec, two_g).expect("spd normal system");
        for k in 0..g {
            r1_out[row * g + k] = x[k];
            r2_out[row * g + k] = x[g + k];
        }
    }
    (r1_out, r2_out)
}

/// The migrated bodies against the pre-change `f64` oracles on one seeded case, at a derived bound.
///
/// # Complexity
/// Temporal the sum of the three oracles' (`O(o·n²)` dominated); spatial `O(o·n)`.
#[test]
fn learn_fold_icm_agrees_with_the_f64_oracles() {
    let (o, n) = (4_usize, 256);
    let mut rng = Lcg(0x51CE_0FAC);
    let hl = toeplitz(n, 0.4);
    let p = scaled_plane(rand_plane(o, n, &mut rng), n, &mut rng);
    let p2 = scaled_plane(rand_plane(o, n, &mut rng), n, &mut rng);
    let ws = rand_mat(o, n, &mut rng);
    let u = 2f64.powi(-32);

    // plane_w: one grid rounding per operand conversion and per multiply, the normalised transform
    // amplifying an input error by at most √128 and rounding each output once, then the `·c` and
    // `·r` multiplies. Bound `(√128·u·(1 + max|s·d|) + u)·max|r|·max|c|`: the grid quantum times
    // the operation's condition growth, `u` a full unit to cover the half-unit roundings.
    let got = unq(&plane_w(pref(&p), o, n).expect("plane_w"));
    let want = ref_plane_w(pref(&p), o, n);
    let (s_f, d_f, r_f, c_f) = (unq(&p.s), unq(&p.d), unq(&p.r), unq(&p.c));
    let sd_max = (0..o * n).fold(0.0_f64, |m, k| m.max((s_f[k] * d_f[k % n]).abs()));
    let wbound = (128f64.sqrt() * u * (1.0 + sd_max) + u) * max_abs(&r_f) * max_abs(&c_f);
    let worst = (0..o * n).fold(0.0_f64, |m, k| m.max((got[k] - want[k]).abs()));
    assert!(
        worst <= wbound,
        "plane_w differs from the f64 oracle by {worst} > derived bound {wbound}"
    );
    println!("icm oracle: plane_w worst {worst:.3e} bound {wbound:.3e}");

    // j_rows: `D = Ws − W` carries one unit per entry (two conversions), the `D·Hl` product the
    // matmul bound `pe` plus the `n·u·max|Hl|` of `D`'s own error, the `·D` reduction carries
    // `max|D|` of that per term, and the row sum rounds once.
    let dmax = ws
        .iter()
        .zip(&want)
        .fold(0.0_f64, |m, (a, b)| m.max((a - b).abs()))
        + u;
    let pe = product_tolerance(n, dmax, max_abs(&hl)) + n as f64 * u * max_abs(&hl);
    let psum = (0..o).fold(0.0_f64, |m, row| {
        m.max(
            (0..n)
                .map(|i| {
                    (0..n)
                        .map(|k| (ws[row * n + k] - want[row * n + k]) * hl[k * n + i])
                        .sum::<f64>()
                        .abs()
                })
                .sum::<f64>(),
        )
    });
    let jbound = u * (psum + (n as f64) * dmax * pe + 1.0);
    let got_j = q32::from_grid(j_rows(&q(&ws), &q(&want), &q(&hl), o, n, None).expect("j_rows"));
    let want_j = ref_j_rows(&ws, &want, &hl, o, n, None);
    let jworst = (got_j - want_j).abs();
    assert!(
        jworst <= jbound,
        "j_rows differs from the f64 oracle by {jworst} > derived bound {jbound}"
    );
    println!("icm oracle: j_rows worst {jworst:.3e} bound {jbound:.3e}");

    // r_step2: `x` solves the grid system `(M̃ + ρI)x = t̃`; against the oracle's exact `x*`,
    // `|x − x*|∞ ≤ ‖M⁻¹‖∞·(ε_M·Σ|x*| + ε_t + ρ·Σ|x*|) + u/2`, `ε_M = u(1 + max_b Σ_j|P_bj|)`
    // (the inner `kvec` rounding amplified by `P_b`), `ε_t = u`, `ρ = 1e-12·mean(diag M)`.
    let (r1, r2) = r_step2(&q(&ws), &q(&hl), pref(&p), pref(&p2), o, n).expect("r_step2");
    let (r1, r2) = (unq(&r1), unq(&r2));
    let (q1, q2) = ref_r_step2(&ws, &hl, pref(&p), pref(&p2), o, n);
    let g = n / GROUP;
    let two_g = 2 * g;
    let wh = old_matmul(&ws, &hl, o, n, n);
    let mut xbound = 0.0_f64;
    let mut worst_x = 0.0_f64;
    for row in 0..o {
        let (mmat, _) = row_normal_system(pref(&p), pref(&p2), &hl, &wh, row, n);
        let mean_diag = (0..two_g).map(|k| mmat[k * two_g + k]).sum::<f64>() / two_g as f64;
        // ‖M⁻¹‖∞ by `2G` exact f64 solves (columns of the inverse).
        let mut inv_norm = 0.0_f64;
        for b in 0..two_g {
            let mut e = vec![0.0_f64; two_g];
            e[b] = 1.0;
            let z = ref_solve_spd(&mmat, &e, two_g).expect("spd inverse");
            inv_norm = inv_norm.max(z.iter().fold(0.0_f64, |m, v| m + v.abs()));
        }
        let pabs = block_abs_sum(pref(&p), row, n).max(block_abs_sum(pref(&p2), row, n));
        let xsum: f64 = (0..g)
            .map(|k| q1[row * g + k].abs() + q2[row * g + k].abs())
            .sum();
        let bound =
            inv_norm * (u * (1.0 + pabs) * xsum + u + (1.0e-12 * mean_diag + u) * xsum) + 0.5 * u;
        xbound = xbound.max(bound);
        for k in 0..g {
            worst_x = worst_x.max((r1[row * g + k] - q1[row * g + k]).abs());
            worst_x = worst_x.max((r2[row * g + k] - q2[row * g + k]).abs());
        }
    }
    assert!(
        worst_x <= xbound,
        "r_step2 differs from the f64 oracle by {worst_x} > derived bound {xbound}"
    );
    println!("icm oracle: r_step2 worst {worst_x:.3e} bound {xbound:.3e}");
}

/// The 128-point integer transform is its own inverse: `H₁₂₈/√128` squares to the identity, so two
/// `plane_w` passes with unit scales return the grid input.
///
/// The bound is derived from the two normalisations. One pass is `round(y·C/2³²)` with
/// `C = 379625062` and `|C − 2^28.5| = 0.497` (`src/engine/hadamard.rs`'s `C` table), so its error is
/// `≤ |out|·ε + u/2` with `ε = 0.497/2³²`; the transform's ∞-norm is `√GROUP`, so two passes give
/// `|out − in| ≤ sd·ε·(GROUP + √GROUP) + (u/2)(1 + √GROUP)`, `sd = max |s·d|` the input magnitude and
/// `u = 2⁻³²` the grid quantum. `s` is a codebook plane (±2), so `sd = max |s|` and the unit `r`, `c`,
/// `d` multiplications are exact.
///
/// # Complexity
/// Temporal `O(o·n·log2(GROUP))` per pass (two passes); spatial `O(o·n)`.
#[test]
fn learn_fold_icm_transform_is_self_inverse_on_the_grid() {
    let (o, n) = (3_usize, 512);
    let mut rng = Lcg(0x51CE_5E1F);
    let g = n / GROUP;
    let unit = Plane {
        s: rand_plane(o, n, &mut rng).s,
        idx: vec![0_u16; o * n / BLOCK],
        r: vec![1_i64 << 32; o * g],
        c: vec![1_i64 << 32; n],
        d: vec![1_i64 << 32; n],
    };
    let once = plane_w(pref(&unit), o, n).expect("first transform");
    let twice_p = Plane {
        s: once,
        ..unit.clone()
    };
    let twice = plane_w(pref(&twice_p), o, n).expect("second transform");
    let u = 2f64.powi(-32);
    let eps = 0.497 / 2f64.powi(32);
    let sd = max_abs(&unq(&unit.s)).max(max_abs(&unq(&unit.d)));
    let root = 128f64.sqrt();
    let bound = sd * eps * (128f64 + root) + 0.5 * u * (1.0 + root);
    let src = unq(&unit.s);
    let out = unq(&twice);
    let worst = (0..o * n).fold(0.0_f64, |m, k| m.max((out[k] - src[k]).abs()));
    assert!(
        worst <= bound,
        "two transforms differ from the identity by {worst} > derived bound {bound}"
    );
    println!(
        "icm oracle: transform round trip worst {worst:.3e} ({:.1}u) bound {bound:.3e}",
        worst / u
    );
}

/// The unnormalised 128-point butterfly (the engine's own stages, before its normalisation).
fn fwht_unnormalised(x: &mut [i64]) {
    let mut half = 1;
    while half < x.len() {
        for base in (0..x.len()).step_by(2 * half) {
            for j in base..base + half {
                let (a, b) = (x[j], x[j + half]);
                x[j] = a + b;
                x[j + half] = a - b;
            }
        }
        half *= 2;
    }
}

/// The nearest integer to `z/√128`: the bracket `128(n−½)² ≤ z² ≤ 128(n+½)²` (i.e. `±128n + 32`)
/// around the `2^28.5` estimate. A tie needs `z = 4(2n+1)√2`, no integer.
fn exact_div_sqrt128(z: i64) -> i64 {
    let mag = i128::from(z.unsigned_abs());
    let sq = mag * mag;
    let mut n = (mag as f64 / 128f64.sqrt()).round() as i128;
    loop {
        if sq > 128 * n * n + 128 * n + 32 {
            n += 1;
        } else if n > 0 && sq < 128 * n * n - 128 * n + 32 {
            n -= 1;
        } else {
            break;
        }
    }
    if z < 0 {
        -n as i64
    } else {
        n as i64
    }
}

/// The engine's `fwht128` against the exact integer normalisation `round(z/√128)`.
///
/// The engine's epilogue is `round(z·C/2³²)` with `C = 379625062` and `|C − 2^28.5| = 0.497`, a relative
/// offset `ε = 0.497/379625062`; that shifts the exact value by `|z|·ε` and each side then rounds once,
/// so the two differ by `≤ |z|·ε + 1` words. `z` is the unnormalised butterfly output, `|z| ≤
/// √128·max|s|` by the transform's ∞-norm.
#[test]
fn learn_fold_icm_engine_transform_agrees_with_the_exact_normalisation() {
    let n = 1280usize;
    let mut rng = Lcg(0x1280_5EED);
    let words: Vec<i64> = (0..n)
        .map(|_| (rng.uniform(-2.0, 2.0) * 2f64.powi(32)) as i64)
        .collect();
    let mut got = words.clone();
    apeiron::engine::fwht_fixed(&mut got, None, GROUP).expect("engine transform");
    let eps = 0.497 / 379_625_062.0;
    let max_word = words.iter().fold(0_i64, |m, v| m.max(v.abs())) as f64;
    let bound = 128f64.sqrt() * max_word * eps + 1.0;
    let mut worst = 0.0_f64;
    for (block, out) in got.chunks(GROUP).enumerate() {
        let mut z = words[block * GROUP..(block + 1) * GROUP].to_vec();
        fwht_unnormalised(&mut z);
        for (k, &zi) in z.iter().enumerate() {
            let want = exact_div_sqrt128(zi);
            let err = (out[k] - want).abs() as f64;
            worst = worst.max(err);
            assert!(
                err <= bound,
                "block {block} entry {k}: engine {} vs exact {want} = {err} words > {bound}",
                out[k]
            );
        }
    }
    println!(
        "icm oracle: engine transform vs exact round(z/√128), worst {worst:.1} words, bound {bound:.1}"
    );
}

/// The number of positions at which the `u16` code plane disagrees with a `u8` baseline.
fn idx_diffs(got: &[u16], want: &[u8]) -> usize {
    got.iter()
        .zip(want)
        .filter(|(a, b)| **a != u16::from(**b))
        .count()
}

#[test]
fn learn_fold_icm_matmul_matches_the_f64_products() {
    // The shared Q32.32 product against the test-local copy of the old `f64` body, on the two seeded
    // shapes and on scalings whose entries force the power-of-two shift (`max|·| > 8`).
    for &(o, n, seed) in &[
        (16_usize, 256_usize, 0x51CE_0001_u64),
        (32, 512, 0x51CE_0002),
    ] {
        let (ws, hl, p1, _p2) = seeded_case(o, n, seed);
        let (ws_f, hl_f) = (unq(&ws), unq(&hl));
        for scale in [1.0_f64, 1.0e2, 1.0e3] {
            let a: Vec<f64> = ws_f.iter().map(|v| v * scale).collect();
            let b: Vec<f64> = hl_f.iter().map(|v| v * scale).collect();
            let got = unq(&matmul(&q(&a), &q(&b), o, n, n).expect("matmul"));
            let want = old_matmul(&a, &b, o, n, n);
            let bound = product_tolerance(n, max_abs(&a), max_abs(&b));
            let worst = (0..o * n).fold(0.0_f64, |m, i| m.max((got[i] - want[i]).abs()));
            assert!(
                worst <= bound,
                "o={o} n={n} scale={scale}: worst |matmul − f64| {worst} > bound {bound}"
            );
        }
        // A plane block, the second operand shape the step uses.
        let b = {
            let (s_f, r_f) = (unq(&p1.s), unq(&p1.r));
            let mut x = s_f;
            for row in 0..o {
                for col in 0..n {
                    x[row * n + col] *= r_f[row * (n / GROUP) + col / GROUP];
                }
            }
            x
        };
        let got = unq(&matmul(&q(&b), &hl, o, n, n).expect("matmul"));
        let want = old_matmul(&b, &hl_f, o, n, n);
        let bound = product_tolerance(n, max_abs(&b), max_abs(&hl_f));
        let worst = (0..o * n).fold(0.0_f64, |m, i| m.max((got[i] - want[i]).abs()));
        assert!(
            worst <= bound,
            "o={o} n={n} block product: worst {worst} > bound {bound}"
        );
    }
}

#[test]
fn learn_fold_icm_decision_safety_matches_the_pre_change_planes() {
    // The single-plane step and the two-plane joint step on both seeded instances: the returned codes
    // are the pre-change codes, or, where a tie flipped, the objective is no higher.
    for &(o, n, seed, hstep, h1, h2) in &[
        (
            16_usize,
            256_usize,
            0x51CE_0001_u64,
            STEP_16_256_IDX,
            JOINT_16_256_IDX1,
            JOINT_16_256_IDX2,
        ),
        (
            32,
            512,
            0x51CE_0002,
            STEP_32_512_IDX,
            JOINT_32_512_IDX1,
            JOINT_32_512_IDX2,
        ),
    ] {
        let (ws, hl, p1, p2) = seeded_case(o, n, seed);
        let base = unhex(hstep);
        let out = e8_icm_step(&ws, &hl, pref(&p1), o, n, 3, 0).expect("step");
        assert_eq!(out.idx.len(), base.len(), "plane {o}x{n} size");
        let diffs = idx_diffs(&out.idx, &base);
        let j_after = plane_j(&ws, &out, &hl, o, n);
        let j_before = plane_j(&ws, &plane_from_idx(&base, &p1, o, n), &hl, o, n);
        assert!(
            diffs == 0 || j_after <= j_before + tol(j_before),
            "{o}x{n} step: {diffs} code indices differ from the pre-change plane and J rose \
             {j_before} -> {j_after}"
        );
        let (b1, b2) = (unhex(h1), unhex(h2));
        let (q1, q2, st) = joint(&ws, &hl, pref(&p1), pref(&p2), o, n, 3, 0.0);
        let diffs = idx_diffs(&q1.idx, &b1) + idx_diffs(&q2.idx, &b2);
        let j_after = pair_j(&ws, &q1, &q2, &hl, o, n);
        let j_before = pair_j(
            &ws,
            &plane_from_idx(&b1, &p1, o, n),
            &plane_from_idx(&b2, &p2, o, n),
            &hl,
            o,
            n,
        );
        assert!(
            diffs == 0 || j_after <= j_before + tol(j_before),
            "{o}x{n} joint: {diffs} code indices differ from the pre-change planes and J rose \
             {j_before} -> {j_after}"
        );
        assert!(
            st.j_after <= st.j_before + tol(st.j_before),
            "{o}x{n} joint stats: J {} -> {}",
            st.j_before,
            st.j_after
        );
    }
}

#[test]
fn learn_fold_icm_decision_safety_r_step2_is_no_worse_than_the_pre_change_fit() {
    // `r_step2` returns group scales, not codes; the decision-safety property is the objective of
    // the two fitted planes, `J` with the new `(r1, r2)` against `J` with the pre-change pair.
    for &(o, n, seed, r1b, r2b) in &[
        (
            16_usize,
            256_usize,
            0x51CE_0001_u64,
            R_16_256_R1,
            R_16_256_R2,
        ),
        (32, 512, 0x51CE_0002, R_32_512_R1, R_32_512_R2),
    ] {
        let (ws, hl, p1, p2) = seeded_case(o, n, seed);
        let (r1, r2) = r_step2(&ws, &hl, pref(&p1), pref(&p2), o, n).expect("r_step2");
        let fitted = |r1: Vec<i64>, r2: Vec<i64>| {
            let mut a = p1.clone();
            a.r = r1;
            let mut b = p2.clone();
            b.r = r2;
            pair_j(&ws, &a, &b, &hl, o, n)
        };
        let j_after = fitted(r1.clone(), r2.clone());
        let j_before = fitted(q(r1b), q(r2b));
        assert!(
            j_after <= j_before + 1e-9 * (1.0 + j_before.abs()),
            "{o}x{n} r_step2: fitted J rose {j_before} -> {j_after}"
        );
    }
}

#[test]
fn learn_fold_icm_refuses_unrepresentable_products_and_shapes() {
    // Shape and empty-dimension refusals are the module's shape refusal.
    let (sh, ov) = (Some(IcmError::Shape), Some(IcmError::Overflow));
    assert_eq!(matmul(&[], &[], 0, 1, 1).err(), sh);
    assert_eq!(matmul(&[1i64], &[1i64], 1, 1, 0).err(), sh);
    assert_eq!(matmul(&[1i64], &[1i64; 4], 1, 1, 2).err(), sh);
    assert_eq!(matmul(&[1i64; 2], &[1i64], 2, 2, 1).err(), sh);
    // A representable product of operands far above the `gemm_nn` input bound is accepted and the
    // exact power-of-two scaling is undone: `10⁴ · 2·10⁴ = 2·10⁸` round-trips through the grid.
    let big = matmul(&q(&[1.0e4]), &q(&[2.0e4]), 1, 1, 1).expect("representable");
    assert_eq!(unq(&big), vec![2.0e8]);
    // An unrepresentably large product leaves the `i64` grid after the exact undo: refused, no panic.
    assert_eq!(matmul(&[i64::MAX], &[i64::MAX], 1, 1, 1).err(), ov);
    assert_eq!(matmul(&[i64::MAX; 4], &[i64::MAX], 4, 1, 1).err(), ov);
    // The same refusal through a public step: an overflowing `ws` is an error, not a panic.
    let (o, n) = (4_usize, 256);
    let hl = q(&toeplitz(n, 0.4));
    let p = rand_plane(o, n, &mut Lcg(17));
    let ws = vec![i64::MAX; o * n];
    assert_eq!(e8_icm_step(&ws, &hl, pref(&p), o, n, 1, 0).err(), ov);
    assert_eq!(
        e8_joint_icm_2plane(&ws, &hl, pref(&p), pref(&p), o, n, 1, 0.0, None, 256, None).err(),
        ov
    );
    assert_eq!(r_step2(&ws, &hl, pref(&p), pref(&p), o, n).err(), ov);
}

/// The `e8_icm_step` timing at the fold shape, release only:
/// `cargo test --release --locked --test learn_fold_icm -- --ignored --nocapture`.
///
/// Measures the pre-change single-threaded `f64` product and the new Q32.32 product on the same
/// `Ws @ Hl` operands, then three `e8_icm_step` runs, and prints the mean and min–max. The three-run
/// protocol is the harness's: a single run is a sample, not a result.
#[test]
#[ignore = "release-mode timing at o = 896, n = 4864"]
fn learn_fold_icm_timing_at_the_fold_shape() {
    use std::time::Instant;
    let (o, n) = (896_usize, 4864);
    let mut rng = Lcg(0x7E_51CE);
    let hl = toeplitz(n, 0.37);
    let p = rand_plane(o, n, &mut rng);
    let ws = rand_mat(o, n, &mut rng);
    let (ws_q, hl_q) = (q(&ws), q(&hl));

    let t = Instant::now();
    let old = old_matmul(&ws, &hl, o, n, n);
    let old_secs = t.elapsed().as_secs_f64();

    let mut share = Vec::new();
    for _ in 0..3 {
        let t = Instant::now();
        let new = unq(&matmul(&ws_q, &hl_q, o, n, n).expect("matmul"));
        share.push(t.elapsed().as_secs_f64());
        assert!(new.iter().all(|v| v.is_finite()) && old.iter().all(|v| v.is_finite()));
        let tol = product_tolerance(n, max_abs(&ws), max_abs(&hl));
        let worst = (0..o * n).fold(0.0_f64, |m, i| m.max((new[i] - old[i]).abs()));
        assert!(worst <= tol, "timing-shape product off by {worst} > {tol}");
    }

    let mut runs = Vec::new();
    for _ in 0..3 {
        let t = Instant::now();
        let out = e8_icm_step(&ws_q, &hl_q, pref(&p), o, n, 1, 0).expect("step");
        runs.push(t.elapsed().as_secs_f64());
        assert!(out.idx.iter().all(|&v| u16::from(v as u8) == v));
    }
    let mean = runs.iter().sum::<f64>() / runs.len() as f64;
    let min = runs.iter().copied().fold(f64::INFINITY, f64::min);
    let max = runs.iter().copied().fold(0.0_f64, f64::max);
    let share_mean = share.iter().sum::<f64>() / share.len() as f64;
    let share_min = share.iter().copied().fold(f64::INFINITY, f64::min);
    let share_max = share.iter().copied().fold(0.0_f64, f64::max);
    println!(
        "icm timing o={o} n={n} threads=8: matmul(Ws@Hl) old f64 1t {old_secs:.2}s; \
         new {share_mean:.2}s [{share_min:.2}, {share_max:.2}]; \
         e8_icm_step(sweeps=1) {mean:.2}s [{min:.2}, {max:.2}] over 3 runs; \
         pre-change reference 29.4s/call"
    );
    assert!(
        share_mean < old_secs,
        "the shared Q32.32 matmul ({share_mean:.2}s) did not beat the scalar f64 product \
         ({old_secs:.2}s) on the same operands"
    );
}

#[test]
fn learn_fold_icm_j_is_non_increasing_per_sweep() {
    for &(arm, seed) in &[(Arm::Step, 0x1ACE_5EED_u64), (Arm::Joint, 0x2B1C_0DE5)] {
        for &o in &[4_usize, 8, 16] {
            let (ws, hl, p1, p2) = arm_case(arm, o, 256, seed);
            let j_init = match arm {
                Arm::Step => plane_j(&ws, &p1, &hl, o, 256),
                Arm::Joint => pair_j(&ws, &p1, &p2, &hl, o, 256),
            };
            let mut prev = j_init;
            for sweeps in 1..=3 {
                let (_, _, j, st) = run_arm(arm, &ws, &hl, &p1, &p2, o, 256, sweeps, 0.0, &hl);
                assert!(
                    j <= prev + tol(prev),
                    "{arm:?} o={o} sweeps={sweeps}: J rose {prev} -> {j}"
                );
                if let Some(st) = st {
                    assert!(
                        st.j_after <= st.j_before + tol(st.j_before),
                        "{arm:?} o={o} sweeps={sweeps}: stats J {} -> {}",
                        st.j_before,
                        st.j_after
                    );
                    assert!(
                        (st.j_after - j).abs() <= tol(j),
                        "{arm:?} o={o} sweeps={sweeps}: stats J {} != measured {j}",
                        st.j_after
                    );
                }
                prev = j;
            }
            assert!(
                prev <= j_init + tol(j_init),
                "{arm:?} o={o}: final J {prev} > initial {j_init}"
            );
            assert!(prev < j_init, "{arm:?} o={o}: no move at all ({j_init})");
        }
    }
}

#[test]
fn learn_fold_icm_g_row_weights_the_reported_j() {
    let (o, n) = (5_usize, 256);
    let hl = toeplitz(n, 0.4);
    let mut rng = Lcg(99);
    let p1 = rand_plane(o, n, &mut rng);
    let p2 = rand_plane(o, n, &mut rng);
    let ws = rand_mat(o, n, &mut rng);
    let g_row: Vec<f64> = (0..o).map(|_| rng.uniform(0.25, 2.0)).collect();
    let unweighted = pair_j(&q(&ws), &p1, &p2, &q(&hl), o, n);
    let (_q1, _q2, st) = e8_joint_icm_2plane(
        &q(&ws),
        &q(&hl),
        pref(&p1),
        pref(&p2),
        o,
        n,
        0,
        0.0,
        Some(&q(&g_row)),
        256,
        None,
    )
    .expect("joint");
    // With sweeps = 0 nothing moves, so j_before = j_after = the weighted sum of the row norms.
    assert_eq!(st.j_after, st.j_before);
    assert_eq!(st.moves, 0);
    let expect = ref_j_rows(&ws, &unq(&pair_w(&p1, &p2, o, n)), &hl, o, n, Some(&g_row));
    assert!((st.j_before - expect).abs() <= tol(expect));
    assert!(st.j_before > 0.0 && unweighted > 0.0);
}

/// `p` with block `m` of row 0 set to codebook row `j`.
fn set_code(p: &Plane, m: usize, j: usize) -> Plane {
    let mut out = p.clone();
    out.s[m * BLOCK..(m + 1) * BLOCK].copy_from_slice(&codebook()[j]);
    out.idx[m] = j as u16;
    out
}

/// `p` with block `m` of row 0 replaced by code `(idx[m] + delta) % CODE_LEN`.
fn perturb(p: &Plane, m: usize, delta: usize) -> Plane {
    set_code(p, m, (p.idx[m] as usize + delta) % CODE_LEN)
}

#[test]
fn learn_fold_icm_recovers_a_representable_target() {
    // One plane, and the sum of two: a target the arm can represent is recovered exactly (every block
    // code equal and `J` at zero) after one perturbed block. Sweep 0 visits the blocks in reverse
    // order, so the last block is repaired first and every later block sees the exact state.
    for arm in [Arm::Step, Arm::Joint] {
        let (o, n) = (6_usize, 256);
        let hl = q(&toeplitz(n, 0.4));
        let mut rng = Lcg(if arm == Arm::Step { 7 } else { 13 });
        let truth1 = rand_plane(o, n, &mut rng);
        let truth2 = rand_plane(o, n, &mut rng);
        let ws = match arm {
            Arm::Step => plane_w(pref(&truth1), o, n).expect("plane_w"),
            Arm::Joint => pair_w(&truth1, &truth2, o, n),
        };
        let init = perturb(&truth1, n / BLOCK - 1, if arm == Arm::Step { 1 } else { 5 });
        let j_init = match arm {
            Arm::Step => plane_j(&ws, &init, &hl, o, n),
            Arm::Joint => pair_j(&ws, &init, &truth2, &hl, o, n),
        };
        assert!(
            j_init > 1e-6,
            "{arm:?}: the perturbation must be visible, got {j_init}"
        );
        let (out1, out2, j_out, st) = run_arm(arm, &ws, &hl, &init, &truth2, o, n, 2, 0.0, &hl);
        assert!(
            j_out < 1e-6 * (1.0 + j_init),
            "{arm:?}: J {j_init} -> {j_out}, expected ~0"
        );
        if let Some(st) = st {
            assert!(
                st.j_after < 1e-6 * (1.0 + st.j_before),
                "{arm:?}: stats J after {j_out}"
            );
        }
        for (got, want, plane) in [(&out1, &truth1, 1_u8), (&out2, &truth2, 2)] {
            for (k, (&g, &w)) in got.idx.iter().zip(&want.idx).enumerate() {
                assert_eq!(g, w, "{arm:?}: block {k} plane {plane} not recovered");
            }
        }
    }
}

#[test]
fn learn_fold_icm_block_argmin_is_exact() {
    // The single plane at 4 x 256: brute-force row 0's last block (the first block sweep 0 visits)
    // over all 256 codewords, every other block held at `init`.
    {
        let (o, n) = (4_usize, 256);
        let hl = toeplitz(n, 0.35);
        let mut rng = Lcg(11);
        let init = rand_plane(o, n, &mut rng);
        let ws = rand_mat(o, n, &mut rng);
        let m = n / BLOCK - 1;
        let out = e8_icm_step(&q(&ws), &q(&hl), pref(&init), o, n, 1, 0).expect("step");
        let (wsq, hlq) = (q(&ws), q(&hl));
        let mut best = usize::MAX;
        let mut best_j = f64::INFINITY;
        for j in 0..CODE_LEN {
            let jj = plane_j(&wsq, &set_code(&init, m, j), &hlq, o, n);
            if jj < best_j {
                best_j = jj;
                best = j;
            }
        }
        assert_eq!(
            out.idx[m] as usize, best,
            "row 0 block {m}: chose {} not the brute-force argmin {best}",
            out.idx[m]
        );
        let j_out = plane_j(&wsq, &out, &hlq, o, n);
        assert!(
            j_out <= best_j + tol(best_j),
            "J {j_out} above the brute-force block minimum {best_j}"
        );
    }
    // The plane pair at 1 x 128 with only block `m` free: 256 x 256 candidate pairs. One row keeps
    // the pair brute force affordable.
    {
        let (o, n) = (1_usize, 128);
        let hl = toeplitz(n, 0.3);
        let mut rng = Lcg(23);
        let p1 = rand_plane(o, n, &mut rng);
        let p2 = rand_plane(o, n, &mut rng);
        let ws = rand_mat(o, n, &mut rng);
        let m = 5_usize;
        let (out1, out2, st) = e8_joint_icm_2plane(
            &q(&ws),
            &q(&hl),
            pref(&p1),
            pref(&p2),
            o,
            n,
            1,
            0.0,
            None,
            256,
            Some(&[m]),
        )
        .expect("joint");
        let (wsq, hlq) = (q(&ws), q(&hl));
        let mut best = (0_usize, 0_usize);
        let mut best_j = f64::INFINITY;
        for i1 in 0..CODE_LEN {
            let w1 = plane_w(pref(&set_code(&p1, m, i1)), o, n).expect("w1");
            for i2 in 0..CODE_LEN {
                let w2 = plane_w(pref(&set_code(&p2, m, i2)), o, n).expect("w2");
                let mut w = w1.clone();
                for k in 0..o * n {
                    w[k] += w2[k];
                }
                let jj = q32::from_grid(j_rows(&wsq, &w, &hlq, o, n, None).expect("j"));
                if jj < best_j {
                    best_j = jj;
                    best = (i1, i2);
                }
            }
        }
        assert_eq!(
            (out1.idx[m] as usize, out2.idx[m] as usize),
            best,
            "block {m}: chose {:?} not the brute-force pair argmin {best:?}",
            (out1.idx[m], out2.idx[m])
        );
        assert!(
            (st.j_after - best_j).abs() <= tol(best_j),
            "joint J {} != brute-force block minimum {best_j}",
            st.j_after
        );
    }
}

#[test]
fn learn_fold_icm_r_step2_satisfies_normal_equations() {
    let (o, n) = (5_usize, 256);
    let hl = toeplitz(n, 0.45);
    let mut rng = Lcg(31);
    let p1 = rand_plane(o, n, &mut rng);
    let p2 = rand_plane(o, n, &mut rng);
    let ws = rand_mat(o, n, &mut rng);
    let (r1, r2) = r_step2(&q(&ws), &q(&hl), pref(&p1), pref(&p2), o, n).expect("r_step2");
    let g = n / GROUP;
    assert_eq!(r1.len(), o * g);
    assert_eq!(r2.len(), o * g);
    let wh = unq(&matmul(&q(&ws), &q(&hl), o, n, n).expect("wh"));
    // The `wh` above is the product the body under test uses (the shared Q32.32 `matmul`).
    //
    // Derivation of the bound. The body solves `(M + ρI)x = t` on the `Q32.32` grid, `U = 2⁻³²`.
    // (i) each `M[a][b]` is assembled as `kvec[b][j] = round(Σ_t P_bt H_tj)` then
    // `round(Σ_j kvec[b][j] P_aj)`; the inner rounding is `≤ U/2` and is amplified by
    // `Σ_j|P_bj|`, the outer by one more `U/2`, so `|M̃ − M| ≤ U(1 + max_b Σ_j|P_bj|)`;
    // (ii) `|t̃ − t| ≤ U/2`; (iii) the ridge is `ρ = 1e-12·mean(diag M̃)`, itself floored at one raw
    // unit, and enters the solved system but not this test's `M`; (iv) `x` is a grid word, so it
    // differs from the exact solution of `M̃x = t̃` by at most `U/2` per coordinate. Summing,
    //   |Mx − t|_a ≤ U·(Σ_b|M_ab| + (1 + max_b Σ_j|P_bj|)·Σ_b|x_b| + 1) + ρ·Σ_b|x_b|,
    // which is the grid quantum times the row condition growth, and is what the assert below uses.
    let u = 2f64.powi(-32);
    let two_g = 2 * g;
    let mut worst = 0.0_f64;
    let mut mrow_max = 0.0_f64;
    let mut pabs_max = 0.0_f64;
    let mut xsum_max = 0.0_f64;
    let mut ridge_max = 0.0_f64;
    for row in 0..o {
        let mut x = vec![0.0_f64; two_g];
        for k in 0..g {
            x[k] = q32::from_grid(r1[row * g + k]);
            x[g + k] = q32::from_grid(r2[row * g + k]);
        }
        xsum_max = xsum_max.max(x.iter().fold(0.0_f64, |m, v| m + v.abs()));
        pabs_max = pabs_max
            .max(block_abs_sum(pref(&p1), row, n))
            .max(block_abs_sum(pref(&p2), row, n));
        let (mmat, tvec) = row_normal_system(pref(&p1), pref(&p2), &hl, &wh, row, n);
        let mrow = (0..two_g).fold(0.0_f64, |m, a_| {
            m.max((0..two_g).map(|b| mmat[a_ * two_g + b].abs()).sum::<f64>())
        });
        mrow_max = mrow_max.max(mrow);
        let mean_diag = (0..two_g).map(|k| mmat[k * two_g + k]).sum::<f64>() / two_g as f64;
        ridge_max = ridge_max.max(mean_diag);
        for a_ in 0..two_g {
            let mx: f64 = (0..two_g).map(|b| mmat[a_ * two_g + b] * x[b]).sum();
            worst = worst.max((mx - tvec[a_]).abs());
        }
    }
    let bound =
        u * (mrow_max + (1.0 + pabs_max) * xsum_max + 1.0) + (1.0e-12 * ridge_max + u) * xsum_max;
    assert!(
        worst <= bound,
        "r_step2 normal-equation residual {worst} > derived grid bound {bound}"
    );
}

#[test]
fn learn_fold_icm_icm_ridge_metric_is_non_increasing() {
    let (o, n) = (4_usize, 256);
    let hl = q(&toeplitz(n, 0.4));
    let hr = ridged_metric(&hl, n, 0.2).expect("ridged");
    assert!(
        (0..n).all(|i| hr[i * n + i] > hl[i * n + i]),
        "the ridge must raise the diagonal"
    );
    let mut rng = Lcg(41);
    let p0 = rand_plane(o, n, &mut rng);
    let ws = q(&rand_mat(o, n, &mut rng));
    let mut prev = plane_j(&ws, &p0, &hr, o, n);
    for sweeps in 1..=2 {
        let (_, _, j, _) = run_arm(Arm::Step, &ws, &hl, &p0, &p0, o, n, sweeps, 0.2, &hr);
        assert!(
            j <= prev + tol(prev),
            "ridged sweeps={sweeps}: J rose {prev} -> {j}"
        );
        prev = j;
    }
    let (_, _, j, st) = run_arm(Arm::Joint, &ws, &hl, &p0, &p0, o, n, 1, 0.2, &hr);
    let st = st.expect("joint stats");
    assert!(j <= st.j_before + tol(st.j_before));
    assert!((st.j_after - j).abs() <= tol(j));
}

#[test]
fn learn_fold_icm_row_chunk_does_not_change_the_result() {
    let (o, n) = (8_usize, 256);
    let hl = q(&toeplitz(n, 0.4));
    let mut rng = Lcg(53);
    let p1 = rand_plane(o, n, &mut rng);
    let p2 = rand_plane(o, n, &mut rng);
    let ws = q(&rand_mat(o, n, &mut rng));
    let (a1, a2, ast) =
        e8_joint_icm_2plane(&ws, &hl, pref(&p1), pref(&p2), o, n, 2, 0.0, None, 1, None)
            .expect("chunk 1");
    let (b1, b2, bst) = joint(&ws, &hl, pref(&p1), pref(&p2), o, n, 2, 0.0);
    assert_eq!(a1.s, b1.s);
    assert_eq!(a1.idx, b1.idx);
    assert_eq!(a2.s, b2.s);
    assert_eq!(a2.idx, b2.idx);
    assert_eq!(ast, bst);
}

#[test]
fn learn_fold_icm_nearest_index_matches_the_codebook_rule() {
    let cb = codebook();
    for j in 0..CODE_LEN {
        assert_eq!(nearest_index(&cb[j]), Ok(j), "row {j}");
    }
}

#[test]
fn learn_fold_icm_refuses_ragged_shapes() {
    let (o, n) = (2_usize, 256);
    let hl = q(&toeplitz(n, 0.4));
    let p = rand_plane(o, n, &mut Lcg(3));
    let ws = q(&rand_mat(o, n, &mut Lcg(5)));
    let good = plane_w(pref(&p), o, n).expect("plane_w");
    assert_eq!(good.len(), o * n);
    let sh = Some(IcmError::Shape);
    let step_err =
        |pp: PlaneRef<'_>, ws: &[i64], hl: &[i64]| e8_icm_step(ws, hl, pp, o, n, 1, 0).err();
    let joint_err = |pp: PlaneRef<'_>| {
        e8_joint_icm_2plane(&ws, &hl, pp, pref(&p), o, n, 1, 0.0, None, 256, None).err()
    };
    let r_err = |pp: PlaneRef<'_>, hl: &[i64]| r_step2(&ws, hl, pp, pref(&p), o, n).err();
    // A short `ws`, `hl`, or any plane slice is a shape refusal.
    assert_eq!(step_err(pref(&p), &ws[1..], &hl), sh);
    assert_eq!(step_err(pref(&p), &ws, &hl[1..]), sh);
    let s1 = PlaneRef {
        s: &p.s[1..],
        ..pref(&p)
    };
    assert_eq!(step_err(s1, &ws, &hl), sh);
    let i1 = PlaneRef {
        idx: &p.idx[1..],
        ..pref(&p)
    };
    assert_eq!(joint_err(i1), sh);
    let r1 = PlaneRef {
        r: &p.r[1..],
        ..pref(&p)
    };
    assert_eq!(step_err(r1, &ws, &hl), sh);
    let c1 = PlaneRef {
        c: &p.c[1..],
        ..pref(&p)
    };
    assert_eq!(r_err(c1, &hl), sh);
    let d1 = PlaneRef {
        d: &p.d[1..],
        ..pref(&p)
    };
    assert_eq!(step_err(d1, &ws, &hl), sh);
    assert_eq!(r_err(pref(&p), &hl[1..]), sh);
    assert_eq!(j_rows(&ws, &good, &hl, o, n, Some(&[1i64])).err(), sh);
    // `n` must be a positive multiple of BLOCK and GROUP.
    let bad_n = 64_usize;
    let q = Plane {
        s: vec![0_i64; o * bad_n],
        idx: vec![0; o * (bad_n / BLOCK)],
        r: Vec::new(),
        c: vec![1_i64 << 32; bad_n],
        d: vec![1_i64 << 32; bad_n],
    };
    let (bad_ws, bad_hl) = (vec![0_i64; o * bad_n], vec![0_i64; bad_n * bad_n]);
    let wd = Some(IcmError::Width);
    assert_eq!(
        e8_icm_step(&bad_ws, &bad_hl, pref(&q), o, bad_n, 1, 0).err(),
        wd
    );
    assert_eq!(plane_w(pref(&q), o, bad_n).err(), wd);
    // Zero chunk and out-of-range block subset.
    assert_eq!(
        e8_joint_icm_2plane(&ws, &hl, pref(&p), pref(&p), o, n, 1, 0.0, None, 0, None).err(),
        Some(IcmError::ZeroChunk)
    );
    assert_eq!(
        e8_joint_icm_2plane(
            &ws,
            &hl,
            pref(&p),
            pref(&p),
            o,
            n,
            1,
            0.0,
            None,
            256,
            Some(&[n / BLOCK])
        )
        .err(),
        Some(IcmError::BlockOutOfRange)
    );
}
