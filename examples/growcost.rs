// The build path with growth: no reserve, so every rebuild pays the tag fill.
// ARM=ap|fold BITS=n
use apeiron::map::Words;
use foldhash::fast::RandomState;
use std::hint::black_box;
use std::time::Instant;
fn main(){
    let bits: u32 = std::env::var("BITS").ok().and_then(|v|v.parse().ok()).unwrap_or(20);
    let n = 1usize << bits;
    let mut s = 0x243F_6A88_85A3_08D3u64;
    let ks: Vec<u64> = (0..n).map(|_|{s^=s<<13;s^=s>>7;s^=s<<17;s}).collect();
    let arm = std::env::var("ARM").unwrap_or_else(|_|"ap".into());
    let reps: usize = std::env::var("REPS").ok().and_then(|v|v.parse().ok()).unwrap_or(20);
    let mut acc=0usize; let t=Instant::now();
    for _ in 0..reps {
        if arm=="fold" {
            let mut m: hashbrown::HashMap<u64,u64,RandomState>=hashbrown::HashMap::default();
            for &k in &ks { m.insert(k,k); } acc+=m.len();
        } else {
            let mut m: Words<u64,u64>=Words::new();
            for &k in &ks { m.insert(k,k); } acc+=m.len();
        }
    }
    let e=t.elapsed();
    println!("{arm} {bits}: {:.2} ms/build  ({})", e.as_secs_f64()*1000.0/reps as f64, black_box(acc));
}
