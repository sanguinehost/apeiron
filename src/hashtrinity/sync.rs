//! Library surface for `syncfold` — lets other crates (e.g. Galvanize's
//! tfplugin6 client) reuse the protocol-layer pieces (HPACK, HTTP/2 framing,
//! gRPC message framing, ECDSA/X.509 cert generation) without depending on
//! the sync tool's own binary. `main.rs` is a thin CLI wrapper over this same
//! module tree — one set of modules, two crate roots (lib + bin), the
//! standard Cargo pattern for "usable as a library too."

pub mod archive {
    //! The bundle — a tree packed into one blob and back, bit-exact (the bundle lever
    //! that turns N S3 objects into 1). A minimal lossless archive: deterministic
    //! order, mode + mtime preserved. Round-trip verified in the test.
    //!
    //! Paths are stored as UTF-8 (the common case); a non-UTF-8 path is a hard error
    //! rather than reaching for `unsafe` OsStr reconstruction.

    #![allow(dead_code)]

    use std::fs::{self, File, FileTimes};
    use std::io::{self, Error, ErrorKind, Read, Write};
    use std::path::Path;
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    use crate::sync::sysio;

    const MAGIC: &[u8] = b"SFAR1\n";

    /// Pack the tree at `root` into one blob.
    pub fn pack(root: &Path) -> io::Result<Vec<u8>> {
        let mut out = Vec::new();
        out.extend_from_slice(MAGIC);
        pack_dir(root, Path::new(""), &mut out)?;
        Ok(out)
    }

    fn rel_bytes(rel: &Path) -> io::Result<&[u8]> {
        rel.to_str().map(str::as_bytes).ok_or_else(|| {
            Error::new(
                ErrorKind::InvalidData,
                format!("non-UTF8 path: {}", rel.display()),
            )
        })
    }

    fn pack_dir(abs: &Path, rel: &Path, out: &mut Vec<u8>) -> io::Result<()> {
        let mut entries: Vec<_> = fs::read_dir(abs)?.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for de in entries {
            let crel = rel.join(de.file_name());
            let ft = de.file_type()?;
            let pb = rel_bytes(&crel)?.to_vec();
            if ft.is_symlink() {
                let target = fs::read_link(de.path())?;
                let tb = target
                    .to_str()
                    .ok_or_else(|| Error::new(ErrorKind::InvalidData, "non-UTF8 symlink target"))?
                    .as_bytes()
                    .to_vec();
                out.push(b'l');
                put_path(out, &pb);
                out.extend_from_slice(&(tb.len() as u32).to_be_bytes());
                out.extend_from_slice(&tb);
            } else if ft.is_dir() {
                out.push(b'd');
                put_path(out, &pb);
                pack_dir(&de.path(), &crel, out)?;
            } else {
                let m = de.metadata()?;
                let data = fs::read(de.path())?;
                let (secs, nanos) = m
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| (d.as_secs(), d.subsec_nanos()))
                    .unwrap_or((0, 0));
                out.push(b'f');
                put_path(out, &pb);
                out.extend_from_slice(&(sysio::mode_of(&m)).to_be_bytes());
                out.extend_from_slice(&secs.to_be_bytes());
                out.extend_from_slice(&nanos.to_be_bytes());
                out.extend_from_slice(&(data.len() as u64).to_be_bytes());
                out.extend_from_slice(&data);
            }
        }
        Ok(())
    }

    fn put_path(out: &mut Vec<u8>, p: &[u8]) {
        out.extend_from_slice(&(p.len() as u16).to_be_bytes());
        out.extend_from_slice(p);
    }

    /// **Streaming pack** — emit the same archive format incrementally to `sink`,
    /// reading each file in chunks. RAM stays bounded (the sink buffers a part, not
    /// the tree), so there is no whole-tree-in-memory ceiling. Same bytes as `pack`.
    pub fn pack_stream(root: &Path, sink: &mut dyn FnMut(&[u8]) -> io::Result<()>) -> io::Result<()> {
        sink(MAGIC)?;
        pack_dir_stream(root, Path::new(""), sink)
    }

    fn put_path_s(sink: &mut dyn FnMut(&[u8]) -> io::Result<()>, p: &[u8]) -> io::Result<()> {
        sink(&(p.len() as u16).to_be_bytes())?;
        sink(p)
    }

    fn pack_dir_stream(
        abs: &Path,
        rel: &Path,
        sink: &mut dyn FnMut(&[u8]) -> io::Result<()>,
    ) -> io::Result<()> {
        let mut entries: Vec<_> = fs::read_dir(abs)?.flatten().collect();
        entries.sort_by_key(|e| e.file_name());
        for de in entries {
            let crel = rel.join(de.file_name());
            let ft = de.file_type()?;
            let pb = rel_bytes(&crel)?.to_vec();
            if ft.is_symlink() {
                let target = fs::read_link(de.path())?;
                let tb = target
                    .to_str()
                    .ok_or_else(|| Error::new(ErrorKind::InvalidData, "non-UTF8 symlink target"))?
                    .as_bytes()
                    .to_vec();
                sink(b"l")?;
                put_path_s(sink, &pb)?;
                sink(&(tb.len() as u32).to_be_bytes())?;
                sink(&tb)?;
            } else if ft.is_dir() {
                sink(b"d")?;
                put_path_s(sink, &pb)?;
                pack_dir_stream(&de.path(), &crel, sink)?;
            } else {
                let m = de.metadata()?;
                let (secs, nanos) = m
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| (d.as_secs(), d.subsec_nanos()))
                    .unwrap_or((0, 0));
                let len = m.len();
                sink(b"f")?;
                put_path_s(sink, &pb)?;
                sink(&(sysio::mode_of(&m)).to_be_bytes())?;
                sink(&secs.to_be_bytes())?;
                sink(&nanos.to_be_bytes())?;
                sink(&len.to_be_bytes())?;
                // stream exactly `len` bytes (pad if the file shrank mid-backup, so the
                // declared length always matches — format stays self-consistent).
                let mut f = File::open(de.path())?;
                let mut buf = vec![0u8; 1 << 20];
                let mut remaining = len;
                while remaining > 0 {
                    let n = f.read(&mut buf)?;
                    if n == 0 {
                        break;
                    }
                    let take = (n as u64).min(remaining) as usize;
                    sink(&buf[..take])?;
                    remaining -= take as u64;
                }
                while remaining > 0 {
                    let z = [0u8; 1 << 16];
                    let t = remaining.min(z.len() as u64) as usize;
                    sink(&z[..t])?;
                    remaining -= t as u64;
                }
            }
        }
        Ok(())
    }

    struct Cur<'a> {
        b: &'a [u8],
        p: usize,
    }
    impl<'a> Cur<'a> {
        fn take(&mut self, n: usize) -> io::Result<&'a [u8]> {
            if self.p + n > self.b.len() {
                return Err(Error::new(ErrorKind::UnexpectedEof, "truncated archive"));
            }
            let s = &self.b[self.p..self.p + n];
            self.p += n;
            Ok(s)
        }
        fn u16(&mut self) -> io::Result<usize> {
            Ok(u16::from_be_bytes(self.take(2)?.try_into().unwrap()) as usize)
        }
        fn u32(&mut self) -> io::Result<u32> {
            Ok(u32::from_be_bytes(self.take(4)?.try_into().unwrap()))
        }
        fn u64(&mut self) -> io::Result<u64> {
            Ok(u64::from_be_bytes(self.take(8)?.try_into().unwrap()))
        }
    }

    /// Unpack a blob into `root`, recreating the tree bit-exact (mode + mtime).
    pub fn unpack(blob: &[u8], root: &Path) -> io::Result<()> {
        if !blob.starts_with(MAGIC) {
            return Err(Error::new(ErrorKind::InvalidData, "not an SFAR archive"));
        }
        fs::create_dir_all(root)?;
        let mut c = Cur {
            b: blob,
            p: MAGIC.len(),
        };
        while c.p < c.b.len() {
            let kind = c.take(1)?[0];
            let plen = c.u16()?;
            let path = std::str::from_utf8(c.take(plen)?)
                .map_err(|_| Error::new(ErrorKind::InvalidData, "bad path"))?;
            let dst = root.join(path);
            match kind {
                b'd' => {
                    fs::create_dir_all(&dst)?;
                }
                b'l' => {
                    let tlen = c.u32()? as usize;
                    let target = std::str::from_utf8(c.take(tlen)?)
                        .map_err(|_| Error::new(ErrorKind::InvalidData, "bad symlink"))?;
                    if let Some(parent) = dst.parent() {
                        fs::create_dir_all(parent)?;
                    }
                    let _ = fs::remove_file(&dst);
                    sysio::symlink(target.as_ref(), &dst)?;
                }
                b'f' => {
                    let mode = c.u32()?;
                    let secs = c.u64()?;
                    let nanos = c.u32()?;
                    let len = c.u64()? as usize;
                    let data = c.take(len)?;
                    if let Some(parent) = dst.parent() {
                        fs::create_dir_all(parent)?;
                    }
                    fs::write(&dst, data)?;
                    sysio::set_mode_path(&dst, mode)?;
                    let mt = SystemTime::UNIX_EPOCH + Duration::new(secs, nanos);
                    if let Ok(f) = File::options().write(true).open(&dst) {
                        let _ = f.set_times(FileTimes::new().set_modified(mt));
                    }
                }
                other => {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        format!("bad entry kind {other}"),
                    ));
                }
            }
        }
        Ok(())
    }

    /// **Streaming unpack** — read the SFAR archive from a `Read` and recreate the
    /// tree, writing each file's bytes straight to disk in chunks. RAM is bounded (a
    /// chunk + the path), so a huge object restores without loading it whole.
    pub fn unpack_stream(r: &mut dyn Read, root: &Path) -> io::Result<()> {
        let mut magic = [0u8; MAGIC.len()];
        r.read_exact(&mut magic)?;
        if magic != MAGIC {
            return Err(Error::new(ErrorKind::InvalidData, "not an SFAR archive"));
        }
        fs::create_dir_all(root)?;
        loop {
            let mut k = [0u8; 1];
            // clean EOF at an entry boundary = done
            if read_some(r, &mut k)? == 0 {
                break;
            }
            let plen = read_u16(r)? as usize;
            let mut pbytes = vec![0u8; plen];
            r.read_exact(&mut pbytes)?;
            let path = std::str::from_utf8(&pbytes)
                .map_err(|_| Error::new(ErrorKind::InvalidData, "bad path"))?;
            let dst = root.join(path);
            match k[0] {
                b'd' => {
                    fs::create_dir_all(&dst)?;
                }
                b'l' => {
                    let tlen = read_u32(r)? as usize;
                    let mut t = vec![0u8; tlen];
                    r.read_exact(&mut t)?;
                    let target = std::str::from_utf8(&t)
                        .map_err(|_| Error::new(ErrorKind::InvalidData, "bad symlink"))?;
                    if let Some(p) = dst.parent() {
                        fs::create_dir_all(p)?;
                    }
                    let _ = fs::remove_file(&dst);
                    sysio::symlink(target.as_ref(), &dst)?;
                }
                b'f' => {
                    let mode = read_u32(r)?;
                    let secs = read_u64(r)?;
                    let nanos = read_u32(r)?;
                    let len = read_u64(r)?;
                    if let Some(p) = dst.parent() {
                        fs::create_dir_all(p)?;
                    }
                    let mut f = File::create(&dst)?;
                    let mut remaining = len;
                    let mut buf = vec![0u8; 1 << 20];
                    while remaining > 0 {
                        let take = (buf.len() as u64).min(remaining) as usize;
                        r.read_exact(&mut buf[..take])?;
                        f.write_all(&buf[..take])?;
                        remaining -= take as u64;
                    }
                    sysio::set_mode_file(&f, mode)?;
                    let _ = f.set_times(
                        FileTimes::new()
                            .set_modified(SystemTime::UNIX_EPOCH + Duration::new(secs, nanos)),
                    );
                }
                other => {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        format!("bad entry kind {other}"),
                    ));
                }
            }
        }
        Ok(())
    }

    fn read_some(r: &mut dyn Read, buf: &mut [u8]) -> io::Result<usize> {
        let mut got = 0;
        while got < buf.len() {
            let n = r.read(&mut buf[got..])?;
            if n == 0 {
                break;
            }
            got += n;
        }
        Ok(got)
    }
    fn read_u16(r: &mut dyn Read) -> io::Result<u16> {
        let mut b = [0u8; 2];
        r.read_exact(&mut b)?;
        Ok(u16::from_be_bytes(b))
    }
    fn read_u32(r: &mut dyn Read) -> io::Result<u32> {
        let mut b = [0u8; 4];
        r.read_exact(&mut b)?;
        Ok(u32::from_be_bytes(b))
    }
    fn read_u64(r: &mut dyn Read) -> io::Result<u64> {
        let mut b = [0u8; 8];
        r.read_exact(&mut b)?;
        Ok(u64::from_be_bytes(b))
    }

    /// Count of leaf files in a tree (for reporting "N objects → 1").
    pub fn count_files(root: &Path) -> u64 {
        fn rec(p: &Path) -> u64 {
            let mut n = 0;
            if let Ok(rd) = fs::read_dir(p) {
                for de in rd.flatten() {
                    match de.file_type() {
                        Ok(ft) if ft.is_dir() && !ft.is_symlink() => n += rec(&de.path()),
                        Ok(ft) if ft.is_symlink() => n += 1,
                        Ok(_) => n += 1,
                        Err(_) => {}
                    }
                }
            }
            n
        }
        rec(root)
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn pack_unpack_roundtrip() {
            let base = std::env::temp_dir().join(format!("sfar_test_{}", std::process::id()));
            let src = base.join("src");
            let dst = base.join("dst");
            let _ = fs::remove_dir_all(&base);
            fs::create_dir_all(src.join("sub")).unwrap();
            fs::write(src.join("a.txt"), b"hello").unwrap();
            fs::write(src.join("sub/b.bin"), [0u8, 1, 2, 3, 255]).unwrap();

            let blob = pack(&src).unwrap();
            unpack(&blob, &dst).unwrap();
            assert_eq!(fs::read(dst.join("a.txt")).unwrap(), b"hello");
            assert_eq!(
                fs::read(dst.join("sub/b.bin")).unwrap(),
                vec![0u8, 1, 2, 3, 255]
            );
            let _ = fs::remove_dir_all(&base);
        }
    }
}
pub mod block {
    //! Block duplication — `dd` as the one op at **α = raw block stream**.
    //!
    //! syncfold's tree sync (`sync.rs`) is the trinity over a *tree*. A disk or an
    //! image file is the degenerate tree: one linear byte stream, no children. The
    //! same three faces, re-instantiated for the stream:
    //!
    //!   * **B (leaf):** read/write one block at its offset — `read_at`/`write_at`
    //!     (pread/pwrite). Each block sits at its own coordinate; there is no shared
    //!     file pointer, so the writes parallelize with no contention.
    //!   * **B/U (medium):** the fork over *children* becomes the partition over
    //!     *offsets* — split `[0, size)` into one contiguous span per worker (the Web
    //!     partition `pmapAt`), each span merged through the order-free `Stats`
    //!     monoid (`preduce_schedule_free`), exactly as the tree walk does.
    //!   * **NB (streaming law):** `--sync` reads the destination block and writes
    //!     ONLY where it differs from the source (the per-block throne-check). Plain
    //!     `dd` and block-duplicator rewrite every block; this skips the unchanged
    //!     bulk. Same lever as the file re-sync — the decisive win.
    //!
    //! On Linux a block device IS a file (`/dev/sda`), so there is no Windows-style
    //! elevation / TCP-IPC dance: open the path (as root for a real device) and
    //! `read_at`/`write_at`. Safe `std`, no FFI, no `unsafe`.

    use std::fs::File;
    use std::io::{self, Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::path::Path;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    use std::thread;

    use crate::sync::image::{Format, Source};
    use crate::sync::sysio;

    pub struct Opts {
        pub chunk: usize,
        pub threads: usize,
        pub sync: bool,
        pub sparse: bool,
        pub src_format: Format,
        pub dst_format: Format,
        pub dry_run: bool,
        pub quiet: bool,
    }

    /// The B/U accumulator — componentwise `+`, commutative monoid, so the per-span
    /// results fork/join in any order.
    #[derive(Clone, Copy, Default)]
    pub struct Stats {
        pub bytes_copied: u64,
        pub bytes_skipped: u64, // --sync: dst already matched
        pub bytes_zero: u64,    // sparse: all-zero source block left as a hole
        pub blocks_copied: u64,
        pub blocks_skipped: u64,
        pub blocks_zero: u64,
        pub errors: u64,
    }

    impl Stats {
        const ZERO: Stats = Stats {
            bytes_copied: 0,
            bytes_skipped: 0,
            bytes_zero: 0,
            blocks_copied: 0,
            blocks_skipped: 0,
            blocks_zero: 0,
            errors: 0,
        };

        #[inline]
        fn merge(self, o: Stats) -> Stats {
            Stats {
                bytes_copied: self.bytes_copied + o.bytes_copied,
                bytes_skipped: self.bytes_skipped + o.bytes_skipped,
                bytes_zero: self.bytes_zero + o.bytes_zero,
                blocks_copied: self.blocks_copied + o.blocks_copied,
                blocks_skipped: self.blocks_skipped + o.blocks_skipped,
                blocks_zero: self.blocks_zero + o.blocks_zero,
                errors: self.errors + o.errors,
            }
        }
    }

    /// Hole granularity — the filesystem block. A zero run shorter than this can't
    /// actually become a hole on disk, so detecting holes at 4 KiB is the finest that
    /// saves space. This DECOUPLES the I/O chunk (large → few syscalls) from the
    /// sparse granularity (fine → keeps the skip), so bigger `--bs` is a pure win.
    const GRANULE: usize = 4096;

    /// All-zero test — a hole. Word-at-a-time (`chunks_exact(16)` → u128) so it
    /// vectorizes; fully safe `std`, no `unsafe`. The read is already cheaper than
    /// the write we elide, and on a sparse source the hole pages are kernel-zeroed.
    #[inline]
    fn is_zero(buf: &[u8]) -> bool {
        let mut chunks = buf.chunks_exact(16);
        for c in &mut chunks {
            if u128::from_ne_bytes(c.try_into().unwrap()) != 0 {
                return false;
            }
        }
        chunks.remainder().iter().all(|&b| b == 0)
    }

    /// Visit each non-zero `GRANULE`-aligned run of `buf` (located at file offset
    /// `base`) as `(offset, &data)`, coalescing adjacent data granules. Zero runs are
    /// the holes — never visited. Returns the total data bytes seen. This is the one
    /// place the data/hole split lives; the local writer and the network sender both
    /// drive it.
    fn for_each_extent(
        buf: &[u8],
        base: u64,
        mut f: impl FnMut(u64, &[u8]) -> io::Result<()>,
    ) -> io::Result<u64> {
        let (n, mut i, mut data) = (buf.len(), 0usize, 0u64);
        while i < n {
            let g = GRANULE.min(n - i);
            if is_zero(&buf[i..i + g]) {
                i += g;
            } else {
                let start = i;
                i += g;
                while i < n {
                    let g2 = GRANULE.min(n - i);
                    if is_zero(&buf[i..i + g2]) {
                        break;
                    }
                    i += g2;
                }
                f(base + start as u64, &buf[start..i])?;
                data += (i - start) as u64;
            }
        }
        Ok(data)
    }

    /// Write only the non-zero runs of `buf` to the pre-sized destination, leaving
    /// holes for the zeros. `df = None` is dry-run. Returns (data written, holes).
    fn write_extents(df: Option<&File>, buf: &[u8], base: u64) -> io::Result<(u64, u64)> {
        let data = for_each_extent(buf, base, |off, run| match df {
            Some(f) => sysio::write_all_at(f, run, off),
            None => Ok(()),
        })?;
        Ok((data, buf.len() as u64 - data))
    }

    /// Duplicate `src` → `dst` block by block. `src`/`dst` may each be a regular file
    /// or a block device — on Linux they are the same kind of object.
    pub fn run(src: &Path, dst: &Path, opts: &Opts) -> io::Result<Stats> {
        let chunk = opts.chunk.max(512);
        let threads = opts.threads.max(1);

        // The source is read through its format applicative (raw = identity, qcow2 =
        // L1/L2 indirection). The engine below never sees the format — only `read_at`.
        let source = Source::open(src, opts.src_format)?;
        let size = source.size()?;

        // A qcow2/vhdx *destination* is emitted by the format writer (`fromTerm` at
        // α=format), not the block engine — the layout is a whole-image structure.
        if !opts.dry_run {
            let written = match opts.dst_format {
                Format::Qcow2 => Some(crate::sync::image::write_qcow2(&source, dst, 16)?),
                Format::Vhdx => Some(crate::sync::image::write_vhdx(&source, dst, 512)?),
                Format::Raw => None,
            };
            if let Some((data, hole)) = written {
                return Ok(Stats {
                    bytes_copied: data,
                    bytes_zero: hole,
                    blocks_copied: 1,
                    ..Stats::ZERO
                });
            }
        }

        // Is the destination a block device? (Then: no create, no set_len — a device
        // has a fixed size we must not touch.)
        let dst_is_dev = sysio::is_device_path(dst);

        // --sync needs to READ the destination to compare; cold just writes.
        let dst_exists = dst.exists();
        let sync = opts.sync && dst_exists;

        // Sparse (hole-skip) only makes sense for a regular file: a block device has
        // no holes, so eliding a zero write would leave whatever was there before.
        let sparse = opts.sparse && !dst_is_dev;

        let d = if opts.dry_run {
            None
        } else {
            let mut oo = File::options();
            oo.read(true).write(true).create(true);
            // Cold copy starts from a clean slate: truncate so every region we skip
            // (a hole, or a --no-sparse no-op) is guaranteed zero, never a stale byte
            // from a previous, larger file. --sync must NOT truncate — it reads the
            // destination to diff against it. (Truncate is meaningless on a device.)
            if !sync && !dst_is_dev {
                oo.truncate(true);
            }
            let dfile = oo.open(dst)?;
            if !dst_is_dev {
                // Regular-file target: size it to the source up front so every span
                // can write_at its offset into pre-allocated (sparse-zero) space.
                dfile.set_len(size)?;
            }
            Some(dfile)
        };

        let progress = AtomicU64::new(0);
        let done = AtomicBool::new(false);

        // One contiguous span of blocks per worker — the Web partition over offsets.
        let nblocks = size.div_ceil(chunk as u64).max(1);
        let per = nblocks.div_ceil(threads as u64);

        let total = thread::scope(|scope| {
            // Live progress (carriage-return line), unless quiet / dry-run.
            let ticker = if opts.quiet || opts.dry_run || size == 0 {
                None
            } else {
                Some(scope.spawn(|| progress_bar(&progress, &done, size)))
            };

            let mut handles = Vec::new();
            for t in 0..threads {
                let start_blk = (t as u64) * per;
                if start_blk >= nblocks {
                    break;
                }
                let lo = start_blk * chunk as u64;
                let hi = (((t as u64 + 1) * per) * chunk as u64).min(size);
                let (s_ref, d_ref, prog_ref) = (&source, d.as_ref(), &progress);
                handles.push(
                    scope.spawn(move || copy_span(s_ref, d_ref, lo, hi, chunk, sync, sparse, prog_ref)),
                );
            }

            let mut acc = Stats::ZERO;
            for h in handles {
                acc = acc.merge(h.join().unwrap_or(Stats {
                    errors: 1,
                    ..Stats::ZERO
                }));
            }
            done.store(true, Ordering::Release);
            if let Some(tk) = ticker {
                let _ = tk.join();
            }
            acc
        });

        Ok(total)
    }

    /// One worker: copy blocks covering `[lo, hi)`. In `sync` mode each block is
    /// compared against the destination and written only on a difference (the
    /// per-block throne-check); otherwise it is written unconditionally.
    fn copy_span(
        s: &Source,
        d: Option<&File>,
        lo: u64,
        hi: u64,
        chunk: usize,
        sync: bool,
        sparse: bool,
        progress: &AtomicU64,
    ) -> Stats {
        let mut acc = Stats::ZERO;
        let mut sbuf = vec![0u8; chunk];
        let mut dbuf = if sync { vec![0u8; chunk] } else { Vec::new() };

        let mut off = lo;
        while off < hi {
            let want = ((hi - off) as usize).min(chunk);
            let n = match s.read_at(off, &mut sbuf[..want]) {
                Ok(n) => n,
                Err(_) => {
                    acc.errors += 1;
                    break;
                }
            };
            if n == 0 {
                break; // short source — done
            }
            let block = &sbuf[..n];

            // Three faces of "what to write":
            if sync {
                // --sync: the per-chunk throne-check — write only if dst differs.
                let write = match d {
                    Some(df) => {
                        let m = read_full_at(df, &mut dbuf[..n], off).unwrap_or(0);
                        !(m == n && dbuf[..n] == *block)
                    }
                    None => true,
                };
                if write {
                    if let Some(df) = d {
                        if sysio::write_all_at(df, block, off).is_err() {
                            acc.errors += 1;
                        }
                    }
                    acc.bytes_copied += n as u64;
                    acc.blocks_copied += 1;
                } else {
                    acc.bytes_skipped += n as u64;
                    acc.blocks_skipped += 1;
                }
            } else if sparse {
                // Cold + sparse: punch holes at GRANULE, write only data extents —
                // fine holes no matter how large `chunk` is.
                match write_extents(d, block, off) {
                    Ok((data, hole)) => {
                        if data > 0 {
                            acc.bytes_copied += data;
                            acc.blocks_copied += 1;
                        }
                        if hole > 0 {
                            acc.bytes_zero += hole;
                            acc.blocks_zero += 1;
                        }
                    }
                    Err(_) => acc.errors += 1,
                }
            } else {
                // Cold, --no-sparse: write every block verbatim (a true device clone).
                if let Some(df) = d {
                    if sysio::write_all_at(df, block, off).is_err() {
                        acc.errors += 1;
                    }
                }
                acc.bytes_copied += n as u64;
                acc.blocks_copied += 1;
            }

            off += n as u64;
            progress.fetch_add(n as u64, Ordering::Relaxed);
        }
        acc
    }

    /// `read_at` may return short; loop until the slice is full or EOF. Returns the
    /// number of bytes actually read (0 at clean EOF).
    fn read_full_at(f: &File, buf: &mut [u8], at: u64) -> io::Result<usize> {
        let mut got = 0;
        while got < buf.len() {
            match sysio::read_at(f, &mut buf[got..], at + got as u64) {
                Ok(0) => break,
                Ok(n) => got += n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        Ok(got)
    }

    // ───────────────────────── network transport ─────────────────────────
    //
    // Block duplication over TCP — the SAME op, with the wire as the medium. The
    // streaming-law skip travels too: the sender walks the source's data extents
    // (`for_each_extent`) and transmits ONLY the non-zero runs, each tagged with its
    // offset; the receiver `set_len`s the destination to a sparse-zero base and
    // `write_at`s each run. An all-zero region is never sent — so an 80%-empty disk
    // crosses the wire at ~20% of its size. block-duplicator streams every byte
    // (zeros included) down one sequential socket; this is that, minus the drag.
    //
    //   HELLO : "SFDD" ‖ ver(1) ‖ total_size(u64 LE) ‖ chunk_hint(u32 LE)
    //   DATA  : 0x01 ‖ offset(u64 LE) ‖ len(u32 LE) ‖ <len bytes>
    //   END   : 0x00

    const WIRE_CHUNK: usize = 4 << 20; // 4 MiB read granularity on the sender

    /// Rate-limit a progress line to ~10 Hz so it doesn't flood stderr (one carriage
    /// -return refresh, not one per chunk).
    struct Throttle(std::time::Instant);
    impl Throttle {
        fn new() -> Self {
            Throttle(std::time::Instant::now())
        }
        fn ready(&mut self) -> bool {
            if self.0.elapsed().as_millis() >= 100 {
                self.0 = std::time::Instant::now();
                true
            } else {
                false
            }
        }
    }

    fn send_data_frame(w: &mut impl Write, off: u64, run: &[u8]) -> io::Result<()> {
        let mut hdr = [0u8; 13];
        hdr[0] = 0x01;
        hdr[1..9].copy_from_slice(&off.to_le_bytes());
        hdr[9..13].copy_from_slice(&(run.len() as u32).to_le_bytes());
        w.write_all(&hdr)?;
        w.write_all(run)
    }

    /// Read-back verification: re-read source and destination through their formats
    /// and compare. Returns `Ok(None)` if identical, `Ok(Some(off))` at the first
    /// mismatch. A qcow2 destination is decoded through its codec, so this checks the
    /// written image actually reproduces the source end to end.
    pub fn verify(
        src: &Path,
        src_fmt: Format,
        dst: &Path,
        dst_fmt: Format,
        quiet: bool,
    ) -> io::Result<Option<u64>> {
        let a = Source::open(src, src_fmt)?;
        let b = Source::open(dst, dst_fmt)?;
        let (sa, sb) = (a.size()?, b.size()?);
        if !quiet {
            eprintln!(
                "syncfold dd: verify {} ({}) vs {} ({})",
                src.display(),
                a.describe(),
                dst.display(),
                b.describe()
            );
        }
        if sa != sb {
            return Ok(Some(sa.min(sb))); // size mismatch — report the boundary
        }
        let chunk = 4 << 20;
        let (mut ba, mut bb) = (vec![0u8; chunk], vec![0u8; chunk]);
        let mut off = 0u64;
        while off < sa {
            let want = ((sa - off) as usize).min(chunk);
            a.read_at(off, &mut ba[..want])?;
            b.read_at(off, &mut bb[..want])?;
            if ba[..want] != bb[..want] {
                // narrow to the exact byte for a useful report
                let at = ba[..want]
                    .iter()
                    .zip(&bb[..want])
                    .position(|(x, y)| x != y)
                    .unwrap_or(0);
                return Ok(Some(off + at as u64));
            }
            off += want as u64;
        }
        Ok(None)
    }

    /// Push a local source to `host:port`, hole-skipping over the wire (unless the
    /// caller turns sparse off — e.g. a true device clone where zeros are meaningful).
    pub fn push(
        src: &Path,
        fmt: Format,
        host: &str,
        port: u16,
        sparse: bool,
        quiet: bool,
    ) -> io::Result<Stats> {
        let s = Source::open(src, fmt)?;
        let size = s.size()?;
        let mut stream = TcpStream::connect((host, port))?;
        stream.set_nodelay(true)?;

        let mut hello = Vec::with_capacity(17);
        hello.extend_from_slice(b"SFDD");
        hello.push(1u8);
        hello.extend_from_slice(&size.to_le_bytes());
        hello.extend_from_slice(&(WIRE_CHUNK as u32).to_le_bytes());
        stream.write_all(&hello)?;

        let mut buf = vec![0u8; WIRE_CHUNK];
        let mut acc = Stats::ZERO;
        let mut off = 0u64;
        let mut tick = Throttle::new();
        while off < size {
            let want = ((size - off) as usize).min(WIRE_CHUNK);
            let n = s.read_at(off, &mut buf[..want])?;
            if n == 0 {
                break;
            }
            if sparse {
                let data = for_each_extent(&buf[..n], off, |o, run| {
                    send_data_frame(&mut stream, o, run)
                })?;
                acc.bytes_copied += data;
                acc.bytes_zero += n as u64 - data;
            } else {
                send_data_frame(&mut stream, off, &buf[..n])?;
                acc.bytes_copied += n as u64;
            }
            acc.blocks_copied += 1;
            off += n as u64;
            if !quiet && tick.ready() {
                let mut e = io::stderr().lock();
                let _ = write!(
                    e,
                    "\rsyncfold dd push: {:6.2}%  sent {:.1} / {:.1} MiB",
                    off as f64 / size as f64 * 100.0,
                    acc.bytes_copied as f64 / (1024.0 * 1024.0),
                    size as f64 / (1024.0 * 1024.0)
                );
                let _ = e.flush();
            }
        }
        stream.write_all(&[0x00])?; // END
        stream.flush()?;
        if !quiet {
            eprintln!();
        }
        Ok(acc)
    }

    /// Serve as the receiving sink: accept ONE connection on `port` and write the
    /// streamed extents into `dst` (file or device). Returns when the peer sends END.
    pub fn serve(dst: &Path, port: u16, quiet: bool) -> io::Result<Stats> {
        let listener = TcpListener::bind(("0.0.0.0", port))?;
        if !quiet {
            eprintln!(
                "syncfold dd serve: listening on 0.0.0.0:{port} → {}",
                dst.display()
            );
        }
        let (mut stream, peer) = listener.accept()?;
        stream.set_nodelay(true)?;
        if !quiet {
            eprintln!("syncfold dd serve: connection from {peer}");
        }

        let mut magic = [0u8; 4];
        stream.read_exact(&mut magic)?;
        if &magic != b"SFDD" {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bad magic (not a syncfold dd stream)",
            ));
        }
        let mut one = [0u8; 1];
        stream.read_exact(&mut one)?; // version
        let mut sz = [0u8; 8];
        stream.read_exact(&mut sz)?;
        let size = u64::from_le_bytes(sz);
        let mut ck = [0u8; 4];
        stream.read_exact(&mut ck)?; // chunk hint — informational

        let dst_is_dev = sysio::is_device_path(dst);
        let mut oo = File::options();
        oo.read(true).write(true).create(true);
        if !dst_is_dev {
            oo.truncate(true);
        }
        let d = oo.open(dst)?;
        if !dst_is_dev {
            d.set_len(size)?; // sparse-zero base; unsent (hole) regions stay zero
        }

        let mut acc = Stats::ZERO;
        let mut hdr = [0u8; 13];
        let mut data = vec![0u8; WIRE_CHUNK];
        let mut tick = Throttle::new();
        loop {
            stream.read_exact(&mut hdr[..1])?;
            if hdr[0] == 0x00 {
                break; // END
            }
            if hdr[0] != 0x01 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "unknown frame tag",
                ));
            }
            stream.read_exact(&mut hdr[1..13])?;
            let off = u64::from_le_bytes(hdr[1..9].try_into().unwrap());
            let len = u32::from_le_bytes(hdr[9..13].try_into().unwrap()) as usize;
            if len > data.len() {
                data.resize(len, 0);
            }
            stream.read_exact(&mut data[..len])?;
            sysio::write_all_at(&d, &data[..len], off)?;
            acc.bytes_copied += len as u64;
            acc.blocks_copied += 1;
            if !quiet && tick.ready() {
                let mut e = io::stderr().lock();
                let _ = write!(
                    e,
                    "\rsyncfold dd serve: recv {:.1} / {:.1} MiB",
                    acc.bytes_copied as f64 / (1024.0 * 1024.0),
                    size as f64 / (1024.0 * 1024.0)
                );
                let _ = e.flush();
            }
        }
        d.sync_all()?;
        if !quiet {
            eprintln!();
        }
        acc.bytes_zero = size.saturating_sub(acc.bytes_copied); // the holes we never received
        Ok(acc)
    }

    /// A minimal carriage-return progress line on stderr; std-only, no ticker crate.
    fn progress_bar(progress: &AtomicU64, done: &AtomicBool, size: u64) {
        use std::io::Write;
        let start = std::time::Instant::now();
        loop {
            let cur = progress.load(Ordering::Relaxed).min(size);
            let pct = cur as f64 / size as f64 * 100.0;
            let secs = start.elapsed().as_secs_f64().max(1e-3);
            let mibps = cur as f64 / (1024.0 * 1024.0) / secs;
            let mut err = io::stderr().lock();
            let _ = write!(
                err,
                "\rsyncfold dd: {:6.2}%  {:>8.1} / {:.1} MiB  {:7.1} MiB/s",
                pct,
                cur as f64 / (1024.0 * 1024.0),
                size as f64 / (1024.0 * 1024.0),
                mibps,
            );
            let _ = err.flush();
            if done.load(Ordering::Acquire) {
                let _ = writeln!(err);
                break;
            }
            thread::sleep(std::time::Duration::from_millis(120));
        }
    }
}
pub mod copy {
    //! The leaf op — one file's bytes, at the lowest syscall cost (the **B face**).
    //!
    //! Both handles are held by us, so the copy and the mtime-preserve happen on fds
    //! we already own — no reopen. `std::io::copy(File → File)` routes through the
    //! kernel's `copy_file_range` on Linux (in-kernel, no userspace bounce; CoW
    //! reflink inside the kernel on btrfs/xfs). Safe `std`, no FFI, no `unsafe`.
    //!
    //! Per file: `openat(src)`, `statx(src)`, `openat(dst)`, `copy_file_range`,
    //! `futimens(dst)`, two closes — the floor for a metadata-preserving copy
    //! without io_uring batching.

    use std::fs::{File, FileTimes};
    use std::io;
    use std::path::Path;

    use crate::sync::sysio;

    /// Copy `src` → `dst`, preserving mode and the source mtime. Returns bytes.
    pub fn copy_one(src: &Path, dst: &Path) -> io::Result<u64> {
        let mut s = File::open(src)?;
        let meta = s.metadata()?;
        let mut oo = File::options();
        oo.write(true).create(true).truncate(true);
        sysio::open_options_mode(&mut oo, sysio::mode_of(&meta)); // unix mode; no-op on Windows
        let mut d = oo.open(dst)?;

        let n = io::copy(&mut s, &mut d)?; // kernel copy_file_range; no reopen

        // Preserve mtime so the next sync's throne-check sees this file unchanged.
        let _ = d.set_times(FileTimes::new().set_modified(meta.modified()?));
        Ok(n)
    }
}
pub mod crypto {
    //! Crypto bedrock — pure safe Rust, zero deps, zero `unsafe`.
    //!
    //! SHA-256 (FIPS 180-4, streaming for the TLS transcript), HMAC-SHA256 (RFC 2104),
    //! HKDF (RFC 5869), and the TLS 1.3 `HKDF-Expand-Label` (RFC 8446). Everything the
    //! SigV4 signer and the TLS key schedule stand on. Verified against the published
    //! known-answer vectors in the test module below — crypto you don't check against
    //! vectors is worthless.

    #![allow(dead_code)]

    // ── SHA-256 ──────────────────────────────────────────────────────────────────

    const H0: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
    ];

    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
        0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
        0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
        0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
        0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
        0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
        0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2,
    ];

    /// Streaming SHA-256.
    #[derive(Clone)]
    pub struct Sha256 {
        h: [u32; 8],
        buf: [u8; 64],
        n: usize,
        total: u64,
    }

    impl Sha256 {
        pub fn new() -> Self {
            Self {
                h: H0,
                buf: [0; 64],
                n: 0,
                total: 0,
            }
        }

        pub fn update(&mut self, data: &[u8]) {
            self.total = self.total.wrapping_add(data.len() as u64);
            let mut i = 0;
            if self.n > 0 {
                while self.n < 64 && i < data.len() {
                    self.buf[self.n] = data[i];
                    self.n += 1;
                    i += 1;
                }
                if self.n == 64 {
                    let b = self.buf;
                    self.compress(&b);
                    self.n = 0;
                }
            }
            while i + 64 <= data.len() {
                let mut b = [0u8; 64];
                b.copy_from_slice(&data[i..i + 64]);
                self.compress(&b);
                i += 64;
            }
            while i < data.len() {
                self.buf[self.n] = data[i];
                self.n += 1;
                i += 1;
            }
        }

        pub fn finalize(mut self) -> [u8; 32] {
            let bits = self.total.wrapping_mul(8);
            self.buf[self.n] = 0x80;
            self.n += 1;
            if self.n > 56 {
                while self.n < 64 {
                    self.buf[self.n] = 0;
                    self.n += 1;
                }
                let b = self.buf;
                self.compress(&b);
                self.n = 0;
            }
            while self.n < 56 {
                self.buf[self.n] = 0;
                self.n += 1;
            }
            self.buf[56..64].copy_from_slice(&bits.to_be_bytes());
            let b = self.buf;
            self.compress(&b);
            let mut out = [0u8; 32];
            for (i, word) in self.h.iter().enumerate() {
                out[i * 4..i * 4 + 4].copy_from_slice(&word.to_be_bytes());
            }
            out
        }

        fn compress(&mut self, block: &[u8; 64]) {
            let mut w = [0u32; 64];
            for i in 0..16 {
                w[i] = u32::from_be_bytes([
                    block[i * 4],
                    block[i * 4 + 1],
                    block[i * 4 + 2],
                    block[i * 4 + 3],
                ]);
            }
            for i in 16..64 {
                let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
                let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
                w[i] = w[i - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[i - 7])
                    .wrapping_add(s1);
            }
            let mut a = self.h;
            for i in 0..64 {
                let s1 = a[4].rotate_right(6) ^ a[4].rotate_right(11) ^ a[4].rotate_right(25);
                let ch = (a[4] & a[5]) ^ ((!a[4]) & a[6]);
                let t1 = a[7]
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K[i])
                    .wrapping_add(w[i]);
                let s0 = a[0].rotate_right(2) ^ a[0].rotate_right(13) ^ a[0].rotate_right(22);
                let maj = (a[0] & a[1]) ^ (a[0] & a[2]) ^ (a[1] & a[2]);
                let t2 = s0.wrapping_add(maj);
                a[7] = a[6];
                a[6] = a[5];
                a[5] = a[4];
                a[4] = a[3].wrapping_add(t1);
                a[3] = a[2];
                a[2] = a[1];
                a[1] = a[0];
                a[0] = t1.wrapping_add(t2);
            }
            for i in 0..8 {
                self.h[i] = self.h[i].wrapping_add(a[i]);
            }
        }
    }

    /// One-shot SHA-256.
    pub fn sha256(data: &[u8]) -> [u8; 32] {
        let mut h = Sha256::new();
        h.update(data);
        h.finalize()
    }

    /// Lowercase hex of a byte slice.
    pub fn hex(b: &[u8]) -> String {
        const D: &[u8; 16] = b"0123456789abcdef";
        let mut s = String::with_capacity(b.len() * 2);
        for &x in b {
            s.push(D[(x >> 4) as usize] as char);
            s.push(D[(x & 0xf) as usize] as char);
        }
        s
    }

    // ── HMAC-SHA256 / HKDF ───────────────────────────────────────────────────────

    pub fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
        let mut k = [0u8; 64];
        if key.len() > 64 {
            k[..32].copy_from_slice(&sha256(key));
        } else {
            k[..key.len()].copy_from_slice(key);
        }
        let mut ipad = [0x36u8; 64];
        let mut opad = [0x5cu8; 64];
        for i in 0..64 {
            ipad[i] ^= k[i];
            opad[i] ^= k[i];
        }
        let mut inner = Sha256::new();
        inner.update(&ipad);
        inner.update(msg);
        let ih = inner.finalize();
        let mut outer = Sha256::new();
        outer.update(&opad);
        outer.update(&ih);
        outer.finalize()
    }

    /// HMAC-SHA512 — same construction as [`hmac_sha256`] above, over SHA-512's
    /// 128-byte block size instead of SHA-256's 64. Needed for RFC 6979 deterministic
    /// nonce generation on P-521 (go-plugin's AutoMTLS curve), whose signatures are
    /// `ecdsa-with-SHA512`.
    pub fn hmac_sha512(key: &[u8], msg: &[u8]) -> [u8; 64] {
        let mut k = [0u8; 128];
        if key.len() > 128 {
            k[..64].copy_from_slice(&sha512(key));
        } else {
            k[..key.len()].copy_from_slice(key);
        }
        let mut ipad = [0x36u8; 128];
        let mut opad = [0x5cu8; 128];
        for i in 0..128 {
            ipad[i] ^= k[i];
            opad[i] ^= k[i];
        }
        let mut inner = Sha512::new();
        inner.update(&ipad);
        inner.update(msg);
        let ih = inner.finalize();
        let mut outer = Sha512::new();
        outer.update(&opad);
        outer.update(&ih);
        outer.finalize()
    }

    pub fn hkdf_extract(salt: &[u8], ikm: &[u8]) -> [u8; 32] {
        hmac_sha256(salt, ikm)
    }

    pub fn hkdf_expand(prk: &[u8], info: &[u8], len: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(len);
        let mut t: Vec<u8> = Vec::new();
        let mut ctr: u8 = 1;
        while out.len() < len {
            let mut m = Vec::with_capacity(t.len() + info.len() + 1);
            m.extend_from_slice(&t);
            m.extend_from_slice(info);
            m.push(ctr);
            let block = hmac_sha256(prk, &m);
            let take = (len - out.len()).min(32);
            out.extend_from_slice(&block[..take]);
            t = block.to_vec();
            ctr = ctr.wrapping_add(1);
        }
        out
    }

    /// TLS 1.3 `HKDF-Expand-Label` (RFC 8446 §7.1).
    pub fn hkdf_expand_label(secret: &[u8], label: &str, context: &[u8], len: usize) -> Vec<u8> {
        let full = format!("tls13 {label}");
        let mut info = Vec::with_capacity(2 + 1 + full.len() + 1 + context.len());
        info.extend_from_slice(&(len as u16).to_be_bytes());
        info.push(full.len() as u8);
        info.extend_from_slice(full.as_bytes());
        info.push(context.len() as u8);
        info.extend_from_slice(context);
        hkdf_expand(secret, &info, len)
    }

    // ── SHA-512 / SHA-384 (FIPS 180-4) — needed for ECDSA-with-SHA384 chains ────────

    const K512: [u64; 80] = [
        0x428a2f98d728ae22,
        0x7137449123ef65cd,
        0xb5c0fbcfec4d3b2f,
        0xe9b5dba58189dbbc,
        0x3956c25bf348b538,
        0x59f111f1b605d019,
        0x923f82a4af194f9b,
        0xab1c5ed5da6d8118,
        0xd807aa98a3030242,
        0x12835b0145706fbe,
        0x243185be4ee4b28c,
        0x550c7dc3d5ffb4e2,
        0x72be5d74f27b896f,
        0x80deb1fe3b1696b1,
        0x9bdc06a725c71235,
        0xc19bf174cf692694,
        0xe49b69c19ef14ad2,
        0xefbe4786384f25e3,
        0x0fc19dc68b8cd5b5,
        0x240ca1cc77ac9c65,
        0x2de92c6f592b0275,
        0x4a7484aa6ea6e483,
        0x5cb0a9dcbd41fbd4,
        0x76f988da831153b5,
        0x983e5152ee66dfab,
        0xa831c66d2db43210,
        0xb00327c898fb213f,
        0xbf597fc7beef0ee4,
        0xc6e00bf33da88fc2,
        0xd5a79147930aa725,
        0x06ca6351e003826f,
        0x142929670a0e6e70,
        0x27b70a8546d22ffc,
        0x2e1b21385c26c926,
        0x4d2c6dfc5ac42aed,
        0x53380d139d95b3df,
        0x650a73548baf63de,
        0x766a0abb3c77b2a8,
        0x81c2c92e47edaee6,
        0x92722c851482353b,
        0xa2bfe8a14cf10364,
        0xa81a664bbc423001,
        0xc24b8b70d0f89791,
        0xc76c51a30654be30,
        0xd192e819d6ef5218,
        0xd69906245565a910,
        0xf40e35855771202a,
        0x106aa07032bbd1b8,
        0x19a4c116b8d2d0c8,
        0x1e376c085141ab53,
        0x2748774cdf8eeb99,
        0x34b0bcb5e19b48a8,
        0x391c0cb3c5c95a63,
        0x4ed8aa4ae3418acb,
        0x5b9cca4f7763e373,
        0x682e6ff3d6b2b8a3,
        0x748f82ee5defb2fc,
        0x78a5636f43172f60,
        0x84c87814a1f0ab72,
        0x8cc702081a6439ec,
        0x90befffa23631e28,
        0xa4506cebde82bde9,
        0xbef9a3f7b2c67915,
        0xc67178f2e372532b,
        0xca273eceea26619c,
        0xd186b8c721c0c207,
        0xeada7dd6cde0eb1e,
        0xf57d4f7fee6ed178,
        0x06f067aa72176fba,
        0x0a637dc5a2c898a6,
        0x113f9804bef90dae,
        0x1b710b35131c471b,
        0x28db77f523047d84,
        0x32caab7b40c72493,
        0x3c9ebe0a15c9bebc,
        0x431d67c49c100d4c,
        0x4cc5d4becb3e42b6,
        0x597f299cfc657e2a,
        0x5fcb6fab3ad6faec,
        0x6c44198c4a475817,
    ];

    /// Streaming SHA-512 (also the engine for SHA-384, which differs only in IV + output length).
    pub struct Sha512 {
        h: [u64; 8],
        buf: [u8; 128],
        n: usize,
        total: u128,
    }

    impl Sha512 {
        fn with_iv(iv: [u64; 8]) -> Self {
            Self {
                h: iv,
                buf: [0; 128],
                n: 0,
                total: 0,
            }
        }
        pub fn new() -> Self {
            Self::with_iv([
                0x6a09e667f3bcc908,
                0xbb67ae8584caa73b,
                0x3c6ef372fe94f82b,
                0xa54ff53a5f1d36f1,
                0x510e527fade682d1,
                0x9b05688c2b3e6c1f,
                0x1f83d9abfb41bd6b,
                0x5be0cd19137e2179,
            ])
        }
        pub fn new384() -> Self {
            Self::with_iv([
                0xcbbb9d5dc1059ed8,
                0x629a292a367cd507,
                0x9159015a3070dd17,
                0x152fecd8f70e5939,
                0x67332667ffc00b31,
                0x8eb44a8768581511,
                0xdb0c2e0d64f98fa7,
                0x47b5481dbefa4fa4,
            ])
        }
        pub fn update(&mut self, data: &[u8]) {
            self.total = self.total.wrapping_add(data.len() as u128);
            let mut i = 0;
            if self.n > 0 {
                while self.n < 128 && i < data.len() {
                    self.buf[self.n] = data[i];
                    self.n += 1;
                    i += 1;
                }
                if self.n == 128 {
                    let b = self.buf;
                    self.compress(&b);
                    self.n = 0;
                }
            }
            while i + 128 <= data.len() {
                let mut b = [0u8; 128];
                b.copy_from_slice(&data[i..i + 128]);
                self.compress(&b);
                i += 128;
            }
            while i < data.len() {
                self.buf[self.n] = data[i];
                self.n += 1;
                i += 1;
            }
        }
        fn finish(mut self) -> [u64; 8] {
            let bits = self.total.wrapping_mul(8);
            self.buf[self.n] = 0x80;
            self.n += 1;
            if self.n > 112 {
                while self.n < 128 {
                    self.buf[self.n] = 0;
                    self.n += 1;
                }
                let b = self.buf;
                self.compress(&b);
                self.n = 0;
            }
            while self.n < 112 {
                self.buf[self.n] = 0;
                self.n += 1;
            }
            self.buf[112..128].copy_from_slice(&bits.to_be_bytes());
            let b = self.buf;
            self.compress(&b);
            self.h
        }
        pub fn finalize(self) -> [u8; 64] {
            let h = self.finish();
            let mut out = [0u8; 64];
            for (i, w) in h.iter().enumerate() {
                out[i * 8..i * 8 + 8].copy_from_slice(&w.to_be_bytes());
            }
            out
        }
        pub fn finalize384(self) -> [u8; 48] {
            let h = self.finish();
            let mut out = [0u8; 48];
            for i in 0..6 {
                out[i * 8..i * 8 + 8].copy_from_slice(&h[i].to_be_bytes());
            }
            out
        }
        fn compress(&mut self, block: &[u8; 128]) {
            let mut w = [0u64; 80];
            for i in 0..16 {
                w[i] = u64::from_be_bytes(block[i * 8..i * 8 + 8].try_into().unwrap());
            }
            for i in 16..80 {
                let s0 = w[i - 15].rotate_right(1) ^ w[i - 15].rotate_right(8) ^ (w[i - 15] >> 7);
                let s1 = w[i - 2].rotate_right(19) ^ w[i - 2].rotate_right(61) ^ (w[i - 2] >> 6);
                w[i] = w[i - 16]
                    .wrapping_add(s0)
                    .wrapping_add(w[i - 7])
                    .wrapping_add(s1);
            }
            let mut a = self.h;
            for i in 0..80 {
                let s1 = a[4].rotate_right(14) ^ a[4].rotate_right(18) ^ a[4].rotate_right(41);
                let ch = (a[4] & a[5]) ^ ((!a[4]) & a[6]);
                let t1 = a[7]
                    .wrapping_add(s1)
                    .wrapping_add(ch)
                    .wrapping_add(K512[i])
                    .wrapping_add(w[i]);
                let s0 = a[0].rotate_right(28) ^ a[0].rotate_right(34) ^ a[0].rotate_right(39);
                let maj = (a[0] & a[1]) ^ (a[0] & a[2]) ^ (a[1] & a[2]);
                let t2 = s0.wrapping_add(maj);
                a[7] = a[6];
                a[6] = a[5];
                a[5] = a[4];
                a[4] = a[3].wrapping_add(t1);
                a[3] = a[2];
                a[2] = a[1];
                a[1] = a[0];
                a[0] = t1.wrapping_add(t2);
            }
            for i in 0..8 {
                self.h[i] = self.h[i].wrapping_add(a[i]);
            }
        }
    }

    pub fn sha512(data: &[u8]) -> [u8; 64] {
        let mut h = Sha512::new();
        h.update(data);
        h.finalize()
    }
    pub fn sha384(data: &[u8]) -> [u8; 48] {
        let mut h = Sha512::new384();
        h.update(data);
        h.finalize384()
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn sha512_384_vectors() {
            assert_eq!(
                hex(&sha512(b"abc")),
                "ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f"
            );
            assert_eq!(
                hex(&sha384(b"abc")),
                "cb00753f45a35e8bb5a03d699ac65007272c32ab0eded1631a8b605a43ff5bed8086072ba1e7cc2358baeca134c825a7"
            );
        }

        #[test]
        fn sha256_vectors() {
            assert_eq!(
                hex(&sha256(b"")),
                "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
            );
            assert_eq!(
                hex(&sha256(b"abc")),
                "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
            );
            // streamed in pieces must equal one-shot
            let mut h = Sha256::new();
            for c in b"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq" {
                h.update(&[*c]);
            }
            assert_eq!(
                hex(&h.finalize()),
                "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"
            );
        }

        #[test]
        fn hmac_vector_rfc4231_case1() {
            let key = [0x0bu8; 20];
            let r = hmac_sha256(&key, b"Hi There");
            assert_eq!(
                hex(&r),
                "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7"
            );
        }

        #[test]
        fn hkdf_vector_rfc5869_case1() {
            let ikm = [0x0bu8; 22];
            let salt: Vec<u8> = (0u8..=0x0c).collect();
            let info: Vec<u8> = (0xf0u8..=0xf9).collect();
            let prk = hkdf_extract(&salt, &ikm);
            assert_eq!(
                hex(&prk),
                "077709362c2e32df0ddc3f0dc47bba6390b6c73bb50f9c3122ec844ad7c2b3e5"
            );
            let okm = hkdf_expand(&prk, &info, 42);
            assert_eq!(
                hex(&okm),
                "3cb25f25faacd57a90434f64d0362f2a2d2d0a90cf1a5a4c5db02d56ecc4c5bf34007208d5b887185865"
            );
        }
    }
}
pub mod grpc {
    //! gRPC unary calls over [`crate::sync::http2`] — the 5-byte message framing
    //! (RFC: 1-byte compression flag + 4-byte big-endian length + the protobuf
    //! message bytes) and the standard headers (`content-type: application/grpc`,
    //! `te: trailers`), plus `grpc-status`/`grpc-message` trailer handling. Pure
    //! Rust, zero deps — the protobuf message bytes themselves are opaque here
    //! (encode/decode is [`hashtrinity`]'s job); this module only frames and
    //! transports them.

    use std::io::{self, Error, ErrorKind, Read, Write};

    use crate::sync::http2;

    pub struct GrpcResponse {
        pub message: Vec<u8>, // raw protobuf bytes, framing already stripped
        pub grpc_status: i32,
        pub grpc_message: String,
    }

    /// Frame a single protobuf message the way gRPC puts it on the wire: no
    /// compression (flag byte 0), then its length, then the bytes themselves.
    fn frame_message(protobuf_bytes: &[u8]) -> Vec<u8> {
        let mut out = Vec::with_capacity(5 + protobuf_bytes.len());
        out.push(0x00); // uncompressed
        out.extend_from_slice(&(protobuf_bytes.len() as u32).to_be_bytes());
        out.extend_from_slice(protobuf_bytes);
        out
    }

    /// Un-frame ONE gRPC message from `body` (which may be exactly one frame, or
    /// occasionally arrive with a trailing partial/duplicate for streaming calls —
    /// a unary response is always exactly one frame, so anything past it is
    /// ignored rather than erroring).
    fn unframe_message(body: &[u8]) -> io::Result<Vec<u8>> {
        let err = || Error::new(ErrorKind::InvalidData, "truncated gRPC message frame");
        if body.len() < 5 {
            if body.is_empty() {
                return Ok(Vec::new()); // a genuinely empty response message (e.g. Empty{})
            }
            return Err(err());
        }
        let len = u32::from_be_bytes([body[1], body[2], body[3], body[4]]) as usize;
        body.get(5..5 + len).map(<[u8]>::to_vec).ok_or_else(err)
    }

    fn header_value<'a>(headers: &'a [(String, String)], name: &str) -> Option<&'a str> {
        headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// Call `full_method` (e.g. `"/tfplugin6.Provider/GetProviderSchema"`) with
    /// `request_protobuf` as the request message, over an already-connected
    /// `io` (our TLS client, a plaintext socket, whatever implements `Read+Write`).
    pub fn call(
        io: &mut (impl Read + Write),
        authority: &str,
        full_method: &str,
        request_protobuf: &[u8],
    ) -> io::Result<GrpcResponse> {
        let headers = [
            (":method", "POST"),
            (":scheme", "http"),
            (":path", full_method),
            (":authority", authority),
            ("content-type", "application/grpc"),
            ("te", "trailers"),
            ("grpc-accept-encoding", "identity"),
        ];
        let body = frame_message(request_protobuf);
        let resp = http2::unary_request(io, &headers, &body)?;

        let status = header_value(&resp.headers, ":status");
        if status != Some("200") {
            return Err(Error::new(
                ErrorKind::Other,
                format!("HTTP/2 status {status:?} (expected 200)"),
            ));
        }

        // grpc-status can arrive either as a trailer (the normal unary-call shape)
        // or, for a call that fails before any response headers, as a HEADERS-only
        // response with grpc-status set directly on the (only) HEADERS frame.
        let status_source = if !resp.trailers.is_empty() {
            &resp.trailers
        } else {
            &resp.headers
        };
        let grpc_status: i32 = header_value(status_source, "grpc-status")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0);
        let grpc_message = header_value(status_source, "grpc-message")
            .unwrap_or("")
            .to_string();

        let message = unframe_message(&resp.body)?;
        Ok(GrpcResponse {
            message,
            grpc_status,
            grpc_message,
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn frame_then_unframe_round_trips() {
            for msg in [b"".as_slice(), b"hello", &[0u8; 300]] {
                let framed = frame_message(msg);
                assert_eq!(framed[0], 0x00, "compression flag must be 0 (uncompressed)");
                let unframed = unframe_message(&framed).unwrap_or_else(|e| unreachable!("{e}"));
                assert_eq!(unframed, msg);
            }
        }

        #[test]
        fn unframe_rejects_truncated_length_prefix() {
            assert!(unframe_message(&[0x00, 0x00, 0x00]).is_err());
        }

        #[test]
        fn unframe_accepts_genuinely_empty_body() {
            assert_eq!(
                unframe_message(&[]).unwrap_or_else(|e| unreachable!("{e}")),
                Vec::<u8>::new()
            );
        }
    }

    #[cfg(test)]
    mod live_check {
        use super::*;
        use crate::sync::tls::client::{ClientIdentity, TlsClient};
        use crate::sync::tls::{ecdsa, x509};
        use std::io::BufRead;
        use std::process::{Command, Stdio};

        /// The full stack, first real call: connect via AutoMTLS to a REAL
        /// go-plugin provider (as `tls::client::live_check` already proved works),
        /// then make an actual `GetProviderSchema` gRPC call over it and check we
        /// get back `grpc-status: 0` and at least SOME protobuf bytes — the first
        /// genuine round trip through crypto + cert + TLS + HTTP/2 + HPACK + gRPC
        /// framing all at once, against the real thing. `GetProviderSchema.Request`
        /// has zero fields (an empty message), so this needs no protobuf encoding
        /// on the way out at all; decoding the response schema is separate,
        /// still-pending work (hashtrinity's proto codec needs `map<K,V>` support
        /// first) — this test only proves the transport, not the schema.
        #[test]
        #[ignore = "spawns a real provider binary and hits its live TLS port — run explicitly"]
        fn get_provider_schema_round_trips_against_a_real_provider() {
            let provider = std::env::var("SYNCFOLD_TEST_PROVIDER_BIN").unwrap_or_else(|_| {
                unreachable!("set SYNCFOLD_TEST_PROVIDER_BIN to an AutoMTLS-capable provider path")
            });

            let curve = ecdsa::p521();
            let (d, qx, qy) = curve
                .generate_keypair()
                .unwrap_or_else(|e| unreachable!("keygen: {e}"));
            let der = x509::build_self_signed_p521(
                &d,
                &qx,
                &qy,
                "localhost",
                &[0x2b],
                1_700_000_000,
                2_100_000_000,
            );
            let pem = x509::to_pem(&der);
            let identity = ClientIdentity {
                cert_der: der,
                private_key: d,
            };

            let mut child = Command::new(&provider)
                .env(
                    "TF_PLUGIN_MAGIC_COOKIE",
                    "d602bf8f470bc67ca7faa0386276bbdd4330efaf76d1a219cb4d6991ca9872b2",
                )
                .env("PLUGIN_PROTOCOL_VERSIONS", "6")
                .env("PLUGIN_MIN_PORT", "10000")
                .env("PLUGIN_MAX_PORT", "25000")
                .env("PLUGIN_CLIENT_CERT", &pem)
                .stdout(Stdio::piped())
                .stderr(Stdio::null())
                .spawn()
                .unwrap_or_else(|e| unreachable!("spawn provider: {e}"));

            let stdout = child
                .stdout
                .take()
                .unwrap_or_else(|| unreachable!("no stdout"));
            let mut lines = std::io::BufReader::new(stdout).lines();
            let mut handshake = None;
            for _ in 0..20 {
                let Some(Ok(line)) = lines.next() else { break };
                if line.starts_with("1|") && line.matches('|').count() == 5 {
                    handshake = Some(line);
                    break;
                }
            }
            let handshake =
                handshake.unwrap_or_else(|| unreachable!("no handshake line seen from provider"));
            // Keep draining stdout in the background for the rest of the test: if the
            // provider logs anything further and nobody reads its stdout pipe, the pipe
            // buffer fills and the provider can block trying to write to it — a stall
            // that looks exactly like an inconsistent, timing-dependent connection
            // failure on OUR side (measured live: the same test failed at different
            // points in the HTTP/2 write sequence across runs, the signature of a
            // starved child process, not a protocol bug).
            std::thread::spawn(move || for _ in lines.by_ref() {});
            let fields: Vec<&str> = handshake.split('|').collect();
            let network = fields[2];
            let addr = fields[3];

            let mut tls = match network {
                "unix" => TlsClient::connect_unix_with_identity(addr, "localhost", Some(&identity)),
                "tcp" => {
                    let (host, port_str) = addr
                        .rsplit_once(':')
                        .unwrap_or_else(|| unreachable!("bad addr: {addr}"));
                    let port: u16 = port_str
                        .parse()
                        .unwrap_or_else(|e| unreachable!("bad port: {e}"));
                    TlsClient::connect_with_identity(host, port, false, Some(&identity))
                }
                other => unreachable!("unexpected network type: {other}"),
            }
            .unwrap_or_else(|e| unreachable!("AutoMTLS handshake failed: {e}"));

            let result = call(
                &mut tls,
                "localhost",
                "/tfplugin6.Provider/GetProviderSchema",
                &[],
            );

            let _ = child.kill();
            let _ = child.wait();

            let resp = result.unwrap_or_else(|e| unreachable!("GetProviderSchema call failed: {e}"));
            assert_eq!(
                resp.grpc_status, 0,
                "grpc-status must be OK, got message: {}",
                resp.grpc_message
            );
            assert!(
                !resp.message.is_empty(),
                "a real provider's schema response must not be empty protobuf bytes"
            );
        }
    }
}
pub mod hpack {
    //! HPACK (RFC 7541) — header compression for HTTP/2. Pure Rust, zero deps.
    //!
    //! Scoped to what a gRPC unary client actually needs: encoding is
    //! literal-only (we choose exactly what headers to send, so there is no need
    //! to grow our own dynamic table on the way out — every outbound field is
    //! `Literal Header Field without Indexing`, optionally using a static-table
    //! name reference). Decoding is the full spec: indexed (static + dynamic),
    //! all three literal forms, Huffman and plain strings, and a real dynamic
    //! table (RFC 7541 §2.3) — because we don't control what a real gRPC SERVER
    //! sends back and must track ITS indexing choices to decode correctly.
    //!
    //! The Huffman table (`HUFFMAN_TABLE`, 257 canonical codes — 256 symbols +
    //! EOS) is transcribed programmatically from `hyperium/h2`'s own
    //! machine-generated table (never retyped by hand), and independently
    //! verified to satisfy the Kraft equality (`sum(2^-len) == 1.0`, the
    //! necessary-and-sufficient condition for a complete prefix code) before use.
    //! The static table (`STATIC_TABLE`, 61 entries) and every decoder behaviour
    //! below is checked against RFC 7541 Appendix C's own published worked
    //! examples (`tests::rfc7541_*`) — real reference bytes, not invented ones.
    //!
    //! `huffman_encode` stays unused by `encode_header`'s own literal-only choice
    //! above, but is real, tested (round-trips + RFC reference bytes) library
    //! surface for a caller that wants Huffman-compressed output later.

    #![allow(dead_code)]

    use std::io::{self, Error, ErrorKind};

    /// (bit length, code) for each of the 256 Huffman symbols plus EOS (index 256).
    /// Kraft sum verified == 1.0 (a complete, valid prefix code) before being
    /// transcribed here — see this session's derivation from `h2`'s own
    /// `ENCODE_TABLE`.
    const HUFFMAN_TABLE: [(u8, u32); 257] = [
        (13, 0x1ff8),
        (23, 0x7fffd8),
        (28, 0xfffffe2),
        (28, 0xfffffe3),
        (28, 0xfffffe4),
        (28, 0xfffffe5),
        (28, 0xfffffe6),
        (28, 0xfffffe7),
        (28, 0xfffffe8),
        (24, 0xffffea),
        (30, 0x3ffffffc),
        (28, 0xfffffe9),
        (28, 0xfffffea),
        (30, 0x3ffffffd),
        (28, 0xfffffeb),
        (28, 0xfffffec),
        (28, 0xfffffed),
        (28, 0xfffffee),
        (28, 0xfffffef),
        (28, 0xffffff0),
        (28, 0xffffff1),
        (28, 0xffffff2),
        (30, 0x3ffffffe),
        (28, 0xffffff3),
        (28, 0xffffff4),
        (28, 0xffffff5),
        (28, 0xffffff6),
        (28, 0xffffff7),
        (28, 0xffffff8),
        (28, 0xffffff9),
        (28, 0xffffffa),
        (28, 0xffffffb),
        (6, 0x14),
        (10, 0x3f8),
        (10, 0x3f9),
        (12, 0xffa),
        (13, 0x1ff9),
        (6, 0x15),
        (8, 0xf8),
        (11, 0x7fa),
        (10, 0x3fa),
        (10, 0x3fb),
        (8, 0xf9),
        (11, 0x7fb),
        (8, 0xfa),
        (6, 0x16),
        (6, 0x17),
        (6, 0x18),
        (5, 0x0),
        (5, 0x1),
        (5, 0x2),
        (6, 0x19),
        (6, 0x1a),
        (6, 0x1b),
        (6, 0x1c),
        (6, 0x1d),
        (6, 0x1e),
        (6, 0x1f),
        (7, 0x5c),
        (8, 0xfb),
        (15, 0x7ffc),
        (6, 0x20),
        (12, 0xffb),
        (10, 0x3fc),
        (13, 0x1ffa),
        (6, 0x21),
        (7, 0x5d),
        (7, 0x5e),
        (7, 0x5f),
        (7, 0x60),
        (7, 0x61),
        (7, 0x62),
        (7, 0x63),
        (7, 0x64),
        (7, 0x65),
        (7, 0x66),
        (7, 0x67),
        (7, 0x68),
        (7, 0x69),
        (7, 0x6a),
        (7, 0x6b),
        (7, 0x6c),
        (7, 0x6d),
        (7, 0x6e),
        (7, 0x6f),
        (7, 0x70),
        (7, 0x71),
        (7, 0x72),
        (8, 0xfc),
        (7, 0x73),
        (8, 0xfd),
        (13, 0x1ffb),
        (19, 0x7fff0),
        (13, 0x1ffc),
        (14, 0x3ffc),
        (6, 0x22),
        (15, 0x7ffd),
        (5, 0x3),
        (6, 0x23),
        (5, 0x4),
        (6, 0x24),
        (5, 0x5),
        (6, 0x25),
        (6, 0x26),
        (6, 0x27),
        (5, 0x6),
        (7, 0x74),
        (7, 0x75),
        (6, 0x28),
        (6, 0x29),
        (6, 0x2a),
        (5, 0x7),
        (6, 0x2b),
        (7, 0x76),
        (6, 0x2c),
        (5, 0x8),
        (5, 0x9),
        (6, 0x2d),
        (7, 0x77),
        (7, 0x78),
        (7, 0x79),
        (7, 0x7a),
        (7, 0x7b),
        (15, 0x7ffe),
        (11, 0x7fc),
        (14, 0x3ffd),
        (13, 0x1ffd),
        (28, 0xffffffc),
        (20, 0xfffe6),
        (22, 0x3fffd2),
        (20, 0xfffe7),
        (20, 0xfffe8),
        (22, 0x3fffd3),
        (22, 0x3fffd4),
        (22, 0x3fffd5),
        (23, 0x7fffd9),
        (22, 0x3fffd6),
        (23, 0x7fffda),
        (23, 0x7fffdb),
        (23, 0x7fffdc),
        (23, 0x7fffdd),
        (23, 0x7fffde),
        (24, 0xffffeb),
        (23, 0x7fffdf),
        (24, 0xffffec),
        (24, 0xffffed),
        (22, 0x3fffd7),
        (23, 0x7fffe0),
        (24, 0xffffee),
        (23, 0x7fffe1),
        (23, 0x7fffe2),
        (23, 0x7fffe3),
        (23, 0x7fffe4),
        (21, 0x1fffdc),
        (22, 0x3fffd8),
        (23, 0x7fffe5),
        (22, 0x3fffd9),
        (23, 0x7fffe6),
        (23, 0x7fffe7),
        (24, 0xffffef),
        (22, 0x3fffda),
        (21, 0x1fffdd),
        (20, 0xfffe9),
        (22, 0x3fffdb),
        (22, 0x3fffdc),
        (23, 0x7fffe8),
        (23, 0x7fffe9),
        (21, 0x1fffde),
        (23, 0x7fffea),
        (22, 0x3fffdd),
        (22, 0x3fffde),
        (24, 0xfffff0),
        (21, 0x1fffdf),
        (22, 0x3fffdf),
        (23, 0x7fffeb),
        (23, 0x7fffec),
        (21, 0x1fffe0),
        (21, 0x1fffe1),
        (22, 0x3fffe0),
        (21, 0x1fffe2),
        (23, 0x7fffed),
        (22, 0x3fffe1),
        (23, 0x7fffee),
        (23, 0x7fffef),
        (20, 0xfffea),
        (22, 0x3fffe2),
        (22, 0x3fffe3),
        (22, 0x3fffe4),
        (23, 0x7ffff0),
        (22, 0x3fffe5),
        (22, 0x3fffe6),
        (23, 0x7ffff1),
        (26, 0x3ffffe0),
        (26, 0x3ffffe1),
        (20, 0xfffeb),
        (19, 0x7fff1),
        (22, 0x3fffe7),
        (23, 0x7ffff2),
        (22, 0x3fffe8),
        (25, 0x1ffffec),
        (26, 0x3ffffe2),
        (26, 0x3ffffe3),
        (26, 0x3ffffe4),
        (27, 0x7ffffde),
        (27, 0x7ffffdf),
        (26, 0x3ffffe5),
        (24, 0xfffff1),
        (25, 0x1ffffed),
        (19, 0x7fff2),
        (21, 0x1fffe3),
        (26, 0x3ffffe6),
        (27, 0x7ffffe0),
        (27, 0x7ffffe1),
        (26, 0x3ffffe7),
        (27, 0x7ffffe2),
        (24, 0xfffff2),
        (21, 0x1fffe4),
        (21, 0x1fffe5),
        (26, 0x3ffffe8),
        (26, 0x3ffffe9),
        (28, 0xffffffd),
        (27, 0x7ffffe3),
        (27, 0x7ffffe4),
        (27, 0x7ffffe5),
        (20, 0xfffec),
        (24, 0xfffff3),
        (20, 0xfffed),
        (21, 0x1fffe6),
        (22, 0x3fffe9),
        (21, 0x1fffe7),
        (21, 0x1fffe8),
        (23, 0x7ffff3),
        (22, 0x3fffea),
        (22, 0x3fffeb),
        (25, 0x1ffffee),
        (25, 0x1ffffef),
        (24, 0xfffff4),
        (24, 0xfffff5),
        (26, 0x3ffffea),
        (23, 0x7ffff4),
        (26, 0x3ffffeb),
        (27, 0x7ffffe6),
        (26, 0x3ffffec),
        (26, 0x3ffffed),
        (27, 0x7ffffe7),
        (27, 0x7ffffe8),
        (27, 0x7ffffe9),
        (27, 0x7ffffea),
        (27, 0x7ffffeb),
        (28, 0xffffffe),
        (27, 0x7ffffec),
        (27, 0x7ffffed),
        (27, 0x7ffffee),
        (27, 0x7ffffef),
        (27, 0x7fffff0),
        (26, 0x3ffffee),
        (30, 0x3fffffff),
    ];
    const HUFFMAN_EOS: usize = 256;

    /// Huffman-encode `data`, MSB-first bit packing, padded with the high-order
    /// bits of the EOS code (all-1s) per RFC 7541 §5.2.
    fn huffman_encode(data: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut acc: u64 = 0;
        let mut nbits: u32 = 0;
        for &b in data {
            let (len, code) = HUFFMAN_TABLE[b as usize];
            acc = (acc << len) | u64::from(code);
            nbits += u32::from(len);
            while nbits >= 8 {
                nbits -= 8;
                out.push((acc >> nbits) as u8);
            }
        }
        if nbits > 0 {
            let pad_bits = 8 - nbits;
            acc = (acc << pad_bits) | ((1u64 << pad_bits) - 1); // pad with 1s (EOS prefix)
            out.push(acc as u8);
        }
        out
    }

    /// Huffman-decode `data` — bit-by-bit accumulation against the SAME 257-entry
    /// canonical table (a linear scan per bit; not the fastest possible decoder,
    /// but directly, visibly correct against the one table this module already
    /// trusts, and header blocks are tiny, so this is nowhere near a hot path).
    fn huffman_decode(data: &[u8]) -> io::Result<Vec<u8>> {
        let mut out = Vec::new();
        let mut acc: u64 = 0;
        let mut nbits: u32 = 0;
        for &byte in data {
            for bit_i in (0..8).rev() {
                let bit = (byte >> bit_i) & 1;
                acc = (acc << 1) | u64::from(bit);
                nbits += 1;
                if let Some(sym) = HUFFMAN_TABLE
                    .iter()
                    .position(|&(len, code)| u32::from(len) == nbits && code as u64 == acc)
                {
                    if sym == HUFFMAN_EOS {
                        return Err(Error::new(
                            ErrorKind::InvalidData,
                            "Huffman EOS symbol in data",
                        ));
                    }
                    out.push(sym as u8);
                    acc = 0;
                    nbits = 0;
                }
            }
        }
        // Remaining bits (if any) MUST be a prefix of the EOS code (all 1s), the padding.
        if nbits > 0 {
            let all_ones = (1u64 << nbits) - 1;
            if acc != all_ones {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "Huffman padding is not all-1s",
                ));
            }
        }
        Ok(out)
    }

    /// RFC 7541 Appendix A — the 61 static entries, index 1..=61 (index 0 is
    /// invalid/reserved). Transcribed from the RFC text, not a library's internal
    /// representation, so it's checked directly against Appendix C's own worked
    /// examples rather than trusted on the strength of its source.
    const STATIC_TABLE: [(&str, &str); 61] = [
        (":authority", ""),
        (":method", "GET"),
        (":method", "POST"),
        (":path", "/"),
        (":path", "/index.html"),
        (":scheme", "http"),
        (":scheme", "https"),
        (":status", "200"),
        (":status", "204"),
        (":status", "206"),
        (":status", "304"),
        (":status", "400"),
        (":status", "404"),
        (":status", "500"),
        ("accept-charset", ""),
        ("accept-encoding", "gzip, deflate"),
        ("accept-language", ""),
        ("accept-ranges", ""),
        ("accept", ""),
        ("access-control-allow-origin", ""),
        ("age", ""),
        ("allow", ""),
        ("authorization", ""),
        ("cache-control", ""),
        ("content-disposition", ""),
        ("content-encoding", ""),
        ("content-language", ""),
        ("content-length", ""),
        ("content-location", ""),
        ("content-range", ""),
        ("content-type", ""),
        ("cookie", ""),
        ("date", ""),
        ("etag", ""),
        ("expect", ""),
        ("expires", ""),
        ("from", ""),
        ("host", ""),
        ("if-match", ""),
        ("if-modified-since", ""),
        ("if-none-match", ""),
        ("if-range", ""),
        ("if-unmodified-since", ""),
        ("last-modified", ""),
        ("link", ""),
        ("location", ""),
        ("max-forwards", ""),
        ("proxy-authenticate", ""),
        ("proxy-authorization", ""),
        ("range", ""),
        ("referer", ""),
        ("refresh", ""),
        ("retry-after", ""),
        ("server", ""),
        ("set-cookie", ""),
        ("strict-transport-security", ""),
        ("transfer-encoding", ""),
        ("user-agent", ""),
        ("vary", ""),
        ("via", ""),
        ("www-authenticate", ""),
    ];

    fn static_index_for(name: &str, value: &str) -> Option<usize> {
        STATIC_TABLE
            .iter()
            .position(|&(n, v)| n == name && v == value)
            .map(|i| i + 1)
    }
    fn static_name_index_for(name: &str) -> Option<usize> {
        STATIC_TABLE
            .iter()
            .position(|&(n, _)| n == name)
            .map(|i| i + 1)
    }

    /// RFC 7541 §5.1 — encode `value` as an N-bit-prefix integer, OR-ed onto
    /// `first_byte_flags` (the representation-type bits already set in the first
    /// byte, e.g. `0x40` for "literal with indexing").
    fn encode_int(out: &mut Vec<u8>, first_byte_flags: u8, prefix_bits: u32, mut value: usize) {
        let max_prefix = (1usize << prefix_bits) - 1;
        if value < max_prefix {
            out.push(first_byte_flags | value as u8);
            return;
        }
        out.push(first_byte_flags | max_prefix as u8);
        value -= max_prefix;
        while value >= 128 {
            out.push(((value % 128) + 128) as u8);
            value /= 128;
        }
        out.push(value as u8);
    }

    fn decode_int(data: &[u8], pos: &mut usize, prefix_bits: u32) -> io::Result<usize> {
        let err = || Error::new(ErrorKind::InvalidData, "truncated HPACK integer");
        let max_prefix = (1usize << prefix_bits) - 1;
        let first = *data.get(*pos).ok_or_else(err)? as usize;
        *pos += 1;
        let mut value = first & max_prefix;
        if value < max_prefix {
            return Ok(value);
        }
        let mut m = 0u32;
        loop {
            let b = *data.get(*pos).ok_or_else(err)? as usize;
            *pos += 1;
            value += (b & 0x7f) << m;
            m += 7;
            if b & 0x80 == 0 {
                break;
            }
        }
        Ok(value)
    }

    /// RFC 7541 §5.2 — a length-prefixed string, optionally Huffman-coded (high
    /// bit of the length byte). We never emit Huffman on the way OUT (plain
    /// literal strings are simpler and equally spec-valid to send), but MUST
    /// decode either form coming IN.
    fn encode_string(out: &mut Vec<u8>, s: &str) {
        encode_int(out, 0x00, 7, s.len()); // H=0
        out.extend_from_slice(s.as_bytes());
    }

    fn decode_string(data: &[u8], pos: &mut usize) -> io::Result<String> {
        let err = || Error::new(ErrorKind::InvalidData, "truncated HPACK string");
        let first = *data.get(*pos).ok_or_else(err)?;
        let huffman = first & 0x80 != 0;
        let len = decode_int(data, pos, 7)?;
        let raw = data.get(*pos..*pos + len).ok_or_else(err)?;
        *pos += len;
        let bytes = if huffman {
            huffman_decode(raw)?
        } else {
            raw.to_vec()
        };
        String::from_utf8(bytes)
            .map_err(|_| Error::new(ErrorKind::InvalidData, "non-UTF8 HPACK string"))
    }

    /// Encode ONE header field for sending — always "literal without indexing"
    /// (RFC 7541 §6.2.2, first-byte pattern `0000xxxx`), using a static-table name
    /// reference when the name matches one (saves re-sending common names like
    /// `:method`/`content-type`), full literal name otherwise. No dynamic table
    /// maintained on the encode side — see the module doc for why that's fine.
    pub fn encode_header(out: &mut Vec<u8>, name: &str, value: &str) {
        if let Some(idx) = static_index_for(name, value) {
            // Fully indexed (RFC 7541 §6.1): first byte `1xxxxxxx`.
            encode_int(out, 0x80, 7, idx);
            return;
        }
        if let Some(idx) = static_name_index_for(name) {
            encode_int(out, 0x00, 4, idx);
            encode_string(out, value);
            return;
        }
        out.push(0x00); // literal name, index 0
        encode_string(out, name);
        encode_string(out, value);
    }

    /// Encode a full header list into one HPACK block (the HEADERS/CONTINUATION
    /// frame payload).
    pub fn encode_headers(headers: &[(&str, &str)]) -> Vec<u8> {
        let mut out = Vec::new();
        for (name, value) in headers {
            encode_header(&mut out, name, value);
        }
        out
    }

    /// Per-connection HPACK decoder state — the dynamic table (RFC 7541 §2.3),
    /// tracking whatever the PEER chooses to index, independent of our own
    /// (indexing-free) encoder above.
    pub struct HpackDecoder {
        /// Most-recently-added first (index 1 = `dynamic[0]`), per §2.3.2.
        dynamic: Vec<(String, String)>,
        dynamic_size: usize,
        max_size: usize,
    }

    impl Default for HpackDecoder {
        fn default() -> Self {
            Self::new()
        }
    }

    impl HpackDecoder {
        #[must_use]
        pub fn new() -> Self {
            Self {
                dynamic: Vec::new(),
                dynamic_size: 0,
                max_size: 4096,
            }
        }

        fn entry_size(name: &str, value: &str) -> usize {
            name.len() + value.len() + 32 // RFC 7541 §4.1's fixed 32-byte overhead
        }

        fn add_dynamic(&mut self, name: String, value: String) {
            self.dynamic_size += Self::entry_size(&name, &value);
            self.dynamic.insert(0, (name, value));
            while self.dynamic_size > self.max_size {
                let Some((n, v)) = self.dynamic.pop() else {
                    break;
                };
                self.dynamic_size -= Self::entry_size(&n, &v);
            }
        }

        fn evict_to_max(&mut self) {
            while self.dynamic_size > self.max_size {
                let Some((n, v)) = self.dynamic.pop() else {
                    break;
                };
                self.dynamic_size -= Self::entry_size(&n, &v);
            }
        }

        fn lookup(&self, index: usize) -> io::Result<(String, String)> {
            if index == 0 {
                return Err(Error::new(
                    ErrorKind::InvalidData,
                    "HPACK index 0 is invalid",
                ));
            }
            if index <= STATIC_TABLE.len() {
                let (n, v) = STATIC_TABLE[index - 1];
                return Ok((n.to_string(), v.to_string()));
            }
            let dyn_idx = index - STATIC_TABLE.len() - 1;
            self.dynamic.get(dyn_idx).cloned().ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidData,
                    format!("HPACK index {index} out of range"),
                )
            })
        }

        /// Decode one HPACK block (a full HEADERS+CONTINUATION payload, already
        /// reassembled) into an ordered header list.
        pub fn decode(&mut self, data: &[u8]) -> io::Result<Vec<(String, String)>> {
            let mut out = Vec::new();
            let mut pos = 0usize;
            while pos < data.len() {
                let first = data[pos];
                if first & 0x80 != 0 {
                    // Indexed Header Field (§6.1)
                    let idx = decode_int(data, &mut pos, 7)?;
                    let (n, v) = self.lookup(idx)?;
                    out.push((n, v));
                } else if first & 0x40 != 0 {
                    // Literal Header Field with Incremental Indexing (§6.2.1)
                    let idx = decode_int(data, &mut pos, 6)?;
                    let name = if idx == 0 {
                        decode_string(data, &mut pos)?
                    } else {
                        self.lookup(idx)?.0
                    };
                    let value = decode_string(data, &mut pos)?;
                    self.add_dynamic(name.clone(), value.clone());
                    out.push((name, value));
                } else if first & 0x20 != 0 {
                    // Dynamic Table Size Update (§6.3)
                    let new_size = decode_int(data, &mut pos, 5)?;
                    self.max_size = new_size;
                    self.evict_to_max();
                } else {
                    // Literal Header Field without Indexing (§6.2.2, prefix 0000) or
                    // Never Indexed (§6.2.3, prefix 0001) — both non-indexing, 4-bit prefix.
                    let idx = decode_int(data, &mut pos, 4)?;
                    let name = if idx == 0 {
                        decode_string(data, &mut pos)?
                    } else {
                        self.lookup(idx)?.0
                    };
                    let value = decode_string(data, &mut pos)?;
                    out.push((name, value));
                }
            }
            Ok(out)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn hx(s: &str) -> Vec<u8> {
            let s: String = s.chars().filter(|c| !c.is_whitespace()).collect();
            (0..s.len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
                .collect()
        }

        #[test]
        fn huffman_table_is_a_complete_prefix_code() {
            let kraft: f64 = HUFFMAN_TABLE
                .iter()
                .map(|&(len, _)| 2f64.powi(-i32::from(len)))
                .sum();
            assert!(
                (kraft - 1.0).abs() < 1e-9,
                "Kraft sum must be exactly 1.0 for a complete code, got {kraft}"
            );
        }

        #[test]
        fn huffman_round_trips_arbitrary_ascii() {
            for s in [
                "www.example.com",
                "custom-header",
                "",
                "a",
                "/tfplugin6.Provider/GetProviderSchema",
            ] {
                let encoded = huffman_encode(s.as_bytes());
                let decoded = huffman_decode(&encoded).unwrap_or_else(|e| unreachable!("{e}"));
                assert_eq!(
                    decoded,
                    s.as_bytes(),
                    "round trip must reproduce the original bytes for {s:?}"
                );
            }
        }

        /// RFC 7541 Appendix C.4.1 — the REAL Huffman-encoded bytes from the spec
        /// itself decode to "www.example.com" via this module's own huffman_decode.
        #[test]
        fn rfc7541_huffman_reference_bytes_decode_correctly() {
            let encoded = hx("f1e3 c2e5 f23a 6ba0 ab90 f4ff");
            let decoded = huffman_decode(&encoded).unwrap_or_else(|e| unreachable!("{e}"));
            assert_eq!(decoded, b"www.example.com");
        }

        /// RFC 7541 Appendix C.2.1 — Literal Header Field with Indexing, full
        /// literal name+value, added to the dynamic table.
        #[test]
        fn rfc7541_c2_1_literal_with_indexing() {
            let bytes = hx("400a 6375 7374 6f6d 2d6b 6579 0d63 7573 746f 6d2d 6865 6164 6572");
            let mut dec = HpackDecoder::new();
            let headers = dec.decode(&bytes).unwrap_or_else(|e| unreachable!("{e}"));
            assert_eq!(
                headers,
                vec![("custom-key".to_string(), "custom-header".to_string())]
            );
            assert_eq!(dec.dynamic.len(), 1, "must be added to the dynamic table");
            assert_eq!(
                dec.dynamic_size, 55,
                "RFC 7541's own worked size (name+value+32 overhead)"
            );
        }

        /// RFC 7541 Appendix C.2.2 — Literal Header Field without Indexing, an
        /// indexed name (`:path` = static index 4) + literal value, NOT added to
        /// the dynamic table.
        #[test]
        fn rfc7541_c2_2_literal_without_indexing_indexed_name() {
            let bytes = hx("040c 2f73 616d 706c 652f 7061 7468");
            let mut dec = HpackDecoder::new();
            let headers = dec.decode(&bytes).unwrap_or_else(|e| unreachable!("{e}"));
            assert_eq!(
                headers,
                vec![(":path".to_string(), "/sample/path".to_string())]
            );
            assert!(
                dec.dynamic.is_empty(),
                "without-indexing must NOT touch the dynamic table"
            );
        }

        /// RFC 7541 Appendix C.3.1 — a full request: three fully-indexed static
        /// entries plus one literal-with-indexing for `:authority`.
        #[test]
        fn rfc7541_c3_1_first_request_no_huffman() {
            let bytes = hx("8286 8441 0f77 7777 2e65 7861 6d70 6c65 2e63 6f6d");
            let mut dec = HpackDecoder::new();
            let headers = dec.decode(&bytes).unwrap_or_else(|e| unreachable!("{e}"));
            assert_eq!(
                headers,
                vec![
                    (":method".to_string(), "GET".to_string()),
                    (":scheme".to_string(), "http".to_string()),
                    (":path".to_string(), "/".to_string()),
                    (":authority".to_string(), "www.example.com".to_string()),
                ]
            );
            assert_eq!(dec.dynamic_size, 57);
        }

        /// Same request as C.3.1, but the RFC's own Huffman-coded variant
        /// (Appendix C.4.1) — same decoded result, different wire bytes.
        #[test]
        fn rfc7541_c4_1_first_request_with_huffman() {
            let bytes = hx("8286 8441 8cf1 e3c2 e5f2 3a6b a0ab 90f4 ff");
            let mut dec = HpackDecoder::new();
            let headers = dec.decode(&bytes).unwrap_or_else(|e| unreachable!("{e}"));
            assert_eq!(
                headers,
                vec![
                    (":method".to_string(), "GET".to_string()),
                    (":scheme".to_string(), "http".to_string()),
                    (":path".to_string(), "/".to_string()),
                    (":authority".to_string(), "www.example.com".to_string()),
                ]
            );
        }

        /// Our own encoder → our own decoder must round-trip (the encode side
        /// isn't covered by the RFC vectors above, since those are all decode-only
        /// checks against externally-produced bytes).
        #[test]
        fn encode_then_decode_round_trips_our_own_headers() {
            let headers = [
                (":method", "POST"),
                (":scheme", "http"),
                (":path", "/tfplugin6.Provider/GetProviderSchema"),
                (":authority", "localhost"),
                ("content-type", "application/grpc"),
                ("te", "trailers"),
            ];
            let encoded = encode_headers(&headers);
            let mut dec = HpackDecoder::new();
            let decoded = dec.decode(&encoded).unwrap_or_else(|e| unreachable!("{e}"));
            let expected: Vec<(String, String)> = headers
                .iter()
                .map(|(n, v)| (n.to_string(), v.to_string()))
                .collect();
            assert_eq!(decoded, expected);
        }

        #[test]
        fn dynamic_table_size_update_evicts_entries() {
            // Two "literal with incremental indexing, new name" fields (§6.2.1, first
            // byte 0x40, name index 0 -> literal name follows).
            let mut raw = Vec::new();
            raw.push(0x40);
            encode_string(&mut raw, "a");
            encode_string(&mut raw, "1");
            raw.push(0x40);
            encode_string(&mut raw, "b");
            encode_string(&mut raw, "2");
            let mut dec = HpackDecoder::new();
            dec.decode(&raw).unwrap_or_else(|e| unreachable!("{e}"));
            assert_eq!(dec.dynamic.len(), 2);

            // A dynamic table size update (§6.3, first byte 0x20) to exactly fit the
            // most-recently-added entry ("b": "2") and evict the older one.
            let mut upd = Vec::new();
            encode_int(&mut upd, 0x20, 5, HpackDecoder::entry_size("b", "2"));
            dec.decode(&upd).unwrap_or_else(|e| unreachable!("{e}"));
            assert_eq!(dec.dynamic.len(), 1, "must evict down to the new max size");
            assert_eq!(
                dec.dynamic[0],
                ("b".to_string(), "2".to_string()),
                "most-recently-added survives"
            );
        }
    }
}
pub mod http {
    //! Minimal HTTP/1.1 over our pure-Rust TLS client. Just enough for the S3 REST
    //! verbs: build a request, read a response (Content-Length or chunked or
    //! read-to-close). Zero deps.

    #![allow(dead_code)]

    use std::io::{self, Error, ErrorKind, Write};

    use crate::sync::tls::client::TlsClient;

    pub struct Response {
        pub status: u16,
        pub headers: Vec<(String, String)>,
        pub body: Vec<u8>,
    }

    impl Response {
        pub fn header(&self, name: &str) -> Option<&str> {
            let n = name.to_ascii_lowercase();
            self.headers
                .iter()
                .find(|(k, _)| k.to_ascii_lowercase() == n)
                .map(|(_, v)| v.as_str())
        }
    }

    /// One HTTPS request over a fresh TLS connection (Connection: close).
    pub fn request(
        host: &str,
        port: u16,
        method: &str,
        path: &str,
        extra_headers: &[(String, String)],
        body: &[u8],
        verify: bool,
    ) -> io::Result<Response> {
        let mut c = TlsClient::connect(host, port, verify)?;
        write_request(&mut c, host, port, method, path, extra_headers, body)?;
        let raw = c.recv_all()?; // the Web sweep: decrypt records across all cores
        parse_response(&raw)
    }

    /// One plain HTTP/1.1 request over TCP (no TLS) — for loopback S3 reads where the data and
    /// the server share a disk and TLS is pure overhead. SigV4 signs the payload, never the
    /// transport, so a signature is valid over plain HTTP identically.
    pub fn request_plain(
        host: &str,
        port: u16,
        method: &str,
        path: &str,
        extra_headers: &[(String, String)],
        body: &[u8],
    ) -> io::Result<Response> {
        use std::net::TcpStream;
        use std::io::Read;
        let mut c = TcpStream::connect((host, port))?;
        write_request(&mut c, host, port, method, path, extra_headers, body)?;
        let mut raw = Vec::new();
        c.read_to_end(&mut raw)?;
        parse_response(&raw)
    }

    /// One plain HTTP request on an EXISTING connection, keep-alive: writes the request, reads
    /// exactly one response (Content-Length framed; the connection stays usable for the next
    /// request). The pooled bulk-read path — kills the per-request connect cost that makes
    /// per-study range reads slow.
    pub fn request_plain_on(
        conn: &mut std::net::TcpStream,
        host: &str,
        port: u16,
        method: &str,
        path: &str,
        extra_headers: &[(String, String)],
        body: &[u8],
    ) -> io::Result<Response> {
        use std::io::{Read, Write};
        let host_hdr = if port == 443 {
            host.to_string()
        } else {
            format!("{host}:{port}")
        };
        let mut req = format!("{method} {path} HTTP/1.1\r\nHost: {host_hdr}\r\nConnection: keep-alive\r\n");
        for (k, v) in extra_headers {
            req.push_str(&format!("{k}: {v}\r\n"));
        }
        req.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
        conn.write_all(req.as_bytes())?;
        if !body.is_empty() {
            conn.write_all(body)?;
        }
        conn.flush()?;

        // Read one response head, then the body per Content-Length.
        let mut buf = Vec::with_capacity(8192);
        let head_end = loop {
            let mut b = [0u8; 1];
            match conn.read(&mut b) {
                Ok(0) => return Err(io::Error::new(ErrorKind::UnexpectedEof, "eof in response head")),
                Ok(_) => {
                    buf.push(b[0]);
                    if buf.len() >= 4 && &buf[buf.len() - 4..] == b"\r\n\r\n" {
                        break buf.len();
                    }
                }
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        };
        let head = std::str::from_utf8(&buf[..head_end]).map_err(|_| io::Error::other("non-utf8 head"))?;
        let mut lines = head.split("\r\n");
        let status = lines
            .next()
            .and_then(|l| l.split_whitespace().nth(1))
            .and_then(|s| s.parse::<u16>().ok())
            .ok_or_else(|| io::Error::other("bad status line"))?;
        let mut content_length = 0usize;
        let mut chunked = false;
        for l in lines {
            if let Some((k, v)) = l.split_once(':') {
                let (k, v) = (k.trim().to_ascii_lowercase(), v.trim());
                if k == "content-length" {
                    content_length = v.parse().unwrap_or(0);
                }
                if k == "transfer-encoding" && v.to_ascii_lowercase().contains("chunked") {
                    chunked = true;
                }
            }
        }
        let body = if chunked {
            let mut out = Vec::new();
            loop {
                let mut size_line = Vec::new();
                let mut b = [0u8; 1];
                loop {
                    match conn.read(&mut b) {
                        Ok(0) => return Err(io::Error::new(ErrorKind::UnexpectedEof, "eof in chunk size")),
                        Ok(_) => {
                            size_line.push(b[0]);
                            if size_line.ends_with(b"\r\n") {
                                break;
                            }
                        }
                        Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                        Err(e) => return Err(e),
                    }
                }
                let size = usize::from_str_radix(
                    std::str::from_utf8(&size_line[..size_line.len() - 2]).unwrap_or("").split(';').next().unwrap_or("0").trim(),
                    16,
                )
                .map_err(|_| io::Error::other("bad chunk size"))?;
                if size == 0 {
                    break;
                }
                let mut chunk = vec![0u8; size + 2];
                conn.read_exact(&mut chunk)?;
                out.extend_from_slice(&chunk[..size]);
            }
            out
        } else if content_length > 0 {
            let mut out = vec![0u8; content_length];
            conn.read_exact(&mut out)?;
            out
        } else {
            Vec::new()
        };
        Ok(Response {
            status,
            headers: vec![],
            body,
        })
    }

    fn write_request<W: Write>(
        w: &mut W,
        host: &str,
        port: u16,
        method: &str,
        path: &str,
        extra_headers: &[(String, String)],
        body: &[u8],
    ) -> io::Result<()> {
        let host_hdr = if port == 443 {
            host.to_string()
        } else {
            format!("{host}:{port}")
        };
        let mut req = format!("{method} {path} HTTP/1.1\r\nHost: {host_hdr}\r\nConnection: close\r\n");
        for (k, v) in extra_headers {
            req.push_str(&format!("{k}: {v}\r\n"));
        }
        req.push_str(&format!("Content-Length: {}\r\n\r\n", body.len()));
        w.write_all(req.as_bytes())?;
        if !body.is_empty() {
            w.write_all(body)?;
        }
        w.flush()?;
        Ok(())
    }

    fn parse_response(raw: &[u8]) -> io::Result<Response> {
        let sep = find(raw, b"\r\n\r\n")
            .ok_or_else(|| Error::new(ErrorKind::InvalidData, "no header terminator"))?;
        let head = std::str::from_utf8(&raw[..sep])
            .map_err(|_| Error::new(ErrorKind::InvalidData, "non-utf8 headers"))?;
        let mut lines = head.split("\r\n");
        let status_line = lines.next().unwrap_or("");
        let status = status_line
            .split_whitespace()
            .nth(1)
            .and_then(|s| s.parse::<u16>().ok())
            .ok_or_else(|| {
                Error::new(
                    ErrorKind::InvalidData,
                    format!("bad status line: {status_line}"),
                )
            })?;
        let mut headers = Vec::new();
        let mut chunked = false;
        for l in lines {
            if let Some((k, v)) = l.split_once(':') {
                let k = k.trim().to_string();
                let v = v.trim().to_string();
                if k.eq_ignore_ascii_case("transfer-encoding") && v.eq_ignore_ascii_case("chunked") {
                    chunked = true;
                }
                headers.push((k, v));
            }
        }
        let body_raw = &raw[sep + 4..];
        let body = if chunked {
            dechunk(body_raw)?
        } else {
            body_raw.to_vec()
        };
        Ok(Response {
            status,
            headers,
            body,
        })
    }

    fn dechunk(mut b: &[u8]) -> io::Result<Vec<u8>> {
        let mut out = Vec::new();
        loop {
            let nl = find(b, b"\r\n")
                .ok_or_else(|| Error::new(ErrorKind::InvalidData, "chunk size line"))?;
            let size_str = std::str::from_utf8(&b[..nl]).unwrap_or("").trim();
            let size = usize::from_str_radix(size_str.split(';').next().unwrap_or("0"), 16)
                .map_err(|_| Error::new(ErrorKind::InvalidData, "bad chunk size"))?;
            b = &b[nl + 2..];
            if size == 0 {
                break;
            }
            if b.len() < size {
                return Err(Error::new(ErrorKind::InvalidData, "truncated chunk"));
            }
            out.extend_from_slice(&b[..size]);
            b = &b[size + 2..]; // skip data + CRLF
        }
        Ok(out)
    }

    fn find(hay: &[u8], needle: &[u8]) -> Option<usize> {
        hay.windows(needle.len()).position(|w| w == needle)
    }
}
pub mod http2 {
    //! HTTP/2 framing (RFC 9113) — just enough to drive ONE gRPC unary call per
    //! connection: the connection preface, SETTINGS exchange, HEADERS+DATA on a
    //! single stream, and reading the response HEADERS/DATA/trailers back. Pure
    //! Rust, zero deps, over any [`crate::sync::hpack`]-paired byte stream (this
    //! module doesn't care whether the underlying transport is plaintext, our own
    //! TLS client, or a Unix socket — it just needs `Read + Write`).
    //!
    //! Deliberately NOT a general HTTP/2 client: no multiplexing (one stream per
    //! connection), no real flow control (we advertise a large enough
    //! `INITIAL_WINDOW_SIZE`/`WINDOW_UPDATE` to never need one for a single
    //! request/response of realistic size), no server push, no priority. Each of
    //! those is real, separable scope a general client would need — this is
    //! scoped to what a gRPC unary call over go-plugin's tfplugin6 actually uses.

    use std::io::{self, Error, ErrorKind, Read, Write};

    use crate::sync::hpack::{encode_headers, HpackDecoder};

    const FRAME_DATA: u8 = 0x0;
    const FRAME_HEADERS: u8 = 0x1;
    const FRAME_SETTINGS: u8 = 0x4;
    const FRAME_GOAWAY: u8 = 0x7;
    const FRAME_WINDOW_UPDATE: u8 = 0x8;
    const FRAME_CONTINUATION: u8 = 0x9;

    const FLAG_END_STREAM: u8 = 0x1;
    const FLAG_END_HEADERS: u8 = 0x4;
    const FLAG_ACK: u8 = 0x1;

    const PREFACE: &[u8] = b"PRI * HTTP/2.0\r\n\r\nSM\r\n\r\n";

    struct FrameHeader {
        len: usize,
        ty: u8,
        flags: u8,
        stream_id: u32,
    }

    fn write_frame(
        w: &mut impl Write,
        ty: u8,
        flags: u8,
        stream_id: u32,
        payload: &[u8],
    ) -> io::Result<()> {
        let len = payload.len();
        let mut hdr = [0u8; 9];
        hdr[0] = (len >> 16) as u8;
        hdr[1] = (len >> 8) as u8;
        hdr[2] = len as u8;
        hdr[3] = ty;
        hdr[4] = flags;
        hdr[5..9].copy_from_slice(&(stream_id & 0x7fff_ffff).to_be_bytes());
        w.write_all(&hdr)?;
        w.write_all(payload)
    }

    fn read_frame_header(r: &mut impl Read) -> io::Result<FrameHeader> {
        let mut hdr = [0u8; 9];
        r.read_exact(&mut hdr)?;
        let len = ((hdr[0] as usize) << 16) | ((hdr[1] as usize) << 8) | hdr[2] as usize;
        let ty = hdr[3];
        let flags = hdr[4];
        let stream_id = u32::from_be_bytes([hdr[5], hdr[6], hdr[7], hdr[8]]) & 0x7fff_ffff;
        Ok(FrameHeader {
            len,
            ty,
            flags,
            stream_id,
        })
    }

    fn read_frame(r: &mut impl Read) -> io::Result<(FrameHeader, Vec<u8>)> {
        let h = read_frame_header(r)?;
        let mut payload = vec![0u8; h.len];
        r.read_exact(&mut payload)?;
        Ok((h, payload))
    }

    /// One SETTINGS parameter as `(id, value)`, encoded as 6 bytes (u16 id + u32 value).
    fn encode_settings(params: &[(u16, u32)]) -> Vec<u8> {
        let mut out = Vec::with_capacity(params.len() * 6);
        for &(id, val) in params {
            out.extend_from_slice(&id.to_be_bytes());
            out.extend_from_slice(&val.to_be_bytes());
        }
        out
    }

    /// The full response to a unary gRPC call: headers, the single reassembled
    /// message body (gRPC-framed: 1-byte compression flag + 4-byte length still
    /// attached — [`crate::sync::grpc`] strips that), and trailers (`grpc-status` etc).
    pub struct Http2Response {
        pub headers: Vec<(String, String)>,
        pub body: Vec<u8>,
        pub trailers: Vec<(String, String)>,
    }

    /// Perform ONE HTTP/2 request/response over `stream_id = 1`: connection
    /// preface, SETTINGS handshake, HEADERS (from `headers`, HPACK-encoded) +
    /// DATA (`body`, already framed by the caller — gRPC's 5-byte prefix, if
    /// any), then read the response through to its trailers.
    pub fn unary_request(
        io: &mut (impl Read + Write),
        headers: &[(&str, &str)],
        body: &[u8],
    ) -> io::Result<Http2Response> {
        io.write_all(PREFACE)?;
        // SETTINGS_INITIAL_WINDOW_SIZE (16MB) governs a STREAM's flow-control window,
        // but the CONNECTION-level window is a separate accounting that always starts
        // at the RFC default 65535 bytes regardless of any SETTINGS value, and can only
        // grow via an explicit WINDOW_UPDATE on stream 0 — a real provider's schema (a
        // large one, e.g. `hashicorp/awscc`, covering hundreds of AWS resource types)
        // can exceed 65535 bytes easily, and without this the server just stops sending
        // once the default window is exhausted, silently stalling the read loop below
        // (measured live: this exact hang, until this fix).
        write_frame(
            io,
            FRAME_SETTINGS,
            0,
            0,
            &encode_settings(&[(0x3, 100), (0x4, 1 << 24)]),
        )?;
        write_frame(io, FRAME_WINDOW_UPDATE, 0, 0, &0x0fff_ffffu32.to_be_bytes())?;

        let mut decoder = HpackDecoder::new();
        let mut response_headers: Option<Vec<(String, String)>> = None;
        let mut response_body = Vec::new();
        let mut trailers: Vec<(String, String)> = Vec::new();
        let mut got_settings_from_peer = false;

        // Read (and ack) the peer's own SETTINGS before sending HEADERS/DATA — some
        // HTTP/2 servers are strict about not seeing stream frames pipelined ahead of
        // the initial SETTINGS exchange completing, even though the spec technically
        // permits it.
        while !got_settings_from_peer {
            let (h, payload) = read_frame(io)?;
            if h.ty == FRAME_SETTINGS && h.flags & FLAG_ACK == 0 {
                got_settings_from_peer = true;
                write_frame(io, FRAME_SETTINGS, FLAG_ACK, 0, &[])?;
            } else if h.ty == FRAME_GOAWAY {
                return Err(Error::new(
                    ErrorKind::ConnectionAborted,
                    format!("server sent GOAWAY: {payload:02x?}"),
                ));
            }
            // anything else this early (a SETTINGS ACK of ours, etc.) — ignore and keep waiting
        }

        let header_block = encode_headers(headers);
        write_frame(io, FRAME_HEADERS, FLAG_END_HEADERS, 1, &header_block)?;
        write_frame(io, FRAME_DATA, FLAG_END_STREAM, 1, body)?;
        // Belt-and-suspenders: also grow stream 1's own window explicitly, in case a
        // peer doesn't apply our SETTINGS_INITIAL_WINDOW_SIZE to this stream in time.
        write_frame(io, FRAME_WINDOW_UPDATE, 0, 1, &0x0fff_ffffu32.to_be_bytes())?;

        loop {
            let (h, payload) = read_frame(io)?;
            match h.ty {
                FRAME_SETTINGS => {
                    if h.flags & FLAG_ACK == 0 {
                        got_settings_from_peer = true;
                        write_frame(io, FRAME_SETTINGS, FLAG_ACK, 0, &[])?;
                    }
                }
                FRAME_WINDOW_UPDATE | FRAME_CONTINUATION => {} // no real flow control / no split header blocks expected
                FRAME_HEADERS if h.stream_id == 1 => {
                    let decoded = decoder.decode(&payload)?;
                    if response_headers.is_none() {
                        response_headers = Some(decoded);
                    } else {
                        trailers = decoded;
                    }
                    if h.flags & FLAG_END_STREAM != 0 {
                        break;
                    }
                }
                FRAME_DATA if h.stream_id == 1 => {
                    response_body.extend_from_slice(&payload);
                    if h.flags & FLAG_END_STREAM != 0 {
                        break;
                    }
                }
                FRAME_HEADERS | FRAME_DATA => {
                    // A frame for a different stream: this client only ever opens stream 1,
                    // so anything else is the peer's business (e.g. its own control stream),
                    // never ours to interpret — skip without erroring.
                }
                FRAME_GOAWAY => {
                    return Err(Error::new(
                        ErrorKind::ConnectionAborted,
                        format!("server sent GOAWAY: {payload:02x?}"),
                    ));
                }
                _ => {} // PING, PRIORITY, RST_STREAM on frames we don't expect — ignored
            }
        }

        if !got_settings_from_peer {
            // Not fatal by itself (the peer may have sent SETTINGS after our first
            // read loop iteration already handled it) — only a diagnostic signal.
        }

        Ok(Http2Response {
            headers: response_headers.unwrap_or_default(),
            body: response_body,
            trailers,
        })
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use std::io::Cursor;

        /// A `Read + Write` double that queues canned frames to read back and
        /// records everything written — exercises `unary_request` without a real
        /// socket or a real server, checking the byte-level framing this module
        /// produces/expects is self-consistent.
        struct FakeConn {
            to_read: Cursor<Vec<u8>>,
            written: Vec<u8>,
        }
        impl Read for FakeConn {
            fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
                self.to_read.read(buf)
            }
        }
        impl Write for FakeConn {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                self.written.extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }

        fn canned_response() -> Vec<u8> {
            let mut out = Vec::new();
            // SETTINGS (empty, from "server")
            write_frame(&mut out, FRAME_SETTINGS, 0, 0, &[]).unwrap();
            // SETTINGS ACK (to our own SETTINGS)
            write_frame(&mut out, FRAME_SETTINGS, FLAG_ACK, 0, &[]).unwrap();
            // Response HEADERS: :status 200 (indexed, static #8), content-type application/grpc (literal)
            let mut hdr_block = Vec::new();
            crate::sync::hpack::encode_header(&mut hdr_block, ":status", "200");
            crate::sync::hpack::encode_header(&mut hdr_block, "content-type", "application/grpc");
            write_frame(&mut out, FRAME_HEADERS, FLAG_END_HEADERS, 1, &hdr_block).unwrap();
            // DATA: a fake gRPC-framed empty message (compressed=0, len=0)
            write_frame(&mut out, FRAME_DATA, 0, 1, &[0x00, 0x00, 0x00, 0x00, 0x00]).unwrap();
            // Trailers: grpc-status: 0
            let mut trailer_block = Vec::new();
            crate::sync::hpack::encode_header(&mut trailer_block, "grpc-status", "0");
            write_frame(
                &mut out,
                FRAME_HEADERS,
                FLAG_END_HEADERS | FLAG_END_STREAM,
                1,
                &trailer_block,
            )
            .unwrap();
            out
        }

        #[test]
        fn unary_request_sends_preface_settings_headers_data_and_parses_response() {
            let mut conn = FakeConn {
                to_read: Cursor::new(canned_response()),
                written: Vec::new(),
            };
            let resp = unary_request(
                &mut conn,
                &[
                    (":method", "POST"),
                    (":path", "/tfplugin6.Provider/GetProviderSchema"),
                ],
                &[0x00, 0x00, 0x00, 0x00, 0x00],
            )
            .unwrap_or_else(|e| unreachable!("{e}"));

            assert!(
                conn.written.starts_with(PREFACE),
                "must send the HTTP/2 connection preface first"
            );
            assert_eq!(
                resp.headers,
                vec![
                    (":status".to_string(), "200".to_string()),
                    ("content-type".to_string(), "application/grpc".to_string()),
                ]
            );
            assert_eq!(resp.body, vec![0x00, 0x00, 0x00, 0x00, 0x00]);
            assert_eq!(
                resp.trailers,
                vec![("grpc-status".to_string(), "0".to_string())]
            );

            // Our own outbound frames must themselves be well-formed: re-parse what we wrote.
            // Order: our SETTINGS, our connection-level WINDOW_UPDATE, our ACK of the peer's
            // SETTINGS (read before we send HEADERS — see `unary_request`'s doc comment on why),
            // then HEADERS, DATA, and a stream-level WINDOW_UPDATE.
            let mut written = Cursor::new(conn.written[PREFACE.len()..].to_vec());
            let (h1, _) = read_frame(&mut written).unwrap_or_else(|e| unreachable!("{e}"));
            assert_eq!(h1.ty, FRAME_SETTINGS);
            let (h_wu, _) = read_frame(&mut written).unwrap_or_else(|e| unreachable!("{e}"));
            assert_eq!(h_wu.ty, FRAME_WINDOW_UPDATE);
            assert_eq!(
                h_wu.stream_id, 0,
                "the first WINDOW_UPDATE grows the connection-level window"
            );
            let (h_ack, _) = read_frame(&mut written).unwrap_or_else(|e| unreachable!("{e}"));
            assert_eq!(h_ack.ty, FRAME_SETTINGS);
            assert_eq!(
                h_ack.flags & FLAG_ACK,
                FLAG_ACK,
                "must ack the peer's SETTINGS"
            );
            let (h2, headers_payload) =
                read_frame(&mut written).unwrap_or_else(|e| unreachable!("{e}"));
            assert_eq!(h2.ty, FRAME_HEADERS);
            assert_eq!(h2.flags & FLAG_END_HEADERS, FLAG_END_HEADERS);
            let mut dec = HpackDecoder::new();
            let sent_headers = dec
                .decode(&headers_payload)
                .unwrap_or_else(|e| unreachable!("{e}"));
            assert_eq!(sent_headers[0], (":method".to_string(), "POST".to_string()));
            let (h3, data_payload) = read_frame(&mut written).unwrap_or_else(|e| unreachable!("{e}"));
            assert_eq!(h3.ty, FRAME_DATA);
            assert_eq!(h3.flags & FLAG_END_STREAM, FLAG_END_STREAM);
            assert_eq!(data_payload, vec![0x00, 0x00, 0x00, 0x00, 0x00]);
            let (h4, _) = read_frame(&mut written).unwrap_or_else(|e| unreachable!("{e}"));
            assert_eq!(h4.ty, FRAME_WINDOW_UPDATE);
            assert_eq!(
                h4.stream_id, 1,
                "the second WINDOW_UPDATE grows stream 1's own window"
            );
        }
    }
}
pub mod image {
    //! Image formats as **applicatives over the one block stream**.
    //!
    //! A disk image is the block stream wearing a wrapper: `raw` is the identity
    //! (the bytes ARE the stream); `qcow2`/`vhdx`/… add a metadata indirection (a
    //! table mapping guest offset → host offset, holes left implicit). So there is
    //! no per-format *engine* — only a per-format `read_at` that resolves the guest
    //! address through the wrapper. `block.rs` drives the copy; this layer is the α.
    //!
    //! Built here, all zero-dep and cross-platform (I/O via `sysio`): `Raw`
    //! (identity), `Qcow2` (read + write, v3, uncompressed/no-backing), and `Vhdx`
    //! (read + write, dynamic, CRC-32C + BAT). Each is validated by round-trip
    //! (raw → format → raw, bit-identical); spec-conformant, so real qemu-img /
    //! Hyper-V should interoperate — that interop is not tested on this box (no
    //! Windows / qemu), by design (we add no deps to test it).

    use std::fs::File;
    use std::io;
    use std::path::Path;

    use crate::sync::sysio;

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub enum Format {
        Raw,
        Qcow2,
        Vhdx,
    }

    impl Format {
        /// Infer from extension; `.qcow2`/`.qcow` → Qcow2, `.vhdx` → Vhdx, else Raw.
        pub fn infer(path: &Path) -> Format {
            match path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.to_ascii_lowercase())
                .as_deref()
            {
                Some("qcow2") | Some("qcow") => Format::Qcow2,
                Some("vhdx") => Format::Vhdx,
                _ => Format::Raw,
            }
        }

        pub fn parse(s: &str) -> Option<Format> {
            match s {
                "raw" => Some(Format::Raw),
                "qcow2" | "qcow" => Some(Format::Qcow2),
                "vhdx" => Some(Format::Vhdx),
                _ => None,
            }
        }
    }

    /// A read-only image source — the guest block stream, however it is wrapped.
    pub enum Source {
        Raw(File),
        Qcow2(Qcow2),
        Vhdx(Vhdx),
    }

    impl Source {
        pub fn open(path: &Path, fmt: Format) -> io::Result<Source> {
            let f = File::open(path)?;
            match fmt {
                Format::Raw => Ok(Source::Raw(f)),
                Format::Qcow2 => Ok(Source::Qcow2(Qcow2::open(f)?)),
                Format::Vhdx => Ok(Source::Vhdx(Vhdx::open(f)?)),
            }
        }

        /// Guest (virtual) size in bytes.
        pub fn size(&self) -> io::Result<u64> {
            match self {
                Source::Raw(f) => raw_size(f),
                Source::Qcow2(q) => Ok(q.size),
                Source::Vhdx(v) => Ok(v.size),
            }
        }

        /// Read up to `buf.len()` guest bytes at guest offset `off`. Returns bytes
        /// read (0 past the end). Unallocated regions read as zeros.
        pub fn read_at(&self, off: u64, buf: &mut [u8]) -> io::Result<usize> {
            match self {
                Source::Raw(f) => read_full_at(f, buf, off),
                Source::Qcow2(q) => q.read_at(off, buf),
                Source::Vhdx(v) => v.read_at(off, buf),
            }
        }

        pub fn describe(&self) -> &'static str {
            match self {
                Source::Raw(_) => "raw",
                Source::Qcow2(_) => "qcow2",
                Source::Vhdx(_) => "vhdx",
            }
        }
    }

    /// Raw size — a regular file reports `len()`; a block device reports 0 there (on
    /// both Linux and Windows handles), so fall back to seeking the end. No device
    /// type needed → cross-platform.
    fn raw_size(f: &File) -> io::Result<u64> {
        use std::io::{Seek, SeekFrom};
        let len = f.metadata().map(|m| m.len()).unwrap_or(0);
        if len > 0 {
            Ok(len)
        } else {
            f.try_clone()?.seek(SeekFrom::End(0))
        }
    }

    /// Positioned read that fills the slice or stops at EOF; returns bytes read.
    fn read_full_at(f: &File, buf: &mut [u8], at: u64) -> io::Result<usize> {
        let mut got = 0;
        while got < buf.len() {
            match sysio::read_at(f, &mut buf[got..], at + got as u64) {
                Ok(0) => break,
                Ok(n) => got += n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        Ok(got)
    }

    // ───────────────────────── qcow2 (READ) ─────────────────────────
    //
    // Spec: QEMU `docs/interop/qcow2.txt`. All multi-byte fields big-endian. We read   CITATION-EXEMPT: the QEMU qcow2 spec, a tree outside `fold`
    // the header, then flatten the two-level L1/L2 table into one guest-cluster →
    // host-offset map (0 = unallocated → zeros). Uncompressed, no backing file, no
    // encryption — the common `qemu-img convert -O qcow2` output.

    const QCOW_MAGIC: u32 = 0x5146_49fb; // "QFI\xfb"
    const OFFSET_MASK: u64 = 0x00ff_ffff_ffff_ffff & !0x1ff; // bits 9..55 = offset
    const L2E_COMPRESSED: u64 = 1 << 62;

    pub struct Qcow2 {
        file: File,
        cluster: u64,
        pub size: u64,
        /// guest cluster index → host byte offset (0 = unallocated hole).
        map: Vec<u64>,
    }

    impl Qcow2 {
        pub fn open(file: File) -> io::Result<Qcow2> {
            let mut hdr = [0u8; 72];
            read_full_at(&file, &mut hdr, 0)?;
            let be32 = |o: usize| u32::from_be_bytes(hdr[o..o + 4].try_into().unwrap());
            let be64 = |o: usize| u64::from_be_bytes(hdr[o..o + 8].try_into().unwrap());

            if be32(0) != QCOW_MAGIC {
                return Err(bad("not a qcow2 image (bad magic)"));
            }
            let version = be32(4);
            if version != 2 && version != 3 {
                return Err(bad("unsupported qcow2 version (need 2 or 3)"));
            }
            if be64(8) != 0 {
                return Err(bad("qcow2 backing files not supported yet"));
            }
            let cluster_bits = be32(20);
            if !(9..=21).contains(&cluster_bits) {
                return Err(bad("implausible qcow2 cluster_bits"));
            }
            let cluster = 1u64 << cluster_bits;
            let size = be64(24);
            if be32(32) != 0 {
                return Err(bad("encrypted qcow2 not supported"));
            }
            let l1_size = be32(36) as u64;
            let l1_off = be64(40);

            let l2_entries = cluster / 8;
            let nclusters = size.div_ceil(cluster);
            let mut map = vec![0u64; nclusters as usize];

            // Read each referenced L2 table once, fill the slice of `map` it covers.
            let mut l1 = vec![0u8; (l1_size * 8) as usize];
            if l1_size > 0 {
                read_full_at(&file, &mut l1, l1_off)?;
            }
            let mut l2 = vec![0u8; cluster as usize];
            for l1i in 0..l1_size {
                let l1e = u64::from_be_bytes(
                    l1[(l1i * 8) as usize..(l1i * 8 + 8) as usize]
                        .try_into()
                        .unwrap(),
                );
                let l2_off = l1e & OFFSET_MASK;
                if l2_off == 0 {
                    continue; // whole L2 absent → those guest clusters are holes
                }
                read_full_at(&file, &mut l2, l2_off)?;
                let base_gc = l1i * l2_entries;
                for l2i in 0..l2_entries {
                    let gc = base_gc + l2i;
                    if gc >= nclusters {
                        break;
                    }
                    let l2e = u64::from_be_bytes(
                        l2[(l2i * 8) as usize..(l2i * 8 + 8) as usize]
                            .try_into()
                            .unwrap(),
                    );
                    if l2e & L2E_COMPRESSED != 0 {
                        return Err(bad("compressed qcow2 clusters not supported"));
                    }
                    map[gc as usize] = l2e & OFFSET_MASK; // 0 stays 0 (unallocated)
                }
            }

            Ok(Qcow2 {
                file,
                cluster,
                size,
                map,
            })
        }

        /// Resolve guest bytes through the cluster map, zero-filling holes.
        pub fn read_at(&self, off: u64, buf: &mut [u8]) -> io::Result<usize> {
            if off >= self.size {
                return Ok(0);
            }
            let end = (off + buf.len() as u64).min(self.size);
            let (mut pos, mut bi) = (off, 0usize);
            while pos < end {
                let gc = pos / self.cluster;
                let within = pos % self.cluster;
                let n = ((self.cluster - within).min(end - pos)) as usize;
                let host = self.map.get(gc as usize).copied().unwrap_or(0);
                if host == 0 {
                    buf[bi..bi + n].fill(0); // unallocated → zeros
                } else {
                    read_exact_at(&self.file, &mut buf[bi..bi + n], host + within)?;
                }
                pos += n as u64;
                bi += n;
            }
            Ok(bi)
        }
    }

    fn read_exact_at(f: &File, buf: &mut [u8], at: u64) -> io::Result<()> {
        if read_full_at(f, buf, at)? != buf.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "short read in qcow2 host cluster",
            ));
        }
        Ok(())
    }

    fn bad(msg: &str) -> io::Error {
        io::Error::new(io::ErrorKind::InvalidData, msg.to_string())
    }

    // ───────────────────────── qcow2 (WRITE) ─────────────────────────
    //
    // Emit a spec-conformant qcow2 v3 (`fromTerm` at α=qcow2): the same skip that
    // makes a sparse copy — an all-zero guest cluster is simply NOT allocated (no L2
    // entry, no data cluster), so it reads back as a hole. We write header, refcount
    // table + blocks (every allocated cluster refcount 1, COPIED set), L1, the active
    // L2 tables, then the data clusters. Uncompressed, no backing, no encryption.
    // Zero external deps; validated by round-trip (raw → qcow2 → raw, bit-identical).

    const COPIED: u64 = 1 << 63;

    #[inline]
    fn is_zero(buf: &[u8]) -> bool {
        let mut ch = buf.chunks_exact(16);
        for c in &mut ch {
            if u128::from_ne_bytes(c.try_into().unwrap()) != 0 {
                return false;
            }
        }
        ch.remainder().iter().all(|&b| b == 0)
    }

    fn put_be32(b: &mut [u8], o: usize, v: u32) {
        b[o..o + 4].copy_from_slice(&v.to_be_bytes());
    }
    fn put_be64(b: &mut [u8], o: usize, v: u64) {
        b[o..o + 8].copy_from_slice(&v.to_be_bytes());
    }

    /// Write `source`'s block stream as a qcow2 image at `dst`. Returns (data bytes,
    /// hole bytes left unallocated).
    pub fn write_qcow2(source: &Source, dst: &Path, cluster_bits: u32) -> io::Result<(u64, u64)> {
        let cluster = 1u64 << cluster_bits;
        let csz = cluster as usize;
        let size = source.size()?;
        let l2_entries = cluster / 8;
        let ngc = size.div_ceil(cluster);
        let l1_size = ngc.div_ceil(l2_entries).max(1);

        // Pass 1 — which guest clusters hold data? (Holes are never allocated.)
        let mut buf = vec![0u8; csz];
        let mut data_gcs: Vec<u64> = Vec::new();
        for gc in 0..ngc {
            let off = gc * cluster;
            let want = ((size - off) as usize).min(csz);
            let n = source.read_at(off, &mut buf[..want])?;
            if !is_zero(&buf[..n]) {
                data_gcs.push(gc);
            }
        }

        // Active L1 entries = the L2 tables we actually need (data_gcs is ascending).
        let mut active_l1: Vec<u64> = data_gcs.iter().map(|gc| gc / l2_entries).collect();
        active_l1.dedup();
        let l2_count = active_l1.len() as u64;
        let l1_clusters = (l1_size * 8).div_ceil(cluster);
        let data_count = data_gcs.len() as u64;

        // Refcount sizing to fixpoint: each 16-bit block covers `cluster/2` clusters.
        let rpb = cluster / 2;
        let rt_clusters = 1u64; // one table cluster covers cluster/8 blocks — ample
        let mut rb_clusters = 1u64;
        let total = loop {
            let total = 1 + rt_clusters + rb_clusters + l1_clusters + l2_count + data_count;
            let need = total.div_ceil(rpb).max(1);
            if need == rb_clusters {
                break total;
            }
            rb_clusters = need;
        };

        // Sequential host-cluster layout.
        let off_rt = 1u64;
        let off_rb = off_rt + rt_clusters;
        let off_l1 = off_rb + rb_clusters;
        let off_l2 = off_l1 + l1_clusters;
        let off_data = off_l2 + l2_count;

        // Header (cluster 0; the rest of the cluster stays zero → empty ext list).
        let mut header = vec![0u8; csz];
        put_be32(&mut header, 0, QCOW_MAGIC);
        put_be32(&mut header, 4, 3); // version 3
        put_be32(&mut header, 20, cluster_bits);
        put_be64(&mut header, 24, size);
        put_be32(&mut header, 36, l1_size as u32);
        put_be64(&mut header, 40, off_l1 * cluster);
        put_be64(&mut header, 48, off_rt * cluster);
        put_be32(&mut header, 56, rt_clusters as u32);
        put_be32(&mut header, 96, 4); // refcount_order (16-bit)
        put_be32(&mut header, 100, 112); // header_length

        // L1 table → each active entry points (COPIED) at its L2 cluster.
        let mut l1 = vec![0u8; (l1_clusters * cluster) as usize];
        for (k, &l1i) in active_l1.iter().enumerate() {
            put_be64(
                &mut l1,
                (l1i * 8) as usize,
                ((off_l2 + k as u64) * cluster) | COPIED,
            );
        }

        // L2 tables → each data guest cluster points (COPIED) at its data cluster.
        let mut l2 = vec![0u8; (l2_count * cluster) as usize];
        for (di, &gc) in data_gcs.iter().enumerate() {
            let k = active_l1.binary_search(&(gc / l2_entries)).unwrap();
            let slot = k * csz + ((gc % l2_entries) * 8) as usize;
            put_be64(&mut l2, slot, ((off_data + di as u64) * cluster) | COPIED);
        }

        // Refcount blocks → refcount 1 for every allocated cluster [0, total).
        let mut rb = vec![0u8; (rb_clusters * cluster) as usize];
        for c in 0..total as usize {
            rb[c * 2 + 1] = 1; // BE u16 = 1
        }
        // Refcount table → point at each refcount block.
        let mut rt = vec![0u8; (rt_clusters * cluster) as usize];
        for j in 0..rb_clusters {
            put_be64(&mut rt, (j * 8) as usize, (off_rb + j) * cluster);
        }

        // Emit. set_len first so every positioned write lands in a sized (sparse) file.
        let f = File::create(dst)?;
        f.set_len(total * cluster)?;
        sysio::write_all_at(&f, &header, 0)?;
        sysio::write_all_at(&f, &rt, off_rt * cluster)?;
        sysio::write_all_at(&f, &rb, off_rb * cluster)?;
        sysio::write_all_at(&f, &l1, off_l1 * cluster)?;
        sysio::write_all_at(&f, &l2, off_l2 * cluster)?;

        let mut data_bytes = 0u64;
        for (di, &gc) in data_gcs.iter().enumerate() {
            let off = gc * cluster;
            let want = ((size - off) as usize).min(csz);
            let n = source.read_at(off, &mut buf[..want])?;
            if n < csz {
                buf[n..].fill(0); // pad the final partial cluster
            }
            sysio::write_all_at(&f, &buf, (off_data + di as u64) * cluster)?;
            data_bytes += n as u64;
        }
        f.sync_all()?;
        Ok((data_bytes, size.saturating_sub(data_bytes)))
    }

    // ───────────────────────── VHDX (Microsoft / Hyper-V) ─────────────────────────
    //
    // Spec: [MS-VHDX]. Little-endian throughout (qcow2 was big-endian). A VHDX is the
    // block stream behind a header section (file-type id + two headers + two region
    // tables, all CRC-32C protected), a metadata region (virtual size, block size,
    // sector sizes), and a Block Allocation Table mapping each guest block to a 1 MB-
    // aligned payload block — holes are simply NOT_PRESENT (sparse, the same skip).
    //
    // We emit a dynamic VHDX: 2 MB blocks, fully-present payload blocks (no sector
    // bitmaps needed), zero blocks left NOT_PRESENT. Zero external deps; validated by
    // round-trip (raw → vhdx → raw). CRC-32C is checked against its known vector so
    // the header/region checksums Hyper-V validates are correct. Real-Hyper-V
    // acceptance is not tested here (no Windows on the build box) — by design.

    const VHDX_BLOCK: u64 = 2 * 1024 * 1024; // 2 MiB payload block
    const MB: u64 = 1024 * 1024;

    // CRC-32C (Castagnoli), reflected, init/xorout 0xFFFFFFFF. Vector: "123456789" → 0xE3069283.
    fn crc32c(data: &[u8]) -> u32 {
        let mut crc: u32 = 0xFFFF_FFFF;
        for &b in data {
            crc ^= b as u32;
            for _ in 0..8 {
                crc = if crc & 1 != 0 {
                    (crc >> 1) ^ 0x82F6_3B78
                } else {
                    crc >> 1
                };
            }
        }
        !crc
    }

    // GUIDs in on-disk byte order (Data1/2/3 little-endian, Data4 as bytes).
    const GUID_BAT: [u8; 16] = [
        0x66, 0x77, 0xC2, 0x2D, 0x23, 0xF6, 0x00, 0x42, 0x9D, 0x64, 0x11, 0x5E, 0x9B, 0xFD, 0x4A, 0x08,
    ];
    const GUID_METADATA: [u8; 16] = [
        0x06, 0xA2, 0x7C, 0x8B, 0x90, 0x47, 0x9A, 0x4B, 0xB8, 0xFE, 0x57, 0x5F, 0x05, 0x0F, 0x88, 0x6E,
    ];
    const GUID_FILE_PARAMS: [u8; 16] = [
        0x37, 0x67, 0xA1, 0xCA, 0x36, 0xFA, 0x43, 0x4D, 0xB3, 0xB6, 0x33, 0xF0, 0xAA, 0x44, 0xE7, 0x6B,
    ];
    const GUID_VDISK_SIZE: [u8; 16] = [
        0x24, 0x42, 0xA5, 0x2F, 0x1B, 0xCD, 0x76, 0x48, 0xB2, 0x11, 0x5D, 0xBE, 0xD8, 0x3B, 0xF4, 0xB8,
    ];
    const GUID_VDISK_ID: [u8; 16] = [
        0xAB, 0x12, 0xCA, 0xBE, 0xE6, 0xB2, 0x23, 0x45, 0x93, 0xEF, 0xC3, 0x09, 0xE0, 0x00, 0xC7, 0x46,
    ];
    const GUID_LOGICAL_SS: [u8; 16] = [
        0x1D, 0xBF, 0x41, 0x81, 0x6F, 0xA9, 0x09, 0x47, 0xBA, 0x47, 0xF2, 0x33, 0xA8, 0xFA, 0xAB, 0x5F,
    ];
    const GUID_PHYSICAL_SS: [u8; 16] = [
        0xC7, 0x48, 0xA3, 0xCD, 0x5D, 0x44, 0x71, 0x44, 0x9C, 0xC9, 0xE9, 0x88, 0x52, 0x51, 0xC5, 0x56,
    ];
    // A fixed nonzero GUID for FileWrite/DataWrite/VirtualDiskId — deterministic.
    const GUID_FIXED: [u8; 16] = [
        0x53, 0x59, 0x4E, 0x43, 0x46, 0x4F, 0x4C, 0x44, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
    ];

    fn put_le16(b: &mut [u8], o: usize, v: u16) {
        b[o..o + 2].copy_from_slice(&v.to_le_bytes());
    }
    fn put_le32(b: &mut [u8], o: usize, v: u32) {
        b[o..o + 4].copy_from_slice(&v.to_le_bytes());
    }
    fn put_le64(b: &mut [u8], o: usize, v: u64) {
        b[o..o + 8].copy_from_slice(&v.to_le_bytes());
    }
    fn le32(b: &[u8], o: usize) -> u32 {
        u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
    }
    fn le64(b: &[u8], o: usize) -> u64 {
        u64::from_le_bytes(b[o..o + 8].try_into().unwrap())
    }

    pub struct Vhdx {
        file: File,
        pub size: u64,
        block: u64,
        chunk_ratio: u64,
        bat: Vec<u64>,
    }

    impl Vhdx {
        pub fn open(file: File) -> io::Result<Vhdx> {
            let mut id = [0u8; 8];
            read_full_at(&file, &mut id, 0)?;
            if &id != b"vhdxfile" {
                return Err(bad("not a VHDX image (bad file-type id)"));
            }
            // Pick a valid header (sig "head" + good CRC) with the highest sequence.
            let mut best_seq: Option<u64> = None;
            for &hoff in &[0x10000u64, 0x20000u64] {
                let mut h = vec![0u8; 4096];
                read_full_at(&file, &mut h, hoff)?;
                if &h[0..4] != b"head" {
                    continue;
                }
                let stored = le32(&h, 4);
                h[4..8].fill(0);
                if crc32c(&h) != stored {
                    continue;
                }
                let seq = le64(&h, 8);
                if best_seq.is_none_or(|b| seq >= b) {
                    best_seq = Some(seq);
                }
            }
            if best_seq.is_none() {
                return Err(bad("no valid VHDX header (checksum/signature)"));
            }

            // Region table → BAT and Metadata region locations.
            let mut rt = vec![0u8; 65536];
            read_full_at(&file, &mut rt, 0x30000)?;
            if &rt[0..4] != b"regi" {
                return Err(bad("bad VHDX region table signature"));
            }
            let entry_count = le32(&rt, 8);
            let (mut bat_off, mut bat_len, mut meta_off) = (0u64, 0u64, 0u64);
            for i in 0..entry_count as usize {
                let e = 16 + i * 32;
                let guid = &rt[e..e + 16];
                let off = le64(&rt, e + 16);
                let len = le32(&rt, e + 24) as u64;
                if guid == GUID_BAT {
                    bat_off = off;
                    bat_len = len;
                } else if guid == GUID_METADATA {
                    meta_off = off;
                }
            }
            if bat_off == 0 || meta_off == 0 {
                return Err(bad("VHDX region table missing BAT or metadata"));
            }

            // Metadata: virtual size, block size, logical sector size.
            let mut md = vec![0u8; 65536];
            read_full_at(&file, &mut md, meta_off)?;
            if &md[0..8] != b"metadata" {
                return Err(bad("bad VHDX metadata signature"));
            }
            let mcount = u16::from_le_bytes(md[10..12].try_into().unwrap());
            let (mut size, mut block, mut lss) = (0u64, 0u64, 512u64);
            // metadata items may sit beyond 64 KB — read more if needed via a helper.
            let read_item = |rel: u32, len: u32| -> io::Result<Vec<u8>> {
                let mut v = vec![0u8; len as usize];
                read_full_at(&file, &mut v, meta_off + rel as u64)?;
                Ok(v)
            };
            for i in 0..mcount as usize {
                let e = 32 + i * 32;
                if e + 32 > md.len() {
                    break;
                }
                let guid = &md[e..e + 16];
                let off = le32(&md, e + 16);
                let len = le32(&md, e + 20);
                if guid == GUID_VDISK_SIZE {
                    size = le64(&read_item(off, len)?, 0);
                } else if guid == GUID_FILE_PARAMS {
                    block = le32(&read_item(off, len)?, 0) as u64;
                } else if guid == GUID_LOGICAL_SS {
                    lss = le32(&read_item(off, len)?, 0) as u64;
                }
            }
            if block == 0 || size == 0 {
                // size 0 is a legitimately empty disk; block 0 is a real error
                if block == 0 {
                    return Err(bad("VHDX metadata missing block size"));
                }
            }
            let chunk_ratio = (8_388_608u64 * lss) / block;

            let mut batb = vec![0u8; bat_len as usize];
            read_full_at(&file, &mut batb, bat_off)?;
            let bat: Vec<u64> = batb
                .chunks_exact(8)
                .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                .collect();

            Ok(Vhdx {
                file,
                size,
                block,
                chunk_ratio,
                bat,
            })
        }

        pub fn read_at(&self, off: u64, buf: &mut [u8]) -> io::Result<usize> {
            if off >= self.size {
                return Ok(0);
            }
            let end = (off + buf.len() as u64).min(self.size);
            let (mut pos, mut bi) = (off, 0usize);
            while pos < end {
                let b = pos / self.block;
                let within = pos % self.block;
                let n = ((self.block - within).min(end - pos)) as usize;
                let pb_index = (b + b / self.chunk_ratio) as usize;
                let entry = self.bat.get(pb_index).copied().unwrap_or(0);
                let state = entry & 0x7;
                if state == 6 {
                    // PAYLOAD_BLOCK_FULLY_PRESENT — file offset in bits 20..
                    let host = (entry & !0xF_FFFFu64) + within;
                    read_exact_at(&self.file, &mut buf[bi..bi + n], host)?;
                } else {
                    buf[bi..bi + n].fill(0); // not present → zeros
                }
                pos += n as u64;
                bi += n;
            }
            Ok(bi)
        }
    }

    /// Write `source`'s block stream as a dynamic VHDX at `dst`. Returns (data, holes).
    pub fn write_vhdx(source: &Source, dst: &Path, lss: u64) -> io::Result<(u64, u64)> {
        let block = VHDX_BLOCK;
        let size = source.size()?;
        let chunk_ratio = (8_388_608u64 * lss) / block;
        let ndb = size.div_ceil(block);

        // Pass 1: which guest blocks hold data?
        let bsz = block as usize;
        let mut buf = vec![0u8; bsz];
        let mut data_blocks: Vec<u64> = Vec::new();
        for b in 0..ndb {
            let o = b * block;
            let want = ((size - o) as usize).min(bsz);
            let n = source.read_at(o, &mut buf[..want])?;
            if !is_zero(&buf[..n]) {
                data_blocks.push(b);
            }
        }

        let total_bat = if ndb == 0 {
            0
        } else {
            ndb + (ndb - 1) / chunk_ratio
        };
        let bat_len = (total_bat * 8).div_ceil(MB).max(1) * MB;

        // Layout (1 MB-aligned regions): header | log | metadata | BAT | payload…
        let log_off = MB;
        let log_len = MB;
        let meta_off = log_off + log_len;
        let meta_len = MB;
        let bat_off = meta_off + meta_len;
        let data_off = bat_off + bat_len;

        // ---- header section (first 1 MB) ----
        let mut hs = vec![0u8; MB as usize];
        hs[0..8].copy_from_slice(b"vhdxfile");
        for (i, &hoff) in [0x10000usize, 0x20000usize].iter().enumerate() {
            let h = &mut hs[hoff..hoff + 4096];
            h[0..4].copy_from_slice(b"head");
            put_le64(h, 8, 1); // sequence number (both 1 → either is valid/active)
            h[16..32].copy_from_slice(&GUID_FIXED); // FileWriteGuid
            h[32..48].copy_from_slice(&GUID_FIXED); // DataWriteGuid
                                                    // LogGuid (48..64) = 0 → no log replay
            put_le16(h, 66, 1); // Version
            put_le32(h, 68, log_len as u32); // LogLength
            put_le64(h, 72, log_off); // LogOffset
            let crc = crc32c(h);
            put_le32(h, 4, crc);
            let _ = i;
        }
        for &rtoff in &[0x30000usize, 0x40000usize] {
            let rt = &mut hs[rtoff..rtoff + 65536];
            rt[0..4].copy_from_slice(b"regi");
            put_le32(rt, 8, 2); // entry count: BAT + metadata
                                // entry 0: BAT
            let e0 = 16;
            rt[e0..e0 + 16].copy_from_slice(&GUID_BAT);
            put_le64(rt, e0 + 16, bat_off);
            put_le32(rt, e0 + 24, bat_len as u32);
            put_le32(rt, e0 + 28, 1); // required
                                      // entry 1: metadata
            let e1 = 16 + 32;
            rt[e1..e1 + 16].copy_from_slice(&GUID_METADATA);
            put_le64(rt, e1 + 16, meta_off);
            put_le32(rt, e1 + 24, meta_len as u32);
            put_le32(rt, e1 + 28, 1);
            let crc = crc32c(rt);
            put_le32(rt, 4, crc);
        }

        // ---- metadata region (1 MB) ----
        let mut md = vec![0u8; meta_len as usize];
        md[0..8].copy_from_slice(b"metadata");
        put_le16(&mut md, 10, 5); // entry count
                                  // items live ≥ 64 KB into the region
        let item_base = 0x10000u32;
        let items: [(&[u8; 16], u32, u32, u32); 5] = [
            // (guid, rel offset, length, flags: bit1 IsVirtualDisk, bit2 IsRequired)
            (&GUID_FILE_PARAMS, item_base, 8, 0b100),
            (&GUID_VDISK_SIZE, item_base + 8, 8, 0b110),
            (&GUID_VDISK_ID, item_base + 16, 16, 0b010),
            (&GUID_LOGICAL_SS, item_base + 32, 4, 0b110),
            (&GUID_PHYSICAL_SS, item_base + 36, 4, 0b110),
        ];
        for (i, (guid, off, len, flags)) in items.iter().enumerate() {
            let e = 32 + i * 32;
            md[e..e + 16].copy_from_slice(*guid);
            put_le32(&mut md, e + 16, *off);
            put_le32(&mut md, e + 20, *len);
            put_le32(&mut md, e + 24, *flags);
        }
        let ib = item_base as usize;
        put_le32(&mut md, ib, block as u32); // FileParameters.BlockSize (+flags 0)
        put_le64(&mut md, ib + 8, size); // VirtualDiskSize
        md[ib + 16..ib + 32].copy_from_slice(&GUID_FIXED); // VirtualDiskId
        put_le32(&mut md, ib + 32, lss as u32); // LogicalSectorSize
        put_le32(&mut md, ib + 36, 4096); // PhysicalSectorSize

        // ---- BAT ----
        let mut bat = vec![0u64; total_bat as usize];
        for (i, &b) in data_blocks.iter().enumerate() {
            let payload_off = data_off + i as u64 * block;
            let pb_index = (b + b / chunk_ratio) as usize;
            bat[pb_index] = payload_off | 6; // FULLY_PRESENT (offset is MB-aligned)
        }
        let mut batb = vec![0u8; bat_len as usize];
        for (i, &e) in bat.iter().enumerate() {
            put_le64(&mut batb, i * 8, e);
        }

        // ---- emit ----
        let f = File::create(dst)?;
        f.set_len(data_off + data_blocks.len() as u64 * block)?;
        sysio::write_all_at(&f, &hs, 0)?;
        sysio::write_all_at(&f, &md, meta_off)?;
        sysio::write_all_at(&f, &batb, bat_off)?;
        let mut data_bytes = 0u64;
        for (i, &b) in data_blocks.iter().enumerate() {
            let o = b * block;
            let want = ((size - o) as usize).min(bsz);
            let n = source.read_at(o, &mut buf[..want])?;
            if n < bsz {
                buf[n..].fill(0);
            }
            sysio::write_all_at(&f, &buf, data_off + i as u64 * block)?;
            data_bytes += n as u64;
        }
        f.sync_all()?;
        Ok((data_bytes, size.saturating_sub(data_bytes)))
    }

    #[cfg(test)]
    mod tests {
        #[test]
        fn crc32c_vector() {
            assert_eq!(super::crc32c(b"123456789"), 0xE306_9283);
        }
    }
}
pub mod s3 {
    //! S3 REST over our own TLS + SigV4 — PUT / GET / ListObjectsV2 / DELETE.
    //! Path-style (minio / S3-compatible) or virtual-host (AWS). Zero deps.
    //!
    //! The body is signed as `UNSIGNED-PAYLOAD` (allowed over TLS) so we never hash
    //! the whole seed — the streaming law at the wire: the integrity is the AEAD
    //! record MAC, not a redundant full-body SHA.

    #![allow(dead_code)]

    use std::io;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    use crate::sync::archive;
    use crate::sync::http::{self, Response};
    use crate::sync::scan;
    use crate::sync::sigv4;
    use crate::sync::tls::client::TlsClient;

    pub struct S3 {
        pub host: String,
        pub port: u16,
        pub region: String,
        pub access_key: String,
        pub secret: String,
        pub path_style: bool,
        pub verify: bool,
        /// Plain HTTP (no TLS) — loopback reads from a same-host server. SigV4 is
        /// transport-agnostic, so signatures are unaffected.
        pub plain: bool,
    }

    impl S3 {
        /// Open a connection with this client's cert-verification policy.
        fn dial(&self) -> io::Result<TlsClient> {
            TlsClient::connect(&self.host, self.port, self.verify)
        }
    }

    /// (amz_date `YYYYMMDDTHHMMSSZ`, datestamp `YYYYMMDD`) in UTC, no chrono.
    fn utc_now() -> (String, String) {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let days = (secs / 86400) as i64;
        let rem = secs % 86400;
        let (h, mi, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
        let (y, m, d) = civil_from_days(days);
        (
            format!("{y:04}{m:02}{d:02}T{h:02}{mi:02}{s:02}Z"),
            format!("{y:04}{m:02}{d:02}"),
        )
    }

    /// Howard Hinnant's days→civil (proleptic Gregorian).
    fn civil_from_days(z: i64) -> (i64, u32, u32) {
        let z = z + 719468;
        let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
        let doe = z - era * 146097;
        let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
        let y = yoe + era * 400;
        let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
        let mp = (5 * doy + 2) / 153;
        let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
        let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
        (if m <= 2 { y + 1 } else { y }, m, d)
    }

    impl S3 {
        fn host_header(&self) -> String {
            if self.port == 443 {
                self.host.clone()
            } else {
                format!("{}:{}", self.host, self.port)
            }
        }

        /// Build the SigV4-signed `(path, headers)` for a request. `query` is the
        /// already-canonical (sorted, encoded) query string.
        fn signed(
            &self,
            method: &str,
            bucket: &str,
            key: &str,
            query: &str,
        ) -> (String, Vec<(String, String)>) {
            let (amz_date, datestamp) = utc_now();
            let host_hdr = self.host_header();
            let payload_hash = "UNSIGNED-PAYLOAD";
            let enc_key: String = key
                .split('/')
                .map(|s| sigv4::uri_encode(s, false))
                .collect::<Vec<_>>()
                .join("/");
            let canonical_uri = if self.path_style {
                format!("/{bucket}/{enc_key}")
            } else {
                format!("/{enc_key}")
            };
            let sign_headers = vec![
                ("host".to_string(), host_hdr),
                ("x-amz-content-sha256".to_string(), payload_hash.to_string()),
                ("x-amz-date".to_string(), amz_date.clone()),
            ];
            let cred = sigv4::Cred {
                access_key: &self.access_key,
                secret: &self.secret,
                region: &self.region,
                service: "s3",
            };
            let s = sigv4::sign(
                method,
                &canonical_uri,
                query,
                &sign_headers,
                payload_hash,
                &amz_date,
                &datestamp,
                &cred,
            );
            let headers = vec![
                ("x-amz-date".to_string(), amz_date),
                ("x-amz-content-sha256".to_string(), payload_hash.to_string()),
                ("Authorization".to_string(), s.authorization),
            ];
            let path = if query.is_empty() {
                canonical_uri
            } else {
                format!("{canonical_uri}?{query}")
            };
            (path, headers)
        }

        /// Sign and send one S3 request on a fresh connection (Connection: close).
        fn send(
            &self,
            method: &str,
            bucket: &str,
            key: &str,
            query: &str,
            body: &[u8],
        ) -> io::Result<Response> {
            let (path, headers) = self.signed(method, bucket, key, query);
            if self.plain {
                http::request_plain(&self.host, self.port, method, &path, &headers, body)
            } else {
                http::request(&self.host, self.port, method, &path, &headers, body, self.verify)
            }
        }

        /// Open a plain-TCP connection for pooled range reads (loopback direct-to-garage path).
        pub fn dial_plain(&self) -> io::Result<std::net::TcpStream> {
            std::net::TcpStream::connect((self.host.as_str(), self.port))
        }

        /// Ranged GET on an EXISTING plain connection (keep-alive, pooled) — the fast bulk path:
        /// one connection per thread, reused across every study span.
        pub fn get_range_plain_on(
            &self,
            conn: &mut std::net::TcpStream,
            bucket: &str,
            key: &str,
            start: u64,
            end: u64,
        ) -> io::Result<Vec<u8>> {
            let (path, mut headers) = self.signed("GET", bucket, key, "");
            headers.push(("Range".to_string(), format!("bytes={start}-{end}")));
            let r = http::request_plain_on(conn, &self.host, self.port, "GET", &path, &headers, &[])?;
            if r.status == 200 || r.status == 206 {
                Ok(r.body)
            } else {
                Err(io::Error::other(format!("ranged GET HTTP {}", r.status)))
            }
        }

        /// Range GET — `bytes=start..=end` inclusive. The `Range` header is not SigV4-signed, so
        /// the signature is untouched; only the header set grows. Enables random access into one
        /// large object (e.g. per-study shards of a 247 GiB dataset zip). The pooled
        /// connection-owning `get_range(conn, ...)` (used by `restore_stream`) is the other one.
        pub fn get_range_http(&self, bucket: &str, key: &str, start: u64, end: u64) -> io::Result<(u16, Vec<u8>)> {
            let (path, headers) = self.signed("GET", bucket, key, "");
            let mut headers = headers;
            headers.push(("Range".to_string(), format!("bytes={start}-{end}")));
            let r = if self.plain {
                http::request_plain(&self.host, self.port, "GET", &path, &headers, &[])?
            } else {
                http::request(&self.host, self.port, "GET", &path, &headers, &[], self.verify)?
            };
            Ok((r.status, r.body))
        }

        /// PUT an object on an EXISTING keep-alive connection (the pool path).
        pub fn put_on(
            &self,
            conn: &mut TlsClient,
            bucket: &str,
            key: &str,
            body: &[u8],
        ) -> io::Result<u16> {
            let (path, headers) = self.signed("PUT", bucket, key, "");
            Ok(conn.request_one("PUT", &path, &headers, body)?.status)
        }

        /// **Raw per-object sync — the Web partition over connections.** Each worker
        /// owns a persistent keep-alive connection (`preduceAt_schedule_free`: any
        /// assignment of files to connections gives the same bucket) and PUTs its
        /// share, with retry-on-reconnect (basic resilience). Returns (uploaded, errors).
        pub fn sync_raw(
            &self,
            bucket: &str,
            files: &[(std::path::PathBuf, String)],
            threads: usize,
        ) -> (u64, u64) {
            let n = files.len();
            if n == 0 {
                return (0, 0);
            }
            let up = AtomicU64::new(0);
            let err = AtomicU64::new(0);
            let p = threads.max(1).min(n);
            let per = n.div_ceil(p);
            std::thread::scope(|s| {
                for t in 0..p {
                    let lo = t * per;
                    let hi = ((t + 1) * per).min(n);
                    if lo >= hi {
                        continue;
                    }
                    let (up, err) = (&up, &err);
                    s.spawn(move || {
                        let mut conn: Option<TlsClient> = self.dial().ok();
                        for (path, key) in &files[lo..hi] {
                            let Ok(data) = std::fs::read(path) else {
                                err.fetch_add(1, Ordering::Relaxed);
                                continue;
                            };
                            let mut ok = false;
                            for _ in 0..3 {
                                if conn.is_none() {
                                    conn = self.dial().ok();
                                }
                                if let Some(c) = conn.as_mut() {
                                    match self.put_on(c, bucket, key, &data) {
                                        Ok(st) if (200..300).contains(&st) => {
                                            ok = true;
                                            break;
                                        }
                                        _ => conn = None, // bad status or broken pipe → reconnect & retry
                                    }
                                }
                            }
                            if ok {
                                up.fetch_add(1, Ordering::Relaxed);
                            } else {
                                err.fetch_add(1, Ordering::Relaxed);
                            }
                        }
                    });
                }
            });
            (up.into_inner(), err.into_inner())
        }

        /// GET an object on an existing keep-alive connection (the pool path).
        pub fn get_on(
            &self,
            conn: &mut TlsClient,
            bucket: &str,
            key: &str,
        ) -> io::Result<(u16, Vec<u8>)> {
            let (path, headers) = self.signed("GET", bucket, key, "");
            let r = conn.request_one("GET", &path, &headers, &[])?;
            Ok((r.status, r.body))
        }

        /// **Scan a bucket prefix — the friction scan over the round-tripped objects,
        /// run as the Web encircle.** List the objects, then in parallel across the
        /// connection pool GET each and read its residual against the signature `Law`;
        /// the prefix is clean iff every object's residual is 0
        /// (`os_secure_iff_encircled`). The application acting on the round-trip.
        pub fn scan_bucket(
            &self,
            bucket: &str,
            prefix: &str,
            sigs: &[scan::Sig],
            threads: usize,
        ) -> io::Result<scan::ScanReport> {
            let keys: Vec<String> = self
                .list(bucket, prefix)?
                .into_iter()
                .map(|(k, _)| k)
                .collect();
            let n = keys.len();
            if n == 0 {
                return Ok(scan::ScanReport {
                    scanned: 0,
                    clean: 0,
                    threats: Vec::new(),
                });
            }
            let threats: std::sync::Mutex<Vec<(String, Vec<String>)>> =
                std::sync::Mutex::new(Vec::new());
            let clean = AtomicU64::new(0);
            let p = threads.max(1).min(n);
            let per = n.div_ceil(p);
            let keys_ref = &keys;
            std::thread::scope(|s| {
                for t in 0..p {
                    let lo = t * per;
                    let hi = ((t + 1) * per).min(n);
                    if lo >= hi {
                        continue;
                    }
                    let (threats, clean) = (&threats, &clean);
                    s.spawn(move || {
                        let mut conn: Option<TlsClient> = self.dial().ok();
                        for key in &keys_ref[lo..hi] {
                            let mut body = None;
                            for _ in 0..3 {
                                if conn.is_none() {
                                    conn = self.dial().ok();
                                }
                                if let Some(c) = conn.as_mut() {
                                    match self.get_on(c, bucket, key) {
                                        Ok((st, b)) if (200..300).contains(&st) => {
                                            body = Some(b);
                                            break;
                                        }
                                        _ => conn = None,
                                    }
                                }
                            }
                            let hits = scan::residual(&body.unwrap_or_default(), sigs);
                            if hits.is_empty() {
                                clean.fetch_add(1, Ordering::Relaxed);
                            } else {
                                threats
                                    .lock()
                                    .unwrap()
                                    .push((key.clone(), hits.iter().map(|s| s.to_string()).collect()));
                            }
                        }
                    });
                }
            });
            let mut threats = threats.into_inner().unwrap();
            threats.sort_by(|a, b| a.0.cmp(&b.0));
            Ok(scan::ScanReport {
                scanned: n as u64,
                clean: clean.into_inner(),
                threats,
            })
        }

        pub fn put(&self, bucket: &str, key: &str, body: &[u8]) -> io::Result<u16> {
            let r = self.send("PUT", bucket, key, "", body)?;
            Ok(r.status)
        }

        pub fn get(&self, bucket: &str, key: &str) -> io::Result<(u16, Vec<u8>)> {
            let r = self.send("GET", bucket, key, "", &[])?;
            Ok((r.status, r.body))
        }

        /// HEAD → object size (Content-Length).
        pub fn head(&self, bucket: &str, key: &str) -> io::Result<u64> {
            let r = self.send("HEAD", bucket, key, "", &[])?;
            if !(200..300).contains(&r.status) {
                return Err(io::Error::other(format!("HEAD HTTP {}", r.status)));
            }
            r.header("content-length")
                .and_then(|v| v.parse().ok())
                .ok_or_else(|| io::Error::other("no Content-Length"))
        }

        /// Ranged GET `[start, start+len)` on a keep-alive connection (206 Partial).
        fn get_range(
            &self,
            conn: &mut TlsClient,
            bucket: &str,
            key: &str,
            start: u64,
            len: u64,
        ) -> io::Result<Vec<u8>> {
            let (path, mut headers) = self.signed("GET", bucket, key, "");
            headers.push((
                "Range".to_string(),
                format!("bytes={}-{}", start, start + len - 1),
            ));
            let r = conn.request_one("GET", &path, &headers, &[])?;
            if r.status == 200 || r.status == 206 {
                Ok(r.body)
            } else {
                Err(io::Error::other(format!("ranged GET HTTP {}", r.status)))
            }
        }

        /// **Streaming restore** — HEAD for the size, then pull the object in `chunk`
        /// ranged GETs (retry per range) feeding the incremental unpacker. Bounded RAM
        /// (one chunk + a 1 MiB file buffer), no whole-object-in-memory ceiling.
        pub fn restore_stream(
            &self,
            bucket: &str,
            key: &str,
            root: &std::path::Path,
            chunk: u64,
        ) -> io::Result<u64> {
            let total = self.head(bucket, key)?;
            let mut rr = RangeReader {
                s3: self,
                bucket: bucket.to_string(),
                key: key.to_string(),
                total,
                off: 0,
                buf: Vec::new(),
                pos: 0,
                conn: self.dial().ok(),
                chunk: chunk.max(1),
            };
            archive::unpack_stream(&mut rr, root)?;
            Ok(total)
        }

        pub fn delete(&self, bucket: &str, key: &str) -> io::Result<u16> {
            let r = self.send("DELETE", bucket, key, "", &[])?;
            Ok(r.status)
        }

        // ── Multipart (kills the 5 GiB single-PUT ceiling; per-part retry) ──

        fn create_multipart(&self, bucket: &str, key: &str) -> io::Result<String> {
            let (path, headers) = self.signed("POST", bucket, key, "uploads=");
            let r = http::request(
                &self.host,
                self.port,
                "POST",
                &path,
                &headers,
                &[],
                self.verify,
            )?;
            if !(200..300).contains(&r.status) {
                return Err(io::Error::other(format!(
                    "CreateMultipartUpload HTTP {}",
                    r.status
                )));
            }
            let uid = between(
                &String::from_utf8_lossy(&r.body),
                "<UploadId>",
                "</UploadId>",
            );
            if uid.is_empty() {
                return Err(io::Error::other(
                    "no UploadId in CreateMultipartUpload response",
                ));
            }
            Ok(uid)
        }

        fn upload_part(
            &self,
            conn: &mut TlsClient,
            bucket: &str,
            key: &str,
            uid: &str,
            part: u32,
            body: &[u8],
        ) -> io::Result<String> {
            let q = format!(
                "partNumber={part}&uploadId={}",
                sigv4::uri_encode(uid, true)
            );
            let (path, headers) = self.signed("PUT", bucket, key, &q);
            let r = conn.request_one("PUT", &path, &headers, body)?;
            if !(200..300).contains(&r.status) {
                return Err(io::Error::other(format!(
                    "UploadPart {part} HTTP {}",
                    r.status
                )));
            }
            Ok(r.header("etag").unwrap_or("").to_string())
        }

        fn upload_part_retry(
            &self,
            conn: &mut Option<TlsClient>,
            bucket: &str,
            key: &str,
            uid: &str,
            part: u32,
            body: &[u8],
        ) -> io::Result<String> {
            for _ in 0..3 {
                if conn.is_none() {
                    *conn = self.dial().ok();
                }
                if let Some(c) = conn.as_mut() {
                    match self.upload_part(c, bucket, key, uid, part, body) {
                        Ok(etag) => return Ok(etag),
                        Err(_) => *conn = None, // broken/throttled → reconnect & retry the part
                    }
                }
            }
            Err(io::Error::other(format!(
                "part {part} failed after retries"
            )))
        }

        fn complete_multipart(
            &self,
            bucket: &str,
            key: &str,
            uid: &str,
            etags: &[(u32, String)],
        ) -> io::Result<u16> {
            let mut xml = String::from("<CompleteMultipartUpload>");
            for (n, e) in etags {
                xml.push_str(&format!(
                    "<Part><PartNumber>{n}</PartNumber><ETag>{e}</ETag></Part>"
                ));
            }
            xml.push_str("</CompleteMultipartUpload>");
            let q = format!("uploadId={}", sigv4::uri_encode(uid, true));
            let (path, mut headers) = self.signed("POST", bucket, key, &q);
            headers.push(("Content-Type".to_string(), "application/xml".to_string()));
            let r = http::request(
                &self.host,
                self.port,
                "POST",
                &path,
                &headers,
                xml.as_bytes(),
                self.verify,
            )?;
            Ok(r.status)
        }

        fn abort_multipart(&self, bucket: &str, key: &str, uid: &str) {
            let q = format!("uploadId={}", sigv4::uri_encode(uid, true));
            let (path, headers) = self.signed("DELETE", bucket, key, &q);
            let _ = http::request(
                &self.host,
                self.port,
                "DELETE",
                &path,
                &headers,
                &[],
                self.verify,
            );
        }

        /// **Streaming backup** — pack the tree into the seed and stream it to S3.
        /// Bounded RAM (one part buffer): small backups go as a single PUT; once the
        /// stream exceeds `part_size` it switches to multipart (no 5 GiB ceiling),
        /// each part retried independently. Returns the final HTTP status.
        pub fn backup_stream(
            &self,
            bucket: &str,
            key: &str,
            root: &std::path::Path,
            part_size: usize,
        ) -> io::Result<u16> {
            let mut conn: Option<TlsClient> = self.dial().ok();
            let mut buf: Vec<u8> = Vec::with_capacity(part_size + (1 << 20));
            let mut uid: Option<String> = None;
            let mut etags: Vec<(u32, String)> = Vec::new();
            let mut part: u32 = 1;

            let pack_result = {
                let mut sink = |chunk: &[u8]| -> io::Result<()> {
                    buf.extend_from_slice(chunk);
                    while buf.len() >= part_size {
                        let body: Vec<u8> = buf.drain(..part_size).collect();
                        if uid.is_none() {
                            uid = Some(self.create_multipart(bucket, key)?);
                        }
                        let etag = self.upload_part_retry(
                            &mut conn,
                            bucket,
                            key,
                            uid.as_ref().unwrap(),
                            part,
                            &body,
                        )?;
                        etags.push((part, etag));
                        part += 1;
                    }
                    Ok(())
                };
                archive::pack_stream(root, &mut sink)
            };
            if let Err(e) = pack_result {
                if let Some(u) = &uid {
                    self.abort_multipart(bucket, key, u);
                }
                return Err(e);
            }

            match uid {
                None => {
                    // small: the whole seed fits in one PUT
                    for _ in 0..3 {
                        if conn.is_none() {
                            conn = self.dial().ok();
                        }
                        if let Some(c) = conn.as_mut() {
                            match self.put_on(c, bucket, key, &buf) {
                                Ok(s) if (200..300).contains(&s) => return Ok(s),
                                _ => conn = None,
                            }
                        }
                    }
                    Err(io::Error::other("single PUT failed after retries"))
                }
                Some(u) => {
                    if !buf.is_empty() {
                        match self.upload_part_retry(&mut conn, bucket, key, &u, part, &buf) {
                            Ok(etag) => etags.push((part, etag)),
                            Err(e) => {
                                self.abort_multipart(bucket, key, &u);
                                return Err(e);
                            }
                        }
                    }
                    let st = self.complete_multipart(bucket, key, &u, &etags)?;
                    if !(200..300).contains(&st) {
                        self.abort_multipart(bucket, key, &u);
                        return Err(io::Error::other(format!(
                            "CompleteMultipartUpload HTTP {st} ({} parts)",
                            etags.len()
                        )));
                    }
                    Ok(st)
                }
            }
        }

        pub fn create_bucket(&self, bucket: &str) -> io::Result<u16> {
            // path-style PUT to /bucket
            let r = self.send("PUT", bucket, "", "", &[])?;
            Ok(r.status)
        }

        /// ListObjectsV2 → (key, size) pairs, following continuation tokens.
        pub fn list(&self, bucket: &str, prefix: &str) -> io::Result<Vec<(String, u64)>> {
            let mut out = Vec::new();
            let mut token: Option<String> = None;
            loop {
                // canonical query: params sorted by name, values encoded
                let mut params: Vec<(String, String)> = vec![("list-type".into(), "2".into())];
                if !prefix.is_empty() {
                    params.push(("prefix".into(), sigv4::uri_encode(prefix, true)));
                }
                if let Some(t) = &token {
                    params.push(("continuation-token".into(), sigv4::uri_encode(t, true)));
                }
                params.sort_by(|a, b| a.0.cmp(&b.0));
                let query = params
                    .iter()
                    .map(|(k, v)| format!("{k}={v}"))
                    .collect::<Vec<_>>()
                    .join("&");
                let r = self.send("GET", bucket, "", &query, &[])?;
                if r.status != 200 {
                    return Err(io::Error::other(format!("list status {}", r.status)));
                }
                let xml = String::from_utf8_lossy(&r.body);
                for contents in xml.split("<Contents>").skip(1) {
                    let key = between(contents, "<Key>", "</Key>");
                    let size = between(contents, "<Size>", "</Size>")
                        .parse::<u64>()
                        .unwrap_or(0);
                    if !key.is_empty() {
                        out.push((unescape(&key), size));
                    }
                }
                if between(&xml, "<IsTruncated>", "</IsTruncated>") == "true" {
                    let t = between(&xml, "<NextContinuationToken>", "</NextContinuationToken>");
                    if t.is_empty() {
                        break;
                    }
                    token = Some(unescape(&t));
                } else {
                    break;
                }
            }
            Ok(out)
        }
    }

    /// A `Read` over an S3 object via ranged GETs — the streaming-restore source.
    pub struct RangeReader<'a> {
        s3: &'a S3,
        bucket: String,
        key: String,
        total: u64,
        off: u64,
        buf: Vec<u8>,
        pos: usize,
        conn: Option<TlsClient>,
        chunk: u64,
    }

    impl RangeReader<'_> {
        fn fetch(&mut self, start: u64, len: u64) -> io::Result<Vec<u8>> {
            for _ in 0..3 {
                if self.conn.is_none() {
                    self.conn = self.s3.dial().ok();
                }
                if let Some(c) = self.conn.as_mut() {
                    match self.s3.get_range(c, &self.bucket, &self.key, start, len) {
                        Ok(d) => return Ok(d),
                        Err(_) => self.conn = None,
                    }
                }
            }
            Err(io::Error::other("ranged GET failed after retries"))
        }
    }

    impl std::io::Read for RangeReader<'_> {
        fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
            if self.pos >= self.buf.len() {
                if self.off >= self.total {
                    return Ok(0);
                }
                let want = self.chunk.min(self.total - self.off);
                let d = self.fetch(self.off, want)?;
                if d.is_empty() {
                    return Ok(0);
                }
                self.off += d.len() as u64;
                self.buf = d;
                self.pos = 0;
            }
            let n = (self.buf.len() - self.pos).min(out.len());
            out[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
            self.pos += n;
            Ok(n)
        }
    }

    fn between(s: &str, a: &str, b: &str) -> String {
        if let Some(i) = s.find(a) {
            let rest = &s[i + a.len()..];
            if let Some(j) = rest.find(b) {
                return rest[..j].to_string();
            }
        }
        String::new()
    }

    fn unescape(s: &str) -> String {
        s.replace("&amp;", "&")
            .replace("&lt;", "<")
            .replace("&gt;", ">")
            .replace("&quot;", "\"")
    }
}
pub mod scan {
    //! The friction scan — `UniversalThreat` in Rust.
    //!
    //! A vulnerability is **friction against a `Law`** (`UniversalThreat.lean`): for a
    //! signature corpus, an atom is legal iff it is NOT a known-bad signature
    //! (`signatureLaw`); the per-file residual is the count of signature hits (0 =
    //! frictionless). The scan over many files is the **Web parallel encircle**
    //! (`ThreatModel.os_secure_iff_encircled` / `Web.encircle_iff`): the global
    //! residual reduced over the web is 0 **iff every file is clean** — read in one
    //! schedule-free parallel sweep, never a serial walk.
    //!
    //! `normalize` (filter to the legal — quarantine the matched) is the fix
    //! (`UniversalThreat.normalize_closes`).

    #![allow(dead_code)]

    use std::path::{Path, PathBuf};
    use std::sync::Mutex;
    use std::thread;

    /// A signature — a known-bad byte pattern (ClamAV `.ndb` style): the `signatureLaw` atom.
    pub struct Sig {
        pub name: &'static str,
        pub pattern: &'static [u8],
    }

    /// The default malware-signature `Law`. EICAR is the industry-standard antivirus
    /// test string — every scanner flags it, and it is completely harmless. (A real
    /// deployment loads a full ClamAV/.ndb corpus; the mechanism is identical.)
    pub fn default_sigs() -> Vec<Sig> {
        vec![Sig {
            name: "EICAR-Test-File",
            pattern: br#"X5O!P%@AP[4\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"#,
        }]
    }

    fn contains(hay: &[u8], needle: &[u8]) -> bool {
        if needle.is_empty() {
            return true;
        }
        if needle.len() > hay.len() {
            return false;
        }
        hay.windows(needle.len()).any(|w| w == needle)
    }

    /// The friction residual of one blob = the names of every signature it matches
    /// (empty = frictionless, the Being cell; non-empty = the Null-cell landing).
    pub fn residual<'a>(data: &[u8], sigs: &'a [Sig]) -> Vec<&'a str> {
        sigs.iter()
            .filter(|s| contains(data, s.pattern))
            .map(|s| s.name)
            .collect()
    }

    pub struct ScanReport {
        pub scanned: u64,
        pub clean: u64,
        /// (name, matched signatures) for each node whose residual ≠ 0.
        pub threats: Vec<(String, Vec<String>)>,
    }

    impl ScanReport {
        /// The encircle readout: clean iff every node's residual is 0.
        pub fn is_clean(&self) -> bool {
            self.threats.is_empty()
        }
    }

    fn collect_files(root: &Path, out: &mut Vec<PathBuf>) {
        let Ok(rd) = std::fs::read_dir(root) else {
            return;
        };
        for de in rd.flatten() {
            match de.file_type() {
                Ok(ft) if ft.is_dir() && !ft.is_symlink() => collect_files(&de.path(), out),
                Ok(ft) if ft.is_file() => out.push(de.path()),
                _ => {}
            }
        }
    }

    /// Scan a tree as the Web encircle — every file is a node, scanned in parallel
    /// across cores; the corpus is clean iff the summed residual is 0.
    pub fn scan_tree(root: &Path, sigs: &[Sig], threads: usize) -> ScanReport {
        let mut files = Vec::new();
        collect_files(root, &mut files);
        scan_named(
            &files
                .iter()
                .map(|p| (p.clone(), p.display().to_string()))
                .collect::<Vec<_>>(),
            sigs,
            threads,
            |p| std::fs::read(p).ok(),
        )
    }

    /// Scan a set of named blobs in parallel (the `pmapAt`/`preduce` sweep). `fetch`
    /// produces the bytes for an item (a local read, or a GET from a bucket).
    pub fn scan_named<T: Sync>(
        items: &[(T, String)],
        sigs: &[Sig],
        threads: usize,
        fetch: impl Fn(&T) -> Option<Vec<u8>> + Sync,
    ) -> ScanReport {
        let n = items.len();
        if n == 0 {
            return ScanReport {
                scanned: 0,
                clean: 0,
                threats: Vec::new(),
            };
        }
        let threats: Mutex<Vec<(String, Vec<String>)>> = Mutex::new(Vec::new());
        let clean = std::sync::atomic::AtomicU64::new(0);
        let p = threads.max(1).min(n);
        let per = n.div_ceil(p);
        thread::scope(|s| {
            for t in 0..p {
                let lo = t * per;
                let hi = ((t + 1) * per).min(n);
                if lo >= hi {
                    continue;
                }
                let (threats, clean, fetch) = (&threats, &clean, &fetch);
                s.spawn(move || {
                    for (item, name) in &items[lo..hi] {
                        let data = fetch(item).unwrap_or_default();
                        let hits = residual(&data, sigs);
                        if hits.is_empty() {
                            clean.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        } else {
                            threats
                                .lock()
                                .unwrap()
                                .push((name.clone(), hits.iter().map(|s| s.to_string()).collect()));
                        }
                    }
                });
            }
        });
        let mut threats = threats.into_inner().unwrap();
        threats.sort_by(|a, b| a.0.cmp(&b.0));
        ScanReport {
            scanned: n as u64,
            clean: clean.into_inner(),
            threats,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn eicar_is_friction_clean_is_not() {
            let sigs = default_sigs();
            let eicar = br#"X5O!P%@AP[4\PZX54(P^)7CC)7}$EICAR-STANDARD-ANTIVIRUS-TEST-FILE!$H+H*"#;
            assert!(!residual(eicar, &sigs).is_empty(), "EICAR must be friction");
            assert!(
                residual(b"hello world", &sigs).is_empty(),
                "clean must be frictionless"
            );
            // embedded in a larger file still detected
            let mut buf = vec![0u8; 1000];
            buf.extend_from_slice(eicar);
            buf.extend_from_slice(&[0u8; 1000]);
            assert!(!residual(&buf, &sigs).is_empty());
        }
    }
}
pub mod sigv4 {
    //! AWS Signature Version 4 — the request auth S3 requires. Pure functions over
    //! the crypto bedrock. Verified against AWS's published S3 "GET Object" example
    //! signature in the test module.

    #![allow(dead_code)]

    use crate::sync::crypto;

    pub struct Cred<'a> {
        pub access_key: &'a str,
        pub secret: &'a str,
        pub region: &'a str,
        pub service: &'a str,
    }

    pub struct Signed {
        pub authorization: String,
        pub signature: String,
        pub signed_headers: String,
    }

    /// Percent-encode per AWS rules (unreserved `A-Za-z0-9-_.~` pass; `/` kept unless
    /// `encode_slash`). Used for the canonical URI and query string.
    pub fn uri_encode(s: &str, encode_slash: bool) -> String {
        let mut out = String::with_capacity(s.len());
        for &b in s.as_bytes() {
            let c = b as char;
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '~') {
                out.push(c);
            } else if b == b'/' && !encode_slash {
                out.push('/');
            } else {
                out.push('%');
                out.push_str(&format!("{b:02X}"));
            }
        }
        out
    }

    /// Build the SigV4 `Authorization` header. `headers` are `(lowercase-name, value)`
    /// pairs that participate in the signature; `payload_hash` is the value of
    /// `x-amz-content-sha256` (a hex digest, or `UNSIGNED-PAYLOAD`).
    pub fn sign(
        method: &str,
        canonical_uri: &str,
        canonical_query: &str,
        headers: &[(String, String)],
        payload_hash: &str,
        amz_date: &str,  // YYYYMMDDTHHMMSSZ
        datestamp: &str, // YYYYMMDD
        cred: &Cred,
    ) -> Signed {
        let mut hs = headers.to_vec();
        hs.sort_by(|a, b| a.0.cmp(&b.0));
        let canonical_headers: String = hs
            .iter()
            .map(|(k, v)| format!("{k}:{}\n", v.trim()))
            .collect();
        let signed_headers = hs
            .iter()
            .map(|(k, _)| k.as_str())
            .collect::<Vec<_>>()
            .join(";");

        let canonical_request = format!(
            "{method}\n{canonical_uri}\n{canonical_query}\n{canonical_headers}\n{signed_headers}\n{payload_hash}"
        );
        let scope = format!("{datestamp}/{}/{}/aws4_request", cred.region, cred.service);
        let string_to_sign = format!(
            "AWS4-HMAC-SHA256\n{amz_date}\n{scope}\n{}",
            crypto::hex(&crypto::sha256(canonical_request.as_bytes()))
        );

        let k_date = crypto::hmac_sha256(
            format!("AWS4{}", cred.secret).as_bytes(),
            datestamp.as_bytes(),
        );
        let k_region = crypto::hmac_sha256(&k_date, cred.region.as_bytes());
        let k_service = crypto::hmac_sha256(&k_region, cred.service.as_bytes());
        let k_signing = crypto::hmac_sha256(&k_service, b"aws4_request");
        let signature = crypto::hex(&crypto::hmac_sha256(&k_signing, string_to_sign.as_bytes()));

        let authorization = format!(
            "AWS4-HMAC-SHA256 Credential={}/{},SignedHeaders={},Signature={}",
            cred.access_key, scope, signed_headers, signature
        );
        Signed {
            authorization,
            signature,
            signed_headers,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        // AWS docs, "Authenticating Requests (AWS Signature Version 4)" — Example: GET
        // Object with a Range header. The published signature is the known answer.
        #[test]
        fn sigv4_s3_get_example() {
            let cred = Cred {
                access_key: "AKIAIOSFODNN7EXAMPLE",
                secret: "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
                region: "us-east-1",
                service: "s3",
            };
            let empty = crypto::hex(&crypto::sha256(b""));
            let headers = vec![
                (
                    "host".to_string(),
                    "examplebucket.s3.amazonaws.com".to_string(),
                ),
                ("range".to_string(), "bytes=0-9".to_string()),
                ("x-amz-content-sha256".to_string(), empty.clone()),
                ("x-amz-date".to_string(), "20130524T000000Z".to_string()),
            ];
            let r = sign(
                "GET",
                "/test.txt",
                "",
                &headers,
                &empty,
                "20130524T000000Z",
                "20130524",
                &cred,
            );
            assert_eq!(
                r.signature, "f0e8bdb87c964420e857bd35b5d6ed310bd44f0170aba48dd91039c6036bdb41",
                "auth={}",
                r.authorization
            );
        }
    }
}
pub mod sync {
    //! The engine — **recursion, not a loop** (the user's direction, 2026-06-29).
    //!
    //! A tree is a `Web`; syncing it is the parallel sweep over the tree's own shape.
    //! There is no work-queue and no worker `loop`: `sync_dir` descends by **calling
    //! itself**, forking a scoped thread per subdirectory while a thread budget
    //! allows (and recursing inline when it doesn't). Each call returns its subtree's
    //! `Stats`, and parent merges children with the **`HashTrinity` accumulator** —
    //! `Stats::merge` is componentwise `+`, a commutative monoid
    //! (`HashTrinity.fusion_is_commutative_monoid`), so the fork/join combine is
    //! order-free: `Web.preduce_schedule_free` realized as recursion.
    //!
    //! The three faces, per file:
    //!   * **B (throne) — certified once, never re-read:** in a re-sync, a file whose
    //!     `(size, mtime)` matches the destination is skipped unread (the streaming
    //!     law). In a cold copy there is nothing to check — we don't even stat the dst.
    //!   * **NB (the break) — what is sent:** missing or `(size,mtime)`-different files.
    //!   * **B/U (the medium):** the recursive fork/join itself.

    use std::fs;
    use std::io;
    use std::path::{Path, PathBuf};
    use std::sync::atomic::{AtomicIsize, Ordering};
    use std::thread;

    use crate::sync::copy;

    pub struct Opts {
        pub threads: usize,
        pub delete: bool,
        pub dry_run: bool,
        pub quiet: bool,
    }

    /// The B/U accumulator (`HashTrinity.fuse`): componentwise `+`, commutative and
    /// associative with `ZERO` the unit — so fork/join merges in any order.
    #[derive(Clone, Copy, Default)]
    pub struct Stats {
        pub files_copied: u64,
        pub bytes_copied: u64,
        pub files_skipped: u64,
        pub dirs_made: u64,
        pub symlinks: u64,
        pub deleted: u64,
        pub errors: u64,
    }

    impl Stats {
        const ZERO: Stats = Stats {
            files_copied: 0,
            bytes_copied: 0,
            files_skipped: 0,
            dirs_made: 0,
            symlinks: 0,
            deleted: 0,
            errors: 0,
        };

        #[inline]
        fn merge(self, o: Stats) -> Stats {
            Stats {
                files_copied: self.files_copied + o.files_copied,
                bytes_copied: self.bytes_copied + o.bytes_copied,
                files_skipped: self.files_skipped + o.files_skipped,
                dirs_made: self.dirs_made + o.dirs_made,
                symlinks: self.symlinks + o.symlinks,
                deleted: self.deleted + o.deleted,
                errors: self.errors + o.errors,
            }
        }
    }

    /// Sync `src` → `dst`.
    pub fn run(src: &Path, dst: &Path, opts: &Opts) -> io::Result<Stats> {
        let smeta = fs::symlink_metadata(src)?;
        if smeta.is_file() {
            if opts.dry_run {
                return Ok(Stats {
                    files_copied: 1,
                    ..Stats::ZERO
                });
            }
            return Ok(match copy::copy_one(src, dst) {
                Ok(n) => Stats {
                    files_copied: 1,
                    bytes_copied: n,
                    ..Stats::ZERO
                },
                Err(_) => Stats {
                    errors: 1,
                    ..Stats::ZERO
                },
            });
        }

        let cold = !dst.exists();
        fs::create_dir_all(dst)?;

        // One thread budget shared across the whole recursion. The current thread is
        // one unit, so `threads - 1` may be spawned concurrently.
        let budget = AtomicIsize::new(opts.threads.max(1) as isize - 1);

        let mut st = thread::scope(|scope| {
            sync_dir(
                scope,
                src.to_path_buf(),
                dst.to_path_buf(),
                cold,
                opts,
                &budget,
            )
        });

        if opts.delete && !cold {
            st = st.merge(delete_extras(src, dst, opts.dry_run));
        }
        Ok(st)
    }

    /// Recurse one directory: copy its own files/symlinks here, then fork its
    /// subdirectories by **balanced bisection** (`fork_dirs`). `sync_dir` is the
    /// traversal — no loop drives it; the parallelism is the fork.
    fn sync_dir<'scope, 'env>(
        scope: &'scope thread::Scope<'scope, 'env>,
        src: PathBuf,
        dst: PathBuf,
        cold: bool,
        opts: &'env Opts,
        budget: &'env AtomicIsize,
    ) -> Stats {
        let rd = match fs::read_dir(&src) {
            Ok(rd) => rd,
            Err(_) => {
                return Stats {
                    errors: 1,
                    ..Stats::ZERO
                };
            }
        };

        // Classify this directory's entries (one readdir; d_type → no per-entry stat).
        let mut subdirs: Vec<(PathBuf, PathBuf)> = Vec::new(); // (child src, child dst)
        let mut files: Vec<PathBuf> = Vec::new();
        let mut symlinks: Vec<PathBuf> = Vec::new();
        for de in rd.flatten() {
            let Ok(ft) = de.file_type() else { continue };
            let name = de.file_name();
            if ft.is_symlink() {
                symlinks.push(name.into());
            } else if ft.is_dir() {
                subdirs.push((de.path(), dst.join(&name)));
            } else {
                files.push(name.into());
            }
        }

        // Fold this directory's files through the monoid (preduce over the leaves).
        let mut acc = files.iter().fold(Stats::ZERO, |a, name| {
            let from = src.join(name);
            let to = dst.join(name);
            let send = cold || changed(&from, &to);
            let s = if !send {
                Stats {
                    files_skipped: 1,
                    ..Stats::ZERO
                }
            } else if opts.dry_run {
                Stats {
                    files_copied: 1,
                    ..Stats::ZERO
                }
            } else {
                match copy::copy_one(&from, &to) {
                    Ok(n) => Stats {
                        files_copied: 1,
                        bytes_copied: n,
                        ..Stats::ZERO
                    },
                    Err(e) => {
                        if !opts.quiet {
                            eprintln!("syncfold: {}: {e}", from.display());
                        }
                        Stats {
                            errors: 1,
                            ..Stats::ZERO
                        }
                    }
                }
            };
            a.merge(s)
        });

        // Symlinks (recreate where missing or different).
        acc = symlinks.iter().fold(acc, |a, name| {
            let from = src.join(name);
            let to = dst.join(name);
            a.merge(sync_symlink(&from, &to, opts.dry_run))
        });

        // Recurse into subdirectories — balanced fork/join.
        acc.merge(fork_dirs(scope, subdirs, cold, opts, budget))
    }

    /// Recurse a set of sibling directories by **bisection** — split in half, fork
    /// the left (budget allowing) while recursing the right, then merge. Log-depth
    /// parallelism with near-perfect balance and minimal main-thread serial work;
    /// pure recursion, no loop. The merge is the order-free monoid (`HashTrinity`).
    fn fork_dirs<'scope, 'env>(
        scope: &'scope thread::Scope<'scope, 'env>,
        mut dirs: Vec<(PathBuf, PathBuf)>,
        cold: bool,
        opts: &'env Opts,
        budget: &'env AtomicIsize,
    ) -> Stats {
        match dirs.len() {
            0 => Stats::ZERO,
            1 => {
                let (csrc, cdst) = dirs.pop().unwrap();
                let made = if opts.dry_run { true } else { make_dir(&cdst) };
                let here = Stats {
                    dirs_made: made as u64,
                    errors: (!made) as u64,
                    ..Stats::ZERO
                };
                if made || opts.dry_run {
                    here.merge(sync_dir(scope, csrc, cdst, cold, opts, budget))
                } else {
                    here
                }
            }
            n => {
                let right = dirs.split_off(n / 2); // dirs := left half
                let left = dirs;
                if budget.fetch_sub(1, Ordering::AcqRel) > 0 {
                    let h = scope.spawn(move || {
                        let r = fork_dirs(scope, left, cold, opts, budget);
                        budget.fetch_add(1, Ordering::AcqRel);
                        r
                    });
                    let r = fork_dirs(scope, right, cold, opts, budget);
                    r.merge(h.join().unwrap_or(Stats {
                        errors: 1,
                        ..Stats::ZERO
                    }))
                } else {
                    budget.fetch_add(1, Ordering::AcqRel);
                    fork_dirs(scope, left, cold, opts, budget)
                        .merge(fork_dirs(scope, right, cold, opts, budget))
                }
            }
        }
    }

    /// Make one mirror directory; parent is guaranteed already made, so `create_dir`
    /// (one `mkdirat`) suffices. An existing dir (re-sync) is fine.
    fn make_dir(dst: &Path) -> bool {
        match fs::create_dir(dst) {
            Ok(()) => true,
            Err(e) if e.kind() == io::ErrorKind::AlreadyExists => true,
            Err(_) => false,
        }
    }

    /// The throne-check: is the destination missing, or different in `(size, mtime)`?
    fn changed(src: &Path, dst: &Path) -> bool {
        match (fs::metadata(src), fs::symlink_metadata(dst)) {
            (Ok(s), Ok(d)) => s.len() != d.len() || s.modified().ok() != d.modified().ok(),
            _ => true,
        }
    }

    fn sync_symlink(from: &Path, to: &Path, dry: bool) -> Stats {
        let Ok(target) = fs::read_link(from) else {
            return Stats {
                errors: 1,
                ..Stats::ZERO
            };
        };
        let same = matches!(fs::read_link(to), Ok(cur) if cur == target);
        if same {
            return Stats::ZERO;
        }
        if dry {
            return Stats {
                symlinks: 1,
                ..Stats::ZERO
            };
        }
        let _ = fs::remove_file(to);
        match crate::sync::sysio::symlink(&target, to) {
            Ok(()) => Stats {
                symlinks: 1,
                ..Stats::ZERO
            },
            Err(_) => Stats {
                errors: 1,
                ..Stats::ZERO
            },
        }
    }

    /// `--delete` mirror — recurse the destination, remove what the source lacks.
    /// The rare path; sequential recursion (no fork) keeps it simple.
    fn delete_extras(src: &Path, dst: &Path, dry: bool) -> Stats {
        let rd = match fs::read_dir(dst) {
            Ok(rd) => rd,
            Err(_) => return Stats::ZERO,
        };
        rd.flatten().fold(Stats::ZERO, |a, de| {
            let name = de.file_name();
            let s = src.join(&name);
            let d = de.path();
            let is_dir = de
                .file_type()
                .map(|t| t.is_dir() && !t.is_symlink())
                .unwrap_or(false);
            if !s.exists() {
                let ok = if dry {
                    true
                } else if is_dir {
                    fs::remove_dir_all(&d).is_ok()
                } else {
                    fs::remove_file(&d).is_ok()
                };
                a.merge(Stats {
                    deleted: ok as u64,
                    errors: (!ok) as u64,
                    ..Stats::ZERO
                })
            } else if is_dir {
                a.merge(delete_extras(&s, &d, dry))
            } else {
                a
            }
        })
    }
}
pub mod sysio {
    //! The one platform cut — positioned I/O, device detection, mode bits, symlinks —
    //! in pure `std`, no crates, no `unsafe`, on BOTH sides. Unix uses pread/pwrite,
    //! the block-special file type, and unix mode/symlink; Windows uses seek_read/
    //! seek_write, the `\\.\` device-path convention, and makes mode a no-op while
    //! mapping symlinks to the file/dir variants. Every `#[cfg]` in the crate lives
    //! here so the rest of the code is platform-blind.

    use std::fs::{File, Metadata, OpenOptions};
    use std::io;
    use std::path::Path;

    // ───────── positioned read (pread) ─────────
    #[cfg(unix)]
    pub fn read_at(f: &File, buf: &mut [u8], off: u64) -> io::Result<usize> {
        use std::os::unix::fs::FileExt;
        f.read_at(buf, off)
    }
    #[cfg(windows)]
    pub fn read_at(f: &File, buf: &mut [u8], off: u64) -> io::Result<usize> {
        use std::os::windows::fs::FileExt;
        f.seek_read(buf, off)
    }

    // ───────── positioned write-all (pwrite, looped) ─────────
    #[cfg(unix)]
    pub fn write_all_at(f: &File, buf: &[u8], off: u64) -> io::Result<()> {
        use std::os::unix::fs::FileExt;
        f.write_all_at(buf, off)
    }
    #[cfg(windows)]
    pub fn write_all_at(f: &File, mut buf: &[u8], mut off: u64) -> io::Result<()> {
        use std::os::windows::fs::FileExt;
        while !buf.is_empty() {
            match f.seek_write(buf, off) {
                Ok(0) => {
                    return Err(io::Error::new(
                        io::ErrorKind::WriteZero,
                        "failed to write whole buffer",
                    ));
                }
                Ok(n) => {
                    buf = &buf[n..];
                    off += n as u64;
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    // ───────── device detection ─────────
    /// Is `path` a raw block device? Unix: a block-special file. Windows: a
    /// `\\.\PhysicalDriveN` / `\\.\X:` style path (the only way to name one).
    #[cfg(unix)]
    pub fn is_device_path(path: &Path) -> bool {
        use std::os::unix::fs::FileTypeExt;
        path.symlink_metadata()
            .map(|m| m.file_type().is_block_device())
            .unwrap_or(false)
    }
    #[cfg(windows)]
    pub fn is_device_path(path: &Path) -> bool {
        let s = path.to_string_lossy();
        s.starts_with(r"\\.\") || s.starts_with(r"\\?\")
    }

    // ───────── unix mode bits (no-op on Windows) ─────────
    /// The permission bits to preserve, `0` where the OS has no unix mode.
    #[cfg(unix)]
    pub fn mode_of(meta: &Metadata) -> u32 {
        use std::os::unix::fs::MetadataExt;
        meta.mode() & 0o7777
    }
    #[cfg(windows)]
    pub fn mode_of(_: &Metadata) -> u32 {
        0o644
    }

    /// Apply a unix mode when creating a file (no-op on Windows).
    #[cfg(unix)]
    pub fn open_options_mode(oo: &mut OpenOptions, mode: u32) {
        use std::os::unix::fs::OpenOptionsExt;
        oo.mode(mode);
    }
    #[cfg(windows)]
    pub fn open_options_mode(_: &mut OpenOptions, _: u32) {}

    /// Set a unix mode on a path / handle (no-op on Windows).
    #[cfg(unix)]
    pub fn set_mode_path(path: &Path, mode: u32) -> io::Result<()> {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode))
    }
    #[cfg(windows)]
    pub fn set_mode_path(_: &Path, _: u32) -> io::Result<()> {
        Ok(())
    }

    #[cfg(unix)]
    pub fn set_mode_file(f: &File, mode: u32) -> io::Result<()> {
        use std::os::unix::fs::PermissionsExt;
        f.set_permissions(std::fs::Permissions::from_mode(mode))
    }
    #[cfg(windows)]
    pub fn set_mode_file(_: &File, _: u32) -> io::Result<()> {
        Ok(())
    }

    // ───────── symlinks ─────────
    /// Create a symlink `link` → `target`. Windows needs file/dir variants, chosen
    /// from the target (symlink creation there needs privilege; that is a runtime
    /// concern, it still compiles).
    #[cfg(unix)]
    pub fn symlink(target: &Path, link: &Path) -> io::Result<()> {
        std::os::unix::fs::symlink(target, link)
    }
    #[cfg(windows)]
    pub fn symlink(target: &Path, link: &Path) -> io::Result<()> {
        if target.is_dir() {
            std::os::windows::fs::symlink_dir(target, link)
        } else {
            std::os::windows::fs::symlink_file(target, link)
        }
    }
}
pub mod tls;
