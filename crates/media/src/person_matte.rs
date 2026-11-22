//! Local, indexed grayscale person mattes. Workers stream frames to a staging file and publish
//! it only after `finish`. The reader does a binary search by presentation key and loads a
//! single coverage plane, so memory stays independent of the length of the source movie.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

use crate::reader::SharedReader;

const MAGIC: [u8; 8] = *b"FCMATTE1";
const HEADER_BYTES: u64 = 24;
const KEY_BYTES: u64 = 8;
const MAX_PIXELS: usize = 1024 * 1024;
const MAX_FILE_BYTES: u64 = 32 * 1024 * 1024 * 1024;
pub const MAX_MATTE_FRAMES: usize = 1_000_000;

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

#[derive(Clone, Copy)]
struct Layout {
    width: u32,
    height: u32,
    pixels: usize,
    frames: u64,
}

impl Layout {
    fn new(width: u32, height: u32, frames: u64) -> io::Result<Self> {
        let pixels = (width as usize)
            .checked_mul(height as usize)
            .filter(|&n| n > 0 && n <= MAX_PIXELS)
            .ok_or_else(|| invalid("unsupported person matte dimensions"))?;
        if frames == 0 || frames > MAX_MATTE_FRAMES as u64 {
            return Err(invalid("unsupported person matte frame count"));
        }
        let layout = Self { width, height, pixels, frames };
        if layout.key_offset(frames) > MAX_FILE_BYTES {
            return Err(invalid("person matte would exceed 32 GiB"));
        }
        Ok(layout)
    }

    /// Length of the header plus `i` coverage planes, bounded by the caps checked in `new`.
    fn plane_offset(self, i: u64) -> u64 {
        HEADER_BYTES + i * self.pixels as u64
    }

    /// A contiguous index after the planes lets reuse validate keys without reading any alpha.
    fn key_offset(self, i: u64) -> u64 {
        self.plane_offset(self.frames) + i * KEY_BYTES
    }
}

/// Stream one matte plane at a time to a new, empty staging file.
pub struct MatteWriter {
    file: BufWriter<File>,
    layout: Layout,
    keys: Vec<i64>,
}

impl MatteWriter {
    pub fn new(file: File, width: u32, height: u32, frame_count: u64) -> io::Result<Self> {
        let layout = Layout::new(width, height, frame_count)?;
        let mut file = BufWriter::new(file);
        file.write_all(&MAGIC)?;
        file.write_all(&width.to_le_bytes())?;
        file.write_all(&height.to_le_bytes())?;
        file.write_all(&frame_count.to_le_bytes())?;
        let mut keys = Vec::new();
        keys.try_reserve_exact(frame_count as usize).map_err(io::Error::other)?;
        Ok(Self { file, layout, keys })
    }

    /// `key` must be strictly increasing, even when the movie has variable frame durations.
    pub fn push(&mut self, key: i64, alpha: &[u8]) -> io::Result<()> {
        if self.keys.last().is_some_and(|&last| key <= last) || self.keys.len() as u64 >= self.layout.frames || alpha.len() != self.layout.pixels {
            return Err(invalid("invalid person matte frame"));
        }
        self.file.write_all(alpha)?;
        self.keys.push(key);
        Ok(())
    }

    pub fn finish(mut self) -> io::Result<()> {
        if self.keys.len() as u64 != self.layout.frames {
            return Err(invalid("incomplete person matte"));
        }
        for key in self.keys {
            self.file.write_all(&key.to_le_bytes())?;
        }
        self.file.flush()?;
        self.file.get_ref().sync_all()
    }
}

pub struct MatteFrame {
    pub width: u32,
    pub height: u32,
    /// Top down, tightly packed, with 0 for transparent and 255 for opaque.
    pub alpha: Vec<u8>,
}

/// Validated header and positional file handle. Clones share the same reader.
#[derive(Clone)]
pub struct MatteReader {
    source: SharedReader,
    layout: Layout,
}

impl MatteReader {
    pub fn open(path: &Path) -> io::Result<Self> {
        #[cfg(any(unix, windows))]
        {
            // A trim extension may atomically replace this path with a larger matte and move
            // its index. Keep this descriptor for the lifetime of its parsed layout, even
            // when the shared media-reader pool is busy opening other files.
            let pool = std::sync::Arc::new(crate::reader::FilePool::new(1));
            let file = crate::reader::FileReader::open_in(path, &pool)?;
            Self::from_reader(std::sync::Arc::new(file))
        }
        #[cfg(not(any(unix, windows)))]
        {
            let _ = path;
            Err(io::Error::new(io::ErrorKind::Unsupported, "person matte cache is unavailable"))
        }
    }

    #[cfg(any(unix, windows, test))]
    fn from_reader(source: SharedReader) -> io::Result<Self> {
        let mut header = [0; HEADER_BYTES as usize];
        source.read_at(0, &mut header)?;
        let (magic, fields) = header.split_at(8);
        if magic != MAGIC {
            return Err(invalid("unrecognized person matte format"));
        }
        let bad_header = || invalid("invalid person matte header");
        let (width, fields) = fields.split_first_chunk::<4>().ok_or_else(bad_header)?;
        let (height, fields) = fields.split_first_chunk::<4>().ok_or_else(bad_header)?;
        let frames = fields.first_chunk::<8>().ok_or_else(bad_header)?;
        let layout = Layout::new(u32::from_le_bytes(*width), u32::from_le_bytes(*height), u64::from_le_bytes(*frames))?;
        if source.len() != layout.key_offset(layout.frames) {
            return Err(invalid("person matte file length does not match its header"));
        }
        Ok(Self { source, layout })
    }

    pub fn frame_count(&self) -> u64 {
        self.layout.frames
    }

    /// Verify a complete source schedule before reusing a cache file.
    pub fn matches_keys(&self, keys: &[i64]) -> io::Result<bool> {
        if keys.len() as u64 != self.layout.frames {
            return Ok(false);
        }
        let mut expected = keys.iter().copied();
        let mut matches = true;
        self.visit_keys(|key| matches &= expected.next() == Some(key))?;
        Ok(matches)
    }

    /// Read the bounded presentation index, checking strict order without reading any alpha.
    pub fn keys(&self) -> io::Result<Vec<i64>> {
        let mut keys = Vec::new();
        keys.try_reserve_exact(self.layout.frames as usize).map_err(io::Error::other)?;
        self.visit_keys(|key| keys.push(key))?;
        Ok(keys)
    }

    /// Whether every requested presentation key has an exact matte. Both schedules must be
    /// strictly increasing. This scans the bounded index without reading any alpha.
    pub fn covers_keys(&self, mut keys: &[i64]) -> io::Result<bool> {
        if keys.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "requested person matte keys are not strictly increasing"));
        }
        self.visit_keys(|key| {
            if let Some((&wanted, rest)) = keys.split_first()
                && wanted == key
            {
                keys = rest;
            }
        })?;
        Ok(keys.is_empty())
    }

    fn visit_keys(&self, mut visit: impl FnMut(i64)) -> io::Result<()> {
        let mut buf = [0; 8192];
        let keys_per_block = buf.len() / KEY_BYTES as usize;
        let mut previous = None;
        for start in (0..self.layout.frames).step_by(keys_per_block) {
            let n = (self.layout.frames - start).min(keys_per_block as u64) as usize * KEY_BYTES as usize;
            let bytes = buf.get_mut(..n).ok_or_else(|| invalid("person matte index block exceeds its buffer"))?;
            self.source.read_at(self.layout.key_offset(start), bytes)?;
            for &stamp in bytes.as_chunks::<8>().0 {
                let key = i64::from_le_bytes(stamp);
                if previous.is_some_and(|prev| key <= prev) {
                    return Err(invalid("person matte index keys are not strictly increasing"));
                }
                previous = Some(key);
                visit(key);
            }
        }
        Ok(())
    }

    pub fn frame(&self, key: i64) -> io::Result<Option<MatteFrame>> {
        let (mut lo, mut hi) = (0, self.layout.frames);
        while lo < hi {
            let i = lo + (hi - lo) / 2;
            let mut stamp = [0; 8];
            self.source.read_at(self.layout.key_offset(i), &mut stamp)?;
            match i64::from_le_bytes(stamp).cmp(&key) {
                std::cmp::Ordering::Less => lo = i + 1,
                std::cmp::Ordering::Greater => hi = i,
                std::cmp::Ordering::Equal => {
                    let mut alpha = Vec::new();
                    alpha.try_reserve_exact(self.layout.pixels).map_err(io::Error::other)?;
                    alpha.resize(self.layout.pixels, 0);
                    self.source.read_at(self.layout.plane_offset(i), &mut alpha)?;
                    return Ok(Some(MatteFrame { width: self.layout.width, height: self.layout.height, alpha }));
                }
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_nonuniform_presentation_keys() {
        let path = std::env::temp_dir().join(format!("filmcraft-matte-{}.bin", std::process::id()));
        let mut writer = MatteWriter::new(File::create(&path).unwrap(), 2, 1, 3).unwrap();
        writer.push(-7, &[10, 20]).unwrap();
        writer.push(20, &[100, 200]).unwrap();
        assert!(writer.push(20, &[5, 6]).is_err());
        writer.push(150, &[255, 0]).unwrap();
        writer.finish().unwrap();
        let reader = MatteReader::open(&path).unwrap();
        assert_eq!(reader.frame_count(), 3);
        assert_eq!(reader.keys().unwrap(), [-7, 20, 150]);
        assert!(reader.matches_keys(&[-7, 20, 150]).unwrap());
        assert!(!reader.matches_keys(&[-7, 20, 151]).unwrap());
        assert!(!reader.matches_keys(&[20, 150]).unwrap());
        assert!(reader.covers_keys(&[20, 150]).unwrap());
        assert!(reader.covers_keys(&[-7, 150]).unwrap());
        assert!(reader.covers_keys(&[]).unwrap());
        assert!(!reader.covers_keys(&[19, 150]).unwrap());
        assert_eq!(reader.covers_keys(&[150, 20]).unwrap_err().kind(), io::ErrorKind::InvalidInput);
        assert_eq!(reader.frame(150).unwrap().unwrap().alpha, [255, 0]);
        assert_eq!(reader.frame(-7).unwrap().unwrap().alpha, [10, 20]);
        assert_eq!(reader.frame(20).unwrap().unwrap().alpha, [100, 200]);
        assert!(reader.frame(19).unwrap().is_none());
        drop(reader);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn coverage_checks_keys_across_index_blocks() {
        let keys: Vec<i64> = (0..1030).map(|i| i * 3).collect();
        let open = |keys: &[i64]| {
            let mut bytes = MAGIC.to_vec();
            bytes.extend_from_slice(&1u32.to_le_bytes());
            bytes.extend_from_slice(&1u32.to_le_bytes());
            bytes.extend_from_slice(&(keys.len() as u64).to_le_bytes());
            bytes.extend(std::iter::repeat_n(255, keys.len()));
            bytes.extend(keys.iter().flat_map(|key| key.to_le_bytes()));
            MatteReader::from_reader(std::sync::Arc::new(crate::reader::MemReader(bytes.into()))).unwrap()
        };
        let reader = open(&keys);
        assert_eq!(reader.keys().unwrap(), keys);
        assert!(reader.covers_keys(&[300, 3069, 3072, 3087]).unwrap());
        assert!(!reader.covers_keys(&[300, 3070, 3087]).unwrap());
        // A repeated or backwards key at a block boundary makes binary search unreliable,
        // even when all requested keys occurred earlier in the index.
        for malformed_key in [3069, 3066] {
            let mut malformed = keys.clone();
            malformed[1024] = malformed_key;
            assert_eq!(open(&malformed).covers_keys(&[300]).unwrap_err().kind(), io::ErrorKind::InvalidData);
            assert_eq!(open(&malformed).keys().unwrap_err().kind(), io::ErrorKind::InvalidData);
        }
    }

    #[cfg(unix)]
    #[test]
    fn open_reader_keeps_its_snapshot_when_an_expansion_replaces_the_path() {
        let dir = std::env::temp_dir().join(format!("filmcraft-matte-replace-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("matte.bin");
        let staging = dir.join("expanded.bin");
        let mut original = MatteWriter::new(File::create(&path).unwrap(), 1, 1, 1).unwrap();
        original.push(7, &[137]).unwrap();
        original.finish().unwrap();
        let open_before = MatteReader::open(&path).unwrap();

        let mut expanded = MatteWriter::new(File::create(&staging).unwrap(), 1, 1, 2).unwrap();
        expanded.push(7, &[137]).unwrap();
        expanded.push(9, &[73]).unwrap();
        expanded.finish().unwrap();
        std::fs::rename(staging, &path).unwrap();
        // Opening this many sources evicts the oldest entry in the shared FileReader pool.
        let other_readers: Vec<_> = (0..=crate::reader::MAX_OPEN_FILES).map(|_| crate::reader::FileReader::open(&path).unwrap()).collect();

        assert_eq!(open_before.keys().unwrap(), [7]);
        assert_eq!(open_before.frame(7).unwrap().unwrap().alpha, [137]);
        assert!(open_before.frame(9).unwrap().is_none());
        let open_after = MatteReader::open(&path).unwrap();
        assert_eq!(open_after.keys().unwrap(), [7, 9]);
        assert_eq!(open_after.frame(9).unwrap().unwrap().alpha, [73]);

        drop((open_before, open_after, other_readers));
        std::fs::remove_file(path).unwrap();
        std::fs::remove_dir(dir).unwrap();
    }

    #[test]
    fn rejects_damaged_headers_and_partial_frames() {
        let mut bytes = MAGIC.to_vec();
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&1u64.to_le_bytes());
        bytes.extend_from_slice(&[255, 0]);
        bytes.extend_from_slice(&42i64.to_le_bytes());
        let open = |bytes: &[u8]| MatteReader::from_reader(std::sync::Arc::new(crate::reader::MemReader(bytes.into())));
        for n in 0..bytes.len() {
            assert!(open(&bytes[..n]).is_err());
        }
        assert!(open(&bytes).is_ok());
        for i in 0..bytes.len() {
            let mut damaged = bytes.clone();
            damaged[i] ^= 0x80;
            assert!(
                std::panic::catch_unwind(|| {
                    if let Ok(reader) = open(&damaged) {
                        let _ = reader.frame(42);
                    }
                })
                .is_ok()
            );
        }
        bytes[8..12].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(open(&bytes).is_err());
    }
}
