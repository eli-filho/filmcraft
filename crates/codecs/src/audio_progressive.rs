//! Opt-in native import preparation: publish a playable prefix, then decode the rest.
//!
//! The ordinary audio opener remains synchronous. Importers using this source must poll
//! `readiness` and keep pending reads out of waveform/render caches.

use filmcraft_media::{MediaError, SharedReader, SharedSource};

/// Prepare standalone MP3, FLAC or AIFF without reading or decoding the whole file first.
/// `None` keeps the ordinary importer for formats without trustworthy length metadata, and
/// when another progressive decoder is already running. Browser imports retain their existing
/// asynchronous byte-fetching path.
pub fn open_progressive_audio(name: &str, reader: SharedReader) -> Result<Option<SharedSource>, MediaError> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        native::open(name, reader)
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = (name, reader);
        Ok(None)
    }
}

#[cfg(not(target_arch = "wasm32"))]
mod native {
    use std::io::{self, Read, Seek, SeekFrom};
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Arc, Mutex, PoisonError, Weak};

    use filmcraft_frame::{AudioBuffer, VideoFrame};
    use filmcraft_media::{
        AudioStreamInfo, FrameRequest, MediaInfo, MediaKind, MediaReadiness, MediaSource, bounded_audio_dimensions, validate_audio_buffer_shape,
    };
    use filmcraft_time::Tick;
    use symphonia::core::codecs::{CODEC_TYPE_MP3, CodecParameters};
    use symphonia::core::errors::Error;
    use symphonia::core::formats::{FormatOptions, FormatReader};
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    use super::{MediaError, SharedReader, SharedSource};
    use crate::audio::{MAX_DECODED_SAMPLES, SymphoniaDecoder, guard_audio};
    use crate::{CodecError, Result};

    // Full-file decoding is CPU and memory intensive: an import batch must not start a worker
    // per file. Other imports use their normal preparation worker while this slot is occupied.
    static DECODING: AtomicBool = AtomicBool::new(false);
    const MAX_PREPARED_CHUNKS: usize = 1_000_000;

    struct Slot;

    impl Slot {
        fn take() -> Option<Self> {
            DECODING.compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire).ok().map(|_| Self)
        }
    }

    impl Drop for Slot {
        fn drop(&mut self) {
            DECODING.store(false, Ordering::Release);
        }
    }

    struct Reader {
        inner: SharedReader,
        position: u64,
    }

    impl Read for Reader {
        fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
            let n = self.inner.len().saturating_sub(self.position).min(buf.len() as u64) as usize;
            if n == 0 {
                return Ok(0);
            }
            self.inner.read_at(self.position, &mut buf[..n])?;
            self.position = self.position.checked_add(n as u64).ok_or_else(|| io::Error::other("audio read offset overflow"))?;
            Ok(n)
        }
    }

    impl Seek for Reader {
        fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
            let position = match from {
                SeekFrom::Start(p) => i128::from(p),
                SeekFrom::Current(d) => i128::from(self.position) + i128::from(d),
                SeekFrom::End(d) => i128::from(self.inner.len()) + i128::from(d),
            };
            self.position = u64::try_from(position).map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "invalid audio seek"))?;
            Ok(self.position)
        }
    }

    impl symphonia::core::io::MediaSource for Reader {
        fn is_seekable(&self) -> bool {
            true
        }
        fn byte_len(&self) -> Option<u64> {
            Some(self.inner.len())
        }
    }

    struct Chunk {
        start: usize,
        frames: usize,
        channels: Vec<Vec<f32>>,
    }

    #[derive(Default)]
    struct Prepared {
        chunks: Vec<Arc<Chunk>>,
        frames: usize,
        done: bool,
        error: Option<String>,
    }

    struct ProgressiveAudio {
        info: MediaInfo,
        rate: u32,
        channels: usize,
        frames: usize,
        prepared: Arc<Mutex<Prepared>>,
    }

    struct Decoder {
        format: Box<dyn FormatReader>,
        decoder: SymphoniaDecoder,
        track: u32,
        channels: usize,
        frames: usize,
    }

    impl Decoder {
        fn next(&mut self, prepared: &mut Prepared) -> Result<bool> {
            let packet = match self.format.next_packet() {
                Ok(packet) => packet,
                Err(Error::IoError(e)) if e.kind() == io::ErrorKind::UnexpectedEof => {
                    if prepared.frames != self.frames {
                        return Err(CodecError::Decode(format!("audio ended after {} samples; its metadata declared {}", prepared.frames, self.frames)));
                    }
                    prepared.done = true;
                    return Ok(false);
                }
                Err(e) => return Err(CodecError::Decode(format!("reading audio: {e}"))),
            };
            if packet.track_id() != self.track {
                return Ok(true);
            }
            // A corrupt packet cannot silently shorten a source whose duration has already
            // been placed on the timeline. Keep its failure visible to the import coordinator.
            let channels = self.decoder.decode(&packet)?.map_err(|e| CodecError::Decode(format!("preparing audio: {e}")))?;
            let frames = channels.first().map_or(0, Vec::len);
            if channels.len() != self.channels || channels.iter().any(|c| c.len() != frames) {
                return Err(CodecError::Decode("audio channel layout changed while preparing the file".into()));
            }
            let end = prepared.frames.checked_add(frames).ok_or_else(|| CodecError::Decode("audio sample count overflow".into()))?;
            if end > self.frames || end.checked_mul(self.channels).is_none_or(|n| n > MAX_DECODED_SAMPLES) {
                return Err(CodecError::Decode("decoded audio exceeds its declared length or memory budget".into()));
            }
            if frames > 0 {
                prepared.chunks.push(Arc::new(Chunk { start: prepared.frames, frames, channels }));
                prepared.frames = end;
            }
            Ok(true)
        }
    }

    pub(super) fn open(name: &str, reader: SharedReader) -> std::result::Result<Option<SharedSource>, MediaError> {
        let ext = name.rsplit('.').next().unwrap_or("").to_ascii_lowercase();
        if !matches!(ext.as_str(), "mp3" | "flac" | "aif" | "aiff" | "aifc") {
            return Ok(None);
        }
        let Some(slot) = Slot::take() else { return Ok(None) };
        guard_audio("opening progressive audio", || prepare(name, &ext, reader, slot, SymphoniaDecoder::new)).map_err(Into::into)
    }

    fn prepare(
        name: &str,
        ext: &str,
        reader: SharedReader,
        slot: Slot,
        make_decoder: impl FnOnce(&CodecParameters) -> Result<SymphoniaDecoder>,
    ) -> Result<Option<SharedSource>> {
        let file_size = reader.len();
        let mss = MediaSourceStream::new(Box::new(Reader { inner: reader, position: 0 }), Default::default());
        let mut hint = Hint::new();
        hint.with_extension(ext);
        let options = FormatOptions { enable_gapless: ext == "mp3", ..FormatOptions::default() };
        let probed =
            symphonia::default::get_probe().format(&hint, mss, &options, &MetadataOptions::default()).map_err(|e| CodecError::Unsupported(e.to_string()))?;
        let format = probed.format;
        let Some(track) = format.default_track().cloned() else { return Ok(None) };
        let p = &track.codec_params;
        // Layer I/II use FilmCraft's own decoder and retain that opener. Missing/estimated
        // timing is unsafe to publish as a fixed timeline duration before decoding finishes.
        if ext == "mp3" && (p.codec != CODEC_TYPE_MP3 || p.delay.is_none()) {
            return Ok(None);
        }
        let (Some(rate), Some(channels), Some(frames)) = (p.sample_rate, p.channels, p.n_frames) else { return Ok(None) };
        let channel_count = channels.count();
        bounded_audio_dimensions(rate, channel_count as u32).map_err(|e| CodecError::Decode(e.to_string()))?;
        if p.time_base.is_some_and(|t| u64::from(t.numer) * u64::from(rate) != u64::from(t.denom)) {
            return Ok(None);
        }
        let Some(frames) = usize::try_from(frames).ok().filter(|n| *n > 0 && n.checked_mul(channel_count).is_some_and(|n| n <= MAX_DECODED_SAMPLES)) else {
            return Ok(None);
        };
        let codec = symphonia::default::get_codecs().get_codec(p.codec).map(|d| d.short_name.to_uppercase()).unwrap_or_else(|| ext.to_uppercase());
        let info = MediaInfo {
            name: name.into(),
            kind: MediaKind::AudioOnly,
            duration: Tick::from_units(frames as i64, i64::from(rate)),
            video: None,
            audio_streams: vec![AudioStreamInfo { sample_rate: rate, channels: channel_count as u32, codec, bits_per_sample: p.bits_per_sample }],
            container: ext.to_uppercase(),
            start_timecode: None,
            file_size: Some(file_size),
        };
        let decoder = make_decoder(p)?;
        let mut decoder = Decoder { format, decoder, track: track.id, channels: channel_count, frames };
        let mut prepared = Prepared::default();
        // Prime enough samples for an ordinary playback buffer, without waiting for a full song.
        let prefix = (rate as usize / 4).max(1).min(frames);
        for _ in 0..4096 {
            if prepared.frames >= prefix || !decoder.next(&mut prepared)? {
                break;
            }
        }
        if prepared.frames == 0 {
            return Err(CodecError::Decode("no audio decoded while preparing the initial prefix".into()));
        }
        let prepared = Arc::new(Mutex::new(prepared));
        if !prepared.lock().unwrap_or_else(PoisonError::into_inner).done {
            let weak = Arc::downgrade(&prepared);
            std::thread::Builder::new()
                .name("filmcraft-audio-import".into())
                .spawn(move || run(decoder, weak, slot))
                .map_err(|e| CodecError::Decode(format!("starting audio preparation: {e}")))?;
        }
        Ok(Some(Arc::new(ProgressiveAudio { info, rate, channels: channel_count, frames, prepared })))
    }

    fn run(mut decoder: Decoder, target: Weak<Mutex<Prepared>>, _slot: Slot) {
        let result = guard_audio("preparing progressive audio", || {
            loop {
                let Some(shared) = target.upgrade() else { return Ok(()) };
                let frames = shared.lock().unwrap_or_else(PoisonError::into_inner).frames;
                drop(shared);
                // Decode and allocate outside the shared lock: playback only waits to clone a
                // few immutable chunks, never for a decoder or a growing whole-file buffer.
                let mut next = Prepared { frames, ..Prepared::default() };
                let more = decoder.next(&mut next)?;
                let Some(target) = target.upgrade() else { return Ok(()) };
                let mut state = target.lock().unwrap_or_else(PoisonError::into_inner);
                // Very short hostile packets must not bypass the sample budget by creating
                // an unbounded number of chunk headers and channel allocations.
                if state.chunks.len().saturating_add(next.chunks.len()) > MAX_PREPARED_CHUNKS {
                    return Err(CodecError::Decode("audio has too many decoded chunks to prepare in memory".into()));
                }
                state.frames = next.frames;
                state.done = next.done;
                state.chunks.extend(next.chunks);
                if !more {
                    return Ok(());
                }
            }
        });
        if let Err(error) = result
            && let Some(target) = target.upgrade()
        {
            target.lock().unwrap_or_else(PoisonError::into_inner).error = Some(error.to_string());
        }
    }

    impl MediaSource for ProgressiveAudio {
        fn info(&self) -> &MediaInfo {
            &self.info
        }

        fn readiness(&self) -> MediaReadiness {
            let prepared = self.prepared.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(error) = &prepared.error {
                MediaReadiness::Failed(error.clone())
            } else if prepared.done {
                MediaReadiness::Ready
            } else {
                MediaReadiness::Loading { available: Tick::from_units(prepared.frames as i64, i64::from(self.rate)), total: self.info.duration }
            }
        }

        fn video_frame(&self, _req: FrameRequest) -> std::result::Result<Arc<VideoFrame>, MediaError> {
            Err(MediaError::NoStream("video"))
        }

        fn audio(&self, start: i64, frames: usize, rate: u32) -> std::result::Result<AudioBuffer, MediaError> {
            bounded_audio_dimensions(rate, self.channels as u32)?;
            validate_audio_buffer_shape(self.channels, frames)?;
            let source_at = |frame: i128| frame * i128::from(self.rate) / i128::from(rate);
            let begin = source_at(i128::from(start).max(0)).min(self.frames as i128) as usize;
            let last = i128::from(start) + frames.saturating_sub(1) as i128;
            if frames == 0 || last < 0 || begin == self.frames {
                return Ok(AudioBuffer::silence(rate, self.channels, frames));
            }
            let interpolation = i128::from((last * i128::from(self.rate)) % i128::from(rate) != 0);
            let end = (source_at(last) + 1 + interpolation).min(self.frames as i128) as usize;
            let chunks = {
                let prepared = self.prepared.lock().unwrap_or_else(PoisonError::into_inner);
                if let Some(error) = &prepared.error {
                    return Err(MediaError::Decode(error.clone()));
                }
                if end > prepared.frames {
                    filmcraft_media::pending::mark();
                    return Err(MediaError::Io("audio is still preparing".into()));
                }
                let first = prepared.chunks.partition_point(|c| c.start.saturating_add(c.frames) <= begin);
                prepared.chunks.get(first..).unwrap_or_default().iter().take_while(|c| c.start < end).cloned().collect::<Vec<_>>()
            };
            let sample = |frame: usize, channel: usize| {
                let index = chunks.partition_point(|c| c.start.saturating_add(c.frames) <= frame);
                chunks.get(index).and_then(|c| c.channels.get(channel).and_then(|v| v.get(frame.saturating_sub(c.start)))).copied().unwrap_or(0.0)
            };
            let mut out = AudioBuffer::silence(rate, self.channels, frames);
            for i in 0..frames {
                let output_frame = i128::from(start) + i as i128;
                if output_frame < 0 {
                    continue;
                }
                let numerator = output_frame * i128::from(self.rate);
                let position = numerator / i128::from(rate);
                if position >= self.frames as i128 {
                    break;
                }
                let at = position as usize;
                let fraction = (numerator % i128::from(rate)) as f32 / rate as f32;
                for (channel, samples) in out.channels.iter_mut().enumerate() {
                    let a = sample(at, channel);
                    let b = sample(at.saturating_add(1).min(self.frames.saturating_sub(1)), channel);
                    if let Some(value) = samples.get_mut(i) {
                        *value = a + (b - a) * fraction;
                    }
                }
            }
            Ok(out)
        }
    }

    #[cfg(test)]
    mod tests {
        use std::sync::Condvar;
        use std::time::{Duration, Instant};

        use filmcraft_media::reader::{ByteReader, MemReader};

        use super::*;

        static WORKER_TEST: Mutex<()> = Mutex::new(());

        fn source(chunks: Vec<Vec<f32>>, total: usize, done: bool) -> ProgressiveAudio {
            let mut frames = 0;
            let chunks = chunks
                .into_iter()
                .map(|samples| {
                    let start = frames;
                    frames += samples.len();
                    Arc::new(Chunk { start, frames: samples.len(), channels: vec![samples] })
                })
                .collect();
            ProgressiveAudio {
                info: MediaInfo {
                    name: "synthetic.wav".into(),
                    kind: MediaKind::AudioOnly,
                    duration: Tick::from_units(total as i64, 48_000),
                    video: None,
                    audio_streams: vec![AudioStreamInfo { sample_rate: 48_000, channels: 1, codec: "PCM".into(), bits_per_sample: Some(16) }],
                    container: "WAV".into(),
                    start_timecode: None,
                    file_size: None,
                },
                rate: 48_000,
                channels: 1,
                frames: total,
                prepared: Arc::new(Mutex::new(Prepared { chunks, frames, done, error: None })),
            }
        }

        #[test]
        fn prepared_prefix_plays_while_later_reads_remain_pending() {
            let source = source(vec![vec![0.25; 4096]], 96_000, false);
            assert!(matches!(source.readiness(), MediaReadiness::Loading { .. }));
            let _ = filmcraft_media::pending::take();
            let audio = source.audio(0, 512, 48_000).unwrap();
            assert!(audio.channels[0].iter().all(|v| *v == 0.25));
            assert!(!filmcraft_media::pending::take());
            assert!(source.audio(96_000 - 512, 512, 48_000).is_err());
            assert!(filmcraft_media::pending::take(), "unfinished audio is retried, never cached as silence");
            assert!(source.audio(i64::MAX, 0, 48_000).unwrap().channels[0].is_empty());
            assert!(source.audio(96_000, 16, 48_000).unwrap().channels[0].iter().all(|v| *v == 0.0));
            assert!(!filmcraft_media::pending::take(), "empty reads and known out-of-range silence need no preparation");
        }

        #[test]
        fn chunk_boundaries_and_resampling_match_the_complete_source() {
            let samples: Vec<f32> = (0..300).map(|i| (i as f32 * 0.2).sin()).collect();
            let source = source(samples.chunks(7).map(<[f32]>::to_vec).collect(), samples.len(), true);
            for rate in [24_000, 44_100, 48_000, 96_000] {
                for start in [-10, 0, 4, 90, 300, i64::MAX] {
                    let actual = source.audio(start, 100, rate).unwrap();
                    if start == i64::MAX {
                        assert!(actual.channels[0].iter().all(|s| *s == 0.0));
                    } else {
                        let expected = crate::audio::read_resampled(std::slice::from_ref(&samples), 48_000, start, 100, rate);
                        for (a, b) in actual.channels[0].iter().zip(&expected.channels[0]) {
                            assert!((a - b).abs() < 1e-5, "rate {rate}, start {start}: {a} != {b}");
                        }
                    }
                }
            }
        }

        #[test]
        fn failures_are_permanent_and_hostile_requests_are_bounded() {
            let source = source(vec![vec![0.0; 16]], 16, false);
            assert!(source.audio(0, 4, 0).is_err());
            assert!(source.audio(0, usize::MAX, 48_000).is_err());
            source.prepared.lock().unwrap().error = Some("synthetic failure".into());
            let _ = filmcraft_media::pending::take();
            assert_eq!(source.readiness(), MediaReadiness::Failed("synthetic failure".into()));
            assert!(matches!(source.audio(0, 4, 48_000), Err(MediaError::Decode(_))));
            assert!(!filmcraft_media::pending::take(), "a failed worker must not leave an infinite loading state");
        }

        struct Gate {
            state: Mutex<(bool, bool)>,
            changed: Condvar,
        }

        impl Gate {
            fn release(&self) {
                self.state.lock().unwrap().1 = true;
                self.changed.notify_all();
            }

            fn wait_until_blocked(&self) {
                let state = self.state.lock().unwrap();
                let (state, _) = self.changed.wait_timeout_while(state, Duration::from_secs(3), |state| !state.0).unwrap();
                assert!(state.0, "worker did not reach the gated part of the file");
            }
        }

        struct Release(Arc<Gate>);
        impl Drop for Release {
            fn drop(&mut self) {
                self.0.release();
            }
        }

        struct GatedReader {
            bytes: MemReader,
            gate: Arc<Gate>,
            panic: bool,
        }

        impl ByteReader for GatedReader {
            fn len(&self) -> u64 {
                self.bytes.len()
            }
            fn read_at(&self, offset: u64, buf: &mut [u8]) -> io::Result<()> {
                if offset >= 256 * 1024 {
                    let mut state = self.gate.state.lock().unwrap();
                    state.0 = true;
                    self.gate.changed.notify_all();
                    while !state.1 {
                        state = self.gate.changed.wait(state).unwrap();
                    }
                    drop(state);
                    assert!(!self.panic, "synthetic reader panic after the playable prefix");
                }
                self.bytes.read_at(offset, buf)
            }
        }

        fn wait_ready(source: &SharedSource) -> MediaReadiness {
            let deadline = Instant::now() + Duration::from_secs(5);
            loop {
                let status = source.readiness();
                if !matches!(status, MediaReadiness::Loading { .. }) && !DECODING.load(Ordering::Acquire) {
                    return status;
                }
                assert!(Instant::now() < deadline, "progressive decoder stayed pending");
                std::thread::yield_now();
            }
        }

        fn gated(panic: bool) -> (SharedReader, Arc<Gate>, Release) {
            let samples = vec![0.25; 48_000 * 8];
            let gate = Arc::new(Gate { state: Mutex::new((false, false)), changed: Condvar::new() });
            let bytes = MemReader(filmcraft_media::wav::write_wav16(&samples, 1, 48_000).into());
            (Arc::new(GatedReader { bytes, gate: gate.clone(), panic }), gate.clone(), Release(gate))
        }

        #[test]
        fn a_slow_tail_does_not_delay_prefix_playback_and_completion_preserves_audio() {
            let _test = WORKER_TEST.lock().unwrap();
            let (reader, gate, _release) = gated(false);
            let slot = Slot::take().unwrap();
            let source = prepare("synthetic.wav", "wav", reader, slot, SymphoniaDecoder::new).unwrap().unwrap();
            gate.wait_until_blocked();
            assert!(matches!(source.readiness(), MediaReadiness::Loading { .. }));
            assert!(Slot::take().is_none(), "only one full-file preparation worker");
            let prefix = source.audio(0, 1024, 48_000).unwrap();
            assert!(prefix.channels[0].iter().all(|v| (*v - 0.25).abs() < 1e-4));
            gate.release();
            assert_eq!(wait_ready(&source), MediaReadiness::Ready);
            let tail = source.audio(48_000 * 7, 1024, 48_000).unwrap();
            assert_eq!(prefix.channels, tail.channels);
        }

        #[test]
        fn a_worker_panic_becomes_failed_instead_of_permanent_loading() {
            let _test = WORKER_TEST.lock().unwrap();
            let (reader, gate, _release) = gated(true);
            let slot = Slot::take().unwrap();
            let source = prepare("synthetic.wav", "wav", reader, slot, SymphoniaDecoder::new).unwrap().unwrap();
            gate.wait_until_blocked();
            gate.release();
            assert!(matches!(wait_ready(&source), MediaReadiness::Failed(error) if error.contains("panic")));
        }

        #[test]
        fn dropping_the_source_cancels_remaining_preparation() {
            let _test = WORKER_TEST.lock().unwrap();
            let (reader, gate, _release) = gated(false);
            let slot = Slot::take().unwrap();
            let source = prepare("synthetic.wav", "wav", reader, slot, SymphoniaDecoder::new).unwrap().unwrap();
            gate.wait_until_blocked();
            drop(source);
            gate.release();
            let deadline = Instant::now() + Duration::from_secs(3);
            while DECODING.load(Ordering::Acquire) {
                assert!(Instant::now() < deadline, "dropping the source did not release its decoder");
                std::thread::yield_now();
            }
        }

        #[test]
        fn unsupported_formats_do_not_start_progressive_work() {
            let reader = Arc::new(MemReader(Arc::from([0u8; 4])));
            assert!(open("sample.aac", reader).unwrap().is_none());
        }

        #[test]
        fn tagged_mp3_has_the_same_gapless_duration_and_samples_as_the_existing_importer() {
            let _test = WORKER_TEST.lock().unwrap();
            let bytes: Arc<[u8]> = crate::audio::gapless_tests::tagged_mp3().into();
            let expected = crate::audio::AudioFileSource::decode("tagged.mp3", bytes.clone()).unwrap();
            let source = open("tagged.mp3", Arc::new(MemReader(bytes))).unwrap().expect("tagged MP3 has exact timing");
            assert_eq!(source.info().duration, expected.info().duration);
            assert_eq!(wait_ready(&source), MediaReadiness::Ready);
            let actual = source.audio(0, 12_000, 44_100).unwrap();
            let expected = expected.audio(0, 12_000, 44_100).unwrap();
            assert_eq!(actual.channels, expected.channels);
        }
    }
}
