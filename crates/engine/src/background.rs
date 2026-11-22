//! Remove Background analyses the source pictures a timeline clip needs, reusing existing
//! coverage. Jobs never edit the project from a worker; expanded caches publish atomically.

use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex, OnceLock, PoisonError};

pub use filmcraft_media::person_matte::MatteFrame as Matte;
use filmcraft_media::person_matte::MatteReader;
use filmcraft_project::{ClipId, ItemId, MediaClip, MediaIdentity, MediaRef, PersonMatteRef, TrackItem, TrackKind};
use filmcraft_time::{FrameRate, Tick, TimeRange};
use serde_json::{Value, json};

use crate::commands::{CommandSpec, NamedKeys, bad};
use crate::{EngineError, Job, Result, Session};

const REMOVE: &str = "clip.removeBackground";
const RESTORE: &str = "clip.restoreBackground";
const TOGGLE: &str = "clip.toggleBackgroundRemoval";
const CACHE_PREFIX: &str = "vision-balanced-v1";
#[cfg(unix)]
const DEVICE_START: usize = CACHE_PREFIX.len() + 1 + 16 + 1 + 16 + 1 + 32 + 1;
#[cfg(unix)]
const DEVICE_END: usize = DEVICE_START + 16;
#[cfg(not(target_arch = "wasm32"))]
const MAX_CONCURRENT_JOBS: usize = 2;

/// A worker-local native segmenter. The engine owns the workflow; the host provides inference.
pub type Segmenter = Box<dyn FnMut(&filmcraft_frame::VideoFrame) -> std::result::Result<Matte, String>>;
pub type SegmenterFactory = fn() -> std::result::Result<Segmenter, String>;

static SEGMENTER: OnceLock<SegmenterFactory> = OnceLock::new();

/// Install the platform implementation at native app or CLI startup.
pub fn register(factory: SegmenterFactory) {
    let _ = SEGMENTER.set(factory);
}

pub(crate) fn commands() -> [CommandSpec; 3] {
    [
        CommandSpec {
            id: REMOVE,
            label: "Remove Background",
            menu: &["Clip"],
            shortcut: None,
            params: r#"{"clip":id?,"wait":bool=false}"#,
            enabled: |s| selected_clip(s, &Value::Null, REMOVE).map(|_| ()).map_err(|e| e.to_string()),
            run: remove,
            journal: true,
            named: NamedKeys::CLIP,
        },
        CommandSpec {
            id: TOGGLE,
            label: "Toggle Background Removal",
            menu: &[],
            shortcut: None,
            params: r#"{"clip":id?}"#,
            enabled: |s| selected_clip(s, &Value::Null, TOGGLE).and_then(|(_, it)| has_matte(it)).map_err(|e| e.to_string()),
            run: toggle,
            journal: true,
            named: NamedKeys::CLIP,
        },
        CommandSpec {
            id: RESTORE,
            label: "Restore Background",
            menu: &["Clip"],
            shortcut: None,
            params: r#"{"clip":id?}"#,
            enabled: |s| selected_clip(s, &Value::Null, RESTORE).and_then(|(_, it)| has_matte(it)).map_err(|e| e.to_string()),
            run: restore,
            journal: true,
            named: NamedKeys::CLIP,
        },
    ]
}

fn selected_clip<'a>(s: &'a Session, params: &Value, command: &str) -> Result<(ItemId, &'a TrackItem)> {
    let seq = s.state.active_sequence.ok_or(EngineError::NoSequence)?;
    let q = s.project.sequence(seq).ok_or(EngineError::NoSequence)?;
    let explicit = params.get("clip").map(|p| p.as_u64().map(ClipId).ok_or_else(|| bad(command, "`clip` must be a clip id"))).transpose()?;
    let clips = if explicit.is_some() { explicit.as_slice() } else { s.state.selection.as_slice() };
    let mut video = clips.iter().filter_map(|&id| {
        let (track, clip) = q.find_item(id)?;
        let track = q.track(track)?;
        (track.kind == TrackKind::Video).then_some((track, clip))
    });
    let (track, clip) = video.next().ok_or_else(|| EngineError::Other("select one video clip on the timeline".into()))?;
    if video.any(|(_, other)| other.id != clip.id) {
        return Err(EngineError::Other("select one video clip on the timeline".into()));
    }
    if track.locked {
        return Err(EngineError::Other("unlock the selected video track first".into()));
    }
    Ok((seq, clip))
}

fn has_matte(clip: &TrackItem) -> Result<()> {
    clip.person_matte.as_ref().map(|_| ()).ok_or_else(|| EngineError::Other("the selected clip has no background removal".into()))
}

fn restore(s: &mut Session, params: &Value) -> Result<Value> {
    let (_, clip) = selected_clip(s, params, RESTORE)?;
    has_matte(clip)?;
    let id = clip.id;
    edit_matte(s, id, "Restore Background", |matte| *matte = None)?;
    Ok(json!({"clip": id.0}))
}

fn toggle(s: &mut Session, params: &Value) -> Result<Value> {
    let (_, clip) = selected_clip(s, params, TOGGLE)?;
    has_matte(clip)?;
    let id = clip.id;
    edit_matte(s, id, "Toggle Background Removal", |matte| {
        if let Some(matte) = matte {
            matte.bypassed = !matte.bypassed;
        }
    })?;
    Ok(json!({"clip": id.0}))
}

fn edit_matte(s: &mut Session, id: ClipId, label: &str, edit: impl FnOnce(&mut Option<PersonMatteRef>)) -> Result<()> {
    s.edit_sequence(label, |q, _, _| {
        let (track, _) = q.find_item(id).ok_or_else(|| EngineError::Other("clip was removed".into()))?;
        let clip = q.track_mut(track).and_then(|t| t.item_mut(id)).ok_or_else(|| EngineError::Other("clip was removed".into()))?;
        edit(&mut clip.person_matte);
        Ok(())
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Target {
    seq: ItemId,
    clip: ClipId,
    item: ItemId,
}

/// A snapshot of the time mapping, including transition handles. Keep the clip itself so that
/// source-time calculations are the renderer's, including reverse and frame holds.
#[derive(Clone)]
struct Coverage {
    clip: TrackItem,
    timeline: TimeRange,
    sequence_rate: FrameRate,
    full: bool,
}

impl Coverage {
    fn new(s: &Session, seq: ItemId, clip: &TrackItem) -> Result<Self> {
        let q = s.project.sequence(seq).ok_or(EngineError::NoSequence)?;
        let (track, _) = q.find_item(clip.id).ok_or_else(|| EngineError::Other("clip was removed".into()))?;
        let track = q.track(track).ok_or_else(|| EngineError::Other("clip track was removed".into()))?;
        let (mut start, mut end) = (clip.start, clip.end());
        for transition in &track.transitions {
            if transition.from == Some(clip.id) || transition.to == Some(clip.id) {
                start = start.min(transition.range().start);
                end = end.max(transition.range().end());
            }
        }
        // Temporal effects can ask for pictures outside the visible interval. Their sampling
        // envelopes are not exposed, so retain full coverage rather than guess and miss a tap.
        let full = clip.effects.iter().any(|effect| effect.enabled && filmcraft_render::vfx::is_temporal(&effect.effect))
            || !clip.speed.is_finite()
            || clip.speed == 0.0
            || clip.duration.0 <= 0
            || q.settings.frame_rate.num <= 0
            || q.settings.frame_rate.den <= 0;
        Ok(Self { clip: clip.clone(), timeline: TimeRange::from_bounds(start, end), sequence_rate: q.settings.frame_rate, full })
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn same_request(&self, other: &Self) -> bool {
        (self.full && other.full)
            || (self.full == other.full
                && self.clip.source_in == other.clip.source_in
                && self.clip.duration == other.clip.duration
                && self.clip.speed == other.clip.speed
                && self.clip.reverse == other.clip.reverse
                && self.clip.frame_hold == other.clip.frame_hold
                && self.clip.time_interpolation == other.clip.time_interpolation
                && self.sequence_rate == other.sequence_rate
                && self.timeline.start - self.clip.start == other.timeline.start - other.clip.start
                && self.timeline.end() - self.clip.start == other.timeline.end() - other.clip.start)
    }

    fn keys(&self, src: &filmcraft_media::SharedSource) -> std::result::Result<Vec<i64>, String> {
        use filmcraft_media::person_matte::MAX_MATTE_FRAMES;
        let all = || src.matte_frame_keys(MAX_MATTE_FRAMES).map_err(|e| e.to_string());
        if self.full {
            return all();
        }
        if let Some(hold) = self.clip.frame_hold {
            return src.matte_frame_key(hold).map(|key| vec![key]).ok_or_else(|| "The held source frame is unavailable".into());
        }
        let a = self.clip.source_time_at(self.timeline.start);
        let b = self.clip.source_time_at(self.timeline.end());
        let span = if self.clip.reverse { TimeRange::from_bounds(b + Tick(1), a + Tick(1)) } else { TimeRange::from_bounds(a, b) };
        if span.is_empty() || (span.start <= Tick::ZERO && span.end() >= src.info().duration) {
            return all();
        }
        let mut keys = src.matte_frame_keys_in_range(span, MAX_MATTE_FRAMES).map_err(|e| e.to_string())?;
        let rate = src.info().frame_rate();
        // Preserve the renderer's exact lookups at both boundaries, including off-grid nested
        // playback and custom export rates. Container rounding can make a request just BEFORE
        // Out display the next picture; retain that guard frame instead of holding a wrong matte.
        for t in [self.timeline.start, self.timeline.end() - Tick(1)] {
            let time = filmcraft_render::video_source_time(&self.clip, t, rate);
            if let Some(key) = src.matte_frame_key(time) {
                keys.push(key);
            }
            if let Some((time, _)) = filmcraft_render::interpolation_blend(&self.clip, t, rate, self.sequence_rate)
                && let Some(key) = src.matte_frame_key(time)
            {
                keys.push(key);
            }
        }
        if self.clip.time_interpolation != filmcraft_project::TimeInterpolation::FrameSampling && rate.frame_duration().0 > 0 {
            // An interpolation tap just inside Out can round to a raw PTS at or beyond it.
            // Include the last eligible nominal tap, without changing the renderer's timing.
            let tail = rate.tick_of(rate.frame_at(self.clip.source_out() - Tick(1)));
            if tail >= span.start
                && tail < self.clip.source_out()
                && let Some(key) = src.matte_frame_key(tail)
            {
                keys.push(key);
            }
        }
        keys.sort_unstable();
        keys.dedup();
        if keys.len() > MAX_MATTE_FRAMES {
            return Err("Too many source frames for background removal".into());
        }
        Ok(keys)
    }
}

pub(super) struct Pending {
    job: Job,
    epoch: u64,
    #[cfg(not(target_arch = "wasm32"))]
    file: String,
    #[cfg(not(target_arch = "wasm32"))]
    coverage: Coverage,
    targets: Vec<Target>,
    result: Arc<Mutex<Option<std::result::Result<Analysis, String>>>>,
}

struct Analysis {
    matte: PersonMatteRef,
    report: filmcraft_export::Report,
    cached: bool,
    source: filmcraft_media::SharedSource,
    covered: Vec<i64>,
}

#[cfg(not(target_arch = "wasm32"))]
struct Analysed {
    matte: PersonMatteRef,
    frames: u64,
    cached: bool,
    source: filmcraft_media::SharedSource,
    covered: Vec<i64>,
}

/// The analysis job targeting this clip in the current document, until it is applied.
pub fn job_for_clip(s: &Session, seq: ItemId, clip: ClipId) -> Option<&Job> {
    s.background_jobs.iter().find(|p| p.epoch == s.document_epoch && p.targets.iter().any(|t| t.seq == seq && t.clip == clip)).map(|p| &p.job)
}

fn source_file(clip: &MediaClip) -> Option<&str> {
    match &clip.media {
        MediaRef::File { path } if !clip.offline && clip.info.kind == filmcraft_media::MediaKind::Movie => Some(path),
        _ => None,
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn remove(s: &mut Session, params: &Value) -> Result<Value> {
    let factory =
        SEGMENTER.get().copied().ok_or_else(|| EngineError::Other("Remove Background requires FilmCraft desktop or CLI on macOS 12 or later".into()))?;
    start(s, params, factory, s.media.matte_dir().ok_or_else(|| EngineError::Other("FilmCraft's local data folder is unavailable".into()))?)
}

#[cfg(target_arch = "wasm32")]
fn remove(_: &mut Session, _: &Value) -> Result<Value> {
    Err(EngineError::Other("Remove Background requires FilmCraft desktop or CLI on macOS 12 or later".into()))
}

#[cfg(not(target_arch = "wasm32"))]
fn start(s: &mut Session, params: &Value, factory: SegmenterFactory, cache: PathBuf) -> Result<Value> {
    let (seq, clip) = selected_clip(s, params, REMOVE)?;
    let (item, source, _) = s.project.resolve_media(clip.item).ok_or_else(|| EngineError::Other("select a clip from an original video file".into()))?;
    let path = source_file(source).ok_or_else(|| EngineError::Other("Remove Background needs an online original video file".into()))?;
    let file = s.media.follow(path).map_err(|e| EngineError::Other(format!("Link the original video before removing its background: {e}")))?;
    let target = Target { seq, clip: clip.id, item };
    let coverage = Coverage::new(s, seq, clip)?;
    let wait = match params.get("wait") {
        Some(v) => v.as_bool().ok_or_else(|| bad(REMOVE, "`wait` must be true or false"))?,
        None => false,
    };
    if let Some(pending) = s.background_jobs.iter_mut().find(|p| p.epoch == s.document_epoch && p.file == file) {
        if !pending.coverage.same_request(&coverage) {
            return Err(EngineError::Other(
                "Background removal is already running for a different clip range. Wait for it or cancel it in Window > Progress.".into(),
            ));
        }
        if wait || pending.job.progress.cancel.load(Ordering::Relaxed) {
            return Err(EngineError::Other("Background removal for this source is already in progress. Check Window > Progress.".into()));
        }
        if !pending.targets.contains(&target) {
            pending.targets.push(target);
        }
        return Ok(json!({"job": pending.job.id, "clip": target.clip.0}));
    }
    // Each job owns a decoder and a Vision sequence. Bound their shared CPU/GPU and memory
    // demand; cancelled jobs keep their slot until their worker has actually returned.
    if s.background_jobs.len() >= MAX_CONCURRENT_JOBS {
        return Err(EngineError::Other("Two background removal jobs are already running. Wait for one to finish or cancel it in Window > Progress.".into()));
    }

    let id = s.jobs.iter().map(|j| j.id).max().unwrap_or(0).checked_add(1).ok_or_else(|| EngineError::Other("too many jobs".into()))?;
    let job = Job { id, label: "Removing Background".into(), progress: Default::default(), result: Default::default() };
    *lock(&job.progress.status) = "Reading original video".into();
    let pending =
        Pending { job: job.clone(), epoch: s.document_epoch, file: file.clone(), coverage: coverage.clone(), targets: vec![target], result: Arc::default() };
    let (progress, result) = (job.progress.clone(), pending.result.clone());
    let worker = Work { file, pool: s.media.clone(), services: s.services.clone(), factory, cache, coverage };
    let run = move || {
        let started = web_time::Instant::now();
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| worker.analyse(&progress)))
            .unwrap_or_else(|_| Err("Internal error during background removal. See the crash log.".into()))
            .map(|Analysed { matte, frames, cached, source, covered }| {
                let seconds = started.elapsed().as_secs_f64();
                let report = filmcraft_export::Report {
                    path: String::new(),
                    frames,
                    seconds,
                    bytes: 0,
                    render_fps: frames as f64 / seconds.max(1e-6),
                    extra_files: Vec::new(),
                };
                Analysis { matte, report, cached, source, covered }
            });
        *lock(&result) = Some(r);
    };
    if !wait {
        std::thread::Builder::new()
            .name("filmcraft-person-matte".into())
            .spawn(run)
            .map_err(|e| EngineError::Other(format!("Could not start background removal: {e}")))?;
        s.jobs.push(job);
        s.background_jobs.push(pending);
        return Ok(json!({"job": id, "clip": target.clip.0}));
    }
    s.jobs.push(job.clone());
    s.background_jobs.push(pending);
    run();
    poll(s);
    let result = lock(&job.result).clone();
    match result {
        Some(Ok(report)) => Ok(json!({"job": id, "clip": target.clip.0, "frames": report.frames})),
        Some(Err(message)) => Err(EngineError::Other(message)),
        None => Err(EngineError::Other("Background removal did not return a result".into())),
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct Work {
    file: String,
    pool: Arc<crate::MediaPool>,
    services: Arc<dyn crate::Services>,
    factory: SegmenterFactory,
    cache: PathBuf,
    coverage: Coverage,
}

#[cfg(not(target_arch = "wasm32"))]
impl Work {
    fn analyse(&self, progress: &filmcraft_export::Progress) -> std::result::Result<Analysed, String> {
        use filmcraft_media::person_matte::{MAX_MATTE_FRAMES, MatteWriter};
        let check_cancel = || {
            if progress.cancel.load(Ordering::Relaxed) { Err("Background removal cancelled".to_string()) } else { Ok(()) }
        };
        check_cancel()?;
        let source_error = |e| format!("Could not read the original video: {e}");
        let cache_error = |e| format!("Could not write the local background-removal cache: {e}");
        let metadata = std::fs::metadata(&self.file).map_err(source_error)?;
        let source = crate::relink::identity_of(&*self.services, &self.file).map_err(source_error)?;
        let cache_id = cache_id_for(&metadata, source).map_err(source_error)?;
        let mut reference = PersonMatteRef { cache_id, source, bypassed: false };
        let src = self.pool.probe_file(&self.file, &*self.services).map_err(|e| format!("Could not open the original video: {e}"))?;
        let all = src.matte_frame_keys(MAX_MATTE_FRAMES).map_err(|e| e.to_string())?;
        let required = self.coverage.keys(&src)?;
        if required.is_empty() {
            return Err("The selected clip has no source video frames to analyse".into());
        }
        check_cancel()?;
        if !valid_source(&self.file, &reference) {
            return Err("The original video changed while opening it. Run Remove Background again.".into());
        }
        std::fs::create_dir_all(&self.cache).map_err(cache_error)?;
        let _cache_locks = lock_cache(&self.cache, &reference.cache_id)?;
        #[cfg(unix)]
        let _previous_source_lock = lock_previous_source(&self.cache, &reference.cache_id, &metadata)?;
        check_cancel()?;
        let attached = self.coverage.clip.person_matte.as_ref().map(|matte| matte.cache_id.as_str());
        let existing = find_existing_matte(&self.cache, &reference.cache_id, attached, &all, &required, check_cancel)?;
        if let Some(existing) = &existing {
            reference.cache_id.clone_from(&existing.id);
        }
        let path = self.cache.join(format!("{}.matte", reference.cache_id));
        let old_keys = existing.as_ref().map_or(&[][..], |matte| matte.keys.as_slice());
        let missing = required.iter().filter(|key| old_keys.binary_search(key).is_err()).count() as u64;
        if missing == 0 {
            progress.total.store(required.len() as u64, Ordering::Relaxed);
            progress.done.store(required.len() as u64, Ordering::Relaxed);
            return Ok(Analysed { matte: reference, frames: required.len() as u64, cached: true, source: src, covered: old_keys.to_vec() });
        }
        progress.total.store(missing, Ordering::Relaxed);
        *lock(&progress.status) = format!("Analysing {missing} uncached source frames used by this clip");
        let mut covered = old_keys.to_vec();
        covered.extend(required);
        covered.sort_unstable();
        covered.dedup();
        let stage = tempfile::NamedTempFile::new_in(&self.cache).map_err(cache_error)?;
        let mut writer: Option<MatteWriter> = None;
        let mut dimensions = None;
        let mut segment: Option<Segmenter> = None;
        let mut done = 0u64;
        for &key in &covered {
            check_cancel()?;
            let matte = if old_keys.binary_search(&key).is_ok() {
                // A later missing interval starts a new Vision sequence. Reusing a handler
                // across omitted cached frames would give it discontinuous temporal history.
                segment = None;
                existing
                    .as_ref()
                    .and_then(|matte| matte.reader.frame(key).ok().flatten())
                    .ok_or_else(|| "The existing background matte could not be read. The cache was left unchanged.".to_owned())?
            } else {
                let frame = src.matte_frame(key).map_err(|e| format!("Could not decode a source picture: {e}"))?;
                if frame.width == 0 || frame.height == 0 || u64::from(frame.width) * u64::from(frame.height) > 8192 * 4320 {
                    return Err("Remove Background requires a source picture no larger than 8192 × 4320 pixels".into());
                }
                if segment.is_none() {
                    segment = Some((self.factory)()?);
                }
                let analyse = segment.as_mut().ok_or_else(|| "Could not create background removal".to_owned())?;
                let matte = analyse(&frame)?;
                done = done.saturating_add(1);
                progress.done.store(done, Ordering::Relaxed);
                *lock(&progress.status) = format!("Analysed {done} of {missing} uncached source frames used by this clip");
                matte
            };
            let size = (matte.width, matte.height);
            if let Some(expected) = dimensions {
                if size != expected {
                    return Err("The source changed video dimensions during background removal".into());
                }
            } else {
                dimensions = Some(size);
                writer = Some(MatteWriter::new(stage.reopen().map_err(cache_error)?, matte.width, matte.height, covered.len() as u64).map_err(cache_error)?);
            }
            writer.as_mut().ok_or_else(|| "Could not create the background matte cache".to_owned())?.push(key, &matte.alpha).map_err(cache_error)?;
        }
        writer.ok_or_else(|| "No background matte frames were produced".to_owned())?.finish().map_err(cache_error)?;
        check_cancel()?;
        if !valid_source(&self.file, &reference) {
            return Err("The original video changed during background removal. Run Remove Background again.".into());
        }
        stage.persist(&path).map_err(|e| cache_error(e.error))?;
        Ok(Analysed { matte: reference, frames: missing, cached: false, source: src, covered })
    }
}

#[cfg(not(target_arch = "wasm32"))]
struct ExistingMatte {
    id: String,
    reader: MatteReader,
    keys: Vec<i64>,
    covered: usize,
}

/// Prefer an attached cache when it covers the request. Otherwise choose the matching file that
/// saves the most inference. Old files retain their names and references across remounts.
#[cfg(not(target_arch = "wasm32"))]
fn find_existing_matte(
    cache: &Path,
    canonical: &str,
    attached: Option<&str>,
    all: &[i64],
    required: &[i64],
    check_cancel: impl Fn() -> std::result::Result<(), String>,
) -> std::result::Result<Option<ExistingMatte>, String> {
    let mut best: Option<ExistingMatte> = None;
    let mut consider = |id: &str| {
        let path = cache.join(format!("{id}.matte"));
        let Some((reader, keys)) = std::fs::metadata(&path).ok().filter(std::fs::Metadata::is_file).and_then(|_| {
            let reader = MatteReader::open(&path).ok()?;
            let keys = reader.keys().ok()?;
            keys.iter().all(|key| all.binary_search(key).is_ok()).then_some((reader, keys))
        }) else {
            return false;
        };
        let covered = required.iter().filter(|key| keys.binary_search(key).is_ok()).count();
        if best.as_ref().is_none_or(|best| covered > best.covered) {
            best = Some(ExistingMatte { id: id.to_owned(), reader, keys, covered });
        }
        covered == required.len()
    };
    check_cancel()?;
    let attached = attached.filter(|id| same_cache_source(canonical, id));
    if attached.is_some_and(&mut consider) || (attached != Some(canonical) && consider(canonical)) {
        return Ok(best);
    }

    #[cfg(unix)]
    {
        // Discovery is needed only when the known names don't cover the clip. Bound both the
        // directory walk and the number of indexes it may open in a damaged cache directory.
        const MAX_ENTRIES: usize = 16_384;
        const MAX_ALIASES: usize = 64;
        let list_error = |e| format!("Could not look for existing background mattes: {e}");
        let mut aliases = 0;
        for (i, entry) in std::fs::read_dir(cache).map_err(list_error)?.enumerate() {
            check_cancel()?;
            if i >= MAX_ENTRIES {
                return Err("Too many entries in the background-removal cache to look for an existing matte".into());
            }
            let name = entry.map_err(list_error)?.file_name();
            let Some(id) = name.to_str().and_then(|name| name.strip_suffix(".matte")) else { continue };
            if id == canonical || attached == Some(id) || !same_cache_source(canonical, id) {
                continue;
            }
            aliases += 1;
            if aliases > MAX_ALIASES {
                return Err("Too many matching background mattes to reuse this source safely".into());
            }
            if consider(id) {
                break;
            }
        }
    }
    Ok(best)
}

/// Hold both handles through cache publication. Independent sources may write in parallel,
/// but two ranges of one source must not replace each other's newly generated coverage.
#[cfg(not(target_arch = "wasm32"))]
fn lock_cache(cache: &Path, cache_id: &str) -> std::result::Result<(std::fs::File, std::fs::File), String> {
    let cache_error = |e| format!("Could not lock the local background-removal cache: {e}");
    // Previous builds hold this gate exclusively. Sharing it lets new builds analyse different
    // sources together while still excluding an older writer that has no per-source lock.
    let legacy = open_cache_lock(&cache.with_extension("lock")).map_err(cache_error)?;
    legacy.try_lock_shared().map_err(|e| match e {
        std::fs::TryLockError::WouldBlock => "Background removal is using this cache in another session. Wait for that job, then try again.".to_owned(),
        std::fs::TryLockError::Error(e) => cache_error(e),
    })?;
    Ok((legacy, lock_source(cache, cache_id)?))
}

#[cfg(not(target_arch = "wasm32"))]
fn open_cache_lock(path: &Path) -> std::io::Result<std::fs::File> {
    std::fs::File::options().read(true).write(true).create(true).truncate(false).open(path)
}

#[cfg(not(target_arch = "wasm32"))]
fn lock_source(cache: &Path, cache_id: &str) -> std::result::Result<std::fs::File, String> {
    let cache_error = |e| format!("Could not lock the local background-removal cache: {e}");
    // These names come from source metadata, not project-supplied cache references. Keep lock
    // files outside the matte directory and never unlink them: replacing a lock inode races holders.
    let locks = cache.with_extension("locks");
    std::fs::create_dir_all(&locks).map_err(cache_error)?;
    let source = open_cache_lock(&locks.join(format!("{cache_id}.lock"))).map_err(cache_error)?;
    source.try_lock().map_err(|e| match e {
        std::fs::TryLockError::WouldBlock => {
            "Background removal is already analysing this source in another session. Wait for that job, then try again.".to_owned()
        }
        std::fs::TryLockError::Error(e) => cache_error(e),
    })?;
    Ok(source)
}

/// The immediately previous parallel build uses a device-specific source lock. Take its lock too
/// while holding the canonical one, so an older worker cannot race expansion of that filename.
#[cfg(all(unix, not(target_arch = "wasm32")))]
fn lock_previous_source(cache: &Path, canonical: &str, metadata: &std::fs::Metadata) -> std::result::Result<Option<std::fs::File>, String> {
    use std::os::unix::fs::MetadataExt;
    let previous = cache_id_with_device(canonical, metadata.dev()).ok_or_else(|| "Invalid background-removal source identity".to_owned())?;
    (previous != canonical).then(|| lock_source(cache, &previous)).transpose()
}

fn lock<T>(m: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

pub(super) fn poll(s: &mut Session) {
    let mut i = 0;
    while i < s.background_jobs.len() {
        let p = &s.background_jobs[i];
        let old_document = p.epoch != s.document_epoch;
        if old_document {
            p.job.progress.cancel.store(true, Ordering::Relaxed);
        }
        let result = lock(&p.result).take();
        let Some(result) = result else {
            i += 1;
            continue;
        };
        let p = s.background_jobs.remove(i);
        let outcome = if old_document {
            Err("Project changed before background removal was applied".into())
        } else if p.job.progress.cancel.load(Ordering::Relaxed) {
            Err("Background removal cancelled".into())
        } else {
            result.and_then(|analysis| apply(s, &p.targets, &analysis).map(|()| analysis).map_err(|e| e.to_string()))
        };
        let report = match outcome {
            Ok(analysis) => {
                s.toast("Background removed");
                *lock(&p.job.progress.status) = if analysis.cached { "Reused the source matte".into() } else { "Background removed".into() };
                Ok(analysis.report)
            }
            Err(message) => {
                *lock(&p.job.progress.error) = Some(message.clone());
                *lock(&p.job.progress.status) = message.clone();
                if !p.job.progress.cancel.load(Ordering::Relaxed) {
                    s.error_toast(REMOVE, message.clone());
                }
                Err(message)
            }
        };
        *lock(&p.job.result) = Some(report);
        p.job.progress.finished.store(true, Ordering::Release);
    }
}

fn apply(s: &mut Session, targets: &[Target], analysis: &Analysis) -> Result<()> {
    let reference = &analysis.matte;
    let ready: Vec<_> = targets
        .iter()
        .filter_map(|target| {
            let q = s.project.sequence(target.seq)?;
            let (track, clip) = q.find_item(target.clip)?;
            let (root, source, _) = s.project.resolve_media(clip.item)?;
            let file = s.media.follow(source_file(source)?).ok()?;
            let required = Coverage::new(s, target.seq, clip).ok()?.keys(&analysis.source).ok()?;
            let covered = required.iter().all(|key| analysis.covered.binary_search(key).is_ok());
            (root == target.item && !q.track(track)?.locked && valid_source(&file, reference) && covered).then_some((
                *target,
                track,
                clip.person_matte.as_ref() == Some(reference),
            ))
        })
        .collect();
    if ready.is_empty() {
        return Err(EngineError::Other(
            "The selected clip range or original video changed. Select the clip and run Remove Background again to analyse its current range.".into(),
        ));
    }
    if ready.iter().any(|(_, _, applied)| !applied) {
        s.edit("Remove Background", |p, _| {
            for (target, track, _) in &ready {
                let source = p.item_mut(target.item).and_then(|i| i.as_media_mut()).ok_or_else(|| EngineError::Other("original video is missing".into()))?;
                source.identity = Some(reference.source);
                let clip = p
                    .sequence_mut(target.seq)
                    .and_then(|q| q.track_mut(*track))
                    .and_then(|t| t.item_mut(target.clip))
                    .ok_or_else(|| EngineError::Other("selected clip is missing".into()))?;
                clip.person_matte = Some(reference.clone());
            }
            Ok(())
        })?;
    } else {
        // A rerun may have repaired a cache file without changing the saved reference.
        s.bump_view();
    }
    for (target, _, _) in ready {
        s.media.remove(target.item);
    }
    Ok(())
}

pub(super) fn cache_dir() -> Option<PathBuf> {
    crate::autosave::default_data_dir().map(|d| d.join("Person Mattes"))
}

/// The key includes the on-disk revision as well as the project's fast content fingerprint.
/// Modification time detects in-place edits and the inode detects same-name atomic replacements.
fn cache_id(file: impl AsRef<Path>, identity: MediaIdentity) -> std::io::Result<String> {
    cache_id_for(&std::fs::metadata(file)?, identity)
}

fn cache_id_for(meta: &std::fs::Metadata, identity: MediaIdentity) -> std::io::Result<String> {
    if !meta.is_file() || meta.len() != identity.size {
        return Err(std::io::Error::other("the original video changed or is not a regular file"));
    }
    let modified = meta.modified()?.duration_since(std::time::UNIX_EPOCH).map_err(std::io::Error::other)?.as_nanos();
    let key = format!("{CACHE_PREFIX}-{:016x}-{:016x}-{modified:032x}", identity.fingerprint, identity.size);
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        // The operating system may assign a different device number when the same volume is
        // mounted again. Reserve its old slot in the v1 filename so legacy caches can be reused.
        Ok(format!("{key}-0000000000000000-{:016x}-{:016x}-{:016x}", meta.ino(), meta.ctime(), meta.ctime_nsec()))
    }
    #[cfg(not(unix))]
    {
        Ok(key)
    }
}

#[cfg(all(unix, not(target_arch = "wasm32")))]
fn cache_id_with_device(canonical: &str, device: u64) -> Option<String> {
    Some(format!("{}{device:016x}{}", canonical.get(..DEVICE_START)?, canonical.get(DEVICE_END..)?))
}

/// `canonical` is generated from source identity and current file metadata. Matching its exact
/// prefix and suffix and one fixed-width hex field also makes a legacy filename safe to open.
fn same_cache_source(canonical: &str, stored: &str) -> bool {
    if canonical == stored {
        return true;
    }
    #[cfg(unix)]
    {
        stored.len() == canonical.len()
            && stored.get(..DEVICE_START) == canonical.get(..DEVICE_START)
            && stored.get(DEVICE_END..) == canonical.get(DEVICE_END..)
            && stored.get(DEVICE_START..DEVICE_END).is_some_and(|field| field.bytes().all(|b| b.is_ascii_digit() || matches!(b, b'a'..=b'f')))
    }
    #[cfg(not(unix))]
    {
        false
    }
}

fn valid_source(file: &str, reference: &PersonMatteRef) -> bool {
    cache_id(file, reference.source).is_ok_and(|id| same_cache_source(&id, &reference.cache_id))
}

pub(super) fn open_matte(dir: &Path, file: &str, reference: &PersonMatteRef) -> Option<MatteReader> {
    if !valid_source(file, reference) {
        return None;
    }
    MatteReader::open(&dir.join(format!("{}.matte", reference.cache_id))).ok()
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use crate::media_test_util::{make_movie, session_with};
    use filmcraft_media::DemoScene;

    fn empty_matte() -> std::result::Result<Segmenter, String> {
        Ok(Box::new(|_| Ok(Matte { width: 2, height: 2, alpha: vec![0; 4] })))
    }

    fn fixture() -> (tempfile::TempDir, Session, ItemId) {
        let dir = tempfile::tempdir().unwrap();
        let source = dir.path().join("source.mov");
        make_movie(&source, DemoScene::CityNight, 64, 48, 4);
        let (s, _, seq) = session_with(&[&source]);
        s.media.set_matte_dir(Some(dir.path().join("mattes")));
        (dir, s, seq)
    }

    #[test]
    fn attaches_to_video_only_and_reuses_matte_after_cut_and_reopen() {
        let (dir, mut s, seq) = fixture();
        let q = s.project.sequence(seq).unwrap();
        let original = q.video_tracks[0].items[0].clone();
        let audio = q.audio_tracks[0].items[0].clone();
        s.state.selection = vec![original.id, audio.id];
        let before = s.render_program(1.0).unwrap();
        assert!(before.px.iter().skip(3).step_by(4).any(|&a| a > 0.5));
        let cache = s.media.matte_dir().unwrap();
        let result = start(&mut s, &json!({"wait": true}), empty_matte, cache.clone()).unwrap();
        assert_eq!(result["frames"], 4);
        let q = s.project.sequence(seq).unwrap();
        let mut video = q.video_tracks[0].items[0].clone();
        let matte = video.person_matte.take().unwrap();
        assert_eq!(video, original);
        assert_eq!(q.audio_tracks[0].items[0], audio);
        assert!(s.render_program(1.0).unwrap().px.iter().all(|&x| x == 0.0));
        s.execute(TOGGLE, json!({"clip": original.id.0})).unwrap();
        assert_eq!(s.render_program(1.0).unwrap().px, before.px);
        s.undo();
        assert!(s.render_program(1.0).unwrap().px.iter().all(|&x| x == 0.0));
        s.undo();
        assert_eq!(s.render_program(1.0).unwrap().px, before.px);
        s.redo();

        let cut = s.sequence_rate().tick_of(2);
        s.execute("timeline.razor", json!({"time": cut.0, "clip": original.id.0})).unwrap();
        let items = &s.project.sequence(seq).unwrap().video_tracks[0].items;
        assert_eq!(items.len(), 2);
        assert!(items.iter().all(|it| it.person_matte.as_ref() == Some(&matte)));
        let right = items[1].id;
        s.execute(RESTORE, json!({"clip": right.0})).unwrap();
        let no_segment = || Err("cache hit should not start Vision".into());
        start(&mut s, &json!({"clip": right.0, "wait": true}), no_segment, cache.clone()).unwrap();

        let project = dir.path().join("test.fcproj");
        s.execute("file.saveAs", json!({"path": project})).unwrap();
        s.execute("file.open", json!({"path": project})).unwrap();
        assert!(s.render_program_at(1.0, cut).unwrap().px.iter().all(|&x| x == 0.0));
        let path = dir.path().join("out.mov");
        s.execute("file.exportMedia", json!({"path": path, "format": "prores", "wait": true})).unwrap();
        assert!(path.is_file());

        std::fs::remove_file(cache.join(format!("{}.matte", matte.cache_id))).unwrap();
        let missing = dir.path().join("missing.mov");
        let err = s.execute("file.exportMedia", json!({"path": missing, "format": "prores", "wait": true})).unwrap_err();
        assert!(err.to_string().contains("Clip > Remove Background"));
        assert!(!missing.exists());
    }

    #[test]
    fn cancel_discards_staged_matte_without_editing() {
        type Gate = (std::sync::mpsc::Sender<()>, std::sync::mpsc::Receiver<()>);
        static GATE: Mutex<Option<Gate>> = Mutex::new(None);
        fn gated_segmenter() -> std::result::Result<Segmenter, String> {
            let (entered, release) = lock(&GATE).take().ok_or("no test gate")?;
            let mut count = 0;
            Ok(Box::new(move |_| {
                count += 1;
                if count == 2 {
                    entered.send(()).map_err(|e| e.to_string())?;
                    release.recv_timeout(std::time::Duration::from_secs(10)).map_err(|e| e.to_string())?;
                }
                Ok(Matte { width: 1, height: 1, alpha: vec![255] })
            }))
        }
        let (_dir, mut s, seq) = fixture();
        let clip = s.project.sequence(seq).unwrap().video_tracks[0].items[0].id;
        let (entered, waiting) = std::sync::mpsc::channel();
        let (release, released) = std::sync::mpsc::channel();
        *lock(&GATE) = Some((entered, released));
        let cache = s.media.matte_dir().unwrap();
        let revision = s.revision;
        let r = start(&mut s, &json!({"clip": clip.0}), gated_segmenter, cache.clone()).unwrap();
        waiting.recv_timeout(std::time::Duration::from_secs(10)).unwrap();
        s.execute("jobs.cancel", json!({"job": r["job"]})).unwrap();
        release.send(()).unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
        while !s.background_jobs.is_empty() && std::time::Instant::now() < deadline {
            s.execute("jobs.list", json!({})).unwrap();
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        assert!(s.background_jobs.is_empty());
        assert_eq!(s.revision, revision);
        assert_eq!(std::fs::read_dir(cache).unwrap().count(), 0);
    }

    #[test]
    fn same_name_replacement_invalidates_a_reference() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("source.mov");
        std::fs::write(&path, b"old").unwrap();
        let source = crate::relink::identity_of_bytes(b"old");
        let reference = PersonMatteRef { source, cache_id: cache_id(&path, source).unwrap(), bypassed: false };
        assert!(valid_source(path.to_str().unwrap(), &reference));
        let replacement = dir.path().join("replacement.mov");
        std::fs::write(&replacement, b"new").unwrap();
        std::fs::rename(replacement, &path).unwrap();
        assert!(!valid_source(path.to_str().unwrap(), &reference));
    }

    #[test]
    fn invalid_background_removal_parameters_do_not_start_work() {
        let (_dir, mut s, seq) = fixture();
        let clip = s.project.sequence(seq).unwrap().video_tracks[0].items[0].id;
        s.state.selection = vec![clip];
        let cache = s.media.matte_dir().unwrap();
        let revision = s.revision;
        for params in [json!({"clip": -1}), json!({"clip": "video"}), json!({"wait": 123})] {
            assert!(start(&mut s, &params, empty_matte, cache.clone()).is_err());
        }
        assert!(s.jobs.is_empty());
        assert!(s.background_jobs.is_empty());
        assert_eq!(s.revision, revision);
        assert!(!cache.exists());
    }
}

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "background_range_tests.rs"]
mod range_tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "background_parallel_tests.rs"]
mod parallel_tests;

#[cfg(all(test, not(target_arch = "wasm32")))]
#[path = "background_lock_tests.rs"]
mod lock_tests;

#[cfg(all(test, unix, not(target_arch = "wasm32")))]
#[path = "background_identity_tests.rs"]
mod identity_tests;
