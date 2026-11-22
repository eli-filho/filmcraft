//! Immediate editable media acceptance, followed by bounded background preparation.
//!
//! The ordinary edit history records moves/cuts while a file opens. Workers never replace a
//! project snapshot: they fill metadata for a ticket in the same document. Resolved tickets are
//! retained while undo can restore them, so undo cannot bring back a permanently loading clip.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};

use filmcraft_media::{AudioStreamInfo, MediaInfo, MediaKind, MediaReadiness, SharedSource, VideoStreamInfo};
use filmcraft_project::{ItemId, ItemKind, MediaClip, MediaIdentity, MediaRef, PendingImport, Project};
use filmcraft_time::{TICKS_PER_SECOND, Tick};
use serde_json::{Value, json};

use crate::{EngineError, Result, Session};

#[cfg(test)]
#[path = "pending_import_prepare_tests.rs"]
mod prepare_tests;

const MAX_ACTIVE: usize = 2;
const MAX_BATCH: usize = 4096;
const PLACEHOLDER_DURATION: Tick = Tick(5 * TICKS_PER_SECOND);

/// Preparation state for frontend presentation. `ready` means playback can proceed; optional
/// fingerprinting/ingest/transcription may still be running. No percentage is invented for I/O.
#[derive(Clone, Debug, PartialEq, serde::Serialize)]
pub struct ImportStatus {
    pub label: String,
    pub progress: Option<f32>,
    pub ready: bool,
    pub error: Option<String>,
}

#[derive(Default)]
pub(crate) struct Imports {
    owner: Option<Arc<()>>,
    next_ticket: u64,
    tasks: BTreeMap<u64, Task>,
}

struct Task {
    ticket: u64,
    item: ItemId,
    path: String,
    cancel: Arc<AtomicBool>,
    published: Arc<AtomicBool>,
    transcribe: Arc<AtomicBool>,
    transcribe_attempted: bool,
    transcript_rx: Option<mpsc::Receiver<crate::pending_import_transcript::Outcome>>,
    rx: Option<mpsc::Receiver<Message>>,
    source: Option<SharedSource>,
    error: Option<String>,
    phase: String,
    complete: Option<Completed>,
}

#[derive(Clone)]
struct Completed {
    identity: Option<MediaIdentity>,
    transcript: Option<Arc<filmcraft_project::transcript::Transcript>>,
    path: String,
    source: SharedSource,
}

enum Message {
    Opened(SharedSource),
    Phase(&'static str),
    Completed(Completed, Vec<crate::Job>, Vec<crate::proxies::PendingJob>, Vec<String>),
    Failed(String),
}

impl Drop for Imports {
    fn drop(&mut self) {
        for task in self.tasks.values() {
            task.cancel.store(true, Ordering::Relaxed);
        }
    }
}

/// Native media-only requests use pending items. Other imports retain their established path,
/// including the browser's asynchronous Blob handling on WebAssembly.
pub(crate) fn eligible(paths: &[String]) -> bool {
    !cfg!(target_arch = "wasm32") && !paths.is_empty() && paths.iter().all(|p| filmcraft_media::is_importable(std::path::Path::new(p)))
}

/// Accept without touching the filesystem. File extensions only select provisional streams;
/// the worker determines the real format and reports unsupported/corrupt files on the item.
pub(crate) fn start(s: &mut Session, paths: &[String], bin: Option<filmcraft_project::BinId>) -> Result<Value> {
    if paths.len() > MAX_BATCH {
        return Err(EngineError::Other(format!("import at most {MAX_BATCH} files at once")));
    }
    if bin.is_some_and(|id| s.project.root.find_bin(id).is_none()) {
        return Err(EngineError::Other("the import bin no longer exists".into()));
    }
    reset_owner(s);
    let batch = s.imports.next_ticket.checked_add(1).ok_or_else(|| EngineError::Other("too many import requests".into()))?;
    let scaling = s.prefs.media.default_media_scaling.clone();
    let mut accepted = Vec::new();
    for path in paths {
        s.imports.next_ticket = s.imports.next_ticket.checked_add(1).ok_or_else(|| EngineError::Other("too many import requests".into()))?;
        let info = provisional_info(s, path);
        let label = s.prefs.labels.for_media(info.kind, info.has_video(), info.has_audio());
        accepted.push((path.clone(), s.imports.next_ticket, info, label));
        s.media.allow(path);
    }
    let ids = s.edit("Import Media", |project, state| {
        let mut ids = Vec::new();
        for (path, ticket, info, label) in accepted {
            let name = info.name.clone();
            let duration = info.duration;
            let id = project.add_item(
                &name,
                label,
                ItemKind::Media(Box::new(MediaClip {
                    media: MediaRef::File { path: path.clone() },
                    info,
                    interpret: Default::default(),
                    mark_in: None,
                    mark_out: None,
                    markers: Vec::new(),
                    offline: false,
                    proxy: None,
                    identity: None,
                })),
                bin,
            );
            project.pending_imports.insert(
                id,
                PendingImport {
                    ticket,
                    batch,
                    path,
                    provisional_duration: duration,
                    scaling: scaling.clone(),
                    placements: BTreeMap::new(),
                    metadata_ready: false,
                    error: None,
                },
            );
            ids.push(id);
        }
        state.project_selection = ids.clone();
        Ok(ids)
    })?;
    // No file validation, fingerprints or decoder construction is allowed above this boundary.
    s.poll_imports();
    Ok(json!({"items": ids.iter().map(|id| id.0).collect::<Vec<_>>(), "pending": true, "errors": []}))
}

fn provisional_info(s: &Session, path: &str) -> MediaInfo {
    let kind = match crate::media_browser::kind_of(path) {
        Some("audio") => MediaKind::AudioOnly,
        Some("image") => MediaKind::Still,
        _ => MediaKind::Movie,
    };
    let settings = s.active_sequence().map(|q| q.settings.clone()).unwrap_or_default();
    let video = (kind != MediaKind::AudioOnly).then(|| VideoStreamInfo {
        width: settings.width,
        height: settings.height,
        frame_rate: settings.frame_rate,
        par: settings.par,
        codec: String::new(),
        pixel_format: String::new(),
        color: filmcraft_color::ColorInfo::REC709,
        has_alpha: false,
        bitrate: None,
        hdr: None,
        color_assumed: None,
    });
    let audio_streams = if kind == MediaKind::Still {
        Vec::new()
    } else {
        vec![AudioStreamInfo { sample_rate: settings.sample_rate, channels: 2, codec: String::new(), bits_per_sample: None }]
    };
    MediaInfo {
        name: std::path::Path::new(path).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| path.to_string()),
        kind,
        duration: PLACEHOLDER_DURATION,
        video,
        audio_streams,
        container: String::new(),
        start_timecode: None,
        file_size: None,
    }
}

fn reset_owner(s: &mut Session) {
    if !s.imports.owner.as_ref().is_some_and(|owner| Arc::ptr_eq(owner, &s.project_instance)) {
        let next_ticket = s.imports.next_ticket;
        s.imports = Imports { owner: Some(s.project_instance.clone()), next_ticket, tasks: BTreeMap::new() };
    }
}

impl Session {
    /// Poll without blocking, start at most two preparers, and hydrate only accepted identities.
    /// Also resumes pending media restored by undo or a saved/recovered project.
    pub fn poll_imports(&mut self) {
        if cfg!(target_arch = "wasm32") {
            return;
        }
        reset_owner(self);
        register_restored(self);
        note_placements(self);
        collect_messages(self);
        poll_transcripts(self);
        apply_results(self);
        launch_queued(self);
        prune_tasks(self);
    }

    pub fn imports_running(&self) -> bool {
        self.project.pending_imports.iter().any(|(id, p)| self.project.item(*id).is_some() && p.error.is_none())
            || self.imports.tasks.values().any(|task| {
                task_matches_source(&self.project, task)
                    && (task.transcript_rx.is_some()
                        || (!task.transcribe_attempted && task.transcribe.load(Ordering::Acquire) && task.complete.is_some() && task.error.is_none()))
            })
    }

    /// Any preparation beneath an item, including nested sequences (for full-file waveforms).
    pub fn import_status(&self, item: ItemId) -> Option<ImportStatus> {
        status_tree(self, item, None)
    }

    /// Preparation affecting the requested picture/sound, rather than an unrelated later clip.
    pub fn import_status_at(&self, item: ItemId, time: Tick) -> Option<ImportStatus> {
        status_tree(self, item, Some(time))
    }
}

fn register_restored(s: &mut Session) {
    let mut seen = BTreeSet::new();
    let invalid: Vec<_> = s
        .project
        .pending_imports
        .iter()
        .filter_map(|(&item, pending)| {
            let collision =
                !seen.insert(pending.ticket) || s.imports.tasks.get(&pending.ticket).is_some_and(|task| task.item != item || task.path != pending.path);
            (collision && pending.error.is_none()).then_some(item)
        })
        .collect();
    if !invalid.is_empty() {
        let project = Arc::make_mut(&mut s.project);
        for item in invalid {
            if let Some(pending) = project.pending_imports.get_mut(&item) {
                pending.error = Some("Conflicting saved media import identity; relink this file".into());
            }
        }
    }
    for (&item, pending) in &s.project.pending_imports {
        if s.project.item(item).is_none() {
            continue;
        }
        s.imports.next_ticket = s.imports.next_ticket.max(pending.ticket);
        s.imports.tasks.entry(pending.ticket).or_insert_with(|| Task {
            ticket: pending.ticket,
            item,
            path: pending.path.clone(),
            cancel: Arc::new(AtomicBool::new(false)),
            rx: None,
            published: Arc::new(AtomicBool::new(false)),
            transcribe: Arc::new(AtomicBool::new(false)),
            transcribe_attempted: false,
            transcript_rx: None,
            source: None,
            error: pending.error.clone(),
            phase: "Preparing media".into(),
            complete: None,
        });
    }
}

fn collect_messages(s: &mut Session) {
    for (&ticket, task) in &mut s.imports.tasks {
        let Some(rx) = task.rx.take() else { continue };
        let mut done = false;
        loop {
            match rx.try_recv() {
                Ok(Message::Opened(source)) => task.source = Some(source),
                Ok(Message::Phase(phase)) => task.phase = phase.into(),
                Ok(Message::Completed(result, mut jobs, mut pending, warnings)) => {
                    task.source = Some(result.source.clone());
                    task.complete = Some(result);
                    if accepts_completed_jobs(&s.project, task.item, &task.path, ticket) {
                        // Detached sessions number their jobs independently. Translate ids before
                        // publishing, so cancelling one job cannot cancel an unrelated export.
                        let largest = s.jobs.iter().map(|job| job.id).max().unwrap_or(0);
                        match remap_jobs(&mut jobs, &mut pending, largest) {
                            Ok(()) => {
                                s.jobs.extend(jobs);
                                s.media_jobs.extend(pending);
                            }
                            Err(error) => {
                                for job in jobs {
                                    job.progress.cancel.store(true, Ordering::Relaxed);
                                }
                                s.events.push(crate::Event::Toast { message: format!("Could not attach media preparation jobs: {error}"), error: true });
                            }
                        }
                    } else {
                        for job in jobs {
                            job.progress.cancel.store(true, Ordering::Relaxed);
                        }
                    }
                    for message in warnings {
                        s.events.push(crate::Event::Toast { message, error: true });
                    }
                    done = true;
                }
                Ok(Message::Failed(error)) => {
                    task.error = Some(error);
                    done = true;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => {
                    if !done {
                        task.error = Some("Media preparation stopped unexpectedly".into());
                    }
                    done = true;
                    break;
                }
            }
        }
        if !done {
            task.rx = Some(rx);
        }
        if task.source.is_some() && (s.project.pending_imports.get(&task.item).is_none_or(|p| p.path != task.path) || s.project.item(task.item).is_none()) {
            // An undone/deleted drop can still be redone from history. Finish preparing its
            // source without waiting forever for a placement that is no longer visible.
            task.published.store(true, Ordering::Release);
        }
    }
}

pub(crate) fn accepts_completed_jobs(project: &Project, item: ItemId, path: &str, ticket: u64) -> bool {
    project.pending_imports.get(&item).is_some_and(|pending| {
        pending.ticket == ticket
            && pending.path == path
            && pending.error.is_none()
            && project
                .item(item)
                .and_then(|item| item.as_media())
                .is_some_and(|media| !media.offline && matches!(&media.media, MediaRef::File { path: current } if current == path))
    })
}

/// Plan the complete id translation first: a newly assigned id may equal another old id.
/// Refusal leaves both lists unchanged so the caller can cancel every detached worker safely.
pub(crate) fn remap_jobs(jobs: &mut [crate::Job], pending: &mut [crate::proxies::PendingJob], largest: u64) -> std::result::Result<(), String> {
    let mut ids = BTreeMap::new();
    let mut next = largest;
    for job in jobs.iter() {
        next = next.checked_add(1).ok_or("job identity space exhausted")?;
        if ids.insert(job.id, next).is_some() {
            return Err("duplicate detached job identity".into());
        }
    }
    if pending.iter().any(|job| !ids.contains_key(&job.job)) {
        return Err("media output refers to an unknown detached job".into());
    }
    for job in jobs {
        if let Some(&id) = ids.get(&job.id) {
            job.id = id;
        }
    }
    for job in pending {
        if let Some(&id) = ids.get(&job.job) {
            job.job = id;
        }
    }
    Ok(())
}

fn apply_results(s: &mut Session) {
    let mut changed = false;
    let pending: Vec<_> = s.project.pending_imports.iter().map(|(&id, p)| (id, p.clone())).collect();
    for (id, pending) in pending {
        let Some(task) = s.imports.tasks.get(&pending.ticket).filter(|t| t.item == id && t.path == pending.path) else { continue };
        // A deletion is an edit, including its pending marker in the undo snapshot. Never
        // resurrect or mutate the document just because its abandoned worker finishes.
        if s.project.item(id).is_none() {
            continue;
        }
        let same_path = s.project.item(id).and_then(|i| i.as_media()).is_some_and(|m| matches!(&m.media, MediaRef::File { path } if *path == pending.path));
        if !same_path {
            Arc::make_mut(&mut s.project).pending_imports.remove(&id);
            changed = true;
            continue;
        }
        if let Some(error) = &task.error {
            if pending.error.as_ref() != Some(error) {
                if let Some(p) = Arc::make_mut(&mut s.project).pending_imports.get_mut(&id) {
                    p.error = Some(error.clone());
                }
                s.events.push(crate::Event::Toast { message: format!("{}: {error}", pending.path), error: true });
                changed = true;
            }
            continue;
        }
        if let Some(source) = &task.source {
            if !pending.metadata_ready {
                for message in hydrate(Arc::make_mut(&mut s.project), id, &pending, source.info()) {
                    s.events.push(crate::Event::Toast { message, error: false });
                }
                changed = true;
            }
            // Replacing metadata must also invalidate a cached provisional source.
            if !pending.metadata_ready || s.media.cached(id).is_none() {
                s.media.insert_file(id, &pending.path, source.clone());
            }
            task.published.store(true, Ordering::Release);
        }
        if let Some(result) = &task.complete {
            let project = Arc::make_mut(&mut s.project);
            let batch_ready = project
                .pending_imports
                .iter()
                .filter(|(id, p)| p.batch == pending.batch && project.item(**id).is_some())
                .all(|(_, p)| p.metadata_ready || p.error.is_some());
            if !batch_ready {
                continue;
            }
            if let Some(media) = project.item_mut(id).and_then(|i| i.as_media_mut()) {
                media.identity = result.identity;
                media.media = MediaRef::File { path: result.path.clone() };
            }
            if let Some(transcript) = &result.transcript {
                project.transcripts.entry(id).or_insert_with(|| transcript.clone());
            }
            // Keep the batch's placement anchors until every sibling has its real duration.
            // A fast second file must still move with its untouched predecessor when that
            // predecessor's metadata arrives later.
            project.pending_imports.remove(&id);
            if result.path != pending.path {
                s.media.allow(&result.path);
            }
            s.media.insert_file(id, &result.path, result.source.clone());
            changed = true;
        }
    }
    if changed {
        s.bump();
        s.previews.live.publish_project(s.project.clone());
        s.sync_persistence();
    }
}

pub(crate) fn hydrate(project: &mut Project, id: ItemId, pending: &PendingImport, info: &MediaInfo) -> Vec<String> {
    let Some(old_info) = project.item(id).and_then(|item| item.as_media()).map(|media| media.info.clone()) else { return Vec::new() };
    let mut warnings = crate::pending_import_hydrate::reconcile_streams_and_geometry(project, id, &old_info, info, &pending.scaling);
    let mut pending_items = project.pending_imports.clone();
    let durations: BTreeMap<_, _> = project.items.iter().map(|(&id, item)| (id, item.duration())).collect();
    if let Some(media) = project.item_mut(id).and_then(|i| i.as_media_mut()) {
        media.info = info.clone();
    }
    let mut shifted = Vec::new();
    for item in project.items.values_mut() {
        let sequence_id = item.id;
        let ItemKind::Sequence(sequence) = &mut item.kind else { continue };
        let sequence = Arc::make_mut(sequence);
        let rate = sequence.settings.frame_rate;
        for track in sequence.video_tracks.iter_mut().chain(sequence.audio_tracks.iter_mut()) {
            // Only whole, untrimmed instances take the discovered duration. A cut/trim already
            // expresses the editor's intent and must never be replayed over a new snapshot.
            for index in 0..track.items.len() {
                let Some(clip) = track.items.get(index) else { continue };
                if clip.item != id
                    || clip.source_in != Tick::ZERO
                    || clip.speed != 1.0
                    || clip.reverse
                    || clip.frame_hold.is_some()
                    || clip.duration != rate.snap_nearest(pending.provisional_duration).max(rate.frame_duration())
                {
                    continue;
                }
                // Track locks and transition timing are deliberate edits, not metadata defaults.
                // Resizing either endpoint would detach its fade/crossfade from the chosen cut.
                let has_transition = |clip: &filmcraft_project::TrackItem| {
                    track
                        .transitions
                        .iter()
                        .any(|transition| transition.from == Some(clip.id) || transition.to == Some(clip.id) || transition.range().overlaps(&clip.range()))
                };
                if track.locked || has_transition(clip) {
                    warnings.push(format!("{}: kept a locked clip or transition unchanged; the full source is available for manual trimming", info.name));
                    continue;
                }
                let old = clip.duration;
                let mut end = clip.end();
                let mut follow = index + 1;
                // Carry only the contiguous, unedited remainder of this same drop. Its baseline
                // is recorded at placement, so a later user move is never mistaken for a batch.
                while let Some(next) = track.items.get(follow) {
                    let Some(record) = pending_items.get(&next.item).filter(|p| p.batch == pending.batch) else { break };
                    let duration = if record.metadata_ready {
                        durations.get(&next.item).copied().unwrap_or(record.provisional_duration)
                    } else {
                        record.provisional_duration
                    };
                    if next.start != end
                        || next.source_in != Tick::ZERO
                        || next.speed != 1.0
                        || next.reverse
                        || next.frame_hold.is_some()
                        || next.group.is_some()
                        || has_transition(next)
                        || next.duration != rate.snap_nearest(duration).max(rate.frame_duration())
                        || record.placements.get(&next.id) != Some(&(sequence_id, track.id, next.start))
                    {
                        break;
                    }
                    end = next.end();
                    follow += 1;
                }
                let wanted = rate.snap_nearest(info.duration).max(rate.frame_duration());
                let mut delta = wanted - old;
                if let Some(obstacle) = track.items.get(follow) {
                    delta = delta.min(obstacle.start - end);
                }
                // A transition can begin before the next clip. Preserve that occupied interval
                // even when its clip is not one of the provisional instances being carried.
                if let Some(transition_start) = track.transitions.iter().filter(|transition| transition.start >= end).map(|transition| transition.start).min() {
                    delta = delta.min(transition_start - end);
                }
                if delta != wanted - old {
                    warnings.push(format!("{}: the timeline length stops at an existing edit; the full source is available for trimming", info.name));
                }
                if let Some(clip) = track.items.get_mut(index) {
                    clip.duration = (old + delta).max(rate.frame_duration());
                }
                for next in track.items.iter_mut().take(follow).skip(index + 1) {
                    next.start += delta;
                    let location = (sequence_id, track.id, next.start);
                    shifted.push((next.item, next.id, location));
                    if let Some(record) = pending_items.get_mut(&next.item) {
                        record.placements.insert(next.id, location);
                    }
                }
            }
        }
    }
    for (item, clip, placement) in shifted {
        if let Some(pending) = project.pending_imports.get_mut(&item) {
            pending.placements.insert(clip, placement);
        }
    }
    if let Some(p) = project.pending_imports.get_mut(&id) {
        p.metadata_ready = true;
    }
    warnings.sort();
    warnings.dedup();
    warnings
}

fn launch_queued(s: &mut Session) {
    let mut active = s.imports.tasks.values().filter(|task| task.rx.is_some()).count();
    let queued: Vec<_> = s
        .project
        .pending_imports
        .iter()
        .filter_map(|(&id, p)| {
            let task = s.imports.tasks.get(&p.ticket)?;
            (task.rx.is_none() && task.error.is_none() && task.complete.is_none() && s.project.item(id).is_some()).then_some(p.ticket)
        })
        .collect();
    for ticket in queued {
        if active >= MAX_ACTIVE {
            break;
        }
        let Some(task) = s.imports.tasks.get_mut(&ticket) else { continue };
        if task.rx.is_some() || task.complete.is_some() || task.error.is_some() {
            continue;
        }
        let (tx, rx) = mpsc::channel();
        let detect_image_sequence = s.prefs.media.import_image_sequences
            && s.project
                .pending_imports
                .get(&task.item)
                .is_some_and(|pending| s.project.pending_imports.values().filter(|p| p.batch == pending.batch).take(2).count() == 1);
        let work = Work {
            item: task.item,
            path: task.path.clone(),
            cancel: task.cancel.clone(),
            pool: s.media.clone(),
            published: task.published.clone(),
            detect_image_sequence,
            services: s.services.clone(),
            project: s.project.clone(),
            project_path: s.path.clone(),
            prefs: s.prefs.clone(),
        };
        let failed = tx.clone();
        let run = move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| prepare(work, &tx)))
                .unwrap_or_else(|_| Err("Media preparation failed internally; the editable clip was kept".into()));
            if let Err(error) = result {
                let _ = failed.send(Message::Failed(error));
            }
        };
        match std::thread::Builder::new().name("filmcraft-import".into()).spawn(run) {
            Ok(_) => {
                task.rx = Some(rx);
                active += 1;
            }
            Err(error) => task.error = Some(format!("Could not start media preparation: {error}")),
        }
    }
}

struct Work {
    item: ItemId,
    path: String,
    cancel: Arc<AtomicBool>,
    pool: Arc<crate::MediaPool>,
    services: Arc<dyn crate::Services>,
    published: Arc<AtomicBool>,
    detect_image_sequence: bool,
    project: Arc<Project>,
    project_path: Option<String>,
    prefs: crate::autosave::Preferences,
}

/// Optional ingest can launch workers before a later copy/open fails or the import is cancelled.
/// Until their handles reach the editor, this guard remains responsible for cancelling them.
struct DetachedImport(Session);

impl std::ops::Deref for DetachedImport {
    type Target = Session;
    fn deref(&self) -> &Session {
        &self.0
    }
}

impl std::ops::DerefMut for DetachedImport {
    fn deref_mut(&mut self) -> &mut Session {
        &mut self.0
    }
}

impl Drop for DetachedImport {
    fn drop(&mut self) {
        for job in &self.0.jobs {
            job.progress.cancel.store(true, Ordering::Relaxed);
        }
    }
}

struct ImportedStill {
    source: SharedSource,
    info: MediaInfo,
}

impl filmcraft_media::MediaSource for ImportedStill {
    fn info(&self) -> &MediaInfo {
        &self.info
    }
    fn readiness(&self) -> MediaReadiness {
        self.source.readiness()
    }
    fn video_frame(&self, request: filmcraft_media::FrameRequest) -> filmcraft_media::Result<Arc<filmcraft_frame::VideoFrame>> {
        self.source.video_frame(request)
    }
    fn audio(&self, start: i64, frames: usize, rate: u32) -> filmcraft_media::Result<filmcraft_frame::AudioBuffer> {
        self.source.audio(start, frames, rate)
    }
    fn audio_stream(&self, stream: usize, start: i64, frames: usize, rate: u32) -> filmcraft_media::Result<filmcraft_frame::AudioBuffer> {
        self.source.audio_stream(stream, start, frames, rate)
    }
}

fn prepare(work: Work, tx: &mpsc::Sender<Message>) -> std::result::Result<(), String> {
    let check = || if work.cancel.load(Ordering::Relaxed) { Err("Import cancelled".to_string()) } else { Ok(()) };
    check()?;
    let path = work.pool.follow(&work.path).map_err(|e| e.to_string())?;
    let name = std::path::Path::new(&path).file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| path.clone());
    let timebase = crate::settings::timebase_rate(&work.prefs.media.indeterminate_timebase);
    // Number parsing is cheap, but listing/probing siblings belongs exclusively to this worker.
    // Match file.import's single-selected-file detection and preserve its multi-file behavior.
    let image_sequence = work.detect_image_sequence
        && filmcraft_media::sequence::Numbered::parse(&path).is_some()
        && crate::media_pool::image_sequence_frames(&path, &*work.services).is_ok_and(|frames| frames.len() > 1);
    let source = if image_sequence {
        crate::media_pool::open_image_sequence(&path, &*work.services, timebase, &name).map_err(|e| e.to_string())?
    } else {
        let progressive = match work.services.reader(&path) {
            Some(Ok(reader)) => filmcraft_codecs::open_progressive_audio(&name, reader).map_err(|e| e.to_string())?,
            _ => None,
        };
        match progressive {
            Some(source) => source,
            None => work.pool.open_file(&path, &*work.services).map_err(|e| e.to_string())?,
        }
    };
    let source: SharedSource = if source.info().kind == MediaKind::Still {
        let mut info = source.info().clone();
        if let Some(video) = info.video.as_mut() {
            video.frame_rate = timebase;
        }
        Arc::new(ImportedStill { source, info })
    } else {
        source
    };
    check()?;
    tx.send(Message::Opened(source.clone())).map_err(|e| e.to_string())?;
    // Metadata and the first playable prefix are published before unrelated identity/ingest work.
    let _ = tx.send(Message::Phase("Recording media identity"));
    let identity = crate::relink::identity_of(&*work.services, &path);
    let mut warnings = Vec::new();
    let identity = match identity {
        Ok(identity) => Some(identity),
        Err(error) => {
            warnings.push(format!("Media identity: {error}"));
            None
        }
    };
    loop {
        check()?;
        match source.readiness() {
            MediaReadiness::Ready => break,
            MediaReadiness::Failed(error) => return Err(error),
            MediaReadiness::Loading { .. } => std::thread::sleep(std::time::Duration::from_millis(25)),
        }
    }
    // The UI first adopts the source and observes placement. Optional processing must not win
    // the race against the drop that is still being placed during that same UI pass.
    while !work.published.load(Ordering::Acquire) {
        check()?;
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let mut detached = DetachedImport(Session::new(work.services));
    detached.project = work.project;
    detached.path = work.project_path;
    detached.prefs = work.prefs;
    detached.media.add_stored(&detached.project, detached.path.as_deref());
    let project = Arc::make_mut(&mut detached.project);
    if let Some(media) = project.item_mut(work.item).and_then(|i| i.as_media_mut()) {
        media.info = source.info().clone();
        media.identity = identity;
        // Sources opened from saved projects may have document-relative references. The
        // detached session uses the resolved path everywhere, including its cache key.
        media.media = MediaRef::File { path: path.clone() };
    }
    project.pending_imports.clear();
    detached.media.allow(&path);
    detached.media.insert_file(work.item, &path, source.clone());
    check()?;
    // A numbered sequence is many files; the single-file ingest implementation must not copy
    // or transcode only its first frame. Synchronous file.import makes the same exclusion.
    if !image_sequence {
        let _ = tx.send(Message::Phase("Preparing ingest"));
        if let Err(error) = crate::proxies::ingest(&mut detached, &[work.item]) {
            warnings.push(format!("Ingest: {error}"));
        }
    }
    check()?;
    let final_path = detached
        .project
        .item(work.item)
        .and_then(|i| i.as_media())
        .and_then(|m| match &m.media {
            MediaRef::File { path } => Some(path.clone()),
            _ => None,
        })
        .unwrap_or_else(|| path.clone());
    let final_source = if final_path == path { source } else { detached.media.open_file(&final_path, &*detached.services).map_err(|e| e.to_string())? };
    let result = Completed {
        identity,
        transcript: detached.project.transcripts.get(&work.item).cloned(),
        path: if final_path == path { work.path } else { final_path },
        source: final_source,
    };
    let jobs = std::mem::take(&mut detached.jobs);
    let pending = std::mem::take(&mut detached.media_jobs);
    send_completed(tx, result, jobs, pending, warnings)
}

fn send_completed(
    tx: &mpsc::Sender<Message>,
    result: Completed,
    jobs: Vec<crate::Job>,
    pending: Vec<crate::proxies::PendingJob>,
    warnings: Vec<String>,
) -> std::result::Result<(), String> {
    match tx.send(Message::Completed(result, jobs, pending, warnings)) {
        Ok(()) => Ok(()),
        Err(error) => {
            if let Message::Completed(_, jobs, _, _) = error.0 {
                for job in jobs {
                    job.progress.cancel.store(true, Ordering::Relaxed);
                }
            }
            Err("Media preparation receiver closed".into())
        }
    }
}

fn prune_tasks(s: &mut Session) {
    let mut retained: BTreeSet<u64> = s.project.pending_imports.iter().filter(|(id, _)| s.project.item(**id).is_some()).map(|(_, p)| p.ticket).collect();
    for (_, project, _) in s.history.undo.iter().chain(&s.history.redo) {
        retained.extend(project.pending_imports.values().map(|p| p.ticket));
    }
    s.imports.tasks.retain(|ticket, task| {
        let keep = retained.contains(ticket) || task_matches_source(&s.project, task);
        if !keep {
            task.cancel.store(true, Ordering::Relaxed);
        }
        keep
    });
}

/// Observe edits on the editor thread; the worker only reads this decision. Ordinary pending
/// timeline edits must not trigger the synchronous automatic-transcription hook.
pub(crate) fn note_placements(s: &mut Session) {
    if !s.project.pending_imports.is_empty() {
        note_whole_placements(s);
    }
    if !s.prefs.media_analysis.auto_transcribe
        || !s.imports.tasks.values().any(|task| !task.transcribe_attempted && !task.transcribe.load(Ordering::Acquire) && task.error.is_none())
    {
        return;
    }
    let all_imported = s.prefs.media_analysis.auto_transcribe_scope == "allImported";
    // Resolve the placements once for the whole batch, not once per retained imported source.
    let placed: BTreeSet<_> = if all_imported {
        BTreeSet::new()
    } else {
        s.project
            .sequences()
            .filter_map(|item| item.as_sequence())
            .flat_map(|sequence| sequence.audio_tracks.iter().flat_map(|track| &track.items))
            .filter(|clip| clip.enabled)
            .filter_map(|clip| s.project.resolve_media(clip.item).map(|(id, _, _)| id))
            .collect()
    };
    for task in s.imports.tasks.values() {
        if all_imported || placed.contains(&task.item) {
            task.transcribe.store(true, Ordering::Release);
        }
    }
}

fn note_whole_placements(s: &mut Session) {
    let mut new = Vec::new();
    for seq_item in s.project.sequences() {
        let Some(sequence) = seq_item.as_sequence() else { continue };
        for track in sequence.all_tracks() {
            for clip in &track.items {
                let Some(pending) = s.project.pending_imports.get(&clip.item) else { continue };
                if !pending.metadata_ready
                    && !pending.placements.contains_key(&clip.id)
                    && clip.source_in == Tick::ZERO
                    && clip.speed == 1.0
                    && clip.duration
                        == sequence.settings.frame_rate.snap_nearest(pending.provisional_duration).max(sequence.settings.frame_rate.frame_duration())
                {
                    new.push((clip.item, clip.id, (seq_item.id, track.id, clip.start)));
                }
            }
        }
    }
    if !new.is_empty() {
        let project = Arc::make_mut(&mut s.project);
        for (item, clip, location) in new {
            if let Some(pending) = project.pending_imports.get_mut(&item) {
                pending.placements.insert(clip, location);
            }
        }
    }
}

/// A completed bin import can be placed much later. Keep its automatic transcription on the
/// worker path, including after the pending marker disappears, rather than blocking that edit.
pub(crate) fn handles_auto_transcription(s: &Session, item: ItemId) -> bool {
    s.imports.tasks.values().any(|task| task.item == item && task_matches_source(&s.project, task))
}

fn task_matches_source(project: &Project, task: &Task) -> bool {
    if project.pending_imports.get(&task.item).is_some_and(|pending| pending.ticket != task.ticket) {
        return false;
    }
    project.item(task.item).and_then(|item| item.as_media()).is_some_and(|media| {
        !media.offline
            && task.complete.as_ref().is_none_or(|complete| media.identity.is_none() || media.identity == complete.identity)
            && matches!(&media.media, MediaRef::File { path } if *path == task.path || task.complete.as_ref().is_some_and(|complete| *path == complete.path))
    })
}

fn poll_transcripts(s: &mut Session) {
    let mut changed = false;
    for task in s.imports.tasks.values_mut() {
        let Some(rx) = task.transcript_rx.take() else { continue };
        let outcome = match rx.try_recv() {
            Ok(outcome) => outcome,
            Err(mpsc::TryRecvError::Empty) => {
                task.transcript_rx = Some(rx);
                continue;
            }
            Err(mpsc::TryRecvError::Disconnected) => Err("Automatic transcription stopped unexpectedly".into()),
        };
        let applies = task_matches_source(&s.project, task);
        match outcome {
            Ok(transcript) => {
                if let Some(complete) = &mut task.complete {
                    complete.transcript = Some(transcript.clone());
                }
                if applies && !s.project.transcripts.contains_key(&task.item) {
                    Arc::make_mut(&mut s.project).transcripts.insert(task.item, transcript.clone());
                    changed = true;
                }
                // Source discovery survives undo of edits made while recognition was running.
                // This happens only at completion, so a later explicit Delete Transcript stays
                // deleted and can itself be undone normally.
                for (_, project, _) in s.history.undo.iter_mut().chain(&mut s.history.redo) {
                    if task_matches_source(project, task) && !project.transcripts.contains_key(&task.item) {
                        Arc::make_mut(project).transcripts.insert(task.item, transcript.clone());
                    }
                }
            }
            Err(error) if applies => s.events.push(crate::Event::Toast { message: format!("Automatic transcription: {error}"), error: true }),
            Err(_) => {}
        }
    }
    if changed {
        s.bump();
        s.previews.live.publish_project(s.project.clone());
        s.sync_persistence();
    }
    launch_transcript(s);
}

fn launch_transcript(s: &mut Session) {
    // Speech recognition is CPU intensive; at most one automatic recognizer runs at a time.
    if s.imports.tasks.values().any(|task| task.transcript_rx.is_some()) {
        return;
    }
    let candidate = s.imports.tasks.iter().find_map(|(&ticket, task)| {
        (!task.transcribe_attempted
            && task.transcribe.load(Ordering::Acquire)
            && task.error.is_none()
            && task.complete.is_some()
            && task_matches_source(&s.project, task))
        .then_some(ticket)
    });
    let Some(ticket) = candidate else { return };
    let Some(task) = s.imports.tasks.get_mut(&ticket) else { return };
    task.transcribe_attempted = true;
    let Some(complete) = &task.complete else { return };
    if !complete.source.info().has_audio() || s.project.transcripts.contains_key(&task.item) {
        return;
    }
    let (item, path, source, cancel) = (task.item, complete.path.clone(), complete.source.clone(), task.cancel.clone());
    match crate::pending_import_transcript::spawn(s, item, path, source, cancel) {
        Ok(rx) => {
            if let Some(task) = s.imports.tasks.get_mut(&ticket) {
                task.transcript_rx = Some(rx);
            }
        }
        Err(error) => s.events.push(crate::Event::Toast { message: format!("Could not start automatic transcription: {error}"), error: true }),
    }
}

fn status_tree(s: &Session, item: ItemId, time: Option<Tick>) -> Option<ImportStatus> {
    // Normal playback must not walk the timeline once every import has finished.
    if s.project.pending_imports.is_empty() {
        return None;
    }
    let mut stack = vec![(item, time, 0usize)];
    let mut seen = BTreeSet::new();
    let mut available = None;
    while let Some((id, time, depth)) = stack.pop() {
        if seen.len() >= 4096 {
            break;
        }
        if depth >= 16 || !seen.insert((id, time.map(|t| t.0))) {
            continue;
        }
        if let Some((root, _, _)) = s.project.resolve_media(id) {
            if let Some(status) = media_status(s, root, time) {
                if !status.ready {
                    return Some(status);
                }
                available = Some(status);
            }
        } else if let Some(sequence) = s.project.sequence(id) {
            for track in sequence.all_tracks().filter(|track| track.enabled && !track.muted) {
                for clip in track.items.iter().filter(|clip| clip.enabled && time.is_none_or(|t| clip.range().contains(t))) {
                    if stack.len().saturating_add(seen.len()) >= 4096 {
                        break;
                    }
                    stack.push((clip.item, time.map(|t| clip.source_time_at(t)), depth + 1));
                }
            }
        }
    }
    available
}

fn media_status(s: &Session, item: ItemId, time: Option<Tick>) -> Option<ImportStatus> {
    let pending = s.project.pending_imports.get(&item)?;
    let task = s.imports.tasks.get(&pending.ticket).filter(|t| t.item == item && t.path == pending.path);
    let error = pending.error.clone().or_else(|| task.and_then(|t| t.error.clone()));
    if let Some(error) = error {
        return Some(ImportStatus { label: "Media preparation failed".into(), ready: false, progress: None, error: Some(error) });
    }
    let source = task.and_then(|t| t.source.as_ref());
    let mut status = ImportStatus { label: task.map_or("Preparing media", |t| t.phase.as_str()).into(), progress: None, ready: source.is_some(), error: None };
    match source.map(|source| source.readiness()) {
        Some(MediaReadiness::Loading { available, total }) => {
            status.label = "Preparing playback".into();
            status.progress = (total.0 > 0).then(|| (available.0 as f64 / total.0 as f64).clamp(0.0, 1.0) as f32);
            status.ready = time.is_some_and(|t| t.max(Tick::ZERO) + Tick(TICKS_PER_SECOND / 4) < available);
        }
        Some(MediaReadiness::Failed(error)) => {
            status.ready = false;
            status.error = Some(error);
        }
        _ => {}
    }
    Some(status)
}

/// A pending reference must not fall through to synchronous source opening from a paint/audio
/// callback. Renderers already understand the pending signal and retry without caching silence.
pub(crate) struct WaitingSource(pub MediaInfo, pub Option<String>);

impl filmcraft_media::MediaSource for WaitingSource {
    fn info(&self) -> &MediaInfo {
        &self.0
    }
    fn readiness(&self) -> MediaReadiness {
        match &self.1 {
            Some(error) => MediaReadiness::Failed(error.clone()),
            None => MediaReadiness::Loading { available: Tick::ZERO, total: self.0.duration },
        }
    }
    fn video_frame(&self, _: filmcraft_media::FrameRequest) -> filmcraft_media::Result<Arc<filmcraft_frame::VideoFrame>> {
        Err(self.error())
    }
    fn audio(&self, _: i64, _: usize, _: u32) -> filmcraft_media::Result<filmcraft_frame::AudioBuffer> {
        Err(self.error())
    }
}

impl WaitingSource {
    fn error(&self) -> filmcraft_media::MediaError {
        if let Some(error) = &self.1 {
            filmcraft_media::MediaError::Decode(error.clone())
        } else {
            filmcraft_media::pending::mark();
            filmcraft_media::MediaError::Io("Media is still being prepared".into())
        }
    }
}
