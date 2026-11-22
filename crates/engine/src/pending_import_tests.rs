//! A slow media source must not hold the editor hostage. These tests gate the first worker
//! access, edit the ordinary timeline clips while it waits, then deliver the source metadata.

#![cfg(not(target_arch = "wasm32"))]

use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::ThreadId;
use std::time::{Duration, Instant};

use filmcraft_media::SharedReader;
use filmcraft_media::reader::MemReader;
use filmcraft_project::{ClipId, ItemId, Project, TrackItem};
use filmcraft_time::{TICKS_PER_SECOND, Tick};
use serde_json::json;

use crate::{Services, Session};

const PATH: &str = "/pending-import-test/voice.wav";
const MEDIA_DURATION: Tick = Tick(12 * TICKS_PER_SECOND);

/// Reject caller-thread I/O rather than hanging if a regression makes import synchronous.
/// The timeout also lets a failed test release a detached worker without leaking it forever.
struct Gate {
    caller: ThreadId,
    released: Mutex<bool>,
    wake: Condvar,
    caller_reads: AtomicUsize,
    worker_reads: AtomicUsize,
}

impl Gate {
    fn new() -> Self {
        Self {
            caller: std::thread::current().id(),
            released: Mutex::new(false),
            wake: Condvar::new(),
            caller_reads: AtomicUsize::new(0),
            worker_reads: AtomicUsize::new(0),
        }
    }

    fn read(&self) -> io::Result<()> {
        if std::thread::current().id() == self.caller && !*self.released.lock().unwrap() {
            self.caller_reads.fetch_add(1, Ordering::SeqCst);
            return Err(io::Error::new(io::ErrorKind::PermissionDenied, "media I/O on the editor thread"));
        }
        self.worker_reads.fetch_add(1, Ordering::SeqCst);
        let (released, timeout) = self.wake.wait_timeout_while(self.released.lock().unwrap(), Duration::from_secs(10), |ready| !*ready).unwrap();
        if timeout.timed_out() && !*released {
            return Err(io::Error::new(io::ErrorKind::TimedOut, "test did not release media"));
        }
        Ok(())
    }

    fn release(&self) {
        *self.released.lock().unwrap() = true;
        self.wake.notify_all();
    }
}

struct DelayedFiles {
    bytes: Arc<[u8]>,
    gate: Arc<Gate>,
}

impl Services for DelayedFiles {
    fn read_file(&self, _: &str) -> io::Result<Vec<u8>> {
        self.gate.read()?;
        Ok(self.bytes.to_vec())
    }

    fn write_file(&self, _: &str, _: &[u8]) -> io::Result<()> {
        Err(io::Error::new(io::ErrorKind::Unsupported, "test does not write files"))
    }

    fn file_size(&self, _: &str) -> io::Result<u64> {
        self.gate.read()?;
        Ok(self.bytes.len() as u64)
    }

    fn read_range(&self, _: &str, offset: u64, len: usize) -> io::Result<Vec<u8>> {
        self.gate.read()?;
        let start = usize::try_from(offset).unwrap().min(self.bytes.len());
        let end = start.saturating_add(len).min(self.bytes.len());
        Ok(self.bytes[start..end].to_vec())
    }

    fn reader(&self, _: &str) -> Option<io::Result<SharedReader>> {
        Some(self.gate.read().map(|()| Arc::new(MemReader(self.bytes.clone())) as SharedReader))
    }
}

/// Only PCM is needed: no decoder model, external tool, fixture file, or sound device.
fn wav() -> Arc<[u8]> {
    let rate = 8000u32;
    let frames = rate * 12;
    let size = frames * 2;
    let mut out = Vec::with_capacity(size as usize + 44);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + size).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&1u16.to_le_bytes());
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 2).to_le_bytes());
    out.extend_from_slice(&2u16.to_le_bytes());
    out.extend_from_slice(&16u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&size.to_le_bytes());
    for _ in 0..frames {
        out.extend_from_slice(&4096i16.to_le_bytes());
    }
    out.into()
}

struct Editor {
    session: Session,
    gate: Arc<Gate>,
}

impl Editor {
    fn new(bytes: Arc<[u8]>) -> Self {
        let gate = Arc::new(Gate::new());
        let services = Arc::new(DelayedFiles { bytes, gate: gate.clone() });
        let mut session = Session::new(services);
        session.execute("file.newSequence", json!({"name": "Pending media", "fps": 24, "video": 1, "audio": 2})).unwrap();
        Self { session, gate }
    }

    fn import(&mut self) -> ItemId {
        let result = self.session.execute("file.import", json!({"paths": [PATH], "background": true})).unwrap();
        let item = ItemId(result["items"][0].as_u64().expect("acceptance returns an editable project item"));
        assert_eq!(self.gate.caller_reads.load(Ordering::SeqCst), 0, "acceptance performed synchronous media I/O");
        let status = self.session.import_status(item).expect("pending item has a visible status");
        assert!(!status.ready);
        assert!(status.error.is_none());
        item
    }

    fn place(&mut self, item: ItemId) -> ClipId {
        self.session.execute("timeline.place", json!({"item": item.0, "time": 0, "insert": false})).unwrap();
        self.session.active_sequence().unwrap().audio_tracks.iter().flat_map(|track| &track.items).find(|clip| clip.item == item).unwrap().id
    }

    fn finish(&mut self) {
        self.gate.release();
        settle(&mut self.session);
    }
}

impl Drop for Editor {
    fn drop(&mut self) {
        self.gate.release();
    }
}

fn settle(session: &mut Session) {
    let until = Instant::now() + Duration::from_secs(5);
    loop {
        session.poll_imports();
        if !session.imports_running() {
            break;
        }
        assert!(Instant::now() < until, "background import did not settle");
        std::thread::sleep(Duration::from_millis(1));
    }
    session.poll_imports();
}

fn clip(session: &Session, id: ClipId) -> TrackItem {
    session.active_sequence().unwrap().find_item(id).unwrap().1.clone()
}

fn move_to_second_track(session: &mut Session, id: ClipId, at: Tick) {
    session.execute("timeline.move", json!({"moves": [{"clip": id.0, "track": "A2", "time": at.0}], "insert": false, "linked": false})).unwrap();
}

#[test]
fn acceptance_and_moving_do_not_wait_for_media_io() {
    let mut editor = Editor::new(wav());
    let item = editor.import();
    let id = editor.place(item);
    let at = Tick(37 * TICKS_PER_SECOND);
    move_to_second_track(&mut editor.session, id, at);
    assert_eq!(clip(&editor.session, id).start, at);
    assert_eq!(editor.gate.caller_reads.load(Ordering::SeqCst), 0);
    assert!(editor.session.imports_running(), "the source has not been released");
    let undo = editor.session.history.undo.len();

    editor.finish();

    let ready = clip(&editor.session, id);
    assert_eq!(ready.item, item);
    assert_eq!(ready.start, at, "completion must keep the user's move");
    assert_eq!(ready.duration, MEDIA_DURATION, "an untouched full-source span takes the discovered duration");
    assert_eq!(editor.session.active_sequence().unwrap().audio_tracks[1].items[0].id, id);
    assert_eq!(editor.session.history.undo.len(), undo, "preparation creates no user edit");
    assert!(editor.session.import_status(item).is_none_or(|status| status.ready && status.error.is_none()));
    let source = editor.session.source(item).unwrap();
    let audio = source.audio(0, 32, 8000).unwrap();
    assert!(audio.channels[0].iter().any(|sample| *sample > 0.0), "the clip uses the real source after preparation");
}

#[test]
fn the_previous_synchronous_import_path_cannot_accept_a_blocked_source() {
    let mut editor = Editor::new(wav());
    let result = editor.session.execute("file.import", json!({"paths": [PATH]}));
    assert!(editor.gate.caller_reads.load(Ordering::SeqCst) > 0, "the legacy path attempts media I/O before returning item identities");
    if let Ok(result) = result {
        assert!(result["items"].as_array().is_none_or(Vec::is_empty), "a blocked synchronous source cannot become an accepted item");
        assert!(!result["errors"].as_array().unwrap().is_empty());
    }
}

#[test]
fn querying_pending_media_status_does_not_perform_caller_thread_io() {
    let mut editor = Editor::new(wav());
    let item = editor.import();
    editor.place(item);
    let status = editor.session.execute("media.status", json!({"item": item.0})).unwrap();
    assert_eq!(status[0]["status"], "preparing");
    assert_eq!(status[0]["preparation"]["ready"], false);
    assert!(status[0]["preparation"]["error"].is_null());
    assert!(status[0]["preparation"]["label"].as_str().is_some_and(|label| !label.is_empty()));
    assert_eq!(editor.gate.caller_reads.load(Ordering::SeqCst), 0, "opening Properties or querying preparation must not probe the blocked file");
    crate::relink::on_open(&mut editor.session);
    assert_eq!(editor.gate.caller_reads.load(Ordering::SeqCst), 0, "opening a saved pending project must resume its worker instead of scanning the file");
    assert!(editor.session.imports_running());
}

struct DelayedTranscriber {
    gate: Arc<Gate>,
    calls: Arc<AtomicUsize>,
    fail: bool,
}

impl filmcraft_speech::Transcriber for DelayedTranscriber {
    fn id(&self) -> String {
        "pending-test-speech".into()
    }

    fn transcribe(
        &self,
        _: &[f32],
        _: &filmcraft_speech::Options,
        _: filmcraft_speech::ProgressFn,
    ) -> Result<filmcraft_project::Transcript, filmcraft_speech::SpeechError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.gate.read()?;
        if self.fail {
            return Err(filmcraft_speech::SpeechError::Unavailable("test recognizer failed".into()));
        }
        Ok(filmcraft_project::Transcript {
            source: self.id(),
            words: vec![filmcraft_project::Word::new("Ready", Tick::ZERO, Tick(TICKS_PER_SECOND))],
            ..Default::default()
        })
    }
}

fn automatic_speech(editor: &mut Editor, fail: bool) -> (Arc<Gate>, Arc<AtomicUsize>) {
    let gate = Arc::new(Gate::new());
    let calls = Arc::new(AtomicUsize::new(0));
    editor.session.transcriber = Some(Arc::new(DelayedTranscriber { gate: gate.clone(), calls: calls.clone(), fail }));
    editor.session.prefs.media_analysis.auto_transcribe = true;
    editor.session.prefs.media_analysis.auto_transcribe_scope = "sequenceClips".into();
    (gate, calls)
}

#[test]
fn a_completed_bin_import_transcribes_on_later_placement_without_blocking_or_losing_undo() {
    let mut editor = Editor::new(wav());
    let (speech, calls) = automatic_speech(&mut editor, false);
    let item = editor.import();
    editor.finish();
    assert_eq!(calls.load(Ordering::SeqCst), 0, "sequence-only scope does not hold bin imports open");
    assert!(!editor.session.project.pending_imports.contains_key(&item));
    // Late placement remains asynchronous even when the original import left undo history.
    editor.session.history.undo.clear();
    editor.session.history.redo.clear();
    editor.session.poll_imports();
    let id = editor.place(item);
    editor.session.poll_imports();
    move_to_second_track(&mut editor.session, id, Tick(20 * TICKS_PER_SECOND));
    assert_eq!(speech.caller_reads.load(Ordering::SeqCst), 0, "late placement must not run speech recognition on the editor thread");
    assert!(editor.session.imports_running());
    speech.release();
    settle(&mut editor.session);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(editor.session.project.transcripts[&item].words[0].text, "Ready");
    editor.session.execute("edit.undo", json!({})).unwrap();
    assert!(editor.session.project.transcripts.contains_key(&item), "undoing an intervening move preserves discovered source text");
    editor.session.execute("transcript.delete", json!({"items": [item.0]})).unwrap();
    editor.session.poll_imports();
    assert!(!editor.session.project.transcripts.contains_key(&item), "explicit later deletion must not be undone by polling");
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn placement_before_media_completion_also_requests_one_background_transcription() {
    let mut editor = Editor::new(wav());
    let (speech, calls) = automatic_speech(&mut editor, false);
    let item = editor.import();
    editor.place(item);
    speech.release();
    editor.finish();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(speech.caller_reads.load(Ordering::SeqCst), 0);
    assert!(editor.session.project.transcripts.contains_key(&item));
}

#[test]
fn failed_automatic_transcription_is_terminal_and_does_not_hold_media_pending() {
    let mut editor = Editor::new(wav());
    let (speech, calls) = automatic_speech(&mut editor, true);
    let item = editor.import();
    editor.place(item);
    speech.release();
    editor.finish();
    for _ in 0..3 {
        editor.session.poll_imports();
    }
    assert_eq!(calls.load(Ordering::SeqCst), 1, "a failed recognizer is not retried every frame");
    assert!(!editor.session.imports_running());
    assert!(!editor.session.project.transcripts.contains_key(&item));
    assert_eq!(editor.session.project.item(item).unwrap().duration(), MEDIA_DURATION);
}

fn detached_job(id: u64) -> crate::Job {
    crate::Job { id, label: format!("Detached {id}"), progress: Default::default(), result: Default::default() }
}

fn detached_output(job: u64) -> crate::proxies::PendingJob {
    crate::proxies::PendingJob { job, on_done: crate::proxies::OnDone::Nothing, outputs: Default::default() }
}

#[test]
fn detached_job_id_remapping_never_remaps_an_already_translated_output() {
    let mut jobs = vec![detached_job(1), detached_job(2)];
    let mut outputs = vec![detached_output(1), detached_output(2)];
    crate::pending_import::remap_jobs(&mut jobs, &mut outputs, 1).unwrap();
    assert_eq!(jobs.iter().map(|job| job.id).collect::<Vec<_>>(), vec![2, 3]);
    assert_eq!(outputs.iter().map(|output| output.job).collect::<Vec<_>>(), vec![2, 3]);
}

#[test]
fn exhausted_or_ambiguous_detached_job_ids_are_refused_before_mutation() {
    for (ids, largest, output_ids) in [([1, 2], u64::MAX - 1, [1, 2]), ([1, 1], 10, [1, 1]), ([1, 2], 10, [1, 3])] {
        let mut jobs: Vec<_> = ids.into_iter().map(detached_job).collect();
        let mut outputs: Vec<_> = output_ids.into_iter().map(detached_output).collect();
        assert!(crate::pending_import::remap_jobs(&mut jobs, &mut outputs, largest).is_err());
        assert_eq!(jobs.iter().map(|job| job.id).collect::<Vec<_>>(), ids);
        assert_eq!(outputs.iter().map(|output| output.job).collect::<Vec<_>>(), output_ids);
    }
}

#[test]
fn stale_or_offline_sources_do_not_adopt_detached_ingest_outputs() {
    let mut editor = Editor::new(wav());
    let item = editor.import();
    let ticket = editor.session.project.pending_imports[&item].ticket;
    assert!(crate::pending_import::accepts_completed_jobs(&editor.session.project, item, PATH, ticket));
    assert!(!crate::pending_import::accepts_completed_jobs(&editor.session.project, item, PATH, ticket + 1));
    let project = Arc::make_mut(&mut editor.session.project);
    project.item_mut(item).unwrap().as_media_mut().unwrap().offline = true;
    assert!(!crate::pending_import::accepts_completed_jobs(project, item, PATH, ticket));
    let media = project.item_mut(item).unwrap().as_media_mut().unwrap();
    media.offline = false;
    media.media = filmcraft_project::MediaRef::File { path: "/replacement/voice.wav".into() };
    assert!(!crate::pending_import::accepts_completed_jobs(project, item, PATH, ticket));
    project.items.remove(&item);
    assert!(!crate::pending_import::accepts_completed_jobs(project, item, PATH, ticket));
}

#[test]
fn completion_preserves_pending_trim_split_and_move_edits() {
    let mut editor = Editor::new(wav());
    let item = editor.import();
    let id = editor.place(item);
    let one_second = Tick(TICKS_PER_SECOND);
    let delta = one_second - clip(&editor.session, id).duration;
    editor.session.execute("timeline.trim", json!({"clip": id.0, "edge": "out", "delta": delta.0, "linked": false})).unwrap();
    let cut = editor.session.sequence_rate().tick_of(12);
    editor.session.execute("sequence.addEditAllTracks", json!({"time": cut.0})).unwrap();
    let right = editor.session.active_sequence().unwrap().audio_tracks[0].items.iter().find(|clip| clip.id != id).unwrap().id;
    move_to_second_track(&mut editor.session, right, Tick(80 * TICKS_PER_SECOND));
    let before = [clip(&editor.session, id), clip(&editor.session, right)];
    let undo = editor.session.history.undo.len();
    assert_eq!(editor.gate.caller_reads.load(Ordering::SeqCst), 0);

    editor.finish();

    assert_eq!([clip(&editor.session, id), clip(&editor.session, right)], before, "metadata completion must not replay or overwrite timeline edits");
    assert_eq!(editor.session.history.undo.len(), undo);
    editor.session.active_sequence().unwrap().check().unwrap();
}

#[test]
fn undo_redo_after_completion_restore_edits_without_restoring_pending_media() {
    let mut editor = Editor::new(wav());
    let item = editor.import();
    let id = editor.place(item);
    let at = Tick(7 * TICKS_PER_SECOND);
    move_to_second_track(&mut editor.session, id, at);
    editor.finish();
    let after = clip(&editor.session, id);

    editor.session.execute("edit.undo", json!({})).unwrap();
    editor.session.poll_imports();
    assert_eq!(clip(&editor.session, id).start, Tick::ZERO);
    assert_eq!(clip(&editor.session, id).duration, MEDIA_DURATION);
    assert!(editor.session.import_status(item).is_none_or(|status| status.ready && status.error.is_none()));
    editor.session.execute("edit.redo", json!({})).unwrap();
    editor.session.poll_imports();
    assert_eq!(clip(&editor.session, id), after);
    assert!(!editor.session.imports_running(), "undo/redo must not restart a completed import");
}

#[test]
fn undoing_placement_while_loading_keeps_redo_available_after_completion() {
    let mut editor = Editor::new(wav());
    let item = editor.import();
    let id = editor.place(item);
    editor.session.execute("edit.undo", json!({})).unwrap();
    assert!(editor.session.active_sequence().unwrap().find_item(id).is_none());
    let redo = editor.session.history.redo.len();

    editor.finish();

    assert_eq!(editor.session.history.redo.len(), redo, "preparation must not destroy a pending redo");
    assert!(editor.session.active_sequence().unwrap().find_item(id).is_none(), "completion does not reapply an undone placement");
    editor.session.execute("edit.redo", json!({})).unwrap();
    editor.session.poll_imports();
    assert_eq!(clip(&editor.session, id).item, item);
    assert_eq!(clip(&editor.session, id).duration, MEDIA_DURATION);
    assert!(editor.session.import_status(item).is_none_or(|status| status.ready && status.error.is_none()));
}

#[test]
fn deleting_pending_media_does_not_resurrect_it_when_the_worker_finishes() {
    let mut editor = Editor::new(wav());
    let item = editor.import();
    let id = editor.place(item);
    editor.session.execute("timeline.select", json!({"clips": [id.0]})).unwrap();
    editor.session.execute("edit.clear", json!({})).unwrap();
    editor.session.execute("project.delete", json!({"items": [item.0]})).unwrap();
    assert!(editor.session.project.item(item).is_none());
    let before = editor.session.project.items.clone();
    let undo = editor.session.history.undo.len();

    editor.finish();

    assert_eq!(editor.session.project.items, before);
    assert_eq!(editor.session.history.undo.len(), undo);
    assert!(editor.session.project.item(item).is_none());
    editor.session.execute("edit.undo", json!({})).unwrap();
    settle(&mut editor.session);
    assert_eq!(editor.session.project.item(item).unwrap().duration(), MEDIA_DURATION);
}

#[test]
fn an_untouched_batch_keeps_every_files_full_duration_and_order() {
    let mut editor = Editor::new(wav());
    let result = editor.session.execute("file.import", json!({"paths": [PATH, PATH], "background": true})).unwrap();
    let ids: Vec<_> = result["items"].as_array().unwrap().iter().map(|id| ItemId(id.as_u64().unwrap())).collect();
    assert_eq!(ids.len(), 2);
    assert_ne!(ids[0], ids[1], "separate occurrences are separate editable media items");
    editor.session.execute("timeline.place", json!({"items": [ids[0].0, ids[1].0], "time": 0, "insert": false})).unwrap();
    let before: Vec<_> = editor.session.active_sequence().unwrap().audio_tracks[0].items.iter().map(|clip| clip.id).collect();

    editor.finish();

    let clips = &editor.session.active_sequence().unwrap().audio_tracks[0].items;
    assert_eq!(clips.len(), 2);
    assert_eq!(clips.iter().map(|clip| clip.id).collect::<Vec<_>>(), before);
    assert_eq!(clips[0].item, ids[0]);
    assert_eq!(clips[1].item, ids[1]);
    assert_eq!(clips[0].duration, MEDIA_DURATION, "the next provisional clip must not truncate the first source");
    assert_eq!(clips[1].duration, MEDIA_DURATION);
    assert_eq!(clips[0].start, Tick::ZERO);
    assert_eq!(clips[1].start, MEDIA_DURATION, "untouched batch clips stay end-to-end after their real lengths arrive");
    editor.session.active_sequence().unwrap().check().unwrap();
}

/// Deterministic metadata order, independent of operating-system worker scheduling.
fn hydrate_now(editor: &mut Editor, item: ItemId, duration: Tick) -> Vec<String> {
    let pending = editor.session.project.pending_imports[&item].clone();
    let mut info = editor.session.project.item(item).unwrap().as_media().unwrap().info.clone();
    info.duration = duration;
    crate::pending_import::hydrate(Arc::make_mut(&mut editor.session.project), item, &pending, &info)
}

fn place_batch(editor: &mut Editor, count: usize) -> Vec<ItemId> {
    let result = editor.session.execute("file.import", json!({"paths": vec![PATH; count], "background": true})).unwrap();
    let ids: Vec<_> = result["items"].as_array().unwrap().iter().map(|id| ItemId(id.as_u64().unwrap())).collect();
    editor.session.execute("timeline.place", json!({"items": ids.iter().map(|id| id.0).collect::<Vec<_>>(), "time": 0, "insert": false})).unwrap();
    ids
}

#[test]
fn reverse_and_interleaved_metadata_completion_preserve_the_whole_batch() {
    for order in [[2usize, 1, 0], [1, 0, 2], [0, 1, 2]] {
        let mut editor = Editor::new(wav());
        let items = place_batch(&mut editor, 3);
        let identities: Vec<_> = editor.session.active_sequence().unwrap().audio_tracks[0].items.iter().map(|clip| clip.id).collect();
        for index in order {
            assert!(hydrate_now(&mut editor, items[index], MEDIA_DURATION).is_empty());
        }
        let q = editor.session.active_sequence().unwrap();
        let clips = &q.audio_tracks[0].items;
        assert_eq!(clips.iter().map(|clip| clip.id).collect::<Vec<_>>(), identities);
        for (index, clip) in clips.iter().enumerate() {
            assert_eq!(clip.start, Tick(index as i64 * MEDIA_DURATION.0), "completion order {order:?}");
            assert_eq!(clip.duration, MEDIA_DURATION, "completion order {order:?}");
        }
        q.check().unwrap();
    }
}

#[test]
fn repeated_instances_in_a_batch_use_updated_placement_anchors() {
    let mut editor = Editor::new(wav());
    let result = editor.session.execute("file.import", json!({"paths": [PATH, PATH], "background": true})).unwrap();
    let first = ItemId(result["items"][0].as_u64().unwrap());
    let second = ItemId(result["items"][1].as_u64().unwrap());
    editor.session.execute("timeline.place", json!({"items": [first.0, first.0, second.0], "time": 0, "insert": false})).unwrap();
    assert!(hydrate_now(&mut editor, first, MEDIA_DURATION).is_empty());
    assert!(hydrate_now(&mut editor, second, MEDIA_DURATION).is_empty());
    let clips = &editor.session.active_sequence().unwrap().audio_tracks[0].items;
    assert_eq!(clips.len(), 3);
    for (index, clip) in clips.iter().enumerate() {
        assert_eq!(clip.start, Tick(index as i64 * MEDIA_DURATION.0));
        assert_eq!(clip.duration, MEDIA_DURATION);
    }
    editor.session.active_sequence().unwrap().check().unwrap();
}

#[test]
fn duration_discovery_stops_before_unrelated_edits_and_explains_the_cap() {
    let mut editor = Editor::new(wav());
    let first = editor.import();
    let first_clip = editor.place(first);
    let unrelated = editor.import();
    editor.session.execute("timeline.place", json!({"item": unrelated.0, "time": 10 * TICKS_PER_SECOND, "insert": false})).unwrap();
    let obstacle = editor.session.active_sequence().unwrap().audio_tracks[0].items[1].clone();
    let warnings = hydrate_now(&mut editor, first, MEDIA_DURATION);
    assert!(warnings.iter().any(|warning| warning.contains("full source") && warning.contains("existing edit")));
    assert_eq!(clip(&editor.session, first_clip).duration, Tick(10 * TICKS_PER_SECOND));
    assert_eq!(clip(&editor.session, obstacle.id), obstacle);
    assert_eq!(editor.session.project.item(first).unwrap().duration(), MEDIA_DURATION);
    editor.session.active_sequence().unwrap().check().unwrap();
}

#[test]
fn duration_discovery_preserves_locked_tracks_and_user_transition_timing() {
    for locked in [false, true] {
        let mut editor = Editor::new(wav());
        let item = editor.import();
        let id = editor.place(item);
        if locked {
            let sequence = editor.session.state.active_sequence.unwrap();
            Arc::make_mut(&mut editor.session.project).sequence_mut(sequence).unwrap().audio_tracks[0].locked = true;
        } else {
            editor.session.execute("effects.apply", json!({"effect": "constant_power", "clip": id.0, "edge": "out", "frames": 10})).unwrap();
        }
        let before = editor.session.active_sequence().unwrap().audio_tracks[0].clone();
        assert!(!hydrate_now(&mut editor, item, MEDIA_DURATION).is_empty());
        assert_eq!(editor.session.active_sequence().unwrap().audio_tracks[0], before);
        assert_eq!(editor.session.project.item(item).unwrap().duration(), MEDIA_DURATION);
        editor.session.active_sequence().unwrap().check().unwrap();
    }
}

#[test]
fn metadata_does_not_carry_a_batch_neighbor_moved_by_the_editor() {
    let mut editor = Editor::new(wav());
    let items = place_batch(&mut editor, 2);
    let first = editor.session.active_sequence().unwrap().audio_tracks[0].items[0].id;
    let neighbor = editor.session.active_sequence().unwrap().audio_tracks[0].items[1].id;
    let sequence = editor.session.state.active_sequence.unwrap();
    // A five-second move leaves an intentional gap. The baseline still records the original
    // contiguous drop, so import completion must treat this neighbor as an ordinary obstacle.
    let track = editor.session.active_sequence().unwrap().audio_tracks[0].id;
    editor
        .session
        .execute("timeline.move", json!({"moves": [{"clip": neighbor.0, "track": track.0, "time": 10 * TICKS_PER_SECOND}], "insert": false, "linked": false}))
        .unwrap();
    let before = clip(&editor.session, neighbor);
    assert!(!hydrate_now(&mut editor, items[0], MEDIA_DURATION).is_empty());
    assert_eq!(clip(&editor.session, neighbor), before);
    assert_eq!(clip(&editor.session, first).duration, Tick(10 * TICKS_PER_SECOND));
    assert_eq!(editor.session.state.active_sequence, Some(sequence));
    editor.session.active_sequence().unwrap().check().unwrap();
}

#[test]
fn deleting_a_batch_sibling_does_not_hold_the_surviving_import_open() {
    let mut editor = Editor::new(wav());
    let items = place_batch(&mut editor, 2);
    let second_clip = editor.session.active_sequence().unwrap().audio_tracks[0].items[1].id;
    editor.session.execute("timeline.select", json!({"clips": [second_clip.0]})).unwrap();
    editor.session.execute("edit.clear", json!({})).unwrap();
    editor.session.execute("project.delete", json!({"items": [items[1].0]})).unwrap();
    editor.finish();
    assert!(editor.session.project.item(items[1]).is_none());
    assert_eq!(editor.session.project.item(items[0]).unwrap().duration(), MEDIA_DURATION);
    assert!(!editor.session.imports_running());
    editor.session.execute("edit.undo", json!({})).unwrap();
    settle(&mut editor.session);
    assert_eq!(editor.session.project.item(items[1]).unwrap().duration(), MEDIA_DURATION);
}

#[test]
fn background_import_honors_replace_before_accepting_items() {
    let mut editor = Editor::new(wav());
    let before = editor.session.document_epoch;
    let result = editor.session.execute("file.import", json!({"paths": [PATH], "background": true, "replace": true})).unwrap();
    let item = ItemId(result["items"][0].as_u64().unwrap());
    assert!(editor.session.active_sequence().is_none());
    assert_eq!(editor.session.project.items.len(), 1);
    assert!(editor.session.document_epoch > before);
    assert_eq!(editor.gate.caller_reads.load(Ordering::SeqCst), 0);
    editor.finish();
    assert_eq!(editor.session.project.item(item).unwrap().duration(), MEDIA_DURATION);
}

#[test]
fn duplicated_saved_import_tickets_are_rejected_or_finish_without_stranding_an_item() {
    let mut editor = Editor::new(wav());
    let result = editor.session.execute("file.import", json!({"paths": [PATH, PATH], "background": true})).unwrap();
    let ids: Vec<_> = result["items"].as_array().unwrap().iter().map(|id| ItemId(id.as_u64().unwrap())).collect();
    let mut project = (*editor.session.project).clone();
    let repeated = project.pending_imports[&ids[0]].ticket;
    project.pending_imports.get_mut(&ids[1]).unwrap().ticket = repeated;
    let bytes = filmcraft_format::encode(&project, false);
    let Ok(decoded) = filmcraft_format::decode(&bytes) else { return };
    let mut reopened = Session::new(editor.session.services.clone());
    crate::commands::install_project(&mut reopened, decoded.project, None, false);
    editor.gate.release();
    settle(&mut reopened);
    for id in ids {
        assert!(reopened.project.item(id).is_some());
        assert!(reopened.import_status(id).is_none_or(|status| status.ready || status.error.is_some()));
    }
}

#[test]
fn replacing_the_project_discards_old_import_results_even_when_item_ids_repeat() {
    let mut editor = Editor::new(wav());
    let old = editor.import();
    editor.session.execute("file.newProject", json!({"name": "Replacement"})).unwrap();
    editor.session.execute("file.newSequence", json!({"name": "Replacement sequence", "video": 1, "audio": 2})).unwrap();
    let replacement = editor.session.project.clone();
    let epoch = editor.session.document_epoch;

    editor.finish();

    assert_eq!(editor.session.project, replacement, "a result belonging to the old document must be discarded");
    assert_eq!(editor.session.document_epoch, epoch);
    assert!(editor.session.project.item(old).is_none_or(|item| item.as_media().is_none()));
}

#[test]
fn failed_import_keeps_its_editable_clip_and_reports_the_error() {
    let mut editor = Editor::new(Arc::from(&b"not a WAV file"[..]));
    let item = editor.import();
    let id = editor.place(item);
    let at = Tick(19 * TICKS_PER_SECOND);
    move_to_second_track(&mut editor.session, id, at);
    let before = clip(&editor.session, id);

    editor.finish();

    assert_eq!(clip(&editor.session, id), before);
    let status = editor.session.import_status(item).expect("failed placeholder remains inspectable");
    assert!(!status.ready);
    assert!(status.error.is_some_and(|message| !message.is_empty()));
    move_to_second_track(&mut editor.session, id, Tick(23 * TICKS_PER_SECOND));
    assert_eq!(clip(&editor.session, id).start, Tick(23 * TICKS_PER_SECOND));
}

#[test]
fn saving_a_pending_project_preserves_edits_and_resumes_preparation_when_reopened() {
    let mut editor = Editor::new(wav());
    let item = editor.import();
    let id = editor.place(item);
    let at = Tick(11 * TICKS_PER_SECOND);
    move_to_second_track(&mut editor.session, id, at);
    let bytes = filmcraft_format::encode(&editor.session.project, false);
    let Project { name, .. } = &*editor.session.project;
    let decoded = filmcraft_format::decode(&bytes).unwrap().project;
    assert_eq!(&decoded.name, name);
    let mut reopened = Session::new(editor.session.services.clone());
    crate::commands::install_project(&mut reopened, decoded, None, false);
    reopened.poll_imports();
    assert_eq!(clip(&reopened, id).start, at);
    assert!(reopened.import_status(item).is_some_and(|status| !status.ready));

    editor.gate.release();
    settle(&mut reopened);

    assert_eq!(clip(&reopened, id).start, at);
    assert_eq!(clip(&reopened, id).duration, MEDIA_DURATION);
    assert!(reopened.import_status(item).is_none_or(|status| status.ready && status.error.is_none()));
}
