//! The pending worker preserves ordinary import preferences and owns detached jobs until handoff.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use filmcraft_media::MediaKind;
use filmcraft_project::{IngestAction, ItemId};
use filmcraft_time::FrameRate;
use serde_json::json;

use super::*;

fn settle(session: &mut Session) {
    let deadline = Instant::now() + Duration::from_secs(5);
    while session.imports_running() {
        session.poll_imports();
        assert!(Instant::now() < deadline, "pending image import stayed running");
        std::thread::sleep(Duration::from_millis(1));
    }
    session.poll_imports();
}

fn frames(label: &str) -> (std::path::PathBuf, Vec<String>) {
    let dir = crate::media_test_util::tmp_dir(label);
    let paths = (1..=3)
        .map(|i| {
            let path = dir.join(format!("frame{i:04}.png"));
            image::RgbaImage::from_pixel(4, 4, image::Rgba([20 * i, 30, 40, 255])).save(&path).unwrap();
            path.to_string_lossy().into_owned()
        })
        .collect();
    (dir, paths)
}

fn import(session: &mut Session, paths: &[String]) -> Vec<ItemId> {
    let result = session.execute("file.import", json!({"paths": paths, "background": true})).unwrap();
    assert_eq!(result["pending"], true);
    let ids = result["items"].as_array().unwrap().iter().map(|id| ItemId(id.as_u64().unwrap())).collect();
    settle(session);
    ids
}

#[test]
fn single_numbered_import_detects_a_sequence_and_does_not_ingest_only_its_first_frame() {
    let (dir, paths) = frames("pending-sequence-pref");
    let mut session = Session::default();
    session.prefs.media.import_image_sequences = true;
    session.prefs.media.indeterminate_timebase = "25".into();
    let settings = &mut Arc::make_mut(&mut session.project).settings.ingest;
    settings.enabled = true;
    settings.action = IngestAction::Copy;
    let ids = import(&mut session, &paths[..1]);
    let media = session.project.item(ids[0]).unwrap().as_media().unwrap();
    assert_eq!(media.info.kind, MediaKind::ImageSequence);
    assert_eq!(media.info.frame_rate(), FrameRate::FPS_25);
    assert_eq!(media.info.duration, FrameRate::FPS_25.tick_of(3));
    assert!(!dir.join("Ingested Media").exists(), "a sequence cannot ingest just its first still");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn multi_file_import_keeps_stills_and_the_indeterminate_timebase() {
    let (dir, paths) = frames("pending-still-pref");
    let mut session = Session::default();
    session.prefs.media.import_image_sequences = true;
    session.prefs.media.indeterminate_timebase = "24".into();
    let ids = import(&mut session, &paths[..2]);
    for id in ids {
        let media = session.project.item(id).unwrap().as_media().unwrap();
        assert_eq!(media.info.kind, MediaKind::Still);
        assert_eq!(media.info.frame_rate(), FrameRate::FPS_24);
        assert_eq!(session.media.cached(id).unwrap().info().frame_rate(), FrameRate::FPS_24);
    }
    std::fs::remove_dir_all(dir).unwrap();
}

fn make_job() -> (crate::Job, Arc<filmcraft_export::Progress>) {
    let progress = Arc::new(filmcraft_export::Progress::default());
    (crate::Job { id: 1, label: "synthetic ingest".into(), progress: progress.clone(), result: Arc::new(Mutex::new(None)) }, progress)
}

#[test]
fn detached_jobs_are_cancelled_on_unwind_and_rejected_handoff() {
    let (job, progress) = make_job();
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut detached = DetachedImport(Session::default());
        detached.jobs.push(job);
        panic!("synthetic failure after starting ingest");
    }));
    assert!(result.is_err());
    assert!(progress.cancel.load(Ordering::Relaxed));

    let (job, progress) = make_job();
    let session = Session::default();
    let source = Arc::new(WaitingSource(provisional_info(&session, "voice.wav"), None));
    let completed = Completed { identity: None, transcript: None, path: "voice.wav".into(), source };
    let (tx, rx) = mpsc::channel();
    drop(rx);
    assert!(send_completed(&tx, completed, vec![job], Vec::new(), Vec::new()).is_err());
    assert!(progress.cancel.load(Ordering::Relaxed), "a job whose handles were not delivered must stop");
}

#[test]
fn deleted_pending_sources_are_released_when_undo_no_longer_needs_them() {
    let mut session = Session::default();
    let result = start(&mut session, &["/pending-import-test/missing.wav".into()], None).unwrap();
    let item = result["items"][0].as_u64().unwrap();
    assert!(crate::project_tools::project_subset(&session.project, &BTreeSet::new()).pending_imports.is_empty());
    session.execute("project.delete", json!({"items": [item]})).unwrap();
    assert!(session.project.item(ItemId(item)).is_none());
    assert!(!session.project.pending_imports.contains_key(&ItemId(item)));
    session.history.undo.clear();
    session.history.redo.clear();
    session.poll_imports();
    assert!(session.imports.tasks.is_empty(), "a stale pending marker must not retain the deleted source forever");
}
