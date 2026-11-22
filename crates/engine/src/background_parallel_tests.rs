//! Deterministic coverage for simultaneous analysis of different original files.

use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

use filmcraft_media::DemoScene;
use filmcraft_media::person_matte::MatteReader;
use filmcraft_project::{ClipId, ItemId, PersonMatteRef};
use serde_json::{Value, json};

use super::{Matte, Segmenter, job_for_clip, lock, start};
use crate::Session;
use crate::media_test_util::{make_movie, session_with};

const TIMEOUT: Duration = Duration::from_secs(10);

fn empty() -> Matte {
    Matte { width: 2, height: 2, alpha: vec![0; 4] }
}

fn empty_segmenter() -> Result<Segmenter, String> {
    Ok(Box::new(|_| Ok(empty())))
}

struct Gate {
    stop_after: usize,
    entered: Sender<()>,
    release: Receiver<()>,
}

static NEXT_GATE: Mutex<Option<Gate>> = Mutex::new(None);

fn prepare_gate(stop_after: usize) -> (Receiver<()>, Sender<()>) {
    let (entered, waiting) = mpsc::channel();
    let (release, released) = mpsc::channel();
    *lock(&NEXT_GATE) = Some(Gate { stop_after, entered, release: released });
    (waiting, release)
}

fn gated_segmenter() -> Result<Segmenter, String> {
    let Gate { stop_after, entered, release } = lock(&NEXT_GATE).take().ok_or("no test gate")?;
    let mut frames = 0;
    Ok(Box::new(move |_| {
        frames += 1;
        if frames == stop_after {
            entered.send(()).map_err(|e| e.to_string())?;
            release.recv_timeout(TIMEOUT).map_err(|e| e.to_string())?;
        }
        Ok(empty())
    }))
}

fn wait_for_job(session: &mut Session, id: &Value) -> Value {
    let deadline = Instant::now() + TIMEOUT;
    loop {
        let jobs = session.execute("jobs.list", json!({})).unwrap();
        let job = jobs.as_array().unwrap().iter().find(|job| job["id"] == *id).unwrap();
        if job["finished"] == true {
            return job.clone();
        }
        assert!(Instant::now() < deadline, "background removal did not finish: {job}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn matte_for(session: &Session, seq: ItemId, clip: ClipId) -> Option<&PersonMatteRef> {
    session.project.sequence(seq).unwrap().find_item(clip).unwrap().1.person_matte.as_ref()
}

#[test]
fn two_originals_analyse_concurrently_and_cancel_independently() {
    static REJECTED_FACTORIES: AtomicUsize = AtomicUsize::new(0);
    fn unexpected_segmenter() -> Result<Segmenter, String> {
        REJECTED_FACTORIES.fetch_add(1, Ordering::Relaxed);
        Err("this request must not create an inference worker".into())
    }

    let dir = tempfile::tempdir().unwrap();
    let a = dir.path().join("a.mov");
    let b = dir.path().join("b.mov");
    let c = dir.path().join("c.mov");
    make_movie(&a, DemoScene::CityNight, 64, 48, 6);
    make_movie(&b, DemoScene::Dunes, 64, 48, 6);
    make_movie(&c, DemoScene::OceanSunset, 64, 48, 6);
    let (mut session, sources, seq) = session_with(&[&a, &b, &c]);
    let items = &session.project.sequence(seq).unwrap().video_tracks[0].items;
    let (a, b, c) = (items[0].id, items[1].id, items[2].id);
    let cache = dir.path().join("mattes");
    session.media.set_matte_dir(Some(cache.clone()));
    let place_b = |session: &mut Session, frame: i64, source_in: i64, frames: i64| {
        let placed = session
            .execute(
                "timeline.place",
                json!({"item": sources[1].0, "track": "V1", "frame": frame, "sourceIn": {"frame": source_in}, "duration": {"frame": frames}, "audio": false}),
            )
            .unwrap();
        ClipId(placed["clips"][0].as_u64().unwrap())
    };
    let same_cut = place_b(&mut session, 20, 0, 6);
    let other_cut = place_b(&mut session, 30, 1, 2);

    session.execute("timeline.trim", json!({"clip": a.0, "edge": "out", "deltaFrames": -2, "linked": false})).unwrap();
    start(&mut session, &json!({"clip": a.0, "wait": true}), empty_segmenter, cache.clone()).unwrap();
    let old_matte = matte_for(&session, seq, a).unwrap().clone();
    let old_path = cache.join(format!("{}.matte", old_matte.cache_id));
    let old_bytes = std::fs::read(&old_path).unwrap();
    assert_eq!(MatteReader::open(&old_path).unwrap().frame_count(), 4);
    session.execute("timeline.trim", json!({"clip": a.0, "edge": "out", "deltaFrames": 2, "linked": false})).unwrap();

    // A has written a staged frame beyond its old cache by the time B starts.
    let (a_entered, a_release) = prepare_gate(2);
    let a_job = start(&mut session, &json!({"clip": a.0}), gated_segmenter, cache.clone()).unwrap()["job"].clone();
    a_entered.recv_timeout(TIMEOUT).unwrap();
    let (b_entered, b_release) = prepare_gate(1);
    let b_job = start(&mut session, &json!({"clip": b.0}), gated_segmenter, cache.clone()).unwrap()["job"].clone();
    b_entered.recv_timeout(TIMEOUT).unwrap();
    assert_ne!(a_job, b_job);
    assert_eq!(session.background_jobs.len(), 2);

    let reused = start(&mut session, &json!({"clip": same_cut.0}), unexpected_segmenter, cache.clone()).unwrap();
    assert_eq!(reused["job"], b_job);
    let conflict = start(&mut session, &json!({"clip": other_cut.0}), unexpected_segmenter, cache.clone()).unwrap_err();
    assert!(conflict.to_string().contains("different clip range"), "{conflict}");
    let listed = session.jobs.len();
    let limit = start(&mut session, &json!({"clip": c.0}), unexpected_segmenter, cache.clone()).unwrap_err().to_string();
    assert!(limit.to_lowercase().contains("two") || limit.contains('2'), "{limit}");
    assert_eq!(session.jobs.len(), listed);
    assert_eq!(REJECTED_FACTORIES.load(Ordering::Relaxed), 0);

    let revision = session.revision;
    session.execute("jobs.cancel", json!({"job": a_job})).unwrap();
    a_release.send(()).unwrap();
    let cancelled = wait_for_job(&mut session, &a_job);
    assert!(cancelled["result"]["error"].as_str().is_some_and(|message| message.to_lowercase().contains("cancelled")), "{cancelled}");
    assert_eq!(session.revision, revision);
    assert_eq!(matte_for(&session, seq, a), Some(&old_matte));
    assert_eq!(std::fs::read(&old_path).unwrap(), old_bytes);
    assert!(job_for_clip(&session, seq, b).is_some());
    assert!(matte_for(&session, seq, b).is_none());

    b_release.send(()).unwrap();
    let completed = wait_for_job(&mut session, &b_job);
    assert_eq!(completed["result"]["frames"], 6, "{completed}");
    let b_matte = matte_for(&session, seq, b).unwrap();
    assert_eq!(matte_for(&session, seq, same_cut), Some(b_matte));
    assert!(matte_for(&session, seq, other_cut).is_none());
    assert!(matte_for(&session, seq, c).is_none());
    let b_path = cache.join(format!("{}.matte", b_matte.cache_id));
    assert_eq!(MatteReader::open(&b_path).unwrap().frame_count(), 6);
    assert_eq!(std::fs::read(old_path).unwrap(), old_bytes);
    assert_eq!(std::fs::read_dir(cache).unwrap().count(), 2);
    assert!(session.background_jobs.is_empty());
}
