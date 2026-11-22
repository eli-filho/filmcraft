//! Trim coverage through the same edit, preview, and export paths used by the editor.

use std::path::PathBuf;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::time::{Duration, Instant};

use filmcraft_media::person_matte::MatteReader;
use filmcraft_media::{DemoScene, SharedSource};
use filmcraft_project::{ClipId, ItemId, TimeInterpolation, TrackItem};
use filmcraft_time::{Tick, TimeRange};
use serde_json::{Value, json};

use super::{Coverage, Matte, RESTORE, Segmenter, lock, start};
use crate::media_test_util::{make_movie, session_with};
use crate::{Event, Session};

fn fixture() -> (tempfile::TempDir, Session, ItemId, ItemId, ClipId, PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("source.mov");
    make_movie(&source, DemoScene::CityNight, 64, 48, 12);
    let (mut session, sources, seq) = session_with(&[&source]);
    let clip = session.project.sequence(seq).unwrap().video_tracks[0].items[0].id;
    trim(&mut session, clip, "in", 3);
    trim(&mut session, clip, "out", -5);
    let cache = dir.path().join("mattes");
    session.media.set_matte_dir(Some(cache.clone()));
    (dir, session, seq, sources[0], clip, cache)
}

fn trim(session: &mut Session, clip: ClipId, edge: &str, delta: i64) {
    session.execute("timeline.trim", json!({"clip": clip.0, "edge": edge, "deltaFrames": delta, "linked": false})).unwrap();
}

fn item(session: &Session, seq: ItemId, clip: ClipId) -> &TrackItem {
    session.project.sequence(seq).unwrap().find_item(clip).unwrap().1
}

fn original(session: &Session, dir: &tempfile::TempDir) -> SharedSource {
    session.media.probe_file(dir.path().join("source.mov").to_str().unwrap(), &*session.services).unwrap()
}

fn empty() -> Matte {
    Matte { width: 2, height: 2, alpha: vec![0; 4] }
}

fn empty_segmenter() -> Result<Segmenter, String> {
    Ok(Box::new(|_| Ok(empty())))
}

fn wait_for_job(session: &mut Session, id: &Value) -> Value {
    let deadline = Instant::now() + Duration::from_secs(10);
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

#[test]
fn only_the_trim_and_missing_expansion_pictures_run_inference() {
    static CALLS: AtomicUsize = AtomicUsize::new(0);
    fn counted() -> Result<Segmenter, String> {
        Ok(Box::new(|_| {
            CALLS.fetch_add(1, Ordering::Relaxed);
            Ok(empty())
        }))
    }

    let (dir, mut session, seq, _, clip, cache) = fixture();
    let rate = session.sequence_rate();
    let run = json!({"clip": clip.0, "wait": true});
    assert_eq!(item(&session, seq, clip).source_in, rate.tick_of(3));
    assert_eq!(item(&session, seq, clip).source_out(), rate.tick_of(7));
    start(&mut session, &run, counted, cache.clone()).unwrap();
    assert_eq!(CALLS.load(Ordering::Relaxed), 4, "only [3, 7) needs mattes with this source's frame-aligned lookup");
    let original = item(&session, seq, clip).person_matte.clone().unwrap();
    let matte_path = cache.join(format!("{}.matte", original.cache_id));
    assert_eq!(MatteReader::open(&matte_path).unwrap().frame_count(), 4);

    trim(&mut session, clip, "out", -2);
    session.execute(RESTORE, json!({"clip": clip.0})).unwrap();
    let no_segmenter = || Err("a narrower trim should reuse existing coverage".into());
    start(&mut session, &run, no_segmenter, cache.clone()).unwrap();
    assert_eq!(item(&session, seq, clip).person_matte.as_ref(), Some(&original));

    trim(&mut session, clip, "in", -1);
    trim(&mut session, clip, "out", 4);
    assert_eq!(item(&session, seq, clip).source_in, rate.tick_of(2));
    assert_eq!(item(&session, seq, clip).source_out(), rate.tick_of(9));
    assert!(session.render_program_at(1.0, rate.tick_of(3)).unwrap().px.iter().all(|&channel| channel == 0.0));
    assert!(session.render_program_at(1.0, rate.tick_of(2)).unwrap().px.iter().skip(3).step_by(4).any(|&alpha| alpha > 0.5));
    session.poll_persistence();
    assert!(session.drain_events().iter().any(|event| matches!(event, Event::Toast { message, error: true } if message.contains("Remove Background"))));
    let rejected_export = dir.path().join("incomplete.mov");
    let error = session.execute("file.exportMedia", json!({"path": rejected_export, "format": "prores", "wait": true})).unwrap_err();
    assert!(error.to_string().contains("Remove Background"));
    assert!(!rejected_export.exists());

    start(&mut session, &run, counted, cache).unwrap();
    assert_eq!(CALLS.load(Ordering::Relaxed), 7, "only source pictures 2, 7 and 8 were missing");
    assert_eq!(item(&session, seq, clip).person_matte.as_ref(), Some(&original));
    assert_eq!(MatteReader::open(&matte_path).unwrap().frame_count(), 7);
    for frame in 2..9 {
        assert!(session.render_program_at(1.0, rate.tick_of(frame)).unwrap().px.iter().all(|&channel| channel == 0.0), "frame {frame}");
    }
    let output = dir.path().join("covered.mov");
    session.execute("file.exportMedia", json!({"path": output, "format": "prores", "wait": true})).unwrap();
    assert!(output.is_file());
}

#[test]
fn full_source_cache_already_covers_a_trimmed_clip_and_is_not_rewritten() {
    let (_dir, mut session, seq, _, clip, cache) = fixture();
    let run = json!({"clip": clip.0, "wait": true});
    trim(&mut session, clip, "in", -3);
    trim(&mut session, clip, "out", 5);
    start(&mut session, &run, empty_segmenter, cache.clone()).unwrap();
    let reference = item(&session, seq, clip).person_matte.clone().unwrap();
    let path = cache.join(format!("{}.matte", reference.cache_id));
    let before = std::fs::read(&path).unwrap();
    let modified = std::fs::metadata(&path).unwrap().modified().unwrap();
    assert_eq!(MatteReader::open(&path).unwrap().frame_count(), 12);

    trim(&mut session, clip, "in", 3);
    trim(&mut session, clip, "out", -5);
    session.execute(RESTORE, json!({"clip": clip.0})).unwrap();
    start(&mut session, &run, || Err("full-source cache already covers this trim".into()), cache).unwrap();
    assert_eq!(item(&session, seq, clip).person_matte.as_ref(), Some(&reference));
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert_eq!(std::fs::metadata(&path).unwrap().modified().unwrap(), modified);
}

#[test]
fn coverage_follows_retimed_and_held_pictures_in_the_original() {
    let (dir, session, seq, _, clip, _) = fixture();
    let src = original(&session, &dir);
    let keys = src.matte_frame_keys(12).unwrap();
    let rate = session.sequence_rate();
    let mut retimed = item(&session, seq, clip).clone();
    retimed.reverse = true;
    assert_eq!(Coverage::new(&session, seq, &retimed).unwrap().keys(&src).unwrap(), keys[3..7]);

    retimed.reverse = false;
    retimed.speed = 0.5;
    retimed.duration = rate.tick_of(8);
    for interpolation in [TimeInterpolation::FrameBlending, TimeInterpolation::OpticalFlow] {
        retimed.time_interpolation = interpolation;
        let (next, weight) = filmcraft_render::interpolation_blend(&retimed, retimed.start + rate.tick_of(1), rate, rate).unwrap();
        assert_eq!(next, rate.tick_of(4));
        assert_eq!(weight, 0.5);
        assert_eq!(Coverage::new(&session, seq, &retimed).unwrap().keys(&src).unwrap(), keys[3..8]);
    }

    // The held picture need not lie in the moving clip's ordinary [3, 7) source interval.
    retimed.frame_hold = Some(rate.tick_of(9));
    assert_eq!(Coverage::new(&session, seq, &retimed).unwrap().keys(&src).unwrap(), [keys[9]]);
}

#[test]
fn subclip_uses_absolute_parent_keys_and_transition_handles_extend_both_ranges() {
    let (dir, mut session, seq, source, first, _) = fixture();
    let src = original(&session, &dir);
    let keys = src.matte_frame_keys(12).unwrap();
    let subclip = session.execute("clip.makeSubclip", json!({"item": source.0, "startFrame": 4, "endFrame": 8, "restrictTrims": false})).unwrap();
    let placed = session.execute("timeline.place", json!({"item": subclip["item"], "track": "V1", "frame": 7, "audio": false})).unwrap();
    let second = ClipId(placed["clips"][0].as_u64().unwrap());
    let coverage = |session: &Session, clip| Coverage::new(session, seq, item(session, seq, clip)).unwrap().keys(&src).unwrap();
    assert_eq!(coverage(&session, second), keys[4..8]);

    session.execute("effects.apply", json!({"effect": "cross_dissolve", "clip": second.0, "edge": "in", "frames": 4})).unwrap();
    let transition = &session.active_sequence().unwrap().video_tracks[0].transitions[0];
    assert_eq!((transition.from, transition.to), (Some(first), Some(second)));
    assert_eq!(coverage(&session, first), keys[3..9]);
    assert_eq!(coverage(&session, second), keys[2..8]);
}

#[test]
fn off_grid_render_can_need_the_picture_at_the_clips_source_out() {
    use filmcraft_isobmff::{Brand, Mp4Writer, SampleEntry, TrackConfig, WriteSample, WriterOptions};
    use std::io::Cursor;

    let (_dir, mut session, child, _, clip, _) = fixture();
    let rate = session.sequence_rate();
    let mut jpeg = Vec::new();
    image::codecs::jpeg::JpegEncoder::new(&mut jpeg).encode_image(&image::GrayImage::new(64, 48)).unwrap();
    let mut movie = Mp4Writer::new(Cursor::new(Vec::new()), WriterOptions::new(Brand::Mov)).unwrap();
    let track = movie.add_track(TrackConfig::new(SampleEntry::jpeg(64, 48), 1_000)).unwrap();
    let milliseconds = |frame| rate.tick_of(frame).to_rational_round(1, 1_000);
    for frame in 0..12 {
        let duration = u32::try_from(milliseconds(frame + 1) - milliseconds(frame)).unwrap();
        movie.write_sample(track, WriteSample { data: &jpeg, duration, composition_offset: 0, is_sync: true }).unwrap();
    }
    let source = session.media.open_bytes("rounded.mov", movie.finish().unwrap().into_inner().into()).unwrap();
    let keys = source.matte_frame_keys(12).unwrap();
    let required = |session: &Session| Coverage::new(session, child, item(session, child, clip)).unwrap().keys(&source).unwrap();
    let original = item(&session, child, clip);
    let subframe = filmcraft_render::video_source_time(original, original.end() - Tick(1), source.info().frame_rate());
    let raw = TimeRange::from_bounds(original.source_in, original.source_out());
    assert_eq!(source.matte_frame_keys_in_range(raw, 12).unwrap(), keys[3..7]);
    assert_eq!(source.matte_frame_key(subframe), Some(keys[7]));
    assert_eq!(required(&session), keys[3..8], "off-grid sampling needs the extra picture");

    session.execute("file.newSequence", json!({"name": "Parent", "fps": 24, "width": 64, "height": 48})).unwrap();
    session.execute("timeline.place", json!({"item": child.0, "track": "V1", "frame": 0, "audio": false})).unwrap();
    assert_eq!(required(&session), keys[3..8], "the same coverage works when the sequence is nested");
}

type Gate = (Sender<()>, Receiver<()>);

fn gated(gate: &Mutex<Option<Gate>>, stop_after: usize) -> Result<Segmenter, String> {
    let (entered, release) = lock(gate).take().ok_or("no test gate")?;
    let mut count = 0;
    Ok(Box::new(move |_| {
        count += 1;
        if count == stop_after {
            entered.send(()).map_err(|error| error.to_string())?;
            release.recv_timeout(Duration::from_secs(10)).map_err(|error| error.to_string())?;
        }
        Ok(empty())
    }))
}

#[test]
fn pending_range_is_neither_shared_with_a_different_cut_nor_applied_after_expansion() {
    static GATE: Mutex<Option<Gate>> = Mutex::new(None);
    fn pause_on_first_picture() -> Result<Segmenter, String> {
        gated(&GATE, 1)
    }

    let (_dir, mut session, seq, source, clip, cache) = fixture();
    let placed = session
        .execute("timeline.place", json!({"item": source.0, "track": "V1", "frame": 9, "sourceIn": {"frame": 8}, "duration": {"frame": 2}, "audio": false}))
        .unwrap();
    let other = ClipId(placed["clips"][0].as_u64().unwrap());
    let (entered, waiting) = mpsc::channel();
    let (release, released) = mpsc::channel();
    *lock(&GATE) = Some((entered, released));
    let job = start(&mut session, &json!({"clip": clip.0}), pause_on_first_picture, cache.clone()).unwrap();
    waiting.recv_timeout(Duration::from_secs(10)).unwrap();
    let second = start(&mut session, &json!({"clip": other.0}), empty_segmenter, cache);
    trim(&mut session, clip, "out", 1);
    release.send(()).unwrap();
    assert!(second.is_err(), "a job for a different range cannot share the first job");
    let result = wait_for_job(&mut session, &job["job"]);
    assert!(result["result"]["error"].as_str().is_some_and(|message| message.contains("Remove Background")), "{result}");
    assert!(item(&session, seq, clip).person_matte.is_none());
    assert!(item(&session, seq, other).person_matte.is_none());
}

#[test]
fn cancelling_expansion_keeps_the_prior_partial_cache_byte_for_byte() {
    static GATE: Mutex<Option<Gate>> = Mutex::new(None);
    fn pause_on_second_picture() -> Result<Segmenter, String> {
        gated(&GATE, 2)
    }

    let (_dir, mut session, seq, _, clip, cache) = fixture();
    start(&mut session, &json!({"clip": clip.0, "wait": true}), empty_segmenter, cache.clone()).unwrap();
    let reference = item(&session, seq, clip).person_matte.clone().unwrap();
    let matte_path = cache.join(format!("{}.matte", reference.cache_id));
    let before = std::fs::read(&matte_path).unwrap();
    trim(&mut session, clip, "out", 2);
    let revision = session.revision;
    let (entered, waiting) = mpsc::channel();
    let (release, released) = mpsc::channel();
    *lock(&GATE) = Some((entered, released));
    let job = start(&mut session, &json!({"clip": clip.0}), pause_on_second_picture, cache.clone()).unwrap();
    waiting.recv_timeout(Duration::from_secs(10)).unwrap();
    let cancellation = session.execute("jobs.cancel", json!({"job": job["job"]}));
    release.send(()).unwrap();
    cancellation.unwrap();
    let result = wait_for_job(&mut session, &job["job"]);
    assert!(result["result"]["error"].as_str().is_some_and(|message| message.to_lowercase().contains("cancelled")), "{result}");
    assert_eq!(session.revision, revision);
    assert_eq!(item(&session, seq, clip).person_matte.as_ref(), Some(&reference));
    assert_eq!(std::fs::read(&matte_path).unwrap(), before);
    assert_eq!(std::fs::read_dir(cache).unwrap().count(), 1, "cancelled staging file was removed");
}
