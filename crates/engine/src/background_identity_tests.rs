use std::fs::File;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};

use filmcraft_media::DemoScene;
use filmcraft_media::person_matte::MatteWriter;
use filmcraft_project::PersonMatteRef;
use serde_json::json;

use super::{
    CACHE_PREFIX, DEVICE_END, DEVICE_START, cache_id, cache_id_with_device, lock_cache, lock_previous_source, lock_source, open_matte, start, valid_source,
};
use crate::media_test_util::{make_movie, session_with};

fn matte_file(cache: &Path, reference: &PersonMatteRef, keys: &[i64]) -> PathBuf {
    let path = cache.join(format!("{}.matte", reference.cache_id));
    let mut writer = MatteWriter::new(File::create(&path).unwrap(), 1, 1, keys.len() as u64).unwrap();
    for &key in keys {
        writer.push(key, &[0]).unwrap();
    }
    writer.finish().unwrap();
    path
}

#[test]
fn legacy_device_id_opens_and_a_fully_covered_cache_reattaches_without_inference() {
    let temp = tempfile::tempdir().unwrap();
    let source_path = temp.path().join("source.mov");
    make_movie(&source_path, DemoScene::CityNight, 64, 48, 4);
    let file = source_path.to_str().unwrap();
    let (mut session, _, seq) = session_with(&[&source_path]);
    let clip = session.project.sequence(seq).unwrap().video_tracks[0].items[0].id;
    let source = crate::relink::identity_of(&*session.services, file).unwrap();
    let keys = session.media.probe_file(file, &*session.services).unwrap().matte_frame_keys(4).unwrap();
    let canonical = PersonMatteRef { cache_id: cache_id(&source_path, source).unwrap(), source, bypassed: false };
    let current_device = std::fs::metadata(&source_path).unwrap().dev();
    let other_device = if current_device == u64::MAX { u64::MAX - 1 } else { u64::MAX };
    let legacy = PersonMatteRef { cache_id: cache_id_with_device(&canonical.cache_id, other_device).unwrap(), ..canonical.clone() };
    let cache = temp.path().join("mattes");
    std::fs::create_dir(&cache).unwrap();
    session.media.set_matte_dir(Some(cache.clone()));
    // A newer cache may cover less than a legacy full-source cache after a remount.
    let stable_path = matte_file(&cache, &canonical, &keys[..1]);
    let legacy_path = matte_file(&cache, &legacy, &keys);
    let before = std::fs::metadata(&legacy_path).unwrap();
    assert!(valid_source(file, &legacy));
    assert_eq!(open_matte(&cache, file, &canonical).unwrap().keys().unwrap(), keys[..1]);
    assert_eq!(open_matte(&cache, file, &legacy).unwrap().keys().unwrap(), keys);

    start(&mut session, &json!({"clip": clip.0, "wait": true}), || Err("legacy cache hit must skip inference".into()), cache.clone()).unwrap();
    let attached = &session.project.sequence(seq).unwrap().find_item(clip).unwrap().1.person_matte;
    assert_eq!(attached.as_ref(), Some(&legacy));
    assert!(session.render_program(1.0).unwrap().px.iter().all(|&value| value == 0.0));
    let after = std::fs::metadata(&legacy_path).unwrap();
    assert_eq!((after.ino(), after.modified().unwrap()), (before.ino(), before.modified().unwrap()));
    assert_eq!(open_matte(&cache, file, &canonical).unwrap().frame_count(), 1);
    assert!(stable_path.exists());
    assert_eq!(std::fs::read_dir(cache).unwrap().count(), 2);
}

#[test]
fn only_the_device_field_can_differ() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("source.mov");
    std::fs::write(&path, b"unchanged source").unwrap();
    let source = crate::relink::identity_of_bytes(b"unchanged source");
    let canonical = PersonMatteRef { cache_id: cache_id(&path, source).unwrap(), source, bypassed: false };
    let file = path.to_str().unwrap();
    let mut stored = canonical.clone();
    stored.cache_id = cache_id_with_device(&canonical.cache_id, u64::MAX).unwrap();
    assert!(valid_source(file, &stored));

    // Fingerprint, size, mtime, inode, ctime seconds and ctime nanoseconds must still match.
    for at in [CACHE_PREFIX.len() + 1, CACHE_PREFIX.len() + 18, DEVICE_START - 33, DEVICE_END + 1, DEVICE_END + 18, DEVICE_END + 35] {
        let mut bytes = stored.cache_id.as_bytes().to_vec();
        bytes[at] = if bytes[at] == b'0' { b'1' } else { b'0' };
        let damaged = PersonMatteRef { cache_id: String::from_utf8(bytes).unwrap(), ..stored.clone() };
        assert!(!valid_source(file, &damaged), "accepted a non-device mismatch at byte {at}");
    }
    for device in ["000000000000000/", "00000000000000é", "fffffffffffffff", "000000000000000g"] {
        let mut malformed = stored.clone();
        malformed.cache_id.replace_range(DEVICE_START..DEVICE_END, device);
        assert!(!valid_source(file, &malformed), "accepted malformed device field {device}");
        assert!(open_matte(temp.path(), file, &malformed).is_none());
    }
}

#[test]
fn canonical_writer_also_excludes_a_previous_parallel_writer() {
    let temp = tempfile::tempdir().unwrap();
    let source = temp.path().join("source.mov");
    std::fs::write(&source, b"source").unwrap();
    let metadata = std::fs::metadata(&source).unwrap();
    let canonical = cache_id(&source, crate::relink::identity_of_bytes(b"source")).unwrap();
    if metadata.dev() == 0 {
        return; // The previous name is already canonical on this filesystem.
    }
    let previous = cache_id_with_device(&canonical, metadata.dev()).unwrap();
    let cache = temp.path().join("mattes");
    let old = lock_cache(&cache, &previous).unwrap();
    let current = lock_cache(&cache, &canonical).unwrap();
    assert!(lock_previous_source(&cache, &canonical, &metadata).is_err());
    drop(old);
    let previous_lock = lock_previous_source(&cache, &canonical, &metadata).unwrap().unwrap();
    assert!(lock_source(&cache, &previous).is_err());
    drop((previous_lock, current));
}
