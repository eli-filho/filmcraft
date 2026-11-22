use std::fs::{File, TryLockError};
use std::path::Path;

use super::lock_cache;

#[test]
fn different_sources_can_write_concurrently_and_the_same_source_cannot() {
    let temp = tempfile::tempdir().unwrap();
    let cache = temp.path().join("mattes");
    std::fs::create_dir(&cache).unwrap();

    let first = lock_cache(&cache, "first-source").expect("lock the first source");
    let second = lock_cache(&cache, "second-source").expect("lock a different source in the same cache");
    assert!(lock_cache(&cache, "first-source").is_err());
    assert!(lock_cache(&cache, "second-source").is_err());

    drop(first);
    let reopened = lock_cache(&cache, "first-source").expect("lock the first source while the other stays held");
    assert!(lock_cache(&cache, "second-source").is_err());
    drop((reopened, second));
    let _released = lock_cache(&cache, "second-source").expect("lock the second source after release");
}

fn open_legacy_lock(cache: &Path) -> File {
    File::options().read(true).write(true).create(true).truncate(false).open(cache.with_extension("lock")).unwrap()
}

#[test]
fn legacy_exclusive_lock_and_current_shared_locks_exclude_each_other() {
    let temp = tempfile::tempdir().unwrap();
    let cache = temp.path().join("mattes");
    std::fs::create_dir(&cache).unwrap();

    let legacy = open_legacy_lock(&cache);
    legacy.try_lock().expect("take the previous build's exclusive lock");
    assert!(lock_cache(&cache, "first-source").is_err());
    drop(legacy);

    let current = lock_cache(&cache, "first-source").expect("take the current shared lock after release");
    let legacy = open_legacy_lock(&cache);
    assert!(matches!(legacy.try_lock(), Err(TryLockError::WouldBlock)));
    drop(current);
    legacy.try_lock().expect("the previous build can proceed once the shared lock is released");
}
