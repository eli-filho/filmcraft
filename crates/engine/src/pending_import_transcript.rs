//! Automatic speech work runs independently of media acceptance and placement timing.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, mpsc};

use filmcraft_media::SharedSource;
use filmcraft_project::{ItemId, MediaRef, Transcript};

use crate::Session;

pub(crate) type Outcome = std::result::Result<Arc<Transcript>, String>;

pub(crate) fn spawn(session: &Session, item: ItemId, path: String, source: SharedSource, cancel: Arc<AtomicBool>) -> std::io::Result<mpsc::Receiver<Outcome>> {
    let project = session.project.clone();
    let services = session.services.clone();
    let prefs = session.prefs.clone();
    let transcriber = session.transcriber.clone();
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new().name("filmcraft-import-transcript".into()).spawn(move || {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if cancel.load(Ordering::Relaxed) {
                return Err("Automatic transcription cancelled".into());
            }
            let mut detached = Session::new(services);
            detached.project = project;
            detached.prefs = prefs;
            detached.transcriber = transcriber;
            let project = Arc::make_mut(&mut detached.project);
            let media = project.item_mut(item).and_then(|item| item.as_media_mut()).ok_or("The transcription source is unavailable")?;
            media.info = source.info().clone();
            media.media = MediaRef::File { path: path.clone() };
            project.pending_imports.clear();
            detached.media.allow(&path);
            detached.media.insert_file(item, &path, source);
            detached.execute("transcript.generate", serde_json::json!({"items": [item.0]})).map_err(|error| error.to_string())?;
            if cancel.load(Ordering::Relaxed) {
                return Err("Automatic transcription cancelled".into());
            }
            detached.project.transcripts.get(&item).cloned().ok_or_else(|| "No transcript was generated".into())
        }))
        .unwrap_or_else(|_| Err("Automatic transcription failed internally; the editable clip was kept".into()));
        let _ = tx.send(result);
    })?;
    Ok(rx)
}
