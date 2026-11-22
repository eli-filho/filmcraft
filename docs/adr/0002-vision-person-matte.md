# ADR 0002: Apple Vision FFI for person mattes

- **Status:** proposed for the local Remove Background implementation

FilmCraft needs an on-device person matte for ordinary video footage. Apple Vision can segment a person without a point or box prompt. Its Balanced quality targets video analysis and leaves the result in a grayscale pixel buffer. A job analyses the original pictures in presentation order, stores the grayscale planes locally, and leaves the movie and its audio untouched.

For MP4 and MOV, analyse the selected clip's source interval, including transition handles and the pictures the renderer selects at its boundaries. A coarse container clock can require one neighboring picture just before the Out point. Frame holds need their held picture. Matroska and temporal effects whose sampling envelope is not known retain full-source coverage. Keep analysis in the existing background worker: changing processes would not reduce the Vision work, and trim-aware scheduling needs no helper executable or intermediate movie.

The version-one cache remains indexed by original presentation keys and source identity. Reuse covered keys; publish an expanded union atomically under a writer lock, without rewriting a complete cache on a hit. Existing readers keep their opened file snapshot during replacement. Before attaching a result, check coverage against the clip's current timing again. Missing keys show the original picture with an actionable warning and stop export instead of borrowing a neighboring matte.

On Unix, a device number can change when the same volume is remounted. New cache names use zero in that field. Existing references and cache files keep their names. A legacy name can be reused when its fingerprint, file size, modification time, inode, and change time match the current source. Writers lock the canonical identity and the current device's legacy name to coordinate with the previous build.

Allow at most two analysis jobs per session for different originals. Each worker keeps its own Vision sequence and visits that source's pictures in order; matching requests share the pending job, while a different range of the same original waits. An exclusive lock per source protects cache expansion. A shared lock at the previous directory-wide lock path also coordinates with older builds that acquire it exclusively. Keep all lock files in place so replacing an inode cannot bypass a holder. Cancellation affects only its job and frees its slot after the worker returns.

Keep the synchronous Vision and CoreVideo calls in the macOS-only FFI module in `crates/platform`. This is video media processing. The codec rules in [ADR 0001](0001-platform-ffi.md) still apply to codec integration. The same containment requirements apply here. The public segmentation API is safe, checks buffer sizes, returns errors, and exposes Rust-owned pixels. Each job constructs its own Vision request and handler on its worker thread. No Rust callback crosses the Vision boundary.

Use the existing MIT-licensed `objc2` and `objc2-foundation` dependencies for the request selectors, and the existing CoreVideo bindings for pixel buffers. Resolve the segmentation class at runtime so the app can start on older macOS versions. The command returns an availability error when the host has no backend. No model files, model runtime, or media transfer are needed.

EdgeTAM's full video predictor propagates a user prompt. Its published Core ML example applies prompts per frame. That suits a different interaction than automatic person isolation and does not establish full temporal tracking performance on Intel Macs.
