//! Reconcile provisional stream guesses without replaying or replacing the editor's work.

use std::collections::BTreeSet;

use filmcraft_media::MediaInfo;
use filmcraft_project::{ClipId, ItemId, ItemKind, ParamValue, Project, TrackItem, TrackKind, resolve_auto_points};
use filmcraft_time::Tick;

/// Call before swapping the media info or reconciling durations. Added stream instances retain
/// the provisional duration and join the placement anchors for the ordinary duration pass.
pub(crate) fn reconcile_streams_and_geometry(project: &mut Project, id: ItemId, old_info: &MediaInfo, new_info: &MediaInfo, scaling: &str) -> Vec<String> {
    let mut warnings = Vec::new();
    let Some(pending) = project.pending_imports.get(&id).cloned() else { return warnings };
    let roots: BTreeSet<_> = project.items.keys().copied().filter(|&item| project.resolve_media(item).is_some_and(|(root, _, _)| root == id)).collect();
    let sequences: Vec<_> = project.sequences().map(|item| item.id).collect();
    let old_size = old_info.video.as_ref().map(|video| (video.width, video.height)).unwrap_or((1920, 1080));
    let extra_streams = new_info.audio_streams.len().min(filmcraft_media::MAX_AUDIO_STREAMS).saturating_sub(old_info.audio_streams.len());
    for sequence in sequences {
        let Some(q) = project.sequence(sequence) else { continue };
        let rate = q.settings.frame_rate;
        let frame = (q.settings.width, q.settings.height);
        let full_duration = rate.snap_nearest(pending.provisional_duration).max(rate.frame_duration());
        let mut remove = BTreeSet::new();
        let mut expand = Vec::new();
        if q.all_tracks().flat_map(|track| &track.items).any(|clip| roots.contains(&clip.item)) {
            if new_info.has_audio() && !old_info.has_audio() {
                warnings.push(format!("{} also contains audio; place the source again to choose its audio destination", new_info.name));
            }
            if new_info.has_video() && !old_info.has_video() {
                warnings.push(format!("{} also contains video; place the source again to choose its video destination", new_info.name));
            }
        }
        for track in q.all_tracks() {
            for clip in track.items.iter().filter(|clip| clip.item == id) {
                let pristine = pending.placements.contains_key(&clip.id)
                    && clip.duration == full_duration
                    && untouched_content(project, clip, track.kind, frame, old_size, scaling);
                let aligned_pair = !(old_info.has_video() && old_info.has_audio())
                    || clip.link.is_some_and(|link| {
                        q.all_tracks().filter(|other| other.kind != track.kind).flat_map(|other| &other.items).any(|other| {
                            other.item == id
                                && other.link == Some(link)
                                && other.start == clip.start
                                && other.source_in == clip.source_in
                                && other.duration == clip.duration
                        })
                    });
                let absent = match track.kind {
                    TrackKind::Video => old_info.has_video() && !new_info.has_video(),
                    TrackKind::Audio => old_info.has_audio() && !new_info.has_audio(),
                };
                if absent {
                    if pristine && aligned_pair && !track.locked && !track.transitions.iter().any(|transition| transition.range().overlaps(&clip.range())) {
                        remove.insert(clip.id);
                    } else {
                        warnings.push(format!("{} has no {} stream; its edited placeholder was kept", new_info.name, stream_name(track.kind)));
                    }
                } else if extra_streams > 0 && track.kind == TrackKind::Audio && clip.audio_stream == 0 {
                    // An unlinked or independently edited A/V pair is ambiguous: never infer
                    // new placements from its video or recreate a deleted audio instance.
                    if pristine
                        && aligned_pair
                        && !track.locked
                        && !track
                            .transitions
                            .iter()
                            .any(|transition| transition.from == Some(clip.id) || transition.to == Some(clip.id) || transition.range().overlaps(&clip.range()))
                        && old_info.audio_streams.len() == 1
                    {
                        expand.push((track.id, clip.clone()));
                    } else {
                        warnings.push(format!(
                            "{} has additional audio streams; existing edits were kept. Place the source again to add those streams",
                            new_info.name
                        ));
                    }
                }
            }
        }
        if !remove.is_empty()
            && let Some(q) = project.sequence_mut(sequence)
        {
            for track in q.all_tracks_mut() {
                track.items.retain(|clip| !remove.contains(&clip.id));
            }
        }
        for (track, clip) in expand {
            if let Err(error) = add_audio_streams(project, sequence, track, clip, new_info.audio_streams.len()) {
                warnings.push(format!("{}: {error}; existing edits were kept. Place the source again to add its other audio streams", new_info.name));
            }
        }
    }
    if let Some(video) = &new_info.video {
        let source_size = (video.width, video.height);
        let source_par = project
            .item(id)
            .and_then(|item| item.as_media())
            .and_then(|media| media.interpret.par)
            .and_then(filmcraft_project::checked_par)
            .unwrap_or_else(|| filmcraft_project::sane_par(video.par));
        for item in project.items.values_mut() {
            let ItemKind::Sequence(sequence) = &mut item.kind else { continue };
            let sequence = std::sync::Arc::make_mut(sequence);
            let frame = (sequence.settings.width, sequence.settings.height);
            let shown = filmcraft_project::conformed_size(source_size, source_par, sequence.settings.par);
            for clip in sequence.video_tracks.iter_mut().flat_map(|track| &mut track.items).filter(|clip| roots.contains(&clip.item)) {
                for effect in &mut clip.effects {
                    resolve_auto_points(effect, frame, source_size);
                }
                // An explicit transform or animation made while loading belongs to the editor.
                // Scale-to-Frame was already set at placement and may since have been disabled.
                if scaling == "setToFrameSize" && default_scale(clip) {
                    crate::settings::apply_media_scaling(clip, scaling, frame, shown);
                }
            }
        }
    }
    warnings.sort();
    warnings.dedup();
    warnings
}

fn stream_name(kind: TrackKind) -> &'static str {
    match kind {
        TrackKind::Video => "video",
        TrackKind::Audio => "audio",
    }
}

fn default_scale(clip: &TrackItem) -> bool {
    !clip.scale_to_frame
        && clip.effects.iter().find(|effect| effect.effect == "motion").is_some_and(|motion| {
            ["scale", "scale_width"].iter().all(|name| motion.param(name).is_some_and(|param| !param.is_animated() && param.value.as_f64() == Some(100.0)))
                && motion.param("uniform_scale").is_some_and(|param| !param.is_animated() && param.value == ParamValue::Bool(true))
        })
}

fn untouched_content(project: &Project, clip: &TrackItem, kind: TrackKind, frame: (u32, u32), source: (u32, u32), scaling: &str) -> bool {
    let Some(item) = project.item(clip.item) else { return false };
    if clip.name != item.name
        || clip.label != item.label
        || clip.custom_label_color != item.custom_label_color
        || clip.source_in != Tick::ZERO
        || clip.speed != 1.0
        || clip.reverse
        || !clip.enabled
        || clip.group.is_some()
        || !clip.markers.is_empty()
        || clip.gain_db != 0.0
        || clip.frame_hold.is_some()
        || clip.scale_to_frame != (kind == TrackKind::Video && scaling == "scaleToFrameSize")
        || clip.essential.is_some()
        || clip.multicam.is_some()
        || !clip.time_interpolation.is_default()
        || clip.maintain_pitch
        || clip.hold_filters
        || clip.field_options.is_some()
        || !clip.source_channels.is_empty()
        || clip.audio_stream != 0
        || clip.graphic.is_some()
        || clip.person_matte.is_some()
    {
        return false;
    }
    let mut actual = clip.effects.clone();
    let mut expected = match kind {
        TrackKind::Video => filmcraft_project::effect::intrinsic_video(),
        TrackKind::Audio => filmcraft_project::effect::intrinsic_audio(),
    };
    // Compare auto defaults at the same size; NaN itself deliberately does not compare equal.
    for effect in actual.iter_mut().chain(&mut expected) {
        resolve_auto_points(effect, frame, source);
    }
    actual == expected
}

fn add_audio_streams(project: &mut Project, sequence: ItemId, track: filmcraft_project::TrackId, clip: TrackItem, streams: usize) -> Result<(), String> {
    let streams = streams.min(filmcraft_media::MAX_AUDIO_STREAMS);
    let Some(q) = project.sequence(sequence) else { return Err("the sequence is unavailable".into()) };
    let Some(first) = q.audio_tracks.iter().position(|candidate| candidate.id == track) else { return Err("the audio track is unavailable".into()) };
    for offset in 1..streams {
        if let Some(destination) = first.checked_add(offset).and_then(|index| q.audio_tracks.get(index))
            && (destination.locked
                || destination.items.iter().any(|other| other.range().overlaps(&clip.range()))
                || destination.transitions.iter().any(|transition| transition.range().overlaps(&clip.range())))
        {
            return Err("additional audio streams need free, unlocked destination tracks".into());
        }
    }
    let tracks = crate::commands::ensure_audio_tracks(project, sequence, first, streams, "Import audio streams").map_err(|error| error.to_string())?;
    let link = clip.link.unwrap_or_else(|| project.alloc_id());
    if let Some(original) = project.sequence_mut(sequence).and_then(|q| q.track_mut(track)).and_then(|track| track.items.iter_mut().find(|c| c.id == clip.id)) {
        original.link = Some(link);
    }
    for stream in 1..streams {
        let Some(&destination) = first.checked_add(stream).and_then(|index| tracks.get(index)) else { continue };
        let mut added = clip.clone();
        added.id = ClipId(project.alloc_id());
        added.audio_stream = stream;
        added.link = Some(link);
        let anchor = (added.id, added.start);
        if let Some(track) = project.sequence_mut(sequence).and_then(|q| q.track_mut(destination)) {
            track.items.push(added);
            track.sort();
            if let Some(pending) = project.pending_imports.get_mut(&clip.item) {
                pending.placements.insert(anchor.0, (sequence, destination, anchor.1));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Arc;

    use filmcraft_geom::Vec2;
    use filmcraft_media::{AudioStreamInfo, MediaKind, VideoStreamInfo};
    use filmcraft_project::{Keyframe, Label, MediaClip, MediaRef, Param, PendingImport, SequenceSettings};
    use filmcraft_time::{FrameRate, TICKS_PER_SECOND, TimeRange};

    use super::*;

    struct Fixture {
        project: Project,
        sequence: ItemId,
        source: ItemId,
        video: ClipId,
        audio: ClipId,
        old: MediaInfo,
    }

    impl Fixture {
        fn new() -> Self {
            let mut project = Project::new("Hydration tests");
            let sequence = project.new_sequence("Timeline", SequenceSettings { frame_rate: FrameRate::FPS_24, ..Default::default() }, 1, 1, None);
            let old = MediaInfo {
                name: "Portrait.mov".into(),
                kind: MediaKind::Movie,
                duration: Tick(5 * TICKS_PER_SECOND),
                video: Some(VideoStreamInfo {
                    width: 1920,
                    height: 1080,
                    frame_rate: FrameRate::FPS_24,
                    par: (1, 1),
                    codec: String::new(),
                    pixel_format: String::new(),
                    color: filmcraft_color::ColorInfo::REC709,
                    has_alpha: false,
                    bitrate: None,
                    hdr: None,
                    color_assumed: None,
                }),
                audio_streams: vec![AudioStreamInfo { sample_rate: 48000, channels: 2, codec: String::new(), bits_per_sample: None }],
                container: String::new(),
                start_timecode: None,
                file_size: None,
            };
            let source = project.add_item(
                &old.name,
                Label::Forest,
                ItemKind::Media(Box::new(MediaClip {
                    media: MediaRef::File { path: "pending.mov".into() },
                    info: old.clone(),
                    interpret: Default::default(),
                    mark_in: None,
                    mark_out: None,
                    markers: Vec::new(),
                    offline: false,
                    proxy: None,
                    identity: None,
                })),
                None,
            );
            project.pending_imports.insert(
                source,
                PendingImport {
                    ticket: 1,
                    batch: 1,
                    path: "pending.mov".into(),
                    provisional_duration: old.duration,
                    scaling: "none".into(),
                    placements: BTreeMap::new(),
                    metadata_ready: false,
                    error: None,
                },
            );
            let link = project.alloc_id();
            let mut clips = Vec::new();
            for kind in [TrackKind::Video, TrackKind::Audio] {
                let mut clip = project.make_track_item(source, kind, Tick::ZERO, TimeRange::new(Tick::ZERO, old.duration), FrameRate::FPS_24).unwrap();
                clip.link = Some(link);
                let id = clip.id;
                let track = project.sequence_mut(sequence).unwrap().tracks_mut(kind).first_mut().unwrap();
                let track_id = track.id;
                track.items.push(clip);
                project.pending_imports.get_mut(&source).unwrap().placements.insert(id, (sequence, track_id, Tick::ZERO));
                clips.push(id);
            }
            Self { project, sequence, source, video: clips[0], audio: clips[1], old }
        }

        fn clip_mut(&mut self, id: ClipId) -> &mut TrackItem {
            self.project.sequence_mut(self.sequence).unwrap().all_tracks_mut().flat_map(|track| &mut track.items).find(|clip| clip.id == id).unwrap()
        }

        fn clip(&self, id: ClipId) -> &TrackItem {
            self.project.sequence(self.sequence).unwrap().all_tracks().flat_map(|track| &track.items).find(|clip| clip.id == id).unwrap()
        }

        fn hydrate(&mut self, info: &MediaInfo, scaling: &str) -> Vec<String> {
            reconcile_streams_and_geometry(&mut self.project, self.source, &self.old, info, scaling)
        }

        fn portrait(&self) -> MediaInfo {
            let mut info = self.old.clone();
            let video = info.video.as_mut().unwrap();
            video.width = 1080;
            video.height = 1920;
            info
        }

        fn multistream(&self) -> MediaInfo {
            let mut info = self.old.clone();
            info.audio_streams = vec![info.audio_streams[0].clone(); 3];
            info
        }
    }

    #[test]
    fn pending_placement_defers_auto_anchor_and_default_fit_until_real_dimensions() {
        let mut fixture = Fixture::new();
        let q = fixture.project.sequence_mut(fixture.sequence).unwrap();
        q.video_tracks[0].items.clear();
        q.audio_tracks[0].items.clear();
        let vdest = q.video_tracks[0].id;
        let adest = q.audio_tracks[0].id;
        let mut session = crate::Session::new(Arc::new(crate::FsServices));
        session.project = Arc::new(fixture.project);
        session.state.active_sequence = Some(fixture.sequence);
        session.prefs.media.default_media_scaling = "setToFrameSize".into();
        crate::commands::place_item(
            &mut session,
            crate::commands::ItemPlacement {
                item: fixture.source,
                range: TimeRange::new(Tick::ZERO, fixture.old.duration),
                at: Tick::ZERO,
                vdest: Some(vdest),
                adest: Some(adest),
                insert: false,
                label: "Place pending movie",
                audio_split: None,
            },
        )
        .unwrap();
        let clip = &session.active_sequence().unwrap().video_tracks[0].items[0];
        let motion = clip.effects.iter().find(|effect| effect.effect == "motion").unwrap();
        let anchor = motion.param("anchor").unwrap().value.as_vec2().unwrap();
        assert!(anchor.x.is_nan() && anchor.y.is_nan(), "a provisional frame must not become the final source anchor");
        assert_eq!(motion.param("scale").unwrap().value.as_f64(), Some(100.0));
    }

    #[test]
    fn portrait_metadata_centers_the_actual_source_and_applies_default_fit() {
        let mut fixture = Fixture::new();
        let info = fixture.portrait();
        assert!(fixture.hydrate(&info, "setToFrameSize").is_empty());
        let motion = fixture.clip(fixture.video).effects.iter().find(|effect| effect.effect == "motion").unwrap();
        assert_eq!(motion.param("anchor").unwrap().value.as_vec2(), Some(Vec2::new(540.0, 960.0)));
        assert_eq!(motion.param("position").unwrap().value.as_vec2(), Some(Vec2::new(960.0, 540.0)));
        assert_eq!(motion.param("scale").unwrap().value.as_f64(), Some(56.3));
    }

    #[test]
    fn metadata_keeps_user_position_anchor_and_animated_scale() {
        let mut fixture = Fixture::new();
        let info = fixture.portrait();
        let motion = fixture.clip_mut(fixture.video).effects.iter_mut().find(|effect| effect.effect == "motion").unwrap();
        motion.params.insert("position".into(), Param::new(ParamValue::Vec2(Vec2::new(25.0, 40.0))));
        motion.params.insert("anchor".into(), Param::new(ParamValue::Vec2(Vec2::new(13.0, 15.0))));
        motion.param_mut("scale").unwrap().keyframes.push(Keyframe::new(Tick::ZERO, ParamValue::Float(100.0)));
        let before = motion.clone();
        assert!(fixture.hydrate(&info, "setToFrameSize").is_empty());
        assert_eq!(fixture.clip(fixture.video).effects[0], before);
    }

    #[test]
    fn additional_streams_follow_a_moved_whole_pair_with_unique_clip_identities() {
        let mut fixture = Fixture::new();
        let at = Tick(20 * TICKS_PER_SECOND);
        fixture.clip_mut(fixture.video).start = at;
        fixture.clip_mut(fixture.audio).start = at;
        let info = fixture.multistream();
        assert!(fixture.hydrate(&info, "none").is_empty());
        let q = fixture.project.sequence(fixture.sequence).unwrap();
        assert_eq!(q.audio_tracks.len(), 3);
        let clips: Vec<_> = q.audio_tracks.iter().flat_map(|track| &track.items).collect();
        assert_eq!(clips.len(), 3);
        assert_eq!(clips[0].id, fixture.audio);
        assert_eq!(clips.iter().map(|clip| clip.id).collect::<BTreeSet<_>>().len(), 3);
        for (stream, clip) in clips.iter().enumerate() {
            assert_eq!(clip.start, at);
            assert_eq!(clip.audio_stream, stream);
            assert_eq!(clip.link, fixture.clip(fixture.video).link);
            assert!(fixture.project.pending_imports[&fixture.source].placements.contains_key(&clip.id));
        }
        q.check().unwrap();
    }

    #[test]
    fn stream_discovery_preserves_trimmed_audio_and_does_not_recreate_deleted_audio() {
        let mut fixture = Fixture::new();
        let info = fixture.multistream();
        fixture.clip_mut(fixture.audio).duration = Tick(2 * TICKS_PER_SECOND);
        let before = fixture.clip(fixture.audio).clone();
        assert!(!fixture.hydrate(&info, "none").is_empty());
        assert_eq!(fixture.clip(fixture.audio), &before);
        assert_eq!(fixture.project.sequence(fixture.sequence).unwrap().audio_tracks.len(), 1);
        fixture.project.sequence_mut(fixture.sequence).unwrap().audio_tracks[0].items.clear();
        fixture.hydrate(&info, "none");
        assert!(fixture.project.sequence(fixture.sequence).unwrap().audio_tracks.iter().all(|track| track.items.is_empty()));
    }

    #[test]
    fn conflicting_audio_destination_is_unchanged_and_no_extra_tracks_are_created() {
        let mut fixture = Fixture::new();
        let track = filmcraft_project::TrackId(fixture.project.alloc_id());
        let mut obstacle = fixture.clip(fixture.audio).clone();
        obstacle.id = ClipId(fixture.project.alloc_id());
        let mut destination = filmcraft_project::Track::new(track, TrackKind::Audio, "Existing edit".into());
        destination.items.push(obstacle.clone());
        fixture.project.sequence_mut(fixture.sequence).unwrap().audio_tracks.push(destination);
        let next_id = fixture.project.next_id;
        assert!(!fixture.hydrate(&fixture.multistream(), "none").is_empty());
        let q = fixture.project.sequence(fixture.sequence).unwrap();
        assert_eq!(q.audio_tracks.len(), 2);
        assert_eq!(q.audio_tracks[1].items, vec![obstacle]);
        assert_eq!(fixture.project.next_id, next_id, "rejected stream placement must not allocate identities");
    }

    #[test]
    fn absent_audio_removes_only_untouched_provisional_instances() {
        let mut fixture = Fixture::new();
        let mut info = fixture.old.clone();
        info.audio_streams.clear();
        fixture.clip_mut(fixture.audio).gain_db = 4.0;
        let edited = fixture.clip(fixture.audio).clone();
        assert!(!fixture.hydrate(&info, "none").is_empty());
        assert_eq!(fixture.clip(fixture.audio), &edited);
        fixture.clip_mut(fixture.audio).gain_db = 0.0;
        assert!(fixture.hydrate(&info, "none").is_empty());
        assert!(fixture.project.sequence(fixture.sequence).unwrap().audio_tracks[0].items.is_empty());
        assert_eq!(fixture.clip(fixture.video).id, fixture.video);
    }

    #[test]
    fn absent_stream_keeps_an_independently_moved_placeholder() {
        let mut fixture = Fixture::new();
        let mut info = fixture.old.clone();
        info.audio_streams.clear();
        fixture.clip_mut(fixture.audio).start = Tick(4 * TICKS_PER_SECOND);
        let before = fixture.clip(fixture.audio).clone();
        assert!(!fixture.hydrate(&info, "none").is_empty());
        assert_eq!(fixture.clip(fixture.audio), &before);
    }
}
