//! Direct placement and wrap width for caption tracks in the paused Program Monitor.

use egui::{CursorIcon, Id, PointerButton, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2, pos2, vec2};
use filmcraft_engine::captions::caption_bounds;
use filmcraft_project::{CaptionStyle, ClipId, TrackId};
use serde_json::{Value, json};

use crate::FilmcraftApp;
use crate::frames::Target;
use crate::state::Tool;

#[derive(Clone, Copy)]
struct Drag {
    caption: ClipId,
    /// Distance from the press to the centre of the caption, in picture fractions.
    offset: Vec2,
}

#[derive(Clone, Copy)]
struct VisibleCaption {
    track: TrackId,
    caption: ClipId,
    bounds: Rect,
    width: f32,
}

#[derive(Clone, Copy, Default)]
struct WidthDrag {
    start_x: f32,
    width: f32,
}

pub(super) fn overlay(app: &mut FilmcraftApp, ui: &mut egui::Ui, picture: Rect, viewport: Rect, frame: i64) {
    let drag_id = Id::unique("program-caption-drag");
    let clear = |ui: &egui::Ui| ui.data_mut(|d| d.remove::<Drag>(drag_id));
    if app.playback.playing || app.ui.tool != Tool::Selection || app.ui.mask_pen.is_some() || app.ui.gfx_edit.is_some() || !picture.is_positive() {
        clear(ui);
        return;
    }
    let Some(sequence) = app.session.state.active_sequence else { return };
    let Some(shown) = app.program_shown.filter(|k| {
        k.frame == frame
            && k.revision == app.session.revision
            && matches!(k.target, Target::Sequence(id) | Target::SequenceLive(id) | Target::SequencePlan(id, _) if id == sequence)
    }) else {
        clear(ui);
        return;
    };
    let Some(seq) = app.session.active_sequence() else { return };
    let time = seq.settings.frame_rate.tick_of(frame);
    let (w, h) = filmcraft_render::output_size(seq, super::quantize_scale(shown.size as f32 / 1000.0));
    let scale = picture.size() / vec2(w as f32, h as f32);
    let captions: Vec<_> = seq
        .caption_tracks
        .iter()
        .rev() // register the upper track last so it receives the pointer first
        .filter(|track| track.enabled && !track.locked)
        .filter_map(|track| {
            let c = track.caption_at(time)?;
            let b = caption_bounds(c, &track.style, w, h)?;
            let bounds = Rect::from_min_size(picture.min + vec2(b.x as f32, b.y as f32) * scale, vec2(b.w as f32, b.h as f32) * scale);
            let width = track.style.wrap_width.unwrap_or(CaptionStyle::DEFAULT_WRAP_WIDTH).clamp(CaptionStyle::MIN_WRAP_WIDTH, 1.0);
            Some(VisibleCaption { track: track.id, caption: c.id, bounds, width })
        })
        .collect();
    let mut drag: Option<Drag> = ui.data(|d| d.get_temp(drag_id));
    if drag.is_some_and(|d| !captions.iter().any(|c| c.caption == d.caption)) {
        clear(ui);
        drag = None;
    }
    let primary = PointerButton::Primary;
    let label = tl!("Click for width handles. Drag to position all captions on this track.");
    let reach = picture.intersect(viewport).intersect(ui.clip_rect());
    for visible in captions {
        let VisibleCaption { track, caption, bounds, .. } = visible;
        let hit = bounds.expand(3.0).intersect(reach);
        if !hit.is_positive() {
            continue;
        }
        let id = Id::unique(("program-caption", sequence, caption));
        let resp = ui.interact(hit, id, Sense::click_and_drag()).on_hover_cursor(CursorIcon::Grab).on_hover_text(label);
        app.auto.add(&format!("program.caption.{}", caption.0), hit, label);
        if resp.clicked_by(primary) {
            invoke(app, ui, "captions.select", json!({"captions": [caption.0]}));
        }
        if resp.drag_started_by(primary)
            && let Some(press) = ui.input(|i| i.pointer.press_origin()).or(resp.interact_pointer_pos())
        {
            let start = Drag { caption, offset: (bounds.center() - press) / picture.size() };
            ui.data_mut(|d| d.insert_temp(drag_id, start));
            drag = Some(start);
        }
        if let Some(d) = drag.filter(|d| d.caption == caption) {
            let stopped = resp.drag_stopped_by(primary);
            if stopped || ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                clear(ui);
            }
            if (resp.dragged_by(primary) || stopped)
                && !ui.input(|i| i.key_pressed(egui::Key::Escape))
                && let Some(pointer) = resp.interact_pointer_pos()
            {
                let center = bounded_center(picture, bounds.size(), pointer + d.offset * picture.size());
                if stopped {
                    let pos = (center - picture.min) / picture.size();
                    let p = json!({"track": track.0, "position": {"x": pos.x, "y": pos.y}});
                    invoke(app, ui, "captions.setStyle", p);
                } else {
                    ui.ctx().set_cursor_icon(CursorIcon::Grabbing);
                    // Move the placement outline immediately. Commit once on release so the
                    // frame server and the project's undo history see just one edit per drag.
                    let outline = Rect::from_center_size(center, bounds.size());
                    ui.painter().rect_stroke(outline, 2.0, Stroke::new(2.0, app.tokens.accent), StrokeKind::Inside);
                }
            }
        } else {
            if resp.hovered() {
                ui.painter().rect_stroke(bounds.expand(2.0), 2.0, Stroke::new(1.0, app.tokens.accent), StrokeKind::Inside);
            }
            if app.session.state.caption_selection.contains(&caption) {
                width_handles(app, ui, picture, reach, id, visible);
            }
        }
    }
}

fn width_handles(app: &mut FilmcraftApp, ui: &mut egui::Ui, picture: Rect, reach: Rect, id: Id, caption: VisibleCaption) {
    let guide = width_guide(picture, caption.bounds, caption.width);
    let accent = app.tokens.accent;
    let primary = PointerButton::Primary;
    let label = tl!("Drag to change caption width without resizing the text");
    let mut proposed = caption.width;
    for (name, x, sign) in [("left", guide.left(), -1.0), ("right", guide.right(), 1.0)] {
        let handle = Rect::from_center_size(pos2(x, guide.center().y), vec2(14.0, 22.0)).intersect(reach);
        if !handle.is_positive() {
            continue;
        }
        let id = id.with(name);
        let drag_id = id.with("resize");
        let resp = ui.interact(handle, id, Sense::drag()).on_hover_cursor(CursorIcon::ResizeHorizontal).on_hover_text(label);
        app.auto.add(&format!("program.caption.{}.width.{name}", caption.caption.0), handle, label);
        if resp.drag_started_by(primary)
            && let Some(p) = ui.input(|i| i.pointer.press_origin()).or(resp.interact_pointer_pos())
        {
            ui.data_mut(|d| d.insert_temp(drag_id, WidthDrag { start_x: p.x, width: caption.width }));
        }
        let stopped = resp.drag_stopped_by(primary);
        let cancel = ui.input(|i| i.key_pressed(egui::Key::Escape));
        let drag: Option<WidthDrag> = if stopped || cancel { ui.data_mut(|d| d.remove_temp(drag_id)) } else { ui.data(|d| d.get_temp(drag_id)) };
        if !cancel
            && (resp.dragged_by(primary) || stopped)
            && let (Some(d), Some(p)) = (drag, resp.interact_pointer_pos())
        {
            proposed = (d.width + 2.0 * sign * (p.x - d.start_x) / picture.width()).clamp(CaptionStyle::MIN_WRAP_WIDTH, 1.0);
            ui.ctx().set_cursor_icon(CursorIcon::ResizeHorizontal);
            if stopped && (proposed - d.width).abs() > 0.001 {
                invoke(app, ui, "captions.setStyle", json!({"track": caption.track.0, "wrapWidth": proposed}));
            }
        }
    }
    let guide = width_guide(picture, caption.bounds, proposed);
    ui.painter().rect_stroke(guide, 2.0, Stroke::new(1.0, accent), StrokeKind::Inside);
    for x in [guide.left(), guide.right()] {
        let handle = Rect::from_center_size(pos2(x, guide.center().y), vec2(5.0, 15.0));
        ui.painter().rect_filled(handle.intersect(picture), 2.0, accent);
    }
}

fn width_guide(picture: Rect, bounds: Rect, width: f32) -> Rect {
    let size = vec2(width * picture.width(), bounds.height().max(16.0).min(picture.height()));
    Rect::from_center_size(bounded_center(picture, size, bounds.center()), size)
}

fn invoke(app: &mut FilmcraftApp, ui: &egui::Ui, command: &str, params: Value) {
    if let Err(e) = crate::menus::invoke(app, ui.ctx(), command, params) {
        app.ui.status = e;
    }
}

fn bounded_center(picture: Rect, size: Vec2, center: Pos2) -> Pos2 {
    let half = (size / 2.0).min(picture.size() / 2.0);
    let (min, max) = (picture.min + half, picture.max - half);
    // A full-width/height caption has only one possible centre on that axis.
    pos2(center.x.clamp(min.x, max.x.max(min.x)), center.y.clamp(min.y, max.y.max(min.y)))
}
