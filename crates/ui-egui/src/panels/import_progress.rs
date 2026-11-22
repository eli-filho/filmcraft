//! Shared, noninteractive feedback for media that is becoming ready in the background.

use egui::{Color32, Painter, Rect, pos2};

pub(crate) fn label(stage: &str, progress: Option<f32>) -> String {
    match progress.filter(|p| p.is_finite()) {
        Some(p) => format!("{stage} · {:.0}%", p.clamp(0.0, 1.0) * 100.0),
        None => stage.to_string(),
    }
}

pub(crate) fn paint(painter: &Painter, rect: Rect, progress: Option<f32>, color: Color32, time: f64) {
    if rect.width() <= 0.0 || rect.height() <= 0.0 {
        return;
    }
    let painter = painter.with_clip_rect(rect.intersect(painter.clip_rect()));
    painter.rect_filled(rect, 1.0, Color32::from_black_alpha(100));
    let (start, end) = match progress.filter(|p| p.is_finite()) {
        Some(p) => (0.0, p.clamp(0.0, 1.0)),
        // Unknown work size stays indeterminate instead of presenting an invented percentage.
        None => {
            let phase = if time.is_finite() { (time * 0.6).rem_euclid(1.0) as f32 } else { 0.0 };
            (phase * 1.25 - 0.25, phase * 1.25)
        }
    };
    painter.rect_filled(Rect::from_min_max(pos2(rect.min.x + rect.width() * start, rect.min.y), pos2(rect.min.x + rect.width() * end, rect.max.y)), 1.0, color);
}
