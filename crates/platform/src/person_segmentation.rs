//! On-device person segmentation for video frames.

use std::fmt;

#[cfg(target_os = "macos")]
#[allow(unsafe_code)]
mod macos;

/// An 8-bit, top-to-bottom matte in the backend's native resolution.
#[derive(Debug)]
pub struct PersonMatte {
    pub width: u32,
    pub height: u32,
    /// Tightly packed opacity, from 0 (background) to 255 (foreground).
    pub alpha: Vec<u8>,
}

#[derive(Debug)]
pub enum SegmentationError {
    Unavailable,
    InvalidFrame,
    Vision(String),
}

impl fmt::Display for SegmentationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable => f.write_str("Remove Background requires macOS 12 or later with Apple Vision."),
            Self::InvalidFrame => f.write_str("Remove Background needs a valid, tightly packed RGBA8 frame within the supported size."),
            Self::Vision(message) => write!(f, "Apple Vision could not generate a person matte: {message}"),
        }
    }
}

impl std::error::Error for SegmentationError {}

/// Whether the OS provides the person segmentation API. A particular request can still fail.
pub fn available() -> bool {
    #[cfg(target_os = "macos")]
    {
        macos::available()
    }
    #[cfg(not(target_os = "macos"))]
    {
        false
    }
}

/// A single video analysis session at Apple Vision's Balanced quality. Feed frames in presentation
/// order and keep the session on its creating thread so Vision can reuse prior frames.
pub struct PersonSegmenter {
    #[cfg(target_os = "macos")]
    sequence: macos::VisionSequence,
    #[cfg(target_os = "macos")]
    bgra: Vec<u8>,
}

impl PersonSegmenter {
    pub fn new() -> Result<Self, SegmentationError> {
        #[cfg(target_os = "macos")]
        {
            Ok(Self { sequence: macos::VisionSequence::new()?, bgra: Vec::new() })
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(SegmentationError::Unavailable)
        }
    }

    /// Analyze an upright decoded picture without first materializing RGBA and then swizzling
    /// it into Vision's BGRA input. The frame conversion preserves the RGBA path's RGB values
    /// and makes alpha opaque, as Vision's original input preparation does.
    pub fn segment_frame(&mut self, frame: &filmcraft_frame::VideoFrame) -> Result<PersonMatte, SegmentationError> {
        #[cfg(target_os = "macos")]
        {
            pixel_count(frame.width as usize, frame.height as usize).ok_or(SegmentationError::InvalidFrame)?;
            if !frame.to_opaque_bgra8_into(&mut self.bgra) {
                return Err(SegmentationError::InvalidFrame);
            }
            self.sequence.segment_opaque_bgra(frame.width as usize, frame.height as usize, &self.bgra)
        }
        #[cfg(not(target_os = "macos"))]
        {
            let _ = frame;
            Err(SegmentationError::Unavailable)
        }
    }

    /// Analyze one top-to-bottom frame. Vision uses the RGB channels as its input.
    pub fn segment_rgba8(&mut self, width: u32, height: u32, rgba: &[u8]) -> Result<PersonMatte, SegmentationError> {
        let (width, height) = (width as usize, height as usize);
        let len = pixel_count(width, height).and_then(|count| count.checked_mul(4));
        if len != Some(rgba.len()) {
            return Err(SegmentationError::InvalidFrame);
        }
        #[cfg(target_os = "macos")]
        {
            self.sequence.segment(width, height, rgba)
        }
        #[cfg(not(target_os = "macos"))]
        {
            Err(SegmentationError::Unavailable)
        }
    }
}

fn pixel_count(width: usize, height: usize) -> Option<usize> {
    // Cap native allocations and individual dimensions, while allowing 8K and portrait frames.
    if width == 0 || height == 0 || width > 16_384 || height > 16_384 {
        return None;
    }
    width.checked_mul(height).filter(|&count| count <= 8192 * 8192)
}

#[cfg(test)]
mod tests {
    use super::pixel_count;

    #[test]
    fn bounds_matte_dimensions() {
        assert_eq!(pixel_count(1080, 1920), Some(1080 * 1920));
        assert_eq!(pixel_count(7680, 4320), Some(7680 * 4320));
        assert_eq!(pixel_count(0, 1920), None);
        assert_eq!(pixel_count(16_384, 16_384), None);
        assert_eq!(pixel_count(usize::MAX, usize::MAX), None);
    }
}
