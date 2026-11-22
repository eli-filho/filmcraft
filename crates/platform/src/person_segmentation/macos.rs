//! Synchronous bindings to the public Vision request and CoreVideo pixel buffers.

use std::mem::MaybeUninit;
use std::ptr::NonNull;

use objc2::msg_send;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::{AnyClass, AnyObject};
use objc2_core_foundation::{CFDictionary, CFNumber, CFRetained, CFString, CFType};
use objc2_core_video::{
    CVPixelBuffer, CVPixelBufferCreate, CVPixelBufferGetBaseAddress, CVPixelBufferGetBytesPerRow, CVPixelBufferGetDataSize, CVPixelBufferGetHeight,
    CVPixelBufferGetPixelFormatType, CVPixelBufferGetWidth, CVPixelBufferIsPlanar, CVPixelBufferLockBaseAddress, CVPixelBufferLockFlags, CVPixelBufferPool,
    CVPixelBufferUnlockBaseAddress, kCVPixelBufferHeightKey, kCVPixelBufferPixelFormatTypeKey, kCVPixelBufferWidthKey, kCVPixelFormatType_32BGRA,
    kCVPixelFormatType_OneComponent8,
};
use objc2_foundation::{NSArray, NSError};

use super::{PersonMatte, SegmentationError, pixel_count};

// VNGeneratePersonSegmentationRequestQualityLevelBalanced in the Vision SDK.
const BALANCED: usize = 1;

#[link(name = "Vision", kind = "framework")]
unsafe extern "C" {
    // The sequence handler dates to macOS 10.13 and provides a link to Vision even on Big Sur.
    // The person segmentation request, introduced in macOS 12, is looked up at runtime.
    #[link_name = "OBJC_CLASS_$_VNSequenceRequestHandler"]
    static SEQUENCE_REQUEST_HANDLER: AnyClass;
}

fn person_request_class() -> Option<&'static AnyClass> {
    AnyClass::get(c"VNGeneratePersonSegmentationRequest")
}

pub(super) fn available() -> bool {
    person_request_class().is_some()
}

pub(super) struct VisionSequence {
    handler: Retained<AnyObject>,
    request: Retained<AnyObject>,
    requests: Retained<NSArray<AnyObject>>,
    input_pool: Option<InputPool>,
}

impl VisionSequence {
    pub(super) fn new() -> Result<Self, SegmentationError> {
        let class = person_request_class().ok_or(SegmentationError::Unavailable)?;
        autoreleasepool(|_| {
            // SAFETY: the runtime confirmed the class exists, and its public Objective-C API
            // declares +new as returning an owned request.
            let request: Option<Retained<AnyObject>> = unsafe { msg_send![class, new] };
            let request = request.ok_or_else(|| vision_error("cannot create the person segmentation request"))?;
            // SAFETY: this symbol is the system's VNSequenceRequestHandler class object, and the
            // class supports NSObject's +new to initialize an owned instance.
            let handler: Option<Retained<AnyObject>> = unsafe { msg_send![&SEQUENCE_REQUEST_HANDLER, new] };
            let handler = handler.ok_or_else(|| vision_error("cannot create the video request handler"))?;
            // SAFETY: this is our unshared VNGeneratePersonSegmentationRequest. Revision 1 is
            // supported from macOS 12, and Balanced is a valid quality level.
            unsafe {
                let _: () = msg_send![&*request, setRevision: 1_usize];
                let _: () = msg_send![&*request, setQualityLevel: BALANCED];
                let _: () = msg_send![&*request, setOutputPixelFormat: kCVPixelFormatType_OneComponent8];
            }
            let requests = NSArray::from_slice(&[&*request]);
            Ok(Self { handler, request, requests, input_pool: None })
        })
    }

    pub(super) fn segment(&mut self, width: usize, height: usize, rgba: &[u8]) -> Result<PersonMatte, SegmentationError> {
        self.segment_pixels(width, height, Pixels::Rgba(rgba))
    }

    pub(super) fn segment_opaque_bgra(&mut self, width: usize, height: usize, bgra: &[u8]) -> Result<PersonMatte, SegmentationError> {
        self.segment_pixels(width, height, Pixels::OpaqueBgra(bgra))
    }

    fn segment_pixels(&mut self, width: usize, height: usize, pixels: Pixels<'_>) -> Result<PersonMatte, SegmentationError> {
        if self.input_pool.as_ref().is_none_or(|pool| (pool.width, pool.height) != (width, height)) {
            self.input_pool = Some(InputPool::new(width, height)?);
        }
        let pool = self.input_pool.as_ref().map(|pool| &*pool.pool);
        let input = make_input(width, height, pixels, pool)?;
        autoreleasepool(|_| {
            let mut error: Option<Retained<NSError>> = None;
            // SAFETY: the handler, one Vision request and unlocked CoreVideo buffer are valid
            // for this synchronous call. objc2 handles the autoreleasing NSError out parameter.
            let succeeded: bool = unsafe { msg_send![&*self.handler, performRequests: &*self.requests, onCVPixelBuffer: &*input, error: Some(&mut error)] };
            if !succeeded {
                let message = error.map_or_else(|| "the request failed".to_owned(), |err| err.localizedDescription().to_string());
                return Err(vision_error(message));
            }
            // SAFETY: Vision documents results as an array of pixel buffer observations after
            // successful completion of this request.
            let results: Option<Retained<NSArray<AnyObject>>> = unsafe { msg_send![&*self.request, results] };
            let observation = results.and_then(|results| results.firstObject()).ok_or_else(|| vision_error("the request returned no matte"))?;
            // SAFETY: the observation is a VNPixelBufferObservation whose pixelBuffer property
            // returns a valid CoreVideo buffer. objc2 retains the returned object for readback.
            let mask: Option<Retained<CVPixelBuffer>> = unsafe { msg_send![&*observation, pixelBuffer] };
            let mask = mask.ok_or_else(|| vision_error("the observation had no pixel buffer"))?;
            copy_matte(&mask)
        })
    }
}

enum Pixels<'a> {
    Rgba(&'a [u8]),
    // Only segment_frame constructs this variant, from VideoFrame's opaque BGRA conversion.
    OpaqueBgra(&'a [u8]),
}

struct InputPool {
    width: usize,
    height: usize,
    pool: CFRetained<CVPixelBufferPool>,
}

impl InputPool {
    fn new(width: usize, height: usize) -> Result<Self, SegmentationError> {
        pixel_count(width, height).ok_or(SegmentationError::InvalidFrame)?;
        let width_value = CFNumber::new_i64(i64::try_from(width).map_err(|_| SegmentationError::InvalidFrame)?);
        let height_value = CFNumber::new_i64(i64::try_from(height).map_err(|_| SegmentationError::InvalidFrame)?);
        let format = CFNumber::new_i64(i64::from(kCVPixelFormatType_32BGRA));
        // SAFETY: CoreVideo defines these immutable CFString keys on every supported macOS.
        let keys = unsafe { [kCVPixelBufferWidthKey, kCVPixelBufferHeightKey, kCVPixelBufferPixelFormatTypeKey] };
        let attributes = CFDictionary::<CFString, CFType>::from_slices(&keys, &[width_value.as_ref(), height_value.as_ref(), format.as_ref()]);
        let mut out = std::ptr::null_mut();
        // SAFETY: the attributes have the documented CFString/CFNumber types; out is a valid
        // pool out-pointer. A pool recycles only released buffers, so a previous frame retained
        // by Vision cannot be overwritten by preparation of the next frame.
        let status = unsafe { CVPixelBufferPool::create(None, None, Some(attributes.as_ref()), NonNull::from(&mut out)) };
        let out = NonNull::new(out).filter(|_| status == 0).ok_or_else(|| vision_error(format!("could not create the input buffer pool ({status})")))?;
        // SAFETY: successful creation returns a pool retained at +1.
        let pool = unsafe { CFRetained::from_raw(out) };
        Ok(Self { width, height, pool })
    }
}

fn make_input(width: usize, height: usize, pixels: Pixels<'_>, pool: Option<&CVPixelBufferPool>) -> Result<CFRetained<CVPixelBuffer>, SegmentationError> {
    let len = match &pixels {
        Pixels::Rgba(bytes) | Pixels::OpaqueBgra(bytes) => bytes.len(),
    };
    if pixel_count(width, height).and_then(|count| count.checked_mul(4)) != Some(len) {
        return Err(SegmentationError::InvalidFrame);
    }
    let mut out = std::ptr::null_mut();
    let status = match pool {
        // SAFETY: the pool uses the requested BGRA dimensions and out is a valid out-pointer.
        Some(pool) => unsafe { CVPixelBufferPool::create_pixel_buffer(None, pool, NonNull::from(&mut out)) },
        // SAFETY: the caller bounded both dimensions and validated the slice length. Null
        // attributes request the default layout, and out is a valid CoreVideo out-pointer.
        None => unsafe { CVPixelBufferCreate(None, width, height, kCVPixelFormatType_32BGRA, None, NonNull::from(&mut out)) },
    };
    let out = NonNull::new(out).filter(|_| status == 0).ok_or_else(|| vision_error(format!("could not allocate the input buffer ({status})")))?;
    // SAFETY: successful creation returns the pixel buffer retained at +1.
    let input = unsafe { CFRetained::from_raw(out) };
    if CVPixelBufferGetWidth(&input) != width {
        return Err(vision_error("unexpected input buffer width"));
    }
    let row_bytes = width.checked_mul(4).ok_or(SegmentationError::InvalidFrame)?;
    let lock = PixelLock::new(&input, CVPixelBufferLockFlags(0))?;
    let rows = lock.packed_rows(row_bytes, height, kCVPixelFormatType_32BGRA)?;
    match pixels {
        Pixels::Rgba(rgba) => {
            let (pixels, _) = rgba.as_chunks::<4>();
            for (y, src) in pixels.chunks_exact(width).enumerate() {
                // SAFETY: we hold the write lock on a fresh buffer with no foreign aliases. The
                // layout check covers each active row; MaybeUninit permits the initial write.
                let dst = unsafe { std::slice::from_raw_parts_mut(rows.row(y).cast::<MaybeUninit<[u8; 4]>>(), width) };
                for (&[r, g, b, _], dst) in src.iter().zip(dst) {
                    dst.write([b, g, r, 255]);
                }
            }
        }
        Pixels::OpaqueBgra(bgra) => {
            for (y, src) in bgra.chunks_exact(row_bytes).enumerate() {
                // SAFETY: the caller validated exactly width * height BGRA pixels. The source
                // slice and fresh, exclusively write-locked CoreVideo allocation cannot alias,
                // and packed_rows checked that every active destination row fits its stride.
                unsafe { std::ptr::copy_nonoverlapping(src.as_ptr(), rows.row(y), row_bytes) };
            }
        }
    }
    lock.finish()?;
    Ok(input)
}

fn copy_matte(mask: &CVPixelBuffer) -> Result<PersonMatte, SegmentationError> {
    let (width, height) = (CVPixelBufferGetWidth(mask), CVPixelBufferGetHeight(mask));
    let len = pixel_count(width, height).ok_or_else(|| vision_error("invalid matte dimensions"))?;
    let mut alpha = Vec::new();
    alpha.try_reserve_exact(len).map_err(|_| vision_error("could not allocate the matte"))?;
    let lock = PixelLock::new(mask, CVPixelBufferLockFlags::ReadOnly)?;
    let rows = lock.packed_rows(width, height, kCVPixelFormatType_OneComponent8)?;
    for y in 0..height {
        // SAFETY: the buffer stays retained and read-locked; the checked layout covers each
        // active row and Vision has initialized its active samples.
        alpha.extend_from_slice(unsafe { std::slice::from_raw_parts(rows.row(y), width) });
    }
    lock.finish()?;
    // pixel_count caps both dimensions at 16,384.
    Ok(PersonMatte { width: width as u32, height: height as u32, alpha })
}

struct PixelLock<'a> {
    buffer: &'a CVPixelBuffer,
    flags: CVPixelBufferLockFlags,
    unlock_on_drop: bool,
}

impl<'a> PixelLock<'a> {
    fn new(buffer: &'a CVPixelBuffer, flags: CVPixelBufferLockFlags) -> Result<Self, SegmentationError> {
        // SAFETY: this is a retained CoreVideo buffer. Drop unlocks it with the same flags.
        let status = unsafe { CVPixelBufferLockBaseAddress(buffer, flags) };
        if status != 0 {
            return Err(vision_error(format!("could not lock a pixel buffer ({status})")));
        }
        Ok(Self { buffer, flags, unlock_on_drop: true })
    }

    fn packed_rows(&self, row_bytes: usize, height: usize, format: u32) -> Result<PackedRows, SegmentationError> {
        if CVPixelBufferGetPixelFormatType(self.buffer) != format || CVPixelBufferIsPlanar(self.buffer) || CVPixelBufferGetHeight(self.buffer) != height {
            return Err(vision_error("unexpected pixel buffer layout"));
        }
        let stride = CVPixelBufferGetBytesPerRow(self.buffer);
        let len = stride.checked_mul(height).filter(|&len| len <= CVPixelBufferGetDataSize(self.buffer) && len <= isize::MAX as usize);
        if row_bytes > stride || len.is_none() {
            return Err(vision_error("pixel buffer is shorter than its dimensions"));
        }
        let base = NonNull::new(CVPixelBufferGetBaseAddress(self.buffer).cast()).ok_or_else(|| vision_error("pixel buffer is empty"))?;
        Ok(PackedRows { base, stride })
    }

    fn finish(mut self) -> Result<(), SegmentationError> {
        self.unlock_on_drop = false;
        self.unlock()
    }

    fn unlock(&self) -> Result<(), SegmentationError> {
        // SAFETY: this guard holds a successful lock and the same flags used to acquire it.
        let status = unsafe { CVPixelBufferUnlockBaseAddress(self.buffer, self.flags) };
        if status != 0 {
            return Err(vision_error(format!("could not unlock a pixel buffer ({status})")));
        }
        Ok(())
    }
}

impl Drop for PixelLock<'_> {
    fn drop(&mut self) {
        if self.unlock_on_drop
            && let Err(error) = self.unlock()
        {
            log::warn!("{error}");
        }
    }
}

struct PackedRows {
    base: NonNull<u8>,
    stride: usize,
}

impl PackedRows {
    // Both callers iterate only over the checked height used to construct the layout.
    fn row(&self, y: usize) -> *mut u8 {
        self.base.as_ptr().wrapping_add(y * self.stride)
    }
}

fn vision_error(message: impl Into<String>) -> SegmentationError {
    SegmentationError::Vision(message.into())
}

#[cfg(test)]
mod tests {
    use super::super::{PersonSegmenter, SegmentationError, available};
    use super::{CVPixelBuffer, CVPixelBufferLockFlags, InputPool, PixelLock, Pixels, kCVPixelFormatType_32BGRA, make_input};

    fn read_bgra(buffer: &CVPixelBuffer, width: usize, height: usize) -> Vec<u8> {
        let row_bytes = width * 4;
        let lock = PixelLock::new(buffer, CVPixelBufferLockFlags::ReadOnly).expect("read the prepared input");
        let rows = lock.packed_rows(row_bytes, height, kCVPixelFormatType_32BGRA).expect("check the actual CoreVideo row stride");
        let mut bytes = Vec::new();
        for y in 0..height {
            // SAFETY: the read lock holds the buffer in place, the checked stride bounds this
            // active row, and `make_input` initialized all its samples.
            bytes.extend_from_slice(unsafe { std::slice::from_raw_parts(rows.row(y), row_bytes) });
        }
        lock.finish().expect("unlock the prepared input");
        bytes
    }

    #[test]
    fn opaque_bgra_input_matches_the_rgba_swizzle() {
        let (width, height) = (5, 3);
        let rgba: Vec<u8> =
            (0..(width * height) as u8).flat_map(|i| [i.wrapping_mul(37), 255 - i.wrapping_mul(23), i.wrapping_mul(91), i.wrapping_mul(17)]).collect();
        let frame = filmcraft_frame::VideoFrame::rgba8(width as u32, height as u32, rgba.clone());
        let mut bgra = Vec::new();
        assert!(frame.to_opaque_bgra8_into(&mut bgra), "convert the odd-width fixture");
        let original = make_input(width, height, Pixels::Rgba(&rgba), None).expect("make the RGBA input");
        let direct = make_input(width, height, Pixels::OpaqueBgra(&bgra), None).expect("make the opaque BGRA input");
        let expected: Vec<u8> = rgba.as_chunks::<4>().0.iter().flat_map(|&[r, g, b, _]| [b, g, r, 255]).collect();
        assert_eq!(read_bgra(&original, width, height), expected);
        assert_eq!(read_bgra(&direct, width, height), expected);
    }

    #[test]
    fn pool_does_not_recycle_a_retained_input_buffer() {
        let (width, height) = (5, 3);
        let pool = InputPool::new(width, height).expect("create the Vision input pool");
        let first_pixels = [9, 47, 162, 255].repeat(width * height);
        let second_pixels = [222, 18, 36, 255].repeat(width * height);
        let held = make_input(width, height, Pixels::OpaqueBgra(&first_pixels), Some(&pool.pool)).expect("check out the first input");
        let second = make_input(width, height, Pixels::OpaqueBgra(&second_pixels), Some(&pool.pool)).expect("check out another input while the first is held");
        assert_eq!(read_bgra(&held, width, height), first_pixels);
        assert_eq!(read_bgra(&second, width, height), second_pixels);
    }

    #[test]
    fn input_rejects_invalid_dimensions_and_byte_counts() {
        assert!(matches!(InputPool::new(16_385, 1), Err(SegmentationError::InvalidFrame)));
        assert!(matches!(make_input(0, 3, Pixels::OpaqueBgra(&[]), None), Err(SegmentationError::InvalidFrame)));
        assert!(matches!(make_input(5, 3, Pixels::Rgba(&[0; 59]), None), Err(SegmentationError::InvalidFrame)));
        let pool = InputPool::new(5, 3).expect("create the Vision input pool");
        assert!(matches!(make_input(5, 3, Pixels::OpaqueBgra(&[0; 61]), Some(&pool.pool)), Err(SegmentationError::InvalidFrame)));
        assert!(make_input(3, 5, Pixels::OpaqueBgra(&[0; 60]), Some(&pool.pool)).is_err());
    }

    #[test]
    fn analyzes_a_synthetic_frame_on_macos() {
        if !available() {
            return; // The application also supports macOS 11, where this request is absent.
        }
        let mut segmenter = PersonSegmenter::new().expect("create the Vision segmenter");
        assert!(matches!(segmenter.segment_rgba8(161, 120, &[0; 4]), Err(SegmentationError::InvalidFrame)));
        let frame = [12, 96, 180, 255].repeat(161 * 120);
        for _ in 0..2 {
            let mask = segmenter.segment_rgba8(161, 120, &frame).expect("process an opaque test frame");
            assert!(mask.width > 0 && mask.height > 0);
            assert_eq!(mask.alpha.len(), (mask.width * mask.height) as usize);
            assert!(mask.alpha.iter().all(|&value| value < 128));
        }
    }

    #[test]
    fn analyzes_frames_with_reused_pixel_storage_and_dimension_changes() {
        if !available() {
            return;
        }
        use filmcraft_frame::{Chroma, PixelData, VideoFrame};
        use std::sync::Arc;

        let mut segmenter = PersonSegmenter::new().expect("create the Vision segmenter");
        let mut rgba_segmenter = PersonSegmenter::new().expect("create the RGBA reference segmenter");
        let mut previous = None;
        for (width, height) in [(161u32, 120u32), (161, 120), (120, 161)] {
            let pixels = (width * height) as usize;
            let chroma = (width.div_ceil(2) * height.div_ceil(2)) as usize;
            let frame = VideoFrame {
                width,
                height,
                data: PixelData::Yuv8 {
                    planes: [Arc::new(vec![80; pixels]), Arc::new(vec![100; chroma]), Arc::new(vec![180; chroma])],
                    chroma: Chroma::C420,
                    alpha: None,
                },
                color: filmcraft_color::ColorInfo::REC709,
                par: (1, 1),
                pts: filmcraft_time::Tick::ZERO,
            };
            let mask = segmenter.segment_frame(&frame).expect("process the decoded test frame");
            let reference = rgba_segmenter.segment_rgba8(width, height, &frame.to_rgba8()).expect("process the same RGB through the original path");
            let storage = segmenter.bgra.as_ptr();
            if let Some(previous) = previous {
                assert_eq!(storage, previous, "reuse the same-sized pixel allocation");
            }
            previous = Some(storage);
            assert!(mask.width > 0 && mask.height > 0);
            assert_eq!(mask.alpha.len(), (mask.width * mask.height) as usize);
            assert_eq!((mask.width, mask.height), (reference.width, reference.height));
            assert_eq!(mask.alpha, reference.alpha, "preserve Vision's result when reusing input storage");
        }
    }
}
