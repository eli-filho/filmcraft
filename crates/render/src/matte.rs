//! Apply a decoded person matte in upright source space, before Motion places the clip.

use filmcraft_media::person_matte::MatteFrame;
use rayon::prelude::*;

use crate::Image;

pub(crate) fn apply(image: &mut Image, matte: &MatteFrame) {
    let (width, height) = (matte.width as usize, matte.height as usize);
    if width == 0 || height == 0 || width.checked_mul(height) != Some(matte.alpha.len()) || image.w == 0 || image.h == 0 {
        return;
    }
    if (width, height) == (image.w, image.h) {
        image.px.par_chunks_exact_mut(4).zip(matte.alpha.par_iter()).for_each(|(pixel, &alpha)| scale(pixel, alpha as f32));
        return;
    }

    // Vision's plane can be smaller than the decoded picture. Sample at pixel centres so the
    // soft edge follows the source at any preview scale, clamping taps at the frame edges.
    let (sx, sy) = (width as f32 / image.w as f32, height as f32 / image.h as f32);
    let xs: Vec<_> = (0..image.w).map(|x| taps(x, sx, width)).collect();
    image.px.par_chunks_mut(image.w * 4).enumerate().for_each(|(y, row)| {
        let (y0, y1, ty) = taps(y, sy, height);
        for (pixel, &(x0, x1, tx)) in row.chunks_exact_mut(4).zip(&xs) {
            // Both coordinates are clamped to the validated plane dimensions.
            let a = matte.alpha[y0 * width + x0] as f32;
            let b = matte.alpha[y0 * width + x1] as f32;
            let c = matte.alpha[y1 * width + x0] as f32;
            let d = matte.alpha[y1 * width + x1] as f32;
            let top = a + (b - a) * tx;
            let bottom = c + (d - c) * tx;
            scale(pixel, top + (bottom - top) * ty);
        }
    });
}

fn taps(i: usize, scale: f32, len: usize) -> (usize, usize, f32) {
    let x = ((i as f32 + 0.5) * scale - 0.5).clamp(0.0, (len - 1) as f32);
    let a = (x as usize).min(len - 1);
    (a, (a + 1).min(len - 1), x - a as f32)
}

/// The image is premultiplied, so RGB and its existing alpha all get the same coverage.
fn scale(pixel: &mut [f32], coverage: f32) {
    let weight = coverage * (1.0 / 255.0);
    for c in pixel {
        *c *= weight;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matte_scales_premultiplied_rgb_and_alpha_with_bilinear_coverage() {
        let mut image = Image::filled(4, 1, [0.5, 0.25, 0.0, 0.5]);
        apply(&mut image, &MatteFrame { width: 2, height: 1, alpha: vec![0, 255] });
        for (x, coverage) in [0.0, 0.25, 0.75, 1.0].into_iter().enumerate() {
            let expected = [0.5 * coverage, 0.25 * coverage, 0.0, 0.5 * coverage];
            for (actual, expected) in image.get(x, 0).into_iter().zip(expected) {
                assert!((actual - expected).abs() < 1e-7, "pixel {x}: {actual} vs {expected}");
            }
        }
    }
}
