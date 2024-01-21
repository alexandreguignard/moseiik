use moseiik::main::Options;
use moseiik::main::compute_mosaic;
use image::{io::Reader as ImageReader, RgbImage,};
use std::error::Error;

#[cfg(test)]
mod tests {
    use super::*;

    fn open_image(image_path: &str) -> Result<RgbImage, Box<dyn Error>>
    {
        let mut path = image_path.to_owned();
        Ok(ImageReader::open(&path)?.decode()?.into_rgb8())
    }

    #[test]
    fn test_generic() {
        let args = Options {
                image: "assets/kit.jpeg".to_string(),
                output: "assets/test_integration.png".to_string(),
                tiles: "assets/images".to_string(),
                scaling: 1,
                tile_size: 25,
                remove_used: false,
                verbose: false,
                simd: false,
                num_thread: 1,
            };

        compute_mosaic(args);

        let image1_path = "assets/test_integration.png";
        let image2_path = "assets/ground-truth-kit.png";

        let result1 = open_image(image1_path);
        let result2 = open_image(image2_path);

        match (result1, result2) {
            (Ok(im1), Ok(im2)) => {
                assert!(im1 == im2);
            }
            (Err(_err), _) | (_, Err(_err)) => {
                assert!(false);
            }
        }
    }
}
