// On "importe" ce dont on a besoin
use moseiik::main::Options;
use moseiik::main::compute_mosaic;
use image::{io::Reader as ImageReader, RgbImage,};
use std::error::Error;

#[cfg(test)]
mod tests {
    // Permet d'importer tous les symboles utilisés au début du fichier
    use super::*;

    // Fonction permettant de load nos images de test
    fn open_image(image_path: &str) -> Result<RgbImage, Box<dyn Error>>
    {
        // Car image_path n'est pas mutable or il le faut pour la fonction ImageReader::open
        let path = image_path;
        Ok(ImageReader::open(&path)?.decode()?.into_rgb8())
    }

    // x86 avec SIMD
    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_integration_x86_simd() {
        // On défini les options nécessaire à la création de notre mosaic
        let args = Options {
                image: "assets/kit.jpeg".to_string(), // Image sur laquelle on veut créer une mosaic
                output: "assets/test_integration_x86_simd.png".to_string(), // Image que l'on va générer
                tiles: "assets/images".to_string(), // Dossier où se trouvent les tiles
                scaling: 1, // Factor de grandissement
                tile_size: 25, // Taille des tiles
                remove_used: false,
                verbose: true,
                simd: true,
                num_thread: 1,
            };

        // On appelle la fonction compute_mosaic à laquelle on passe nos options pour générer une mosaic
        compute_mosaic(args);

        // On défini les chemins d'accès à nos images de test
        let image1_path = "assets/test_integration_x86_simd.png";
        let image2_path = "assets/ground-truth-kit_x86.png";

        // On load l'image que l'on vient de générer avec l'appelle de la fonction précédente
        let result1 = open_image(image1_path);
        // On load l'image de référence pour réaliser le test
        let result2 = open_image(image2_path);

        // On vérifie qu'on a bien load les deux images
        match (result1, result2) {
            // Si c'est le cas on les compare
            (Ok(im1), Ok(im2)) => {
                assert!(im1 == im2);
            }
            // Sinon on dit que le test n'est pas passé
            (Err(_err), _) | (_, Err(_err)) => {
                assert!(false);
            }
        }
    }

    // x86 sans SIMD
    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_integration_x86() {
        // On défini les options nécessaire à la création de notre mosaic
        let args = Options {
                image: "assets/kit.jpeg".to_string(), // Image sur laquelle on veut créer une mosaic
                output: "assets/test_integration_x86.png".to_string(), // Image que l'on va générer
                tiles: "assets/images".to_string(), // Dossier où se trouvent les tiles
                scaling: 1, // Factor de grandissement
                tile_size: 25, // Taille des tiles
                remove_used: false,
                verbose: false,
                simd: false,
                num_thread: 1,
            };

        // On appelle la fonction compute_mosaic à laquelle on passe nos options pour générer une mosaic
        compute_mosaic(args);

        // On défini les chemins d'accès à nos images de test
        let image1_path = "assets/test_integration_x86.png";
        let image2_path = "assets/ground-truth-kit_x86.png";

        // On load l'image que l'on vient de générer avec l'appelle de la fonction précédente
        let result1 = open_image(image1_path);
        // On load l'image de référence pour réaliser le test
        let result2 = open_image(image2_path);

        // On vérifie qu'on a bien load les deux images
        match (result1, result2) {
            // Si c'est le cas on les compare
            (Ok(im1), Ok(im2)) => {
                assert!(im1 == im2);
            }
            // Sinon on dit que le test n'est pas passé
            (Err(_err), _) | (_, Err(_err)) => {
                assert!(false);
            }
        }
    }

    //arm avec SIMD
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_integration_aarch64_simd() {
        // On défini les options nécessaire à la création de notre mosaic
        let args = Options {
                image: "assets/kit.jpeg".to_string(), // Image sur laquelle on veut créer une mosaic
                output: "assets/test_integration_aarch64_simd.png".to_string(), // Image que l'on va générer
                tiles: "assets/images".to_string(), // Dossier où se trouvent les tiles
                scaling: 1, // Factor de grandissement
                tile_size: 25, // Taille des tiles
                remove_used: false,
                verbose: false,
                simd: true,
                num_thread: 1,
            };

        // On appelle la fonction compute_mosaic à laquelle on passe nos options pour générer une mosaic
        compute_mosaic(args);

        // On défini les chemins d'accès à nos images de test
        let image1_path = "assets/test_integration_aarch64_simd.png";
        let image2_path = "assets/ground-truth-kit_aarch64.png";

        // On load l'image que l'on vient de générer avec l'appelle de la fonction précédente
        let result1 = open_image(image1_path);
        // On load l'image de référence pour réaliser le test
        let result2 = open_image(image2_path);

        // On vérifie qu'on a bien load les deux images
        match (result1, result2) {
            // Si c'est le cas on les compare
            (Ok(im1), Ok(im2)) => {
                assert!(im1 == im2);
            }
            // Sinon on dit que le test n'est pas passé
            (Err(_err), _) | (_, Err(_err)) => {
                assert!(false);
            }
        }
    }

    //arm sans SIMD
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_integration_aarch64() {
        // On défini les options nécessaire à la création de notre mosaic
        let args = Options {
                image: "assets/kit.jpeg".to_string(), // Image sur laquelle on veut créer une mosaic
                output: "assets/test_integration_aarch64.png".to_string(), // Image que l'on va générer
                tiles: "assets/images".to_string(), // Dossier où se trouvent les tiles
                scaling: 1, // Factor de grandissement
                tile_size: 25, // Taille des tiles
                remove_used: false,
                verbose: false,
                simd: false,
                num_thread: 1,
            };

        // On appelle la fonction compute_mosaic à laquelle on passe nos options pour générer une mosaic
        compute_mosaic(args);

        // On défini les chemins d'accès à nos images de test
        let image1_path = "assets/test_integration_aarch64.png";
        let image2_path = "assets/ground-truth-kit_aarch64.png";

        // On load l'image que l'on vient de générer avec l'appelle de la fonction précédente
        let result1 = open_image(image1_path);
        // On load l'image de référence pour réaliser le test
        let result2 = open_image(image2_path);

        // On vérifie qu'on a bien load les deux images
        match (result1, result2) {
            // Si c'est le cas on les compare
            (Ok(im1), Ok(im2)) => {
                assert!(im1 == im2);
            }
            // Sinon on dit que le test n'est pas passé
            (Err(_err), _) | (_, Err(_err)) => {
                assert!(false);
            }
        }
    }
}
