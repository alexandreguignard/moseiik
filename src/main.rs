use clap::Parser;
use image::{
    imageops::{resize, FilterType::Nearest},
    io::Reader as ImageReader,
    GenericImage, GenericImageView, RgbImage,
};
use std::time::Instant;
use std::{
    error::Error,
    fs,
    ops::Deref,
    sync::{Arc, Mutex},
};
use threadpool::ThreadPool;
use threadpool_scope::scope_with;

#[derive(Debug, Parser)]
struct Size {
    width: u32,
    height: u32,
}

#[derive(Parser, Debug, Clone)]
#[command(author, version, about, long_about = None)]
pub struct Options {
    /// Location of the target image
    #[arg(short, long)]
    pub image: String,

    /// Saved result location
    #[arg(short, long, default_value_t=String::from("out.png"))]
    pub output: String,

    /// Location of the tiles
    #[arg(short, long)]
    pub tiles: String,

    /// Scaling factor of the image
    #[arg(long, default_value_t = 1)]
    pub scaling: u32,

    /// Size of the tiles
    #[arg(long, default_value_t = 5)]
    pub tile_size: u32,

    /// Remove used tile
    #[arg(short, long)]
    pub remove_used: bool,

    #[arg(short, long)]
    pub verbose: bool,

    /// Use SIMD when available
    #[arg(short, long)]
    pub simd: bool,

    /// Specify number of threads to use, leave blank for default
    #[arg(short, long, default_value_t = 1)]
    pub num_thread: usize,
}

fn count_available_tiles(images_folder: &str) -> i32 {
    match fs::read_dir(images_folder) {
        Ok(t) => return t.count() as i32,
        Err(_) => return -1,
    };
}

fn prepare_tiles(images_folder: &str, tile_size: &Size, verbose: bool) -> Result<Vec<RgbImage>, Box<dyn Error>> {
    let image_paths = fs::read_dir(images_folder)?;
    let tiles = Arc::new(Mutex::new(Vec::new()));
    let now = Instant::now();
    let pool = ThreadPool::new(num_cpus::get());
    let tile_width = tile_size.width;
    let tile_height = tile_size.height;

    for image_path in image_paths {
        let tiles = Arc::clone(&tiles);
        pool.execute(move || {
            let tile_result =
                || -> Result<RgbImage, Box<dyn Error>> { Ok(ImageReader::open(image_path?.path())?.decode()?.into_rgb8()) };

            let tile = match tile_result() {
                Ok(t) => t,
                Err(_) => return,
            };

            let tile = resize(&tile, tile_width, tile_height, Nearest);
            tiles.lock().unwrap().push(tile)
        });
    }
    pool.join();

    println!(
        "\n{} elements in {} seconds",
        tiles.lock().unwrap().len(),
        now.elapsed().as_millis() as f32 / 1000.0
    );

    if verbose {
        println!("");
    }
    let res = tiles.lock().unwrap().deref().to_owned();
    return Ok(res);
}

fn l1_generic(im1: &RgbImage, im2: &RgbImage) -> i32 {
    im1.iter()
        .zip(im2.iter())
        .fold(0, |res, (a, b)| res + i32::abs((*a as i32) - (*b as i32)))
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse2")]
unsafe fn l1_x86_sse2(im1: &RgbImage, im2: &RgbImage) -> i32 {
    // Only works if data is 16 bytes-aligned, which should be the case.
    // In case of crash due to unaligned data, swap _mm_load_si128 for _mm_loadu_si128.
    use std::arch::x86_64::{
        __m128i,
        _mm_extract_epi16, //SSE2
        _mm_load_si128,    //SSE2
        _mm_sad_epu8,      //SSE2
    };

    let stride = 128 / 8;

    let tile_size = im1.width() * im1.height();
    let nb_sub_pixel = tile_size * 3;

    let im1 = im1.as_raw();
    let im2 = im2.as_raw();

    let mut result: i32 = 0;

    for i in (0..nb_sub_pixel - stride).step_by(stride as usize) {
        // Get pointer to data
        let p_im1: *const __m128i = std::mem::transmute::<*const u8, *const __m128i>(std::ptr::addr_of!(im1[i as usize]));
        let p_im2: *const __m128i = std::mem::transmute::<*const u8, *const __m128i>(std::ptr::addr_of!(im2[i as usize]));

        // Load data to xmm
        let xmm_p1 = _mm_load_si128(p_im1);
        let xmm_p2 = _mm_load_si128(p_im2);

        // Do abs(a-b) and horizontal add, results are stored in lower 16 bits of each 64 bits groups
        let xmm_sub_abs = _mm_sad_epu8(xmm_p1, xmm_p2);

        let res_0 = _mm_extract_epi16(xmm_sub_abs, 0);
        let res_1 = _mm_extract_epi16(xmm_sub_abs, 4);

        result += res_0 + res_1; // + res_2 + res_3;
    }

    // now do the remainder manually
    let remainder = nb_sub_pixel % stride as u32;
    for i in nb_sub_pixel - remainder..nb_sub_pixel {
        let p1: u8 = im1[i as usize];
        let p2: u8 = im2[i as usize];

        result += i32::abs((p1 as i32) - (p2 as i32));
    }

    return result;
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn l1_neon(im1: &RgbImage, im2: &RgbImage) -> i32 {
    use std::arch::aarch64::uint8x16_t;
    use std::arch::aarch64::vabdq_u8; // Absolute subtract
    use std::arch::aarch64::vaddlvq_u8; // horizontal add
    use std::arch::aarch64::vld1q_u8; // Load instruction

    let stride = 128 / 8;

    let tile_size = im1.width() * im1.height();
    let nb_sub_pixel = tile_size * 3;

    let im1 = im1.as_raw();
    let im2 = im2.as_raw();

    let mut result: i32 = 0;

    for i in (0..nb_sub_pixel - stride).step_by(stride as usize) {
        // get pointer to data
        let p_im1: *const u8 = std::ptr::addr_of!(im1[i as usize]);
        let p_im2: *const u8 = std::ptr::addr_of!(im2[i as usize]);

        // load data to xmm
        let xmm1: uint8x16_t = vld1q_u8(p_im1);
        let xmm2: uint8x16_t = vld1q_u8(p_im2);

        // get absolute difference
        let xmm_abs_diff: uint8x16_t = vabdq_u8(xmm1, xmm2);

        // reduce with horizontal add
        result += vaddlvq_u8(xmm_abs_diff) as i32;
    }

    // now do the remainder manually
    let remainder = nb_sub_pixel % stride as u32;
    for i in nb_sub_pixel - remainder..nb_sub_pixel {
        let p1: u8 = im1[i as usize];
        let p2: u8 = im2[i as usize];

        result += i32::abs((p1 as i32) - (p2 as i32));
    }

    return result;
}

unsafe fn get_optimal_l1(simd_flag: bool, verbose: bool) -> unsafe fn(&RgbImage, &RgbImage) -> i32 {
    static mut FN_POINTER: unsafe fn(&RgbImage, &RgbImage) -> i32 = l1_generic;

    static INIT: std::sync::Once = std::sync::Once::new();

    INIT.call_once(|| {
        if simd_flag {
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if is_x86_feature_detected!("sse2") {
                    if verbose {
                        println!("{}[2K\rUsing SSE2 SIMD.", 27 as char);
                    }
                    FN_POINTER = l1_x86_sse2;
                } else {
                    if verbose {
                        println!("{}[2K\rNot using SIMD.", 27 as char);
                    }
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                use std::arch::is_aarch64_feature_detected;
                if is_aarch64_feature_detected!("neon") {
                    if verbose {
                        println!("{}[2K\rUsing NEON SIMD.", 27 as char);
                    }
                    FN_POINTER = l1_neon;
                } else {
                    if verbose {
                        println!("{}[2K\rNot using SIMD.", 27 as char);
                    }
                }
            }
        }
    });

    return FN_POINTER;
}

fn l1(im1: &RgbImage, im2: &RgbImage, simd_flag: bool, verbose: bool) -> i32 {
    return unsafe { get_optimal_l1(simd_flag, verbose)(im1, im2) };
}

fn prepare_target(image_path: &str, scale: u32, tile_size: &Size) -> Result<RgbImage, Box<dyn Error>> {
    let target = ImageReader::open(image_path)?.decode()?.into_rgb8();
    let width = target.width();
    let height = target.height();
    let target = target
        .view(0, 0, width - width % tile_size.width, height - height % tile_size.height)
        .to_image();
    Ok(resize(&target, target.width() * scale, target.height() * scale, Nearest))
}

fn find_best_tile(target: &RgbImage, tiles: &Vec<RgbImage>, simd: bool, verbose: bool) -> usize {
    let mut index_best_tile = 0;
    let mut min_error = i32::MAX;
    for (i, tile) in tiles.iter().enumerate() {
        let error = l1(tile, &target, simd, verbose);
        if error < min_error {
            min_error = error;
            index_best_tile = i;
        }
    }
    return index_best_tile;
}

pub fn compute_mosaic(args: Options) {
    let tile_size = Size {
        width: args.tile_size,
        height: args.tile_size,
    };

    let (target_size, target) = match prepare_target(&args.image, args.scaling, &tile_size) {
        Ok(t) => (
            Size {
                width: t.width(),
                height: t.height(),
            },
            Arc::new(Mutex::new(t)),
        ),
        Err(e) => panic!("Error opening {}. {}", args.image, e),
    };

    let nb_available_tiles = count_available_tiles(&args.tiles);
    let nb_required_tiles: i32 = ((target_size.width / tile_size.width) * (target_size.height / tile_size.height)) as i32;
    if args.remove_used && nb_required_tiles > nb_available_tiles {
        panic!("{} tiles required, found {}.", nb_required_tiles, nb_available_tiles)
    }

    let tiles = &prepare_tiles(&args.tiles, &tile_size, args.verbose).unwrap();
    if args.verbose {
        println!("w: {}, h: {}", target_size.width, target_size.height);
    }

    let now = Instant::now();
    let pool = ThreadPool::new(args.num_thread);
    scope_with(&pool, |scope| {
        for w in 0..target_size.width / tile_size.width {
            let target = Arc::clone(&target);
            scope.execute(move || {
                for h in 0..target_size.height / tile_size.height {
                    if args.verbose {
                        print!(
                            "\rBuild image: {} / {} : {} / {}",
                            w + 1,
                            target_size.width / tile_size.width,
                            h + 1,
                            target_size.height / tile_size.height
                        );
                    }

                    // Crop the tile
                    let target_tile = &(target
                        .lock()
                        .unwrap()
                        .view(tile_size.width * w, tile_size.height * h, tile_size.width, tile_size.height)
                        .to_image());

                    let index_best_tile = find_best_tile(&target_tile, &tiles, args.simd, args.verbose);

                    target
                        .lock()
                        .unwrap()
                        .copy_from(&tiles[index_best_tile], w * tile_size.width, h * tile_size.height)
                        .unwrap();
                }
            });
        }
    });
    println!("\n{} seconds", now.elapsed().as_millis() as f32 / 1000.0);
    target.lock().unwrap().save(args.output).unwrap();
}

fn main() {
    let args = Options::parse();
    compute_mosaic(args);
}

#[cfg(test)]
mod tests {
    // Permet d'importer tous les symboles utilisés au début du fichier
    use super::*;


    //Création d'une fonction permettant d'ouvrir une image
    fn open_image(image_path: &str) -> Result<RgbImage, Box<dyn Error>> {
        // On doit recréer une variable car image_path n'est pas mutable
        let path = image_path;
        // Ouverture de l'image
        let image = ImageReader::open(&path)?.decode()?.into_rgb8();
        Ok(image)
    }
    //TEST 1 
    //Test de la norme l1 sur x86 avec la même image, on s'attend à obtenir 0
    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_l1_x86_sse2_0() {
        // On load notre image
        let image_path = "assets/tiles-small/tile-1.png";
        let result = open_image(image_path);
        match result {
            Ok(target) => {
                unsafe {
                    assert_eq!(l1_x86_sse2(&target, &target), 0);
                }
            }
            Err(_err) => {
                // Une erreur s'est produite
                assert!(false);
            }
        }
    }
    //TEST 2 
    //Test de la norme l1 sur x86 avec deux images différentes
    //Nous avons utilisé un script python permettant le calcul de la norme entre 2 images pour  connaître la valeur attendu
    #[test]
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    fn test_l1_x86_sse2_1() {
        // On load nos deux images
        let image_path1 = "assets/tiles-small/tile-1.png";
        let image_path2 = "assets/tiles-small/tile-2.png";
        let result = open_image(image_path1);
        let result2 = open_image(image_path2);

        // On vérifie que les images ont bien été load
        match (result, result2) {
            (Ok(im1), Ok(im2)) => {
                unsafe {
                    assert_eq!(l1_x86_sse2(&im1, &im2), 2154);//On s'attend à 2154 d'après le scipt python
                }
            }
            (Err(_err), _) | (_, Err(_err)) => {
                // Une erreur s'est produite
                assert!(false);
            }
        }
    }
    //TEST 3 
    //Test de la norme l1 sur Arm avec la même image, on s'attend à obtenir 0
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_l1_neon_0() {
        let image_path = "assets/tiles-small/tile-1.png";
        let result = open_image(image_path);
        match result {
            Ok(target) => {
                unsafe {
                    assert_eq!(l1_neon(&target, &target), 0);
                }
            }
            Err(_err) => {
                // Une erreur s'est produite
                assert!(false);
            }
        }
    }
    //TEST 4 
    //Test de la norme l1 sur Arm avec deux images différentes
    //Nous avons utilisé un script python permettant le calcul de la norme entre 2 images pour  connaître la valeur attendu
    #[test]
    #[cfg(target_arch = "aarch64")]
    fn test_l1_neon_1() {
        let image_path1 = "assets/tiles-small/tile-1.png";
        let image_path2 = "assets/tiles-small/tile-2.png";
        let result = open_image(image_path1);
        let result2 = open_image(image_path2);
        match (result, result2) {
            (Ok(im1), Ok(im2)) => {
                unsafe {
                    assert_eq!(l1_neon(&im1, &im2), 2154);//On s'attend à 2154 d'après le scipt python
                }
            }
            (Err(_err), _) | (_, Err(_err)) => {
                // Une erreur s'est produite
                assert!(false);
            }
        }
    }
    //TEST 5 
    //Test de la norme l1 sur n'importe quelle plateforme avec la même image, on s'attend à obtenir 0
    #[test]
    fn test_l1_generic_0() {
        let image_path = "assets/tiles-small/tile-3.png";
        let result = open_image(image_path);
        match result {
            Ok(target) => {
                assert_eq!(l1_generic(&target, &target), 0);  
            }
            Err(_err) => {
                // Une erreur s'est produite
                assert!(false);
            }
        }
    }
    //TEST 6 
    //Test de la norme l1 sur n'importe quelle plateforme avec deux images différentes
    //Nous avons utilisé un script python permettant le calcul de la norme entre 2 images pour  connaître la valeur attendu
    #[test]
    fn test_l1_generic_1() {
        let image_path1 = "assets/tiles-small/tile-2.png";
        let image_path2 = "assets/tiles-small/tile-4.png";
        let result = open_image(image_path1);
        let result2 = open_image(image_path2);
        match (result, result2) {
            (Ok(im1), Ok(im2)) => {
                assert_eq!(l1_generic(&im1, &im2), 2253);//On s'attend à 2253 d'après le scipt python
            }
            (Err(_err), _) | (_, Err(_err)) => {
                // Une erreur s'est produite
                assert!(false);
            }
        }
    }
    //TEST 7 
    //Test de la fonction prepare_target sur n'importe quelle plateforme
    //Scaling fixé à 5 et tile de base fixé à 5 aussi donc on s'attend à une image de 25 en sortie
    #[test]
    fn test_prepare_target() {
        let image_path1 = "assets/tiles-small/tile-1.png";
        let tile_size = Size {
            width: 5,
            height: 5,
        };

        let target = match prepare_target(image_path1, 5, &tile_size) {
            Ok(t) => {
                assert_eq!(t.width(), 25); // scaling de 5 donc on s'attend à avoir 5x la taille d'origine (5*5=25)
                assert_eq!(t.height(), 25);
            }
            Err(_e) => assert!(false),
        };
    }
    //TEST 8 
    //Test de la fonction prepare_tile sur n'importe quelle plateforme
    //Tile de base en 5x5, taille demandé 30x20 on test donc que la taille en sortie est bien celle demandé
    #[test]
    fn test_prepare_tiles() {
        let folder_path = "assets/tiles-small";
        let tile_size = Size {	//Taille demandée
            width: 30,
            height: 20,
        };

        match prepare_tiles(folder_path, &tile_size, false) {
            Ok(vecteur) => {
                for tile in vecteur {
                    assert_eq!(tile.width(), 30); //Test taille en sortie sur chaque tiles
                    assert_eq!(tile.height(), 20);
                }
            }
            Err(_e) => {
                 // Une erreur s'est produite
                assert!(false);
            }
        }
    }
}

