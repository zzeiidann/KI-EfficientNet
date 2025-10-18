use std::time::Instant;
use anyhow::Result;
use rand::seq::SliceRandom;
use std::{fs, collections::HashMap};
use tch::{nn, nn::ModuleT, nn::OptimizerConfig, Device, Kind, Tensor};
use image::{imageops, DynamicImage};
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};

const IMG_SIZE: u32 = 224;          // Untuk lebih cepat, bisa 160/128
const EPOCHS: i64 = 50;
const BATCH_SIZE: usize = 8;        // Aman untuk RAM kecil; naikin kalau kuat
const LEARNING_RATE: f64 = 0.0001;

// ==== SCALING PARAM ====
const WIDTH_MULT: f64 = 1.0;
const DEPTH_MULT: f64 = 1.0;

// ==== MODEL SAVE PATH ====
const MODEL_DIR: &str = "../models";
const MODEL_PATH: &str = "../models/best_efficientnet_b5.safetensors";

// ====== UTIL ======
fn round_channels(c: i64, wm: f64) -> i64 {
    let mut scaled = (c as f64 * wm).round() as i64;
    let divisor = 8;
    scaled = (scaled + divisor / 2) / divisor * divisor;
    scaled = scaled.max(divisor);
    if scaled < (0.9 * c as f64 * wm) as i64 {
        scaled + divisor
    } else {
        scaled
    }
}

fn round_repeats(r: i64, dm: f64) -> i64 {
    let v = (r as f64 * dm).round() as i64;
    v.max(1)
}

fn normalize_tensor(mut tensor: Tensor) -> Tensor {
    tensor = tensor / 255.0;
    (tensor - 0.5) / 0.5
}

fn augment_image(img: DynamicImage, is_training: bool) -> DynamicImage {
    if !is_training { return img; }
    let mut rng = rand::thread_rng();
    let mut img = img;

    if rand::random::<bool>() { img = img.flipv(); }
    if rand::random::<bool>() { img = img.fliph(); }

    let rotations = [0, 90, 180, 270];
    let angle = *rotations.choose(&mut rng).unwrap();
    img = match angle {
        0 => img,
        90 => imageops::rotate90(&img).into(),
        180 => imageops::rotate180(&img).into(),
        270 => imageops::rotate270(&img).into(),
        _ => img,
    };

    let brightness: f32 = rand::random::<f32>() * 0.2 - 0.1;
    let contrast: f32 = rand::random::<f32>() * 0.2 + 0.9;
    img.adjust_contrast(contrast).brighten((brightness * 255.0) as i32)
}

// ====== DATA LOADING (LABEL MAPPING KONSISTEN) ======
fn list_classes(base: &str) -> Result<Vec<String>> {
    let mut classes: Vec<String> = fs::read_dir(format!("{}/train", base))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .filter_map(|p| p.file_name().and_then(|s| s.to_str()).map(|s| s.to_string()))
        .collect();
    classes.sort(); // penting: urutkan agar konsisten
    Ok(classes)
}

fn load_split_dataset_with_classes(
    base: &str,
    split: &str,
    classes: &[String],
    is_training: bool
) -> Result<(Vec<Tensor>, Vec<i64>)> {
    let mut all_images = Vec::new();
    let mut all_labels = Vec::new();

    let split_path = format!("{}/{}", base, split);
    if !std::path::Path::new(&split_path).exists() {
        println!("⚠ Split '{}' not found at: {}", split, split_path);
        return Ok((all_images, all_labels));
    }

    let index_map: HashMap<&str, i64> = classes
        .iter()
        .enumerate()
        .map(|(i, c)| (c.as_str(), i as i64))
        .collect();

    println!("\n Loading {} dataset...", split.to_uppercase());
    let mut class_entries: Vec<_> = fs::read_dir(&split_path)?.filter_map(|e| e.ok()).collect();
    class_entries.sort_by_key(|e| e.file_name());

    for entry in class_entries {
        let class_dir = entry.path();
        if !class_dir.is_dir() { continue; }

        let class_name = match class_dir.file_name().and_then(|s| s.to_str()) {
            Some(s) => s,
            None => continue,
        };

        let Some(&class_idx) = index_map.get(class_name) else {
            eprintln!("  ⚠ Kelas '{}' tidak ada di daftar global. Lewati.", class_name);
            continue;
        };

        let mut img_paths: Vec<_> = fs::read_dir(&class_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_file())
            .collect();
        img_paths.sort();

        println!("  Class {}: {} ({} images)", class_idx, class_name, img_paths.len());

        let results: Vec<_> = img_paths.par_iter().filter_map(|img_path| {
            match image::open(img_path) {
                Ok(img) => {
                    let img = augment_image(img, is_training);
                    let img = img.resize_exact(IMG_SIZE, IMG_SIZE, imageops::FilterType::Triangle);
                    let img_rgb = img.to_rgb8();
                    let raw = img_rgb.into_raw();
                    let tensor = Tensor::from_slice(&raw)
                        .reshape(&[IMG_SIZE as i64, IMG_SIZE as i64, 3])
                        .to_kind(Kind::Uint8);
                    let tensor = normalize_tensor(tensor.permute(&[2, 0, 1]).to_kind(Kind::Float));
                    Some((tensor, class_idx))
                }
                Err(e) => {
                    eprintln!("  ⚠ Failed to load {:?}: {}", img_path, e);
                    None
                }
            }
        }).collect();

        for (tensor, label) in results {
            all_images.push(tensor);
            all_labels.push(label);
        }
    }

    println!("✓ Loaded {} images\n", all_images.len());
    Ok((all_images, all_labels))
}

// ===== MODEL =====
// Blok MBConv sebagai struct (hindari lifetime/borrow ke Path)
#[derive(Debug)]
struct MBConv {
    conv_expand: Option<nn::Conv2D>,
    bn_expand:   Option<nn::BatchNorm>,
    conv_dw:     nn::Conv2D,
    bn_dw:       nn::BatchNorm,
    se_reduce:   nn::Linear,
    se_expand:   nn::Linear,
    conv_project: nn::Conv2D,
    bn_project:  nn::BatchNorm,
    stride:      i64,
    out_c:       i64,
    drop_connect: f64,
}

impl MBConv {
    fn new(
        vs: &nn::Path,
        in_c: i64,
        out_c: i64,
        expand_ratio: i64,
        kernel: i64,
        stride: i64,
        drop_connect: f64,
    ) -> Self {
        let exp_c = in_c * expand_ratio;

        let conv_expand = if expand_ratio != 1 {
            Some(nn::conv2d(vs, in_c, exp_c, 1, Default::default()))
        } else { None };
        let bn_expand = if expand_ratio != 1 {
            Some(nn::batch_norm2d(vs, exp_c, Default::default()))
        } else { None };

        let conv_dw = nn::conv2d(
            vs, exp_c, exp_c, kernel,
            nn::ConvConfig { padding: (kernel / 2) as i64, stride, groups: exp_c, ..Default::default() }
        );
        let bn_dw = nn::batch_norm2d(vs, exp_c, Default::default());

        let se_reduce = nn::linear(vs, exp_c, (exp_c / 4).max(1), Default::default());
        let se_expand = nn::linear(vs, (exp_c / 4).max(1), exp_c, Default::default());

        let conv_project = nn::conv2d(vs, exp_c, out_c, 1, Default::default());
        let bn_project = nn::batch_norm2d(vs, out_c, Default::default());

        Self {
            conv_expand,
            bn_expand,
            conv_dw,
            bn_dw,
            se_reduce,
            se_expand,
            conv_project,
            bn_project,
            stride,
            out_c,
            drop_connect,
        }
    }
}

impl nn::ModuleT for MBConv {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let mut x = xs.shallow_clone();
        if let (Some(conv_e), Some(bn_e)) = (&self.conv_expand, &self.bn_expand) {
            x = conv_e.forward_t(&x, train);
            x = bn_e.forward_t(&x, train).silu();
        }
        x = self.conv_dw.forward_t(&x, train);
        x = self.bn_dw.forward_t(&x, train).silu();

        let exp_c = x.size()[1];
        let se = x.adaptive_avg_pool2d(&[1, 1]).view([-1, exp_c]);
        let se = self.se_reduce.forward_t(&se, train).silu();
        let se = self.se_expand.forward_t(&se, train).sigmoid().view([-1, exp_c, 1, 1]);
        x = &x * &se;

        x = self.conv_project.forward_t(&x, train);
        x = self.bn_project.forward_t(&x, train);

        if self.stride == 1 && xs.size()[1] == self.out_c {
            let mut out = xs + x;
            if train && self.drop_connect > 0.0 {
                let keep = 1.0 - self.drop_connect;
                let mask = Tensor::rand_like(&out).ge(keep).to_kind(Kind::Float);
                out = (&out / keep) * (1.0 - mask);
            }
            return out;
        }
        x
    }
}

fn efficientnet_b5_scaled(vs: &nn::Path, num_classes: i64, wm: f64, dm: f64) -> nn::SequentialT {
    // Konfigurasi B5 ringkas (expand, out_c, repeats, stride, kernel)
    let base: &[(i64, i64, i64, i64, i64)] = &[
        (1, 24, 2, 1, 3),
        (6, 40, 4, 2, 3),
        (6, 64, 4, 2, 5),
        (6, 128, 6, 2, 3),
        (6, 176, 6, 1, 5),
        (6, 304, 8, 2, 5),
        (6, 512, 2, 1, 3),
    ];

    let mut seq = nn::seq_t();
    let stem = round_channels(48, wm);

    seq = seq
        .add(nn::conv2d(vs, 3, stem, 3, nn::ConvConfig { stride: 2, padding: 1, ..Default::default() }))
        .add(nn::batch_norm2d(vs, stem, Default::default()))
        .add_fn(|x| x.silu());

    let mut in_c = stem;
    for (expand, c, repeats, stride, kernel) in base.iter().copied() {
        let c = round_channels(c, wm);
        let rep = round_repeats(repeats, dm);
        for i in 0..rep {
            let s = if i == 0 { stride } else { 1 };
            let sub = vs.sub(&format!("mb_{}_{}", c, i));
            let mb = MBConv::new(&sub, in_c, c, expand, kernel, s, 0.2);
            seq = seq.add(mb);
            in_c = c;
        }
    }

    let last = round_channels(2048, wm);
    seq = seq
        .add(nn::conv2d(vs, in_c, last, 1, Default::default()))
        .add(nn::batch_norm2d(vs, last, Default::default()))
        .add_fn(|x| x.silu())
        .add_fn(|x| x.adaptive_avg_pool2d(&[1, 1]))
        .add_fn_t(|x, train| x.dropout(0.3, train))
        .add_fn(move |x| x.view([-1, last]))
        .add(nn::linear(vs, last, num_classes, Default::default()));
    seq
}

// ===== DATALOADER-STYLE (streaming) =====
fn batch_from_vec(
    images: &[Tensor],
    labels: &[i64],
    idxs: &[usize],
    device: Device
) -> (Tensor, Tensor) {
    let xs: Vec<Tensor> = idxs.iter().map(|&i| images[i].shallow_clone()).collect();
    let ys: Vec<i64>    = idxs.iter().map(|&i| labels[i]).collect();
    (Tensor::stack(&xs, 0).to(device), Tensor::from_slice(&ys).to(device))
}

fn eval_stream(
    model: &impl nn::ModuleT,
    images: &[Tensor],
    labels: &[i64],
    device: Device,
    batch_size: usize
) -> (f64, f64) {
    let mut tot_loss = 0.0;
    let mut tot_correct = 0i64;
    let mut tot_seen = 0i64;

    for chunk in (0..images.len()).collect::<Vec<_>>().chunks(batch_size) {
        let (bxs, bys) = batch_from_vec(images, labels, chunk, device);
        let logits = model.forward_t(&bxs, false);
        tot_loss += logits.cross_entropy_for_logits(&bys).double_value(&[]);
        let preds = logits.argmax(-1, false);
        tot_correct += preds.eq_tensor(&bys).sum(Kind::Int64).int64_value(&[]);
        tot_seen += bys.size()[0];
    }
    let acc = if tot_seen > 0 { tot_correct as f64 / tot_seen as f64 } else { 0.0 };
    let denom = (tot_seen as f64 / batch_size as f64).max(1.0);
    let loss = if tot_seen > 0 { tot_loss / denom } else { 0.0 };
    (loss, acc)
}

// ===== MAIN =====
fn main() -> Result<()> {
    let start_time = Instant::now();
    tch::set_num_threads(num_cpus::get() as i32);
    tch::set_num_interop_threads(1);
    tch::manual_seed(42);
    
    let device = if tch::Cuda::is_available() {
        Device::Cuda(0)
    } else {
        Device::Cpu
    };
    println!("Using device: {:?}\n", device);
    
    // Daftar kelas global dari TRAIN (urut lexicographically)
    let classes = list_classes("../data")?;
    let num_classes = classes.len() as i64;

    // Load semua split dengan mapping kelas yang sama
    let (train_images, train_labels) = load_split_dataset_with_classes("../data", "train", &classes, true)?;
    let (valid_images, valid_labels) = load_split_dataset_with_classes("../data", "valid", &classes, false)?;
    let (test_images,  test_labels ) = load_split_dataset_with_classes("../data", "test",  &classes, false)?;

    println!(" Dataset summary:");
    println!("  Train: {} images", train_images.len());
    println!("  Valid: {} images", valid_images.len());
    println!("  Test:  {} images", test_images.len());
    println!("  Classes ({}): {:?}\n", num_classes, classes);

    fs::create_dir_all(MODEL_DIR)?;
    println!("Model will be saved to: {}\n", MODEL_PATH);

    // ===== TRAINING =====
    {
        let vs = nn::VarStore::new(device);
        let net = {
            let root = vs.root();
            efficientnet_b5_scaled(&root, num_classes, WIDTH_MULT, DEPTH_MULT)
        };
        let mut opt = nn::Adam::default().build(&vs, LEARNING_RATE)?;

        println!("Starting training...\n");

        let epoch_pb = ProgressBar::new(EPOCHS as u64);
        epoch_pb
            .set_style(
                ProgressStyle::with_template("{spinner:.green} epoch {pos}/{len} {bar:38.cyan/blue} {msg}")
                    .unwrap()
                    .progress_chars("##-"),
            );

        for epoch in 1..=EPOCHS {
            let mut indices: Vec<usize> = (0..train_images.len()).collect();
            indices.shuffle(&mut rand::thread_rng());

            let total_samples = indices.len() as u64;
            let batch_pb = ProgressBar::new(total_samples);
            batch_pb.set_style(
                ProgressStyle::with_template("  [ep {prefix}] {elapsed_precise} {bar:38.green} {pos}/{len} • {per_sec} • eta {eta_precise} • {msg}")
                    .unwrap()
                    .progress_chars("#>-"),
            );
            batch_pb.set_prefix(epoch.to_string());

            let mut train_loss_sum = 0.0;
            let mut train_correct = 0i64;
            let mut seen: u64 = 0;

            for chunk in indices.chunks(BATCH_SIZE) {
                let (bxs, bys) = batch_from_vec(&train_images, &train_labels, chunk, device);

                let logits = net.forward_t(&bxs, true);
                let loss = logits.cross_entropy_for_logits(&bys);
                opt.backward_step(&loss);

                train_loss_sum += loss.double_value(&[]);
                let preds = logits.argmax(-1, false);
                let batch_correct = preds.eq_tensor(&bys).sum(Kind::Int64).int64_value(&[]) as u64;

                let bs = bys.size()[0] as u64;
                seen += bs;
                train_correct += batch_correct as i64;

                let running_acc  = (train_correct as f64) / (seen as f64) * 100.0;
                let running_loss = train_loss_sum / (seen as f64 / BATCH_SIZE as f64);
                batch_pb.set_message(format!("loss {running_loss:.4} • acc {running_acc:.2}%"));
                batch_pb.inc(bs);
            }
            batch_pb.finish_and_clear();

            let (valid_loss, valid_acc) = eval_stream(&net, &valid_images, &valid_labels, device, BATCH_SIZE);
            let train_loss = train_loss_sum / (seen as f64 / BATCH_SIZE as f64);
            let train_acc = (train_correct as f64) / (seen as f64);
            epoch_pb.set_message(format!("train_loss {train_loss:.4} • train_acc {:.2}% • val_loss {valid_loss:.4} • val_acc {:.2}%", train_acc * 100.0, valid_acc * 100.0));
            epoch_pb.inc(1);

        }
        epoch_pb.finish_with_message("Training Done");

        vs.save(MODEL_PATH)?;
        epoch_pb.println(format!("Saved model"));

        drop(opt);
        drop(net);
        drop(vs);
    }

    // ===== TEST / EVALUATION =====
    println!("\n Evaluating on test set...");
    let mut vs = nn::VarStore::new(device);
    let net = {
        let root = vs.root();
        efficientnet_b5_scaled(&root, num_classes, WIDTH_MULT, DEPTH_MULT)
    };
    // Tidak ada borrow aktif terhadap `vs` saat load
    vs.load(MODEL_PATH)?;

    let (test_loss, test_acc) = eval_stream(&net, &test_images, &test_labels, device, BATCH_SIZE);
    println!("✓ Test Loss: {:.4} | Test Accuracy: {:.2}%", test_loss, test_acc * 100.0);

    println!("\n⏱ Total time: {:.2}s", start_time.elapsed().as_secs_f64());
    Ok(())
}
