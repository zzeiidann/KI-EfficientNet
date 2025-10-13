use std::time::Instant;
use anyhow::Result;
use rand::seq::SliceRandom;
use std::fs;
use tch::{nn, nn::ModuleT, nn::OptimizerConfig, Device, Kind, Tensor};
use tch::IndexOp;
use image::{imageops, DynamicImage};
use rayon::prelude::*;
use indicatif::{ProgressBar, ProgressStyle};

const IMG_SIZE: u32 = 224;   // B5 asli biasanya 456
const EPOCHS: i64 = 10;
const BATCH_SIZE: usize = 32;  // Sesuaikan RAM
const LEARNING_RATE: f64 = 0.001;

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

fn load_split_dataset(
    base: &str,
    split: &str,
    is_training: bool
) -> Result<(Vec<Tensor>, Vec<i64>, Vec<String>)> {
    let mut classes = Vec::new();
    let mut all_images = Vec::new();
    let mut all_labels = Vec::new();

    let split_path = format!("{}/{}", base, split);
    if !std::path::Path::new(&split_path).exists() {
        println!("⚠ Split '{}' not found at: {}", split, split_path);
        return Ok((all_images, all_labels, classes));
    }

    println!("\n Loading {} dataset...", split.to_uppercase());

    // Ambil semua entri, tapi hanya proses yang direktori agar aman dari file nyasar
    let entries: Vec<_> = fs::read_dir(&split_path)?
        .filter_map(|e| e.ok())
        .collect();

    for (idx, entry) in entries.iter().enumerate() {
        let class_dir = entry.path();
        if !class_dir.is_dir() {
            eprintln!("  ⚠ Skip non-directory in {}: {:?}", split, class_dir.file_name());
            continue;
        }

        let class_name = class_dir.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        classes.push(class_name.clone());

        let img_paths: Vec<_> = fs::read_dir(&class_dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.is_file())
            .collect();

        println!("  Class {}: {} ({} images)", idx, class_name, img_paths.len());

        let results: Vec<_> = img_paths.par_iter().filter_map(|img_path| {
            match image::open(img_path) {
                Ok(img) => {
                    let img = augment_image(img, is_training);
                    let img = img.resize_exact(IMG_SIZE, IMG_SIZE, imageops::FilterType::Lanczos3);
                    let img_rgb = img.to_rgb8();
                    let raw = img_rgb.into_raw();
                    let tensor = Tensor::from_slice(&raw)
                        .reshape(&[IMG_SIZE as i64, IMG_SIZE as i64, 3])
                        .to_kind(Kind::Uint8);
                    let tensor = normalize_tensor(
                        tensor.permute(&[2, 0, 1]).to_kind(Kind::Float)
                    );
                    Some((tensor, idx as i64))
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

    println!("✓ Loaded {} images from {} classes\n", all_images.len(), classes.len());
    Ok((all_images, all_labels, classes))
}

// ------ MODEL ------

// Rust 2024: gunakan `+ use<>` agar impl tidak mengikat lifetime &nn::Path ke 'static.
fn mbconv_block(
    vs: &nn::Path,
    in_c: i64,
    out_c: i64,
    expand_ratio: i64,
    kernel: i64,
    stride: i64
) -> impl nn::ModuleT + use<> {
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

    nn::func_t(move |xs, train| {
        let mut x = xs.shallow_clone();
        if expand_ratio != 1 {
            x = conv_expand.as_ref().unwrap().forward_t(&x, train);
            x = bn_expand.as_ref().unwrap().forward_t(&x, train).silu();
        }
        x = conv_dw.forward_t(&x, train);
        x = bn_dw.forward_t(&x, train).silu();

        let se = x.adaptive_avg_pool2d(&[1, 1]).view([-1, exp_c]);
        let se = se_reduce.forward_t(&se, train).silu();
        let se = se_expand.forward_t(&se, train).sigmoid().view([-1, exp_c, 1, 1]);
        x = &x * &se;

        x = conv_project.forward_t(&x, train);
        x = bn_project.forward_t(&x, train);

        if stride == 1 && xs.size()[1] == out_c {
            return (xs + x).to_kind(Kind::Float);
        }
        x
    })
}

fn efficientnet_b5(vs: &nn::Path, num_classes: i64) -> impl nn::ModuleT + use<> {
    // (expand, out_c, repeats, stride, kernel)
    let cfg: &[(i64, i64, i64, i64, i64)] = &[
        (1, 24, 2, 1, 3),
        (6, 40, 4, 2, 3),
        (6, 64, 4, 2, 5),
        (6, 128, 6, 2, 3),
        (6, 176, 6, 1, 5),
        (6, 304, 8, 2, 5),
        (6, 512, 2, 1, 3),
    ];

    let mut seq = nn::seq_t();
    let out_channels = 48;

    seq = seq
        .add(nn::conv2d(vs, 3, out_channels, 3, nn::ConvConfig { stride: 2, padding: 1, ..Default::default() }))
        .add(nn::batch_norm2d(vs, out_channels, Default::default()))
        .add_fn(|x| x.silu());

    let mut in_c = out_channels;
    for (expand, c, repeats, stride, kernel) in cfg.iter() {
        let c = *c;
        for i in 0..*repeats {
            let s = if i == 0 { *stride } else { 1 };
            let sub = vs.sub(&format!("mb_{}_{}", c, i));
            let mb = mbconv_block(&sub, in_c, c, *expand, *kernel, s);
            seq = seq.add(mb);
            in_c = c;
        }
    }

    let last_channels = 2048;
    seq = seq
        .add(nn::conv2d(vs, in_c, last_channels, 1, Default::default()))
        .add(nn::batch_norm2d(vs, last_channels, Default::default()))
        .add_fn(|x| x.silu())
        .add_fn(|x| x.adaptive_avg_pool2d(&[1, 1]))
        .add_fn_t(|x, train| x.dropout(0.4, train))
        .add_fn(move |x| x.view([-1, last_channels]))
        .add(nn::linear(vs, last_channels, num_classes, Default::default()));
    seq
}

// ------ UTIL ------

fn shuffle_dataset(xs: &Tensor, ys: &Tensor) -> (Tensor, Tensor) {
    let n = xs.size()[0] as i64;
    let idx = Tensor::randperm(n, (Kind::Int64, xs.device()));
    (xs.index_select(0, &idx), ys.index_select(0, &idx))
}

fn batches(xs: &Tensor, ys: &Tensor, batch_size: usize) -> Vec<(Tensor, Tensor)> {
    let n = xs.size()[0] as usize;
    (0..n).step_by(batch_size).map(|i| {
        let end = (i + batch_size).min(n);
        (xs.i(i as i64..end as i64), ys.i(i as i64..end as i64))
    }).collect()
}

fn evaluate(
    model: &impl nn::ModuleT,
    xs: &Tensor,
    ys: &Tensor,
    device: Device
) -> (f64, f64) {
    let xs = xs.to(device);
    let ys = ys.to(device);

    let logits = model.forward_t(&xs, false);
    let loss = logits.cross_entropy_for_logits(&ys).double_value(&[]);
    let preds = logits.argmax(-1, false);
    let accuracy = preds
        .eq_tensor(&ys)
        .to_kind(Kind::Float)
        .mean(Kind::Float)
        .double_value(&[]);
    (loss, accuracy)
}

// ------ MAIN ------

fn main() -> Result<()> {
    let start_time = Instant::now();
    tch::set_num_threads(num_cpus::get() as i32);
    tch::set_num_interop_threads(1);
    let device = Device::Cpu; // mac ARM: CPU (no CUDA)
    println!(" Using device: {:?}", device);

    // Load datasets
    let (train_images, train_labels, classes) =
        load_split_dataset("../data", "train", true)?;
    let (valid_images, valid_labels, _) =
        load_split_dataset("../data", "valid", false)?;
    let (test_images, test_labels, _) =
        load_split_dataset("../data", "test", false)?;

    let num_classes = classes.len() as i64;
    println!("Dataset summary:");
    println!("  Train: {} images", train_images.len());
    println!("  Valid: {} images", valid_images.len());
    println!("  Test:  {} images", test_images.len());
    println!("  Classes: {} {:?}\n", num_classes, classes);

    // Tensors
    let train_xs = Tensor::stack(&train_images, 0).to(device);
    let train_ys = Tensor::from_slice(&train_labels).to(device);
    let valid_xs = Tensor::stack(&valid_images, 0).to(device);
    let valid_ys = Tensor::from_slice(&valid_labels).to(device);
    let test_xs  = Tensor::stack(&test_images, 0).to(device);
    let test_ys  = Tensor::from_slice(&test_labels).to(device);

    // VarStore
    let mut vs = nn::VarStore::new(device);

    // ---- TRAINING SCOPE ----
    {
        let root = vs.root();
        let net = efficientnet_b5(&root, num_classes);
        let mut opt = nn::Adam::default().build(&vs, LEARNING_RATE)?;

        println!(" Starting training...\n");

        // Progress bar untuk epoch
        let epoch_pb = ProgressBar::new(EPOCHS as u64);
        epoch_pb.set_style(
            ProgressStyle::with_template("{spinner:.green} epoch {pos}/{len} {bar:40.cyan/blue} {msg}")
                .unwrap()
                .progress_chars("##-")
        );

        let mut best_valid_acc = 0.0;

        for epoch in 1..=EPOCHS {
            // Shuffle training tensors
            let (train_xs_shuffled, train_ys_shuffled) = shuffle_dataset(&train_xs, &train_ys);
            let train_batches = batches(&train_xs_shuffled, &train_ys_shuffled, BATCH_SIZE);

            // Progress bar untuk batch (pakai prefix = nomor epoch)
            let batch_pb = ProgressBar::new(train_batches.len() as u64);
            batch_pb.set_style(
                ProgressStyle::with_template("  [ep {prefix}] {elapsed_precise} {bar:40.green} {pos}/{len} • {msg}")
                    .unwrap()
                    .progress_chars("=>-")
            );
            batch_pb.set_prefix(epoch.to_string());

            let mut train_loss_sum = 0.0;
            let mut train_correct = 0i64;
            let mut seen = 0i64;

            for (bxs, bys) in train_batches.iter() {
                let logits = net.forward_t(bxs, true);
                let loss = logits.cross_entropy_for_logits(bys);
                opt.backward_step(&loss);

                train_loss_sum += loss.double_value(&[]);
                let preds = logits.argmax(-1, false);
                let batch_correct = preds.eq_tensor(bys).sum(Kind::Int64).int64_value(&[]);
                train_correct += batch_correct;
                seen += bys.size()[0];

                let running_acc = (train_correct as f64) / (seen as f64) * 100.0;
                let running_loss = train_loss_sum / (batch_pb.position() as f64 + 1.0);
                batch_pb.set_message(format!("loss {running_loss:.4} • acc {running_acc:.2}%"));
                batch_pb.inc(1);
            }
            batch_pb.finish_and_clear();

            let train_acc = train_correct as f64 / train_xs.size()[0] as f64;
            let train_loss = train_loss_sum / (seen as f64 / BATCH_SIZE as f64);

            // Validation
            let (valid_loss, valid_acc) = evaluate(&net, &valid_xs, &valid_ys, device);
            epoch_pb.set_message(format!(
                "train_loss {train_loss:.4} • train_acc {:.2}% • val_loss {valid_loss:.4} • val_acc {:.2}%",
                train_acc * 100.0, valid_acc * 100.0
            ));
            epoch_pb.inc(1);

            // Save best
            if valid_acc > best_valid_acc {
                best_valid_acc = valid_acc;
                std::fs::create_dir_all("../EfficientNetLoad").ok();
                vs.save("../EfficientNetLoad/best_model.safetensors")?;
                epoch_pb.println(format!("  Saved best model (val acc: {:.2}%)", best_valid_acc * 100.0));
            }
        }
        epoch_pb.finish_with_message("training done");
        // net, root, opt drop di sini
    }
    // ---- END TRAINING SCOPE ----

    // Test
    println!("\n🧪 Evaluating on test set...");
    vs.load("../EfficientNetLoad/best_model.safetensors")?;
    let root = vs.root();
    let net = efficientnet_b5(&root, num_classes);

    let (test_loss, test_acc) = evaluate(&net, &test_xs, &test_ys, device);
    println!("✓ Test Loss: {:.4} | Test Accuracy: {:.2}%", test_loss, test_acc * 100.0);

    println!("\n⏱ Total time: {:.2}s", start_time.elapsed().as_secs_f64());
    Ok(())
}
