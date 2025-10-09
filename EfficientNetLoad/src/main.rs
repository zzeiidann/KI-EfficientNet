use anyhow::Result;
use clap::{Parser, Subcommand};
use ort::{session::Session, value::Value};
use std::path::PathBuf;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Predict using ONNX model
    Predict {
        /// Path to ONNX model file
        #[arg(long)]
        model_path: PathBuf,
        
        /// Path to image or directory
        #[arg(long)]
        input: PathBuf,
        
        /// Class names (comma-separated)
        #[arg(long, default_value = "Autistic,Non_Autistic")]
        classes: String,
    },
}

// Image preprocessing
fn preprocess_image(img_path: &PathBuf, img_size: u32) -> Result<Vec<f32>> {
    let img = image::open(img_path)?;
    let img = img.resize_exact(img_size, img_size, image::imageops::FilterType::Lanczos3);
    let img = img.to_rgb8();
    
    // Create flat vector in NCHW format
    let mut data = Vec::with_capacity(3 * img_size as usize * img_size as usize);
    
    // Channel 0 (Red)
    for pixel in img.pixels() {
        data.push((pixel[0] as f32 / 255.0 - 0.5) / 0.5);
    }
    // Channel 1 (Green)
    for pixel in img.pixels() {
        data.push((pixel[1] as f32 / 255.0 - 0.5) / 0.5);
    }
    // Channel 2 (Blue)
    for pixel in img.pixels() {
        data.push((pixel[2] as f32 / 255.0 - 0.5) / 0.5);
    }
    
    Ok(data)
}

// Softmax function
fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.iter().map(|x| x / sum).collect()
}

// Prediction function
fn predict(model_path: PathBuf, input: PathBuf, classes_str: String) -> Result<()> {
    println!("Loading ONNX model...");
    
    let mut session = Session::builder()?
        .commit_from_file(&model_path)?;
    
    let classes: Vec<String> = classes_str.split(',').map(|s| s.to_string()).collect();
    
    // Process input
    if input.is_file() {
        // Single image prediction
        println!("Processing image: {:?}", input);
        let img_data = preprocess_image(&input, 224)?;
        
        // Create tensor - use tuple format (shape, data)
        let shape = vec![1i64, 3, 224, 224];
        let input_tensor = Value::from_array((shape, img_data.into_boxed_slice()))?;
        
        // Run inference
        let outputs = session.run(ort::inputs!["input" => input_tensor])?;
        
        // Get output
        let output_tensor = &outputs[0];
        let (_shape, logits_data) = output_tensor.try_extract_tensor::<f32>()?;
        
        let probs = softmax(logits_data);
        let pred_class = probs
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();
        
        println!("\n=== Prediction Results ===");
        println!("Image: {:?}", input);
        println!("Predicted Class: {}", classes[pred_class]);
        println!("Confidence: {:.2}%", probs[pred_class] * 100.0);
        println!("\nAll Probabilities:");
        for (idx, class_name) in classes.iter().enumerate() {
            println!("  {}: {:.2}%", class_name, probs[idx] * 100.0);
        }
        
    } else if input.is_dir() {
        // Directory prediction
        let files: Vec<_> = std::fs::read_dir(&input)?
            .filter_map(|e| e.ok())
            .filter(|e| {
                if let Some(ext) = e.path().extension() {
                    matches!(ext.to_str(), Some("jpg") | Some("jpeg") | Some("png"))
                } else {
                    false
                }
            })
            .collect();
        
        println!("\nProcessing {} images...\n", files.len());
        
        for file in files {
            let img_data = preprocess_image(&file.path(), 224)?;
            let shape = vec![1i64, 3, 224, 224];
            let input_tensor = Value::from_array((shape, img_data.into_boxed_slice()))?;
            
            let outputs = session.run(ort::inputs!["input" => input_tensor])?;
            let output_tensor = &outputs[0];
            let (_shape, logits_data) = output_tensor.try_extract_tensor::<f32>()?;
            
            let probs = softmax(logits_data);
            let pred_class = probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            
            println!("{:?} -> {} ({:.2}%)", 
                file.file_name(), 
                classes[pred_class],
                probs[pred_class] * 100.0
            );
        }
    }
    
    Ok(())
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    
    match cli.command {
        Commands::Predict {
            model_path,
            input,
            classes,
        } => predict(model_path, input, classes)?,
    }
    
    Ok(())
}