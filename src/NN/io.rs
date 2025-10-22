// src/nn/io.rs
use std::fs::File;
use std::io::{BufWriter, Write};
use csv::ReaderBuilder;
use anyhow::{Result, Context};

/// Structure to hold sentiment data
#[derive(Debug, Clone)]
pub struct SentimentData {
    pub texts: Vec<String>,
    pub labels: Vec<f32>,
}

impl SentimentData {
    pub fn new() -> Self {
        Self {
            texts: Vec::new(),
            labels: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.texts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.texts.is_empty()
    }
}

/// Load sentiment data from CSV file
/// Expected format: text,label
pub fn load_sentiment_csv(path: &str) -> Result<SentimentData> {
    let mut reader = ReaderBuilder::new()
        .has_headers(true)
        .from_path(path)
        .context("Failed to open CSV file")?;

    let mut data = SentimentData::new();

    for result in reader.records() {
        let record = result.context("Failed to read CSV record")?;
        
        if record.len() < 2 {
            continue; // Skip malformed rows
        }

        let text = record.get(0)
            .context("Missing text column")?;
        
        let label_str = record.get(1)
            .context("Missing label column")?;
        
        // Skip empty rows
        if text.trim().is_empty() || label_str.trim().is_empty() {
            continue;
        }
        
        let label: f32 = label_str.parse()
            .context(format!("Failed to parse label: {}", label_str))?;

        data.texts.push(text.to_string());
        data.labels.push(label);
    }

    println!("Loaded {} samples from {}", data.len(), path);
    
    // Print class distribution
    let mut label_counts = std::collections::HashMap::new();
    for &label in &data.labels {
        *label_counts.entry(label as i32).or_insert(0) += 1;
    }
    println!("Class distribution: {:?}", label_counts);

    Ok(data)
}

pub fn write_params_to_txt(
    weights_path: &str,
    biases_path: &str,
    weights: &Vec<Vec<Vec<f32>>>,
    biases: &Vec<Vec<f32>>,
) -> std::io::Result<()> {
    {
        let f = File::create(weights_path)?;
        let mut w = BufWriter::new(f);
        for (l, wmat) in weights.iter().enumerate() {
            let rows = wmat.len();
            let cols = if rows == 0 { 0 } else { wmat[0].len() };
            writeln!(w, "# layer {l} weights (rows x cols = {rows} x {cols})")?;
            for row in wmat {
                for (j, v) in row.iter().enumerate() {
                    if j > 0 {
                        write!(w, " ")?;
                    }
                    write!(w, "{:.6}", v)?;
                }
                writeln!(w)?;
            }
            writeln!(w)?;
        }
    }
    {
        let f = File::create(biases_path)?;
        let mut w = BufWriter::new(f);
        for (l, bvec) in biases.iter().enumerate() {
            writeln!(w, "# layer {l} biases (len = {})", bvec.len())?;
            for (j, v) in bvec.iter().enumerate() {
                if j > 0 {
                    write!(w, " ")?;
                }
                write!(w, "{:.6}", v)?;
            }
            writeln!(w)?;
            writeln!(w)?;
        }
    }
    Ok(())
}
