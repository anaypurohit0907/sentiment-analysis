// src/nn/io.rs
use std::fs::File;
use std::io::{BufWriter, Write};

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
