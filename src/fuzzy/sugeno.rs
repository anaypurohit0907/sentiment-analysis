//! Minimal Sugeno aggregation helper (placeholder).

#[inline]
pub fn sugeno_weighted_average(weights: &[f32], consequents: &[f32]) -> f32 {
	let mut num = 0.0f32;
	let mut den = 0.0f32;
	for (w, c) in weights.iter().zip(consequents.iter()) {
		num += w * c;
		den += w;
	}
	if den > 0.0 { num / den } else { 0.0 }
}

