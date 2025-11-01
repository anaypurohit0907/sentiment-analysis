//! Hybrid neuro-fuzzy: NN logits -> probabilities + scalar cues -> Sugeno rules.

#[inline]
pub fn softmax_temp(z: &[f32], temp: f32) -> Vec<f32> {
    let inv_t = 1.0f32 / temp.max(1e-6);
    let scaled: Vec<f32> = z.iter().map(|&v| v * inv_t).collect();
    let max = scaled
        .iter()
        .fold(f32::NEG_INFINITY, |m, &v| if v > m { v } else { m });
    let exps: Vec<f32> = scaled.iter().map(|&v| (v - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum.max(1e-12)).collect()
}

#[inline]
fn sigmoid(x: f32, center: f32, slope: f32) -> f32 {
    // Smooth step in [0,1]
    (slope * (x - center)).tanh() * 0.5 + 0.5
}

#[derive(Clone, Debug)]
pub struct NeuroRule {
    // (index into membership vector, negate?)
    pub antecedent: Vec<(usize, bool)>,
    pub consequent: f32, // Sugeno constant output
}

pub trait InferLogits {
    // Return raw pre-softmax scores for classes, order must match class_order
    fn forward_logits(&self, x: &[f32]) -> Vec<f32>;
}

// Produces a 6-dim membership vector per sample:
// [0]=μ_neg, [1]=μ_neu, [2]=μ_pos,
// [3]=conf_high (margin high), [4]=ambig_high (entropy high),
// [5]=pos_over_neg_high (z_pos - z_neg high)
pub struct HybridProvider<M: InferLogits> {
    pub model: M,
    pub temperature: f32,
    pub class_order: [usize; 3],

    pub margin_center: f32,
    pub margin_slope: f32,
    pub entropy_center: f32,
    pub entropy_slope: f32,
    pub gap_center: f32,
    pub gap_slope: f32,
}

impl<M: InferLogits> HybridProvider<M> {
    pub fn memberships(&self, x: &[f32]) -> Vec<f32> {
        let logits_raw = self.model.forward_logits(x);
        // Reorder into [neg, neu, pos]
        let mut logits = vec![0.0; 3];
        for (i, &ix) in self.class_order.iter().enumerate() {
            logits[i] = *logits_raw.get(ix).unwrap_or(&0.0);
        }

        let probs = softmax_temp(&logits, self.temperature);
        let mu_neg = probs[0];
        let mu_neu = probs[1];
        let mu_pos = probs[2];

        // top-2 margin (using logits)
        let mut sorted = logits.clone();
        sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
        let margin = if sorted.len() >= 2 { sorted[0] - sorted[1] } else { 0.0 };

        // entropy (natural log)
        let entropy = -probs
            .iter()
            .map(|p| if *p > 0.0 { p * p.ln() } else { 0.0 })
            .sum::<f32>();

        let pos_minus_neg = logits[2] - logits[0];

        // Scalar cues -> [0,1]
        let conf_high = sigmoid(margin, self.margin_center, self.margin_slope);
        let ambig_high = sigmoid(entropy, self.entropy_center, self.entropy_slope);
        let pos_over_neg_high = sigmoid(pos_minus_neg, self.gap_center, self.gap_slope);

        vec![mu_neg, mu_neu, mu_pos, conf_high, ambig_high, pos_over_neg_high]
    }
}

pub struct NeuroSugeno<M: InferLogits> {
    pub provider: HybridProvider<M>,
    pub rules: Vec<NeuroRule>,
}

impl<M: InferLogits> NeuroSugeno<M> {
    pub const DEFAULT_LABEL_THRESHOLD: f32 = 0.33;

    pub fn infer_score(&self, x: &[f32]) -> f32 {
        let mu = self.provider.memberships(x);
        let mut num = 0.0f32;
        let mut den = 0.0f32;
        for r in &self.rules {
            let mut w = 1.0f32;
            for (ix, neg) in &r.antecedent {
                let m = *mu.get(*ix).unwrap_or(&0.0);
                w *= if *neg { 1.0 - m } else { m };
            }
            num += w * r.consequent;
            den += w;
        }
        if den > 0.0 { num / den } else { 0.0 }
    }

    pub fn infer_label(&self, x: &[f32]) -> i32 {
        self.infer_label_with_threshold(x, Self::DEFAULT_LABEL_THRESHOLD)
    }

    pub fn infer_label_with_threshold(&self, x: &[f32], thr: f32) -> i32 {
        let y = self.infer_score(x);
        if y > thr { 1 } else if y < -thr { -1 } else { 0 }
    }

    // For introspection: (rule_index, weight, contribution=weight*consequent)
    pub fn rule_strengths(&self, x: &[f32]) -> Vec<(usize, f32, f32)> {
        let mu = self.provider.memberships(x);
        let mut out = Vec::with_capacity(self.rules.len());
        for (i, r) in self.rules.iter().enumerate() {
            let mut w = 1.0f32;
            for (ix, neg) in &r.antecedent {
                let m = *mu.get(*ix).unwrap_or(&0.0);
                w *= if *neg { 1.0 - m } else { m };
            }
            out.push((i, w, w * r.consequent));
        }
        out
    }
}
