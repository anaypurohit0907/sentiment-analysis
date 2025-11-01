//! Centralized tuned parameters and default rules for the neuro-fuzzy system.

use super::neuro_sugeno::{HybridProvider, InferLogits, NeuroRule};

// Tuned hyperparameters (from user's best-found configuration)
pub const NF_TEMPERATURE: f32 = 0.8;
pub const NF_MARGIN_CENTER: f32 = 0.3;
pub const NF_MARGIN_SLOPE: f32 = 5.0;
pub const NF_ENTROPY_CENTER: f32 = 1.0;
pub const NF_ENTROPY_SLOPE: f32 = 6.0;
pub const NF_GAP_CENTER: f32 = 0.2;
pub const NF_GAP_SLOPE: f32 = 4.0;

// Threshold for mapping Sugeno score to {-1,0,+1}
pub const NF_LABEL_THRESHOLD: f32 = 0.33;

// Class order mapping for logits -> [neg, neu, pos]
pub const NF_CLASS_ORDER: [usize; 3] = [0, 1, 2];

// Default rule base used by both demo and evaluator
pub fn default_rules() -> Vec<NeuroRule> {
    vec![
        NeuroRule { antecedent: vec![(2,false),(0,true),(3,false)], consequent:  1.0 },
        NeuroRule { antecedent: vec![(4,false)],                    consequent:  0.0 },
        NeuroRule { antecedent: vec![(0,false),(2,true),(3,false)], consequent: -1.0 },
        NeuroRule { antecedent: vec![(2,false),(3,true)],           consequent:  0.5 },
        NeuroRule { antecedent: vec![(0,false),(3,true)],           consequent: -0.5 },
        NeuroRule { antecedent: vec![(5,false)],                    consequent:  0.5 },
    ]
}

// Construct a HybridProvider with the tuned parameters
pub fn tuned_provider<M: InferLogits>(model: M) -> HybridProvider<M> {
    HybridProvider {
        model,
        temperature: NF_TEMPERATURE,
        class_order: NF_CLASS_ORDER,
        margin_center: NF_MARGIN_CENTER,
        margin_slope: NF_MARGIN_SLOPE,
        entropy_center: NF_ENTROPY_CENTER,
        entropy_slope: NF_ENTROPY_SLOPE,
        gap_center: NF_GAP_CENTER,
        gap_slope: NF_GAP_SLOPE,
    }
}
