//! Turn dialog notebooks into model input, and saved replies back into messages.
//!
//! A dialog is a Jupyter notebook in aidialog's format: notes, code, raw cells, and prompts whose replies are saved in
//! the prompt cell's output. `input::build` turns the cells above a prompt into Responses input items, with the
//! examples, CRAFTs and `$` values that a host supplies. `cells` converts one cell, `output` renders code outputs and
//! tool results, `media` prepares images and other media, and `tokens` estimates token counts. `reply` reads and
//! writes the saved reply format.

use serde::Deserialize;

pub mod cells;
pub mod input;
pub mod media;
pub mod output;
pub mod reply;
pub mod tokens;

/// What a model can read, as a model catalog entry or fastllm's `get_model_info` gives it
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Caps {
    /// The kinds of input the model reads, such as `text`, `image`, `pdf` or `video`. fastllm names this list
    /// `supported_modalities`.
    #[serde(alias = "supported_modalities")]
    pub input_modalities: Vec<String>,
    /// The model's input window, in tokens
    pub max_input_tokens: Option<u64>,
}
