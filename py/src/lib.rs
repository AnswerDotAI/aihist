//! The Python extension, `aihist._core`. JSON values cross as strings, which the `aihist` package encodes and decodes.

use aihist::{Caps, cells, input, media, output, reply, tokens};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use serde_json::Value;

fn de<T: serde::de::DeserializeOwned>(s: &str) -> PyResult<T> { serde_json::from_str(s).map_err(|e| PyValueError::new_err(e.to_string())) }
fn ser(v: impl serde::Serialize) -> String { serde_json::to_string(&v).expect("values serialize") }
fn caps(s: &str) -> PyResult<Caps> { de(s) }

/// Responses input items for the prompt that `prompt`, an `input::Prompt` as JSON, describes
#[pyfunction]
fn build(prompt: &str) -> PyResult<String> { Ok(ser(input::build(&de(prompt)?).map_err(PyValueError::new_err)?)) }

/// Cell `cell` of notebook `nb` as a model reads it: its media, then its XML
#[pyfunction]
#[pyo3(signature = (nb, cell, caps_, time=None))]
fn cell_parts(nb: &str, cell: &str, caps_: &str, time: Option<&str>) -> PyResult<String> {
    Ok(ser(cells::parts(&de(nb)?, &de(cell)?, &caps(caps_)?, time)))
}

/// The media that cell `cell` of notebook `nb` shows a model: the attachments it refers to, then a code cell's images
#[pyfunction]
fn cell_media(nb: &str, cell: &str, caps_: &str) -> PyResult<String> {
    let (nb, cell): (Value, Value) = (de(nb)?, de(cell)?);
    Ok(ser(cells::media(cells::stored_atts(&nb, &cell), &cell, &cells::msg(&cell), &caps(caps_)?)))
}

/// Cell `cell` as the XML a model reads
#[pyfunction]
#[pyo3(signature = (cell, time=None))]
fn cell_xml(cell: &str, time: Option<&str>) -> PyResult<String> { let cell = de(cell)?; Ok(cells::xml(&cell, &cells::msg(&cell), time)) }

/// Whether the model sees `cell`
#[pyfunction]
fn shown(cell: &str) -> PyResult<bool> { Ok(cells::shown(&de(cell)?)) }

/// A prompt cell's reply as a model's history holds it; `None` for any other cell
#[pyfunction]
fn cell_reply(cell: &str) -> PyResult<Option<String>> {
    let m = cells::msg(&de(cell)?);
    Ok((m.typ == "prompt").then(|| cells::ai_fmt(&m.out)))
}

/// The sigil references in `cell`, as `{vars, cmds, tools}`
#[pyfunction]
fn cell_refs(cell: &str) -> PyResult<String> {
    let r = rgapi::cell_refs(&de(cell)?);
    Ok(ser(serde_json::json!({"vars": r.vars, "cmds": r.cmds, "tools": r.tools})))
}

/// A prompt as the model reads it: the instruction to answer, then the prompt with its id `sid` and `time`
#[pyfunction]
#[pyo3(signature = (task, sid="", time=None))]
fn envelope(task: &str, sid: &str, time: Option<&str>) -> String { cells::envelope(task, sid, time) }

/// `s` in a `<system-reminder>`, as a note for the model that the user may need to know about
#[pyfunction]
fn reminder(s: &str) -> String { input::reminder(s) }

/// Whether `eval` record `r` failed because its expression names nothing
#[pyfunction]
fn is_nameerr(r: &str) -> PyResult<bool> { Ok(input::is_nameerr(&de(r)?)) }

/// The content of the turn that gives a model the `$` values in `records`, and the expressions that name nothing
#[pyfunction]
fn vars(records: &str, caps_: &str) -> PyResult<String> { Ok(ser(input::vars(&de(records)?, &caps(caps_)?))) }

/// Model-input parts for one media item: its `<media>` tag and data, or a `<media-unavailable>` note
#[pyfunction]
#[pyo3(signature = (typ, id, data, caps_, mime=None))]
fn media_item(typ: &str, id: &str, data: Option<Vec<u8>>, caps_: &str, mime: Option<&str>) -> PyResult<String> {
    Ok(ser(media::item(typ, id, data.ok_or_else(|| "media data unavailable".into()), mime, &caps(caps_)?)))
}

/// `data` of type `mime` with its area limited to `max` pixels
#[pyfunction]
fn resize(data: Vec<u8>, mime: &str, max: u32) -> PyResult<Vec<u8>> { media::resize(data, mime, max).map_err(PyValueError::new_err) }

/// The MIME type of `data` from its leading bytes
#[pyfunction]
fn detect_mime(data: &[u8]) -> Option<&'static str> { media::detect_mime(data) }

/// Page count of PDF `data`
#[pyfunction]
fn pdf_pages(data: &[u8]) -> u64 { tokens::pdf_pages(data) }

/// Approximate tokens in `text`
#[pyfunction]
fn count_tokens(text: &str) -> u64 { tokens::count(text) }

/// The model-input parts for the images in `outputs`, notebook outputs as JSON
#[pyfunction]
fn image_parts(outputs: &str, caps_: &str) -> PyResult<String> {
    let outs: Vec<Value> = de(outputs)?;
    let r = output::render(output::stored(&outs), false, false);
    Ok(ser(output::image_parts(&r.images, &caps(caps_)?)))
}

fn result_json(r: output::ToolResult) -> String { ser(serde_json::json!({"output": r.output, "text": r.text, "full": r.full, "md": r.md})) }

/// A tool result from output text and its image parts, as `{output, text, full, md}`
#[pyfunction]
#[pyo3(signature = (text, parts, full=false, md=false))]
fn merge(text: String, parts: &str, full: bool, md: bool) -> PyResult<String> { Ok(result_json(output::merge(text, de(parts)?, full, md))) }

/// The result of a tool call from its notebook `outputs`, as `{output, text, full, md}`. `code` is true when the tool ran
/// code, so a lone `str` repr reads as the string.
#[pyfunction]
#[pyo3(signature = (outputs, caps_, code=false))]
fn tool_result(outputs: &str, caps_: &str, code: bool) -> PyResult<String> {
    let outs: Vec<Value> = de(outputs)?;
    Ok(result_json(output::tool_result(output::stored(&outs), &caps(caps_)?, code)))
}

/// The prose and tool calls of saved reply `text`, as blocks
#[pyfunction]
fn parse_reply(text: &str) -> String { ser(reply::parse(text)) }

/// `text` split at its tool blocks, as `(text before, block JSON)` pairs, with `None` for the text after the last
#[pyfunction]
fn tool_segments(text: &str) -> String { ser(reply::segments(text)) }

/// `text` with the stored result of call `id` replaced by `result`, kept whole. Every other byte stays as it was.
#[pyfunction]
fn set_result(text: &str, id: &str, result: &str) -> String { reply::set_result(text, id, result) }

/// `text` without its fenced blocks whose info string is in `infos`
#[pyfunction]
fn strip_blocks(text: &str, infos: Vec<String>) -> String { reply::strip_blocks(text, &infos.iter().map(String::as_str).collect::<Vec<_>>()) }

/// A turn as a saved reply: `parts`, aidialog `msg2dict` parts as JSON, then `usage`'s block
#[pyfunction]
#[pyo3(signature = (parts, usage=None))]
fn write_reply(parts: &str, usage: Option<&str>) -> PyResult<String> {
    let usage: Option<reply::Usage> = usage.map(de).transpose()?;
    Ok(reply::write(&de::<Vec<reply::Part>>(parts)?, usage.as_ref()))
}

/// A turn's reply as it streams in
#[pyclass]
struct Reply(reply::Reply);

#[pymethods]
impl Reply {
    #[new]
    fn new() -> Self { Self(reply::Reply::default()) }
    /// Add text the model wrote, returning the text to show
    fn text(&mut self, delta: &str) -> String { self.0.text(delta) }
    /// Show that the model is thinking
    fn thinking(&mut self) -> String { self.0.thinking() }
    /// Add a tool call, `msg2dict`'s form of a `ToolUse` as JSON, returning the text to show
    fn call(&mut self, c: &str) -> PyResult<String> { Ok(self.0.call(de(c)?)) }
    /// Add a client call's result, returning its completed block for the stream
    fn result(&mut self, r: &str) -> PyResult<String> { Ok(self.0.result(de(r)?)) }
    /// Add a response's usage, fastllm `UsageStats` fields as JSON
    fn usage(&mut self, u: &str) -> PyResult<()> { self.0.usage += de::<reply::Usage>(u)?; Ok(()) }
    /// The reply to store, with its usage block when `usage`. A `note` ends a reply cut short.
    #[pyo3(signature = (note=None, usage=false))]
    fn finish(&self, note: Option<&str>, usage: bool) -> String { self.0.finish(note, usage) }
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("IM_MAX", media::IM_MAX)?;
    m.add("IMG_TOKENS", tokens::IMG_TOKENS)?;
    m.add("UNSUPPORTED", media::UNSUPPORTED)?;
    m.add("VARS_REPLY", input::VARS_REPLY)?;
    m.add("USAGE_INFO", reply::USAGE_INFO)?;
    m.add_class::<Reply>()?;
    m.add_function(wrap_pyfunction!(build, m)?)?;
    m.add_function(wrap_pyfunction!(cell_parts, m)?)?;
    m.add_function(wrap_pyfunction!(cell_media, m)?)?;
    m.add_function(wrap_pyfunction!(cell_xml, m)?)?;
    m.add_function(wrap_pyfunction!(cell_refs, m)?)?;
    m.add_function(wrap_pyfunction!(shown, m)?)?;
    m.add_function(wrap_pyfunction!(cell_reply, m)?)?;
    m.add_function(wrap_pyfunction!(envelope, m)?)?;
    m.add_function(wrap_pyfunction!(reminder, m)?)?;
    m.add_function(wrap_pyfunction!(is_nameerr, m)?)?;
    m.add_function(wrap_pyfunction!(vars, m)?)?;
    m.add_function(wrap_pyfunction!(media_item, m)?)?;
    m.add_function(wrap_pyfunction!(resize, m)?)?;
    m.add_function(wrap_pyfunction!(detect_mime, m)?)?;
    m.add_function(wrap_pyfunction!(pdf_pages, m)?)?;
    m.add_function(wrap_pyfunction!(count_tokens, m)?)?;
    m.add_function(wrap_pyfunction!(image_parts, m)?)?;
    m.add_function(wrap_pyfunction!(merge, m)?)?;
    m.add_function(wrap_pyfunction!(tool_result, m)?)?;
    m.add_function(wrap_pyfunction!(parse_reply, m)?)?;
    m.add_function(wrap_pyfunction!(tool_segments, m)?)?;
    m.add_function(wrap_pyfunction!(set_result, m)?)?;
    m.add_function(wrap_pyfunction!(strip_blocks, m)?)?;
    m.add_function(wrap_pyfunction!(write_reply, m)?)?;
    Ok(())
}
