//! One notebook cell as a dialog message, and as a model reads it. A cell is a note, code, a raw cell or a prompt, as
//! aidialog's notebook format defines them.

use regex::Regex;
use serde_json::{Map, Value};
use std::collections::HashSet;
use std::sync::LazyLock;

use crate::Caps;
use crate::media::{self, IMG_MIMES, text, xml_attr};
use crate::{output, reply, tokens};

/// Attachments a message refers to: `![...](attachment:id)` and `<!-- last_output:uuid -->`
static ATT_REF: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"!\[[^\]]*\]\(attachment:([^)]+)\)|<!-- last_output:([0-9a-f-]{36}) -->").expect("a valid regex"));

/// A cell's attachments: its own, or for a code cell, the notebook's `metadata.attachments` entry for its id. nbformat
/// allows attachments only on markdown and raw cells, so aidialog's notebook format keeps a code cell's there.
pub fn stored_atts<'a>(nb: &'a Value, cell: &'a Value) -> Option<&'a Map<String, Value>> {
    if let Some(m) = cell["attachments"].as_object() { return Some(m); }
    nb["metadata"]["attachments"][cell["id"].as_str()?].as_object()
}

/// Multiline nbformat text (a string, or a list of strings) as one string.
pub fn text_str(v: &Value) -> String {
    match v { Value::String(s) => s.clone(), Value::Array(a) => a.iter().filter_map(Value::as_str).collect(), _ => String::new() }
}

/// Python's truthiness, for metadata flags such as `skipped: 1`
pub fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// Whether `cell` is a prompt: its metadata has a truthy `solveit_ai`
pub fn is_prompt(cell: &Value) -> bool { truthy(&cell["metadata"]["solveit_ai"]) }

/// A cell's input text, without a prompt's `%%prompt` first line.
pub fn content(cell: &Value) -> String {
    let src = text_str(&cell["source"]);
    if !is_prompt(cell) { return src; }
    match src.split_once('\n') {
        Some((h, rest)) if h.trim() == "%%prompt" => rest.into(),
        None if src.trim() == "%%prompt" => String::new(),
        _ => src,
    }
}

/// A cell as a dialog message
pub struct Msg {
    /// "prompt", "note", "code" or "raw"
    pub typ: &'static str,
    /// The cell's text, without a prompt's `%%prompt` line or reply
    pub content: String,
    /// A prompt's reply, or the text a model sees for code outputs; empty for notes and raw cells
    pub out: String,
}

/// `cell` as a dialog message. A prompt is a code cell with `solveit_ai` metadata, a `%%prompt` first line and its
/// reply in an `is_ai_res` output.
pub fn msg(cell: &Value) -> Msg {
    let src = content(cell);
    let outs = cell["outputs"].as_array().map(Vec::as_slice).unwrap_or_default();
    let kind = cell["cell_type"].as_str().unwrap_or_default();
    if !is_prompt(cell) {
        let typ = match kind { "markdown" => "note", "raw" => "raw", _ => "code" };
        let out = if typ != "code" { String::new() } else { output::render(output::stored(outs), false, true).text };
        return Msg { typ, content: src, out };
    }
    let out = outs.iter().find(|o| truthy(&o["metadata"]["is_ai_res"])).map(|o| text_str(&o["data"]["text/markdown"])).unwrap_or_default();
    Msg { typ: "prompt", content: src, out }
}

/// A reply as the model's history holds it, as aidialog's `ai_fmt` gives it: trimmed, without the interruption
/// line, and marked pending when empty
pub fn ai_fmt(s: &str) -> String {
    let s = s.trim();
    let s = s.strip_suffix(reply::INTERRUPTED).and_then(|t| t.strip_suffix('\n')).unwrap_or(s).trim();
    if s.is_empty() { r#"<output result="pending" reason="incomplete"/>"#.into() } else { s.into() }
}

/// A prompt as the model reads it: an instruction to answer, then the prompt with its message id and `time`
pub fn envelope(task: &str, sid: &str, time: Option<&str>) -> String {
    let mut attrs = if sid.is_empty() { String::new() } else { xml_attr("sid", sid) };
    if let Some(t) = time { attrs += &xml_attr("time", t) }
    format!("<instructions>Respond to the request in the `prompt` below.</instructions>\n<prompt{attrs}>{task}</prompt>")
}

/// `m` as the XML a model reads: a prompt in its envelope, or a `<markdown>`, `<code>` or `<raw>` element with the
/// cell's id, its `time`, its nbdev metadata directives, and any output
pub fn xml(cell: &Value, m: &Msg, time: Option<&str>) -> String {
    let id = cell["id"].as_str().unwrap_or_default();
    if m.typ == "prompt" { return envelope(&m.content, id, time) }
    let tag = if m.typ == "note" { "markdown" } else { m.typ };
    let mut attrs = if id.is_empty() { String::new() } else { xml_attr("id", id) };
    if let Some(t) = time { attrs += &xml_attr("time", t) }
    for (k, v) in cell["metadata"]["nbdev"].as_object().into_iter().flatten() {
        match v.as_str() {
            Some("true" | "") => attrs += &format!(" {k}"),
            Some(v) => attrs += &xml_attr(k, v),
            None => {}
        }
    }
    let out = if m.out.is_empty() { String::new() } else { format!("<out>{}</out>", m.out) };
    format!("<{tag}{attrs}>{}{out}</{tag}>", m.content)
}

/// The media a message shows the model: the attachments its text refers to, then a code cell's output images
pub fn media(atts: Option<&Map<String, Value>>, cell: &Value, m: &Msg, caps: &Caps) -> Vec<Value> {
    let refs: HashSet<&str> =
        ATT_REF.captures_iter(&m.content).filter_map(|c| c.get(1).or_else(|| c.get(2))).map(|g| g.as_str()).collect();
    let mut res = vec![];
    for (id, a) in atts.into_iter().flatten().filter(|(id, _)| refs.contains(id.as_str())) {
        let data = a.as_object().and_then(|o| o.values().next()).map(text_str)
            .ok_or_else(|| "media data unavailable".to_string()).and_then(|s| media::decode(&s));
        res.extend(media::item("content", id, data, None, caps));
    }
    if m.typ == "code" {
        for d in cell["outputs"].as_array().into_iter().flatten().filter_map(|o| o["data"].as_object()) {
            let Some((mime, v)) = tokens::first_mime(d, &IMG_MIMES) else { continue };
            res.extend(media::output_item(mime, &text_str(v), caps));
        }
    }
    res
}

/// Does the model see `cell`? Skipped cells, and raw cells that record a session (`rec_kind`), are hidden.
pub fn shown(cell: &Value) -> bool {
    let meta = &cell["metadata"];
    !truthy(&meta["skipped"]) && !(cell["cell_type"] == "raw" && truthy(&meta["rec_kind"]))
}

/// Cell `cell` of notebook `nb` as a model reads it: its media, then its XML. `time` is the cell's time in the user's
/// time zone.
pub fn parts(nb: &Value, cell: &Value, caps: &Caps, time: Option<&str>) -> Vec<Value> {
    let m = msg(cell);
    let mut res = media(stored_atts(nb, cell), cell, &m, caps);
    res.push(text(xml(cell, &m, time)));
    res
}

