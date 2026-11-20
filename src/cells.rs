//! One notebook cell as a dialog message, and as a model reads it. A cell is a note, code, a raw cell or a prompt, as
//! aidialog's notebook format defines them.

use regex::Regex;
use serde_json::{Map, Value};
use std::collections::HashSet;
use std::sync::LazyLock;

use crate::Caps;
use crate::media::{self, IMG_MIMES, text, xml_attr};
use crate::{output, reply, tokens};

pub use rgapi::cell_refs as refs;

/// Attachments a message refers to: `![...](attachment:id)` and `<!-- last_output:uuid -->`
static ATT_REF: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"!\[[^\]]*\]\(attachment:([^)]+)\)|<!-- last_output:([0-9a-f-]{36}) -->").expect("a valid regex"));

static MEDIA_REF: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"!\[[^\]]*\]\(([^)#]+)#ai\)").expect("a valid regex"));

/// External media links marked #ai, in source order, from a note or prompt's content.
pub fn media_refs(cell: &Value) -> Vec<String> {
    if cell["cell_type"] != "markdown" && !is_prompt(cell) { return vec![]; }
    MEDIA_REF.captures_iter(&content(cell)).map(|c| c[1].to_owned()).filter(|s| !s.starts_with("attachment:")).collect()
}

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
        Value::Null => false, Value::Bool(b) => *b, Value::Number(n) => n.as_f64() != Some(0.0), Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(), Value::Object(o) => !o.is_empty()
    }
}

/// Whether `cell` is a prompt: its metadata has a truthy `solveit_ai`
pub fn is_prompt(cell: &Value) -> bool { truthy(&cell["metadata"]["solveit_ai"]) }

/// The dialog message type, without rendering its content or output.
pub fn kind(cell: &Value) -> &'static str {
    if is_prompt(cell) { "prompt" } else { match cell["cell_type"].as_str() { Some("markdown") => "note", Some("raw") => "raw", _ => "code" } }
}

/// A cell's input text, without a prompt's `%%prompt` first line.
pub fn content(cell: &Value) -> String {
    let src = text_str(&cell["source"]);
    if !is_prompt(cell) { return src; }
    match src.split_once('\n') { Some((h, rest)) if h.trim() == "%%prompt" => rest.into(), None if src.trim() == "%%prompt" => String::new(), _ => src }
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
/// reply in a Markdown output.
pub fn msg(cell: &Value) -> Msg {
    let src = content(cell);
    let outs = cell["outputs"].as_array().map(Vec::as_slice).unwrap_or_default();
    let typ = kind(cell);
    if !is_prompt(cell) {
        let out = if typ != "code" { String::new() } else { output::render(output::stored(outs), false, true).text };
        return Msg { typ, content: src, out };
    }
    let out = outs.iter().find_map(|o| o.get("data")?.get("text/markdown")).map(text_str).unwrap_or_default();
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

/// `m` as the XML a model reads: a prompt in its envelope, or a `<markdown>`, `<code>`, `<raw>` or `<diff>` element
/// with the cell's id, its `time`, its review or nbdev metadata directives, and any output
pub fn xml(cell: &Value, m: &Msg, time: Option<&str>) -> String {
    let id = cell["id"].as_str().unwrap_or_default();
    if m.typ == "prompt" { return envelope(&m.content, id, time); }
    let review = (m.typ == "raw").then(|| cell["metadata"]["review"].as_object()).flatten();
    let tag = if review.is_some() { "diff" } else if m.typ == "note" { "markdown" } else { m.typ };
    let mut attrs = if id.is_empty() { String::new() } else { xml_attr("id", id) };
    if let Some(t) = time { attrs += &xml_attr("time", t) }
    if let Some(r) = review {
        for k in ["path", "kind", "status", "file_status", "baseline", "cell_id", "cell_type"] {
            if let Some(v) = r.get(k).and_then(Value::as_str).filter(|s| !s.is_empty()) { attrs += &xml_attr(k, v); }
        }
    }
    for (k, v) in cell["metadata"]["nbdev"].as_object().into_iter().flatten() {
        match v.as_str() { Some("true" | "") => attrs += &format!(" {k}"), Some(v) => attrs += &xml_attr(k, v), None => {} }
    }
    let out = if m.out.is_empty() { String::new() } else { format!("<out>{}</out>", m.out) };
    format!("<{tag}{attrs}>{}{out}</{tag}>", m.content)
}

/// Referenced attachments, marked links, then code output images.
pub async fn media(nb: &Value, cell: &Value, caps: &Caps, ctx: &media::Context<'_>) -> Vec<Value> {
    let source = content(cell);
    let refs: HashSet<&str> =
        ATT_REF.captures_iter(&source).filter_map(|c| c.get(1).or_else(|| c.get(2))).map(|g| g.as_str().strip_suffix("#ai").unwrap_or(g.as_str())).collect();
    let mut res = vec![];
    for (id, a) in stored_atts(nb, cell).into_iter().flatten().filter(|(id, _)| refs.contains(id.as_str())) {
        let entry = a.as_object().and_then(|o| o.iter().next());
        let data = entry.map(|(_, v)| text_str(v)).ok_or_else(|| "media data unavailable".to_string()).and_then(|s| media::decode(&s));
        res.extend(media::item("content", id, data, entry.map(|(mime, _)| mime.as_str()), caps));
    }
    for reference in media_refs(cell) { res.extend(media::item("content", &reference, media::load(&reference, ctx).await, None, caps)); }
    if cell["cell_type"] == "code" && !is_prompt(cell) {
        for d in cell["outputs"].as_array().into_iter().flatten().filter_map(|o| o["data"].as_object()) {
            let Some((mime, v)) = tokens::first_mime(d, &IMG_MIMES) else { continue; };
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

/// Cell `cell` of notebook `nb` as a model reads it: its media, then its XML, with its host-supplied timestamp.
pub async fn parts(nb: &Value, cell: &Value, caps: &Caps, time: Option<&str>, ctx: &media::Context<'_>) -> Vec<Value> {
    let m = msg(cell);
    let mut res = media(nb, cell, caps, ctx).await;
    res.push(text(xml(cell, &m, time)));
    res
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn review_diff_xml() {
        let hunk = json!({
            "id": "a1b2c3d4", "cell_type": "raw", "source": "@@ -10,2 +10,2 @@\n-old\n+new\n",
            "metadata": {"review": {
                "head": "deadbeef", "baseline": ":0", "path": "src/auth.py", "kind": "hunk",
                "status": "pending", "file_status": "modified", "cell_id": null, "cell_type": null
            }}
        });
        assert_eq!(kind(&hunk), "raw");
        assert!(shown(&hunk));
        assert_eq!(
            xml(&hunk, &msg(&hunk), Some("2026-11-20T09:00:00Z")),
            "<diff id=\"a1b2c3d4\" time=\"2026-11-20T09:00:00Z\" path=\"src/auth.py\" kind=\"hunk\" status=\"pending\" file_status=\"modified\" baseline=\":0\">@@ -10,2 +10,2 @@\n-old\n+new\n</diff>"
        );

        let nb_cell = json!({
            "id": "c9d8e7f6", "cell_type": "raw", "source": "@@ -1,1 +1,1 @@\n-x = 1\n+x = 2\n",
            "metadata": {"review": {
                "baseline": ":0", "path": "nbs/00_core.ipynb", "kind": "cell",
                "status": "accepted", "file_status": "modified", "cell_id": "955b9784", "cell_type": "code"
            }}
        });
        assert_eq!(
            xml(&nb_cell, &msg(&nb_cell), None),
            "<diff id=\"c9d8e7f6\" path=\"nbs/00_core.ipynb\" kind=\"cell\" status=\"accepted\" file_status=\"modified\" baseline=\":0\" cell_id=\"955b9784\" cell_type=\"code\">@@ -1,1 +1,1 @@\n-x = 1\n+x = 2\n</diff>"
        );

        let rejected = json!({
            "id": "r9", "cell_type": "raw", "source": "@@ -0,0 +1,1 @@\n+hi\n",
            "metadata": {"review": {"path": "a&b<1>.py", "kind": "hunk", "status": "rejected", "file_status": "added", "baseline": ":0"}}
        });
        assert_eq!(
            xml(&rejected, &msg(&rejected), None),
            "<diff id=\"r9\" path=\"a&amp;b&lt;1&gt;.py\" kind=\"hunk\" status=\"rejected\" file_status=\"added\" baseline=\":0\">@@ -0,0 +1,1 @@\n+hi\n</diff>"
        );

        let plain = json!({"id": "p1", "cell_type": "raw", "source": "plain raw", "metadata": {}});
        assert_eq!(xml(&plain, &msg(&plain), None), "<raw id=\"p1\">plain raw</raw>");

        let non_obj = json!({"id": "p2", "cell_type": "raw", "source": "raw", "metadata": {"review": false}});
        assert_eq!(xml(&non_obj, &msg(&non_obj), None), "<raw id=\"p2\">raw</raw>");

        let note = json!({"id": "n1", "cell_type": "markdown", "source": "note", "metadata": {"review": {"path": "x.py"}}});
        assert_eq!(xml(&note, &msg(&note), None), "<markdown id=\"n1\">note</markdown>");
    }
}
