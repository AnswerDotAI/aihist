//! Export dialog cells as Markdown documents, plain-text parts, or Python scripts.

use percent_encoding::{AsciiSet, CONTROLS, utf8_percent_encode};
use regex::Regex;
use serde_json::Value;
use std::sync::LazyLock;

use crate::cells::{self, ai_fmt, kind, msg, text_str, truthy};
use crate::output::{render_md, render_outputs_ai, render_text};
use crate::reply::{TOOL_INFO, USAGE_INFO, fenced_ch, fmt_tools, strip_blocks};

/// Characters encoded by Python's `urllib.parse.quote(s)` (default `safe='/'`)
const QUOTE_SET: &AsciiSet = &CONTROLS.add(b' ').add(b'!').add(b'"').add(b'#').add(b'$').add(b'%').add(b'&').add(b'\'').add(b'(').add(b')').add(b'*').add(b'+')
    .add(b',').add(b':').add(b';').add(b'<').add(b'=').add(b'>').add(b'?').add(b'@').add(b'[').add(b'\\').add(b']').add(b'^').add(b'`').add(b'{').add(b'|')
    .add(b'}');

static DIRECTIVE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?m)^\s*#\|\s*exports?\b").expect("valid regex"));

fn quote_att(id: &str) -> String { utf8_percent_encode(id, QUOTE_SET).to_string() }

/// Rewrite `attachment:` references in `text` as data URIs from `cell`'s stored attachments
fn inline_atts(nb: Option<&Value>, cell: &Value, mut text: String) -> String {
    let atts = nb.and_then(|n| cells::stored_atts(n, cell)).or_else(|| cell["attachments"].as_object());
    let Some(atts) = atts else { return text; };
    for (id, by_mime) in atts {
        let Some((mime, data_v)) = by_mime.as_object().and_then(|m| m.iter().next()) else { continue; };
        let b64 = text_str(data_v);
        let uri = format!("data:{mime};base64,{b64}");
        let raw_ref = format!("attachment:{id}");
        let enc_ref = format!("attachment:{}", quote_att(id));
        text = text.replace(&raw_ref, &uri).replace(&enc_ref, &uri);
    }
    text
}

/// Whether `cell` carries an nbdev `export` or `exports` directive in metadata or source comments
pub fn is_exported(cell: &Value) -> bool {
    let nbdev = &cell["metadata"]["nbdev"];
    if nbdev.get("export").is_some() || nbdev.get("exports").is_some() { return true; }
    DIRECTIVE_RE.is_match(&cells::content(cell))
}

/// One cell as Markdown, resolving prompt/code attachments from `nb` when present; `None` for empty code
pub fn msg2md_in(nb: Option<&Value>, cell: &Value, language: &str, weave: bool) -> Option<String> {
    let m = msg(cell);
    match m.typ {
        "note" => Some(inline_atts(nb, cell, m.content)),
        "raw" => Some(fenced_ch(&m.content, "", '`')),
        "prompt" => {
            let mut parts = vec![fenced_ch(&inline_atts(nb, cell, m.content), " prompt", ':')];
            if !m.out.trim().is_empty() { parts.push(fenced_ch(&inline_atts(nb, cell, fmt_tools(&m.out)), " reply", ':')); }
            Some(parts.join("\n\n"))
        }
        _ => {
            let outs = cell["outputs"].as_array().map(Vec::as_slice).unwrap_or_default();
            if weave { let text = render_text(outs, false, false); return (!text.is_empty()).then_some(text); }
            let mut parts = Vec::new();
            if !m.content.trim().is_empty() { parts.push(fenced_ch(&m.content, language, '`')); }
            let rendered = render_md(outs, true);
            if !rendered.is_empty() { parts.push(fenced_ch(&rendered, " output", ':')); }
            (!parts.is_empty()).then(|| parts.join("\n\n"))
        }
    }
}

/// One cell as Markdown with `language` code fences and stored attachments inlined; `None` for empty code
pub fn msg2md(cell: &Value, language: &str) -> Option<String> { msg2md_in(None, cell, language, false) }

fn filter_exported(cells: &[Value]) -> Vec<&Value> {
    let exp: Vec<&Value> = cells.iter().filter(|c| is_exported(c)).collect();
    if !exp.is_empty() { exp } else { cells.iter().filter(|c| !truthy(&c["metadata"]["pinned"])).collect() }
}

/// A notebook as one Markdown document, with optional export filtering and `weave` output-only code cells
pub fn dlg2md_nb(nb: &Value, language: &str, exportfilter: bool, weave: bool) -> String {
    let cells = nb["cells"].as_array().map(Vec::as_slice).unwrap_or_default();
    let selected: Vec<&Value> = if exportfilter { filter_exported(cells) } else { cells.iter().collect() };
    let joined = selected.into_iter().filter_map(|c| msg2md_in(Some(nb), c, language, weave)).filter(|s| !s.is_empty()).collect::<Vec<_>>().join("\n\n");
    format!("{joined}\n")
}

/// Cells as one Markdown document with `language` code fences and stored attachments inlined
pub fn dlg2md(cells: &[Value], language: &str) -> String {
    let joined = cells.iter().filter_map(|c| msg2md(c, language)).filter(|s| !s.is_empty()).collect::<Vec<_>>().join("\n\n");
    format!("{joined}\n")
}

/// Input source or plain-text output for copying one message part (`"input"` or `"output"`)
pub fn msg2text(cell: &Value, part: &str) -> Result<String, &'static str> {
    let m = msg(cell);
    if part == "input" { return Ok(m.content); }
    if part != "output" { return Err("Choose input or output text"); }
    if m.typ == "prompt" { return Ok(ai_fmt(&strip_blocks(&m.out, &[TOOL_INFO, USAGE_INFO]))); }
    if m.typ == "code" {
        let outs = cell["outputs"].as_array().map(Vec::as_slice).unwrap_or_default();
        return Ok(render_outputs_ai(outs, false));
    }
    Ok(String::new())
}

fn comment_lines(text: &str, prefix: &str) -> String {
    let trimmed = text.trim();
    if trimmed.is_empty() { return String::new(); }
    trimmed.lines().map(|l| format!("{prefix} {l}")).collect::<Vec<_>>().join("\n")
}

/// Marked exported Python code, or when `full`, every cell with non-code and outputs commented
pub fn msgs2py(cells: &[Value], full: bool) -> String {
    if !full {
        return cells.iter().filter(|c| kind(c) == "code" && is_exported(c)).map(cells::content).collect::<Vec<_>>().join("\n\n");
    }
    let mut parts = Vec::new();
    for cell in cells {
        let m = msg(cell);
        let id = cell["id"].as_str().unwrap_or("");
        parts.push(format!("# --- Cell: {id} ({}) ---", m.typ));
        parts.push(if m.typ == "code" { m.content.clone() } else { comment_lines(&m.content, "#") });
        if m.typ == "code" {
            let out = msg2text(cell, "output").unwrap_or_default();
            if !out.is_empty() { parts.push(comment_lines(&out, "#>")); }
        } else if m.typ == "prompt" && cell["outputs"].as_array().is_some_and(|a| !a.is_empty()) {
            parts.push("# --- Assistant Response ---".to_owned());
            parts.push(comment_lines(&msg2text(cell, "output").unwrap_or_default(), "#"));
        }
    }
    parts.join("\n\n")
}
