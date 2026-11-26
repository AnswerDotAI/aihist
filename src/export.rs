//! Export dialog cells as structured typed parts, Markdown documents, plain-text parts, or Python scripts.

use percent_encoding::{AsciiSet, CONTROLS, utf8_percent_encode};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::BTreeMap;
use std::sync::LazyLock;

use crate::cells::{self, ai_fmt, kind, msg, text_str, truthy};
use crate::output::{OutputPart, output_parts, render_outputs_ai, render_text};
use crate::reply::{ReplyPart, TOOL_INFO, USAGE_INFO, fenced_ch, reply_parts, strip_blocks, tool_md, usage_md};

/// Characters encoded by Python's `urllib.parse.quote(s)` (default `safe='/'`)
const QUOTE_SET: &AsciiSet = &CONTROLS.add(b' ').add(b'!').add(b'"').add(b'#').add(b'$').add(b'%').add(b'&').add(b'\'').add(b'(').add(b')').add(b'*').add(b'+')
    .add(b',').add(b':').add(b';').add(b'<').add(b'=').add(b'>').add(b'?').add(b'@').add(b'[').add(b'\\').add(b']').add(b'^').add(b'`').add(b'{').add(b'|')
    .add(b'}');

static DIRECTIVE_RE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?m)^\s*#\|\s*exports?\b").expect("valid regex"));

/// Options controlling structured notebook export (`export_nb`).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default)]
pub struct ExportOpts { pub language: Option<String>, pub exportfilter: bool, pub weave: bool, pub frontmatter: bool, pub att_data: bool }

impl Default for ExportOpts {
    fn default() -> Self { Self { language: None, exportfilter: false, weave: false, frontmatter: true, att_data: true } }
}

/// One stored cell attachment: MIME type and optional whitespace-free base64 payload.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExportAttachment { pub mime: String, #[serde(default, skip_serializing_if = "Option::is_none")] pub data: Option<String> }

/// One typed piece of an exported cell's source or output.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase")]
pub enum ExportPart {
    Md { text: String }, Code { lang: String, text: String }, Text { text: String },
    Html { html: String, #[serde(default, skip_serializing_if = "Option::is_none")] text: Option<String> },
    Image { mime: String, data: String, #[serde(default, skip_serializing_if = "Option::is_none")] text: Option<String> }, Tool { data: Value },
    Usage { data: Value }
}

/// One dialog cell projected into typed source/output parts, stored attachments, and cell metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExportCell {
    pub id: String, pub kind: String, pub src: Vec<ExportPart>, pub out: Vec<ExportPart>, pub attachments: BTreeMap<String, ExportAttachment>,
    pub meta: Map<String, Value>
}

/// A notebook projected into optional leading YAML frontmatter and structured typed cells.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExportDoc { pub frontmatter: String, pub language: String, pub cells: Vec<ExportCell> }

fn quote_att(id: &str) -> String { utf8_percent_encode(id, QUOTE_SET).to_string() }

fn cell_attachments(nb: Option<&Value>, cell: &Value, att_data: bool) -> BTreeMap<String, ExportAttachment> {
    let atts = nb.and_then(|n| cells::stored_atts(n, cell)).or_else(|| cell["attachments"].as_object());
    let mut out = BTreeMap::new();
    for (id, by_mime) in atts.into_iter().flatten() {
        let Some((mime, data_v)) = by_mime.as_object().and_then(|m| m.iter().next()) else { continue; };
        let data = att_data.then(|| text_str(data_v).split_whitespace().collect());
        out.insert(id.clone(), ExportAttachment { mime: mime.clone(), data });
    }
    out
}

/// Rewrite stored `attachment:` references as data URIs for flat Markdown export, longest name first.
fn inline_atts(atts: &BTreeMap<String, ExportAttachment>, mut text: String) -> String {
    if atts.is_empty() || !text.contains("attachment:") { return text; }
    let mut refs: Vec<(String, String)> = Vec::with_capacity(atts.len() * 2);
    for (id, att) in atts {
        let Some(data) = &att.data else { continue; };
        let uri = format!("data:{};base64,{data}", att.mime);
        let enc = quote_att(id);
        refs.push((id.clone(), uri.clone()));
        if enc != *id { refs.push((enc, uri)); }
    }
    refs.sort_by_key(|a| std::cmp::Reverse(a.0.len()));
    for (k, uri) in &refs { text = text.replace(&format!("attachment:{k}"), uri); }
    text
}

fn has_mapping_pair(line: &str) -> bool {
    line.char_indices().any(|(i, c)| c == ':' && !line[..i].trim().is_empty() && line[i + 1..].chars().next().is_none_or(char::is_whitespace))
}

fn split_frontmatter(src: &str) -> Option<(String, &str)> {
    let mut lines = src.split_inclusive('\n');
    let first = lines.next()?;
    if first.trim_end() != "---" { return None; }
    let (mut pos, mut mapping) = (first.len(), false);
    for line in lines {
        pos += line.len();
        let t = line.trim_end();
        if t == "---" || t == "..." { return mapping.then(|| (format!("{}\n", src[..pos].trim_end()), &src[pos..])); }
        let t = t.trim();
        if t.is_empty() || t.starts_with('#') { continue; }
        if !mapping {
            if !has_mapping_pair(t) { return None; }
            mapping = true;
        }
    }
    None
}

fn resolve_language(nb: Option<&Value>, language: Option<&str>) -> String {
    if let Some(l) = language.filter(|s| !s.is_empty()) { return l.to_owned(); }
    if let Some(nb) = nb {
        if let Some(l) = nb["metadata"]["kernelspec"]["language"].as_str().filter(|s| !s.is_empty()) { return l.to_owned(); }
        if let Some(l) = nb["metadata"]["language_info"]["name"].as_str().filter(|s| !s.is_empty()) { return l.to_owned(); }
    }
    "python".into()
}

/// Whether `cell` carries an nbdev `export` or `exports` directive in metadata or source comments
pub fn is_exported(cell: &Value) -> bool {
    let nbdev = &cell["metadata"]["nbdev"];
    if nbdev.get("export").is_some() || nbdev.get("exports").is_some() { return true; }
    DIRECTIVE_RE.is_match(&cells::content(cell))
}

fn export_cell_with(nb: Option<&Value>, cell: &Value, language: &str, weave: bool, att_data: bool) -> Option<ExportCell> {
    let id = cell["id"].as_str().unwrap_or_default().to_owned();
    let attachments = cell_attachments(nb, cell, att_data);
    let meta = cell["metadata"].as_object().cloned().unwrap_or_default();
    let m = msg(cell);
    let md = |text: String| if text.is_empty() { Vec::new() } else { vec![ExportPart::Md { text }] };
    match m.typ {
        "note" => Some(ExportCell { id, kind: "note".into(), src: md(m.content), out: Vec::new(), attachments, meta }),
        "raw" => {
            let src = if m.content.is_empty() { Vec::new() } else { vec![ExportPart::Code { lang: String::new(), text: m.content }] };
            Some(ExportCell { id, kind: "raw".into(), src, out: Vec::new(), attachments, meta })
        }
        "prompt" => {
            let out = reply_parts(&m.out).into_iter().map(|p| match p {
                ReplyPart::Md(text) => ExportPart::Md { text }, ReplyPart::Tool(data) => ExportPart::Tool { data },
                ReplyPart::Usage(data) => ExportPart::Usage { data }
            }).collect();
            Some(ExportCell { id, kind: "prompt".into(), src: md(m.content), out, attachments, meta })
        }
        _ => {
            let outs = cell["outputs"].as_array().map(Vec::as_slice).unwrap_or_default();
            if weave {
                let text = render_text(outs, false, false);
                return (!text.is_empty()).then(|| ExportCell { id, kind: "code".into(), src: vec![ExportPart::Md { text }], out: Vec::new(), attachments, meta
                });
            }
            let src = if m.content.trim().is_empty() { Vec::new() } else { vec![ExportPart::Code { lang: language.to_owned(), text: m.content }] };
            let out: Vec<ExportPart> = output_parts(outs, true).into_iter().map(|p| match p {
                OutputPart::Text(text) => ExportPart::Text { text }, OutputPart::Md(text) => ExportPart::Md { text },
                OutputPart::Html { html, text } => ExportPart::Html { html, text },
                OutputPart::Image { mime, data, text } => ExportPart::Image { mime, data, text }
            }).collect();
            if src.is_empty() && out.is_empty() { None } else { Some(ExportCell { id, kind: "code".into(), src, out, attachments, meta }) }
        }
    }
}

/// Project one cell into typed source/output parts; `None` for empty code cells.
pub fn export_cell(nb: Option<&Value>, cell: &Value, language: &str, weave: bool) -> Option<ExportCell> { export_cell_with(nb, cell, language, weave, true) }

fn parts_to_md(parts: &[ExportPart], atts: &BTreeMap<String, ExportAttachment>) -> String {
    parts.iter().map(|p| match p {
        ExportPart::Md { text } => inline_atts(atts, text.clone()), ExportPart::Code { lang, text } => fenced_ch(text, lang, '`'),
        ExportPart::Text { text } => fenced_ch(text, "output", '`'), ExportPart::Html { html, .. } => fenced_ch(html, "{=html}", '`'),
        ExportPart::Image { mime, data, .. } => format!("![](data:{mime};base64,{data})"), ExportPart::Tool { data } => tool_md(data),
        ExportPart::Usage { data } => usage_md(data)
    }).filter(|s| !s.is_empty()).collect::<Vec<_>>().join("\n\n")
}

fn cell_to_md(c: &ExportCell, weave: bool) -> String {
    let (src, out) = (parts_to_md(&c.src, &c.attachments), parts_to_md(&c.out, &c.attachments));
    match c.kind.as_str() {
        "note" => src,
        "raw" => { if src.is_empty() { fenced_ch("", "", '`') } else { src } }
        "prompt" => {
            let mut parts = vec![fenced_ch(&src, " prompt", ':')];
            if !out.is_empty() { parts.push(fenced_ch(&out, " reply", ':')); }
            parts.join("\n\n")
        }
        _ if weave => src,
        _ => {
            let mut parts = Vec::new();
            if !src.is_empty() { parts.push(src); }
            if !out.is_empty() { parts.push(fenced_ch(&out, " output", ':')); }
            parts.join("\n\n")
        }
    }
}

fn filter_exported<'a>(cells: &[&'a Value]) -> Vec<&'a Value> {
    let exp: Vec<_> = cells.iter().copied().filter(|c| is_exported(c)).collect();
    if !exp.is_empty() { exp } else { cells.iter().copied().filter(|c| !truthy(&c["metadata"]["pinned"])).collect() }
}

/// Project a notebook into optional leading YAML frontmatter and structured typed cells.
pub fn export_nb(nb: &Value, opts: &ExportOpts) -> ExportDoc {
    let language = resolve_language(Some(nb), opts.language.as_deref());
    let all = nb["cells"].as_array().map(Vec::as_slice).unwrap_or_default();
    let (mut frontmatter, mut owned_first, mut start) = (String::new(), None, 0usize);
    if opts.frontmatter && let Some(first) = all.first() {
        let k = kind(first);
        if (k == "raw" || k == "note") && let Some((fm, rest)) = split_frontmatter(&cells::content(first)) {
            if rest.trim().is_empty() {
                frontmatter = fm;
                start = 1;
            } else if k == "note" {
                frontmatter = fm;
                let mut c = first.clone();
                c["source"] = Value::String(rest.trim_start_matches(['\r', '\n']).to_owned());
                owned_first = Some(c);
                start = 1;
            }
        }
    }
    let body: Vec<&Value> = owned_first.iter().chain(&all[start..]).collect();
    let selected = if opts.exportfilter { filter_exported(&body) } else { body };
    let cells = selected.into_iter().filter_map(|c| export_cell_with(Some(nb), c, &language, opts.weave, opts.att_data)).collect();
    ExportDoc { frontmatter, language, cells }
}

/// One cell as Markdown, resolving prompt/code attachments from `nb` when present; `None` for empty code
pub fn msg2md_in(nb: Option<&Value>, cell: &Value, language: &str, weave: bool) -> Option<String> {
    let lang = resolve_language(nb, (!language.is_empty()).then_some(language));
    export_cell(nb, cell, &lang, weave).map(|c| cell_to_md(&c, weave))
}

/// One cell as Markdown with `language` code fences and stored attachments inlined; `None` for empty code
pub fn msg2md(cell: &Value, language: &str) -> Option<String> { msg2md_in(None, cell, language, false) }

/// A notebook as one Markdown document, with optional export filtering and `weave` output-only code cells
pub fn dlg2md_nb(nb: &Value, language: &str, exportfilter: bool, weave: bool) -> String {
    let opts = ExportOpts { language: (!language.is_empty()).then(|| language.to_owned()), exportfilter, weave, ..Default::default() };
    let doc = export_nb(nb, &opts);
    let mut parts = Vec::new();
    if !doc.frontmatter.is_empty() { parts.push(doc.frontmatter.trim_end().to_owned()); }
    for c in &doc.cells { let s = cell_to_md(c, weave); if !s.is_empty() { parts.push(s); } }
    format!("{}\n", parts.join("\n\n"))
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
