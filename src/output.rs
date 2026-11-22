//! Kernel outputs as a model reads them: a code cell's outputs, and the result of a tool call that runs code.

use regex::Regex;
use serde_json::{Value, json};
use std::sync::LazyLock;

use crate::Caps;
use crate::cells::text_str;
use crate::media::{self, IMG_MIMES};
use crate::tokens::{TEXT_MIMES, first_mime};

/// `s` without ANSI escape sequences: CSI sequences such as colours, and OSC sequences ending in BEL or `ESC \`
pub fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\x1b' { out.push(c); continue; }
        match chars.next() {
            Some('[') => { while let Some(x) = chars.next() { if ('@'..='~').contains(&x) { break; } } }
            Some(']') => {
                while let Some(x) = chars.next() { if x == '\x07' { break; } if x == '\x1b' && chars.next_if_eq(&'\\').is_some() { break; } }
            }
            Some(_) | None => {}
        }
    }
    out
}

const TB_MAXLEN: usize = 180;

/// One traceback line, capped at `TB_MAXLEN`. `None` drops it, because an over-long line of `~` and `^` markers means
/// nothing once cut. `File ` and `Cell ` locations are never cut.
fn tb_line(l: &str) -> Option<String> {
    if l.chars().count() <= TB_MAXLEN { return Some(l.to_string()); }
    if !l.trim().is_empty() && l.chars().all(|c| "~^ ".contains(c)) { return None; }
    if l.trim_start().starts_with("File ") || l.trim_start().starts_with("Cell ") { return Some(l.to_string()); }
    Some(l.chars().take(TB_MAXLEN).collect::<String>() + "…")
}

/// Traceback chunks, ANSI-stripped and joined. With `cap`, over-long lines are capped in all but the final chunk, which is
/// the exception message and survives whole, as fastcore's `render_text` caps them.
fn traceback(tb: &[Value], cap: bool) -> String {
    let chunks: Vec<String> = tb.iter().map(|c| strip_ansi(&text_str(c))).collect();
    if !cap { return chunks.join("\n"); }
    let Some((last, rest)) = chunks.split_last() else { return String::new(); };
    rest.iter().map(|c| c.split('\n').filter_map(tb_line).collect::<Vec<_>>().join("\n")).chain([last.clone()]).collect::<Vec<_>>().join("\n")
}

/// Text kept in one rendered reply. Longer text keeps its end, where results and exception messages are.
pub const TEXT_MAX: usize = 1 << 20;

/// `text` without its middle when it's longer than `head + tail` bytes: the first `head` and the last `tail` bytes, cut at
/// character boundaries, around a line giving the number of bytes removed.
pub fn keep_ends(text: String, head: usize, tail: usize) -> String {
    if text.len() <= head + tail { return text; }
    let (a, b) = (text.floor_char_boundary(head), text.ceil_char_boundary(text.len() - tail));
    let sep = if a > 0 { "\n" } else { "" };
    format!("{}{sep}[{} bytes of output removed]\n{}", &text[..a], b - a, &text[b..])
}

/// Outputs as a model reads them
pub struct Rendered {
    /// The text outputs, tagged as `render` describes, keeping the last `TEXT_MAX` bytes
    pub text: String,
    /// The MIME type and base64 data of each displayed image, one per output at most
    pub images: Vec<(&'static str, String)>,
    /// An output's `full` flag bypasses ordinary shortening, not the tool-text ceiling
    pub full: bool,
    /// The text is a single Markdown output
    pub md: bool,
}

/// A cell's stored outputs for `render`: each output's type and fields
pub fn stored(outs: &[Value]) -> impl Iterator<Item = (&str, &Value)> { outs.iter().map(|o| (o["output_type"].as_str().unwrap_or_default(), o)) }

/// The text and images in `outs`, each output's type and fields. Each stream shows as a terminal shows it. With `cap`,
/// over-long traceback lines are cut, for a person's terminal. With `code`, the outputs come from running code, and an
/// `execute_result` whose only data is the repr of a Python `str` shows the string itself. A kernel's `call` result
/// already holds a string's text. One output of plain text or Markdown is the text alone; several outputs, or one of
/// another type, are each tagged with their kind and MIME type.
pub fn render<'a>(outs: impl Iterator<Item = (&'a str, &'a Value)>, cap: bool, code: bool) -> Rendered {
    let mut texts: Vec<(String, Option<&'static str>, String)> = vec![];
    let (mut images, mut full) = (vec![], false);
    for (typ, o) in outs {
        match typ {
            "stream" => {
                let tag = o["name"].as_str().unwrap_or("stdout").to_string();
                let text = text_str(&o["text"]);
                if let Some((last, None, body)) = texts.last_mut() && last == &tag { body.push_str(&text) } else { texts.push((tag, None, text)) }
            }
            "error" => {
                let text = o["traceback"].as_array().map(|v| traceback(v, cap)).filter(|s| !s.is_empty())
                    .unwrap_or_else(|| format!("{}: {}", text_str(&o["ename"]), text_str(&o["evalue"])));
                texts.push(("error".into(), None, text));
            }
            "display_data" | "execute_result" => {
                full |= o["metadata"]["full"] == true;
                let data = o["data"].as_object();
                if let Some((mime, value)) = data.and_then(|d| first_mime(d, &TEXT_MIMES)) {
                    let s = text_str(value);
                    let s =
                        if code && typ == "execute_result" && mime == "text/plain" && data.is_some_and(|d| d.len() == 1) { py_str(&s).unwrap_or(s) } else { s };
                    texts.push((typ.into(), Some(mime), strip_ansi(&s)));
                }
                if let Some((mime, value)) = data.and_then(|d| first_mime(d, &IMG_MIMES)) { images.push((mime, text_str(value))) }
            }
            _ => {}
        }
    }
    let mut nonempty: Vec<_> = texts.into_iter().map(|(tag, mime, body)| {
        let body = if mime.is_none() { strip_ansi(&clean_cli(&body)) } else { body };
        (tag, mime, body)
    }).filter(|(_, _, s)| !s.is_empty()).collect();
    let md = matches!(nonempty.as_slice(), [(_, Some("text/markdown"), _)]);
    let bare = matches!(nonempty.as_slice(), [(_, None | Some("text/plain" | "text/markdown"), _)]);
    let text = if bare { nonempty.remove(0).2 } else {
        nonempty.into_iter().map(|(tag, mime, body)| {
            let attr = mime.filter(|&m| m != "text/plain").map(|m| format!(" mime=\"{m}\"")).unwrap_or_default();
            format!("<{tag}{attr}>\n{}\n</{tag}>", body.trim_end_matches('\n'))
        }).collect::<Vec<_>>().join("\n")
    };
    Rendered { text: keep_ends(text, 0, TEXT_MAX), images, full, md }
}

/// The string that `s`, the repr of a Python `str`, shows; `None` when `s` is anything else
fn py_str(s: &str) -> Option<String> {
    let q = s.chars().next().filter(|c| matches!(c, '\'' | '"'))?;
    let mut cs = s.strip_prefix(q)?.strip_suffix(q)?.chars();
    let hex = |cs: &mut std::str::Chars<'_>, n: usize| char::from_u32(u32::from_str_radix(&cs.take(n).collect::<String>(), 16).ok()?);
    let mut out = String::new();
    while let Some(c) = cs.next() {
        if c == q { return None; }
        out.push(if c != '\\' { c } else {
            match cs.next()? {
                'n' => '\n', 't' => '\t', 'r' => '\r', e @ ('\\' | '\'' | '"') => e, 'x' => hex(&mut cs, 2)?, 'u' => hex(&mut cs, 4)?, 'U' => hex(&mut cs, 8)?,
                _ => return None
            }
        });
    }
    Some(out)
}

/// Terminal output as it appears on screen: the last alternate screen, `\r` overwrites and `\b` backspaces
fn clean_cli(s: &str) -> String {
    let s = if s.contains("\x1b[?1049h") { match s.rsplit_once("\x1b[?1049l") { Some((_, after)) => after, None => return String::new() } } else { s };
    s.replace("\r\n", "\n").split('\n').map(|l| {
        let l = l.trim_end_matches('\r');
        let mut cs: Vec<char> = l.rsplit('\r').next().unwrap_or(l).chars().collect();
        while let Some(i) = cs.iter().position(|&c| c == '\x08') { if i == cs.len() - 1 { cs.truncate(i) } else { cs.drain(i.saturating_sub(1)..=i); } }
        cs.into_iter().collect::<String>()
    }).collect::<Vec<_>>().join("\n")
}

/// The MIME type a `ToolResponse` displays as: its content as a list of Responses content parts, which a tool result
/// passes to the model unchanged
pub const TOOL_RESPONSE_MIME: &str = "application/vnd.aidialog.tool-response+json";

/// A tool call's result
pub struct ToolResult {
    /// For the model: the text alone, or Responses content parts when there is media
    pub output: Value,
    /// For the stored reply, with a compact `<media>` tag for each image
    pub text: String,
    /// Never cut the stored text
    pub full: bool,
    /// The stored text is Markdown
    pub md: bool,
}

/// Model-input parts for `images`, each a MIME type and its base64 data: an image's `<media>` tag and the image, or a
/// `<media-unavailable>` note when a model with `caps` can't read it
pub fn image_parts(images: &[(&str, String)], caps: &Caps) -> Vec<Value> { images.iter().flat_map(|(mime, b64)| media::output_item(mime, b64, caps)).collect() }

/// A tool result from rendered output `text` and its `image_parts`. When `parts` holds no media, the result is the
/// text, then each note. Otherwise it's the parts, then the text. `full` and `md` are `Rendered`'s flags.
pub fn merge(text: String, mut parts: Vec<Value>, full: bool, md: bool) -> ToolResult {
    if !parts.iter().any(|p| p["type"] != "input_text") {
        let notes = parts.iter().filter_map(|p| p["text"].as_str().map(String::from));
        let text = std::iter::once(text).chain(notes).filter(|s| !s.is_empty()).collect::<Vec<_>>().join("\n\n");
        return ToolResult { output: text.clone().into(), text, full, md };
    }
    if !text.is_empty() { parts.push(media::text(text)) }
    let text = parts.iter().map(media::ctext).collect::<Vec<_>>().join("\n");
    ToolResult { output: parts.into(), text, full, md: false }
}

/// The result of a tool call from its outputs, each output's type and fields. An output whose data has
/// `TOOL_RESPONSE_MIME` gives its Responses content parts unchanged and in order. The other outputs give the images a
/// model with `caps` can read, shrunk to fit, then the text. `code` is `render`'s flag: true when the tool ran code,
/// false for a kernel's `call`.
pub fn tool_result<'a>(outs: impl Iterator<Item = (&'a str, &'a Value)>, caps: &Caps, code: bool) -> ToolResult {
    let (given, rest): (Vec<_>, Vec<_>) = outs.partition(|(_, o)| o["data"][TOOL_RESPONSE_MIME].is_array());
    let Rendered { text, images, full, md } = render(rest.into_iter(), false, code);
    let mut parts = image_parts(&images, caps);
    parts.extend(given.iter().flat_map(|(_, o)| o["data"][TOOL_RESPONSE_MIME].as_array().into_iter().flatten().cloned()));
    merge(text, parts, full, md)
}

static DISPLAY_ENV: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"^\\begin\{(align|equation|gather|multline|eqnarray)").expect("valid regex"));

/// Canonicalize a `text/latex` payload to `\(...\)`/`\[...\]` delimiters, or `$...$`/`$$...$$` when `dollars`
pub fn normalize_text_latex(s: &str, dollars: bool) -> String {
    if s.is_empty() { return String::new(); }
    let (disp, tex) = if let Some(inner) = s.strip_prefix(r"\[").and_then(|t| t.strip_suffix(r"\]")) { (true, inner) }
    else if let Some(inner) = s.strip_prefix(r"\(").and_then(|t| t.strip_suffix(r"\)")) { (false, inner) }
    else if let Some(inner) = s.strip_prefix("$$").and_then(|t| t.strip_suffix("$$")) { (true, inner) }
    else if let Some(inner) = s.strip_prefix('$').and_then(|t| t.strip_suffix('$')) { (false, inner) }
    else { (DISPLAY_ENV.is_match(s), s) };
    match (dollars, disp) {
        (true, true) => format!("$${tex}$$"),
        (true, false) => format!("${tex}$"),
        (false, true) => format!(r"\[{tex}\]"),
        (false, false) => format!(r"\({tex}\)"),
    }
}

/// Concatenate stdout, stderr and markdown streams, keeping `error` and `execute_result` outputs at the end
pub(crate) fn concat_streams(outputs: &[Value]) -> Vec<Value> {
    let (mut stdout, mut stderr, mut markdown) = (String::new(), String::new(), String::new());
    let (mut has_out, mut has_err, mut has_md) = (false, false, false);
    let (mut res, mut exec_results) = (Vec::new(), Vec::new());
    for out in outputs {
        match out["output_type"].as_str().unwrap_or("") {
            "stream" => {
                let text = text_str(&out["text"]);
                match out["name"].as_str().unwrap_or("") {
                    "stdout" => { stdout.push_str(&text); has_out = true; }
                    "stderr" => { stderr.push_str(&text); has_err = true; }
                    "markdown" => { markdown.push_str(&text); has_md = true; }
                    _ => res.push(out.clone()),
                }
            }
            "error" | "execute_result" => exec_results.push(out.clone()),
            _ => res.push(out.clone()),
        }
    }
    if has_out { res.push(json!({ "output_type": "stream", "name": "stdout", "text": clean_cli(&stdout) })); }
    if has_err { res.push(json!({ "output_type": "stream", "name": "stderr", "text": clean_cli(&stderr) })); }
    if has_md { res.push(json!({ "output_type": "display_data", "data": { "text/markdown": markdown }, "metadata": {} })); }
    res.extend(exec_results);
    res
}

/// Preferred MIME type and content for a Jupyter output dict
pub(crate) fn preferred_msg_out(out: &Value, html1st: bool, include_imgs: bool) -> (&'static str, String) {
    match out["output_type"].as_str().unwrap_or("") {
        "stream" => ("text/plain", text_str(&out["text"])),
        "error" => ("text/plain", out["traceback"].as_array().map(|a| a.iter().map(text_str).collect::<Vec<_>>().join("\n")).unwrap_or_default()),
        "execute_result" | "display_data" => {
            let Some(data) = out["data"].as_object() else { return ("text/plain", String::new()); };
            let mut order: Vec<&'static str> = if html1st {
                vec!["text/html", "text/markdown", "application/javascript", "text/latex"]
            } else {
                vec!["text/markdown", "text/html", "application/javascript", "text/latex"]
            };
            if include_imgs { order.extend(["image/jpeg", "image/png", "image/webp", "image/svg+xml"]); }
            order.push("text/plain");
            for mt in order {
                if let Some(v) = data.get(mt) {
                    let s = text_str(v);
                    if !s.is_empty() { return (mt, s); }
                }
            }
            ("text/plain", String::new())
        }
        _ => ("text/plain", format!("Error: Failed to parse unknown output - {out}")),
    }
}

/// Plain-text rendering of a Jupyter output list, as `aidialog.dialog.render_outputs_ai`
pub fn render_outputs_ai(outputs: &[Value], dollars: bool) -> String {
    concat_streams(outputs)
        .iter()
        .map(|o| {
            let (mime, d) = preferred_msg_out(o, false, false);
            match mime {
                "text/plain" => strip_ansi(&d),
                "text/html" | "text/markdown" => d,
                "text/latex" => normalize_text_latex(&d, dollars),
                "application/javascript" => format!("<script>{d}</script>"),
                _ => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Render notebook outputs as Markdown: text in `output` fences, HTML in `{=html}`, Markdown inlined, images as data URIs
pub fn render_md(outputs: &[Value], html1st: bool) -> String {
    let (mut parts, mut buf) = (Vec::new(), String::new());
    let flush = |buf: &mut String, parts: &mut Vec<String>| {
        let trimmed = buf.trim_end();
        if !trimmed.is_empty() { parts.push(crate::reply::fenced_ch(trimmed, "output", '`')); }
        buf.clear();
    };
    for o in concat_streams(outputs) {
        let (mime, d) = preferred_msg_out(&o, html1st, true);
        if d.is_empty() { continue; }
        match mime {
            "text/plain" => {
                let s = strip_ansi(&d);
                buf.push_str(&s);
                if !s.ends_with('\n') { buf.push('\n'); }
            }
            "text/html" | "image/svg+xml" => {
                flush(&mut buf, &mut parts);
                parts.push(crate::reply::fenced_ch(d.trim(), "{=html}", '`'));
            }
            "application/javascript" => {
                flush(&mut buf, &mut parts);
                parts.push(crate::reply::fenced_ch(&format!("<script>{d}</script>"), "{=html}", '`'));
            }
            "text/markdown" | "text/latex" => {
                flush(&mut buf, &mut parts);
                parts.push(d.trim().to_owned());
            }
            "image/jpeg" | "image/png" | "image/webp" => {
                flush(&mut buf, &mut parts);
                let b64: String = d.split_whitespace().collect();
                parts.push(format!("![](data:{mime};base64,{b64})"));
            }
            _ => {}
        }
    }
    flush(&mut buf, &mut parts);
    parts.join("\n\n")
}

/// Concise ANSI-stripped text for outputs, wrapped in XML-ish tags when multiple outputs appear
pub fn render_text(outputs: &[Value], html1st: bool, cap_tb: bool) -> String {
    let mut items: Vec<(String, String)> = Vec::new();
    for out in concat_streams(outputs) {
        let typ = out["output_type"].as_str().unwrap_or("");
        let (mime, raw) = if typ == "error" && cap_tb {
            ("text/plain", traceback(out["traceback"].as_array().map(Vec::as_slice).unwrap_or_default(), true))
        } else {
            preferred_msg_out(&out, html1st, false)
        };
        let d = strip_ansi(&raw);
        if d.is_empty() { continue; }
        let tag = if typ == "stream" { out["name"].as_str().unwrap_or("stream") } else { typ };
        let attr = if matches!(typ, "execute_result" | "display_data") && mime != "text/plain" {
            format!(r#" mime="{mime}""#)
        } else {
            String::new()
        };
        let body = if d.ends_with('\n') { format!("\n{d}") } else { format!("\n{d}\n") };
        items.push((d, format!("<{tag}{attr}>{body}</{tag}>")));
    }
    if items.is_empty() { return String::new(); }
    if items.len() == 1 { return items.remove(0).0; }
    items.into_iter().map(|(_, tagged)| tagged).collect::<Vec<_>>().join("\n")
}
