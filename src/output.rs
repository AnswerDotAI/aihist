//! Kernel outputs as a model reads them: a code cell's outputs, and the result of a tool call that runs code.

use serde_json::Value;

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
    if !cap { return chunks.join("\n") }
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
                    let s = if code && typ == "execute_result" && mime == "text/plain" && data.is_some_and(|d| d.len() == 1) { py_str(&s).unwrap_or(s) } else { s };
                    texts.push((typ.into(), Some(mime), strip_ansi(&s)));
                }
                if let Some((mime, value)) = data.and_then(|d| first_mime(d, &IMG_MIMES)) { images.push((mime, text_str(value))) }
            }
            _ => {}
        }
    }
    let mut nonempty: Vec<_> = texts.into_iter()
        .map(|(tag, mime, body)| { let body = if mime.is_none() { strip_ansi(&clean_cli(&body)) } else { body }; (tag, mime, body) })
        .filter(|(_, _, s)| !s.is_empty()).collect();
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
        if c == q { return None }
        out.push(if c != '\\' { c } else { match cs.next()? {
            'n' => '\n', 't' => '\t', 'r' => '\r', e @ ('\\' | '\'' | '"') => e,
            'x' => hex(&mut cs, 2)?, 'u' => hex(&mut cs, 4)?, 'U' => hex(&mut cs, 8)?,
            _ => return None,
        }});
    }
    Some(out)
}

/// Terminal output as it appears on screen: the last alternate screen, `\r` overwrites and `\b` backspaces
fn clean_cli(s: &str) -> String {
    let s = if s.contains("\x1b[?1049h") {
        match s.rsplit_once("\x1b[?1049l") { Some((_, after)) => after, None => return String::new() }
    } else { s };
    s.replace("\r\n", "\n").split('\n').map(|l| {
        let l = l.trim_end_matches('\r');
        let mut cs: Vec<char> = l.rsplit('\r').next().unwrap_or(l).chars().collect();
        while let Some(i) = cs.iter().position(|&c| c == '\x08') {
            if i == cs.len() - 1 { cs.truncate(i) } else { cs.drain(i.saturating_sub(1)..=i); }
        }
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
pub fn image_parts(images: &[(&str, String)], caps: &Caps) -> Vec<Value> {
    images.iter().flat_map(|(mime, b64)| media::output_item(mime, b64, caps)).collect()
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keep_ends_cuts_the_middle_at_char_boundaries() {
        assert_eq!(keep_ends("short".into(), 5, 5), "short");
        let kept = keep_ends("€ab€cd€".into(), 4, 5);
        assert_eq!(kept, "€a\n[4 bytes of output removed]\ncd€");
    }
}
