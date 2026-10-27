//! Approximate token counts for cells: a character-class estimate calibrated against o200k,
//! plus fixed charges for images and PDF pages. Counts are derived on demand, never stored.

use base64::{Engine, engine::general_purpose::STANDARD};
use regex::bytes::Regex;
use serde_json::{Map, Value, json};
use std::io::Read;
use std::sync::LazyLock;

use crate::cells::text_str;
use crate::output::strip_ansi;

/// Tokens charged per raster image, whatever its size
pub const IMG_TOKENS: u64 = 1200;
/// Tokens charged per PDF page
pub const PDF_PAGE_TOKENS: u64 = 2663;
/// Text mimes in reading preference order; the first one present is the one counted
pub const TEXT_MIMES: [&str; 5] = ["text/markdown", "text/html", "application/javascript", "text/latex", "text/plain"];

/// The first of `mimes` that `data` has a non-empty value for, with its value, as fastcore's `preferred_out` picks it
pub fn first_mime<'a>(data: &'a Map<String, Value>, mimes: &[&'static str]) -> Option<(&'static str, &'a Value)> {
    let full = |v: &&Value| match v { Value::String(s) => !s.is_empty(), Value::Array(a) => !a.is_empty(), Value::Null => false, _ => true };
    mimes.iter().find_map(|m| data.get(*m).filter(full).map(|v| (*m, v)))
}

/// Approximate tokens in `text`, within about 20% of o200k for most notebook cells. A run of
/// ASCII alphanumerics counts one token up to 12 chars; longer runs count `len/1.7` with a
/// digit in them (hex, base64, hashes) and `len/5` without; digit-only runs count `len/3`.
/// Any other non-space character counts one, a run of the same character once.
pub fn count(text: &str) -> u64 {
    let mut n = 0.0f64;
    let mut it = text.chars().peekable();
    while let Some(c) = it.next() {
        if c.is_ascii_alphanumeric() || c == '_' {
            let (mut len, mut digits, mut has_digit) = (1.0f64, c.is_ascii_digit(), c.is_ascii_digit());
            while let Some(&d) = it.peek() && (d.is_ascii_alphanumeric() || d == '_') {
                it.next();
                len += 1.0;
                if d.is_ascii_digit() { has_digit = true } else { digits = false }
            }
            n += if digits { (len / 3.0).ceil() } else if len <= 12.0 { 1.0 } else if has_digit { len / 1.7 } else { len / 5.0 };
        } else if !c.is_whitespace() {
            while it.peek() == Some(&c) { it.next(); }
            n += 1.0;
        }
    }
    n.round() as u64
}

static PDF_COUNT: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?s)/Type\s*/Pages[^>]*?/Count\s+(\d+)|/Count\s+(\d+)[^>]*?/Type\s*/Pages").expect("a valid regex"));
static OBJ_STM: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)/Type\s*/ObjStm.*?stream\r?\n").expect("a valid regex"));

/// Page count of PDF `b`: the largest `/Count` on a `/Pages` node, in the clear or inside an
/// inflated object stream.
pub fn pdf_pages(b: &[u8]) -> u64 {
    let inflated: Vec<Vec<u8>> = OBJ_STM.find_iter(b).map(|m| {
        let mut out = vec![];
        let _ = flate2::read::ZlibDecoder::new(&b[m.end()..]).read_to_end(&mut out);
        out
    }).collect();
    std::iter::once(b).chain(inflated.iter().map(Vec::as_slice)).flat_map(|o| PDF_COUNT.captures_iter(o))
        .filter_map(|c| c.get(1).or(c.get(2)).and_then(|m| std::str::from_utf8(m.as_bytes()).ok()?.parse::<u64>().ok())).max().unwrap_or(0)
}

/// Tokens for a mime bundle, an output's `data` or one attachment: its preferred text as text,
/// `IMG_TOKENS` per raster image, `PDF_PAGE_TOKENS` per page of a PDF.
pub fn bundle(data: &Map<String, Value>) -> u64 {
    let text = first_mime(data, &TEXT_MIMES).map_or(0, |(_, v)| count(&text_str(v)));
    let media: u64 = data.iter().map(|(mime, v)| match mime.as_str() {
        "application/pdf" => pdf_pages(&STANDARD.decode(text_str(v).trim()).unwrap_or_default()) * PDF_PAGE_TOKENS,
        m if m.starts_with("image/") && m != "image/svg+xml" => IMG_TOKENS, _ => 0
    }).sum();
    text + media
}

/// Tokens for a code cell's outputs: stream text, ANSI-stripped tracebacks, and each rich
/// output's bundle.
pub fn outputs(outs: &[Value]) -> u64 {
    outs.iter().map(|o| match o["output_type"].as_str() {
        Some("stream") => count(&text_str(&o["text"])),
        Some("error") => o["traceback"].as_array().map_or(0, |tb| tb.iter().map(|l| count(&strip_ansi(&text_str(l)))).sum()),
        _ => o["data"].as_object().map_or(0, bundle),
    }).sum()
}

/// `tokens` for `cell`: `source` always, `outputs` when the cell has them, and `attachments` when `atts`, the cell's
/// attachments as `cells::stored_atts` finds them, has any
pub fn cell(cell: &Value, atts: Option<&Map<String, Value>>) -> Value {
    let mut t = json!({"source": count(&text_str(&cell["source"]))});
    if let Some(outs) = cell["outputs"].as_array() { t["outputs"] = outputs(outs).into(); }
    if let Some(atts) = atts.filter(|m| !m.is_empty()) { t["attachments"] = atts.values().filter_map(Value::as_object).map(bundle).sum::<u64>().into(); }
    t
}

/// The sum of `cell`'s token counts
pub fn total(cell: &Value, atts: Option<&Map<String, Value>>) -> u64 {
    self::cell(cell, atts).as_object().into_iter().flat_map(|t| t.values()).filter_map(Value::as_u64).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn counts_by_character_class() {
        assert_eq!(count("def fetch_daily(lat, lon): pass"), 9);
        assert_eq!(count("**======turn: 8======**"), 7);
        assert_eq!(count("2024-01-01"), 6);
        assert_eq!(count("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nG"), 34);
        assert_eq!(count("  \n\t"), 0);
    }

    #[test]
    fn pdf_pages_reads_clear_and_object_streams() {
        assert_eq!(pdf_pages(b"%PDF-1.4\n1 0 obj << /Type /Pages /Kids [] /Count 3 >> endobj"), 3);
        let mut z = flate2::write::ZlibEncoder::new(vec![], flate2::Compression::default());
        z.write_all(b"1 0 << /Type /Pages /Count 7 >>").unwrap();
        let mut pdf = b"%PDF-1.5\n2 0 obj << /Type /ObjStm /N 1 /First 4 >> stream\n".to_vec();
        pdf.extend(z.finish().unwrap());
        pdf.extend(b"\nendstream endobj");
        assert_eq!(pdf_pages(&pdf), 7);
        assert_eq!(pdf_pages(b"not a pdf"), 0);
    }

    #[test]
    fn media_charge_by_constant() {
        let data: Map<String, Value> = serde_json::from_value(json!({"text/plain": "<Figure>", "image/png": "aGk="})).unwrap();
        assert_eq!(bundle(&data), IMG_TOKENS + count("<Figure>"));
        let pdf = STANDARD.encode(b"%PDF-1.4 << /Type /Pages /Count 2 >>");
        let data: Map<String, Value> = serde_json::from_value(json!({"application/pdf": pdf})).unwrap();
        assert_eq!(bundle(&data), 2 * PDF_PAGE_TOKENS);
        let cell_v = json!({"id": "c", "cell_type": "markdown", "source": "# hi"});
        assert_eq!(cell(&cell_v, None), json!({"source": 2}));
    }
}
