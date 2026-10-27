//! Media in model input: MIME detection, `<media>` tags, resizing images, and Responses content parts.

use base64::Engine;
use serde_json::{Value, json};

use crate::Caps;

/// Largest image area, in pixels, sent to a model: four OpenAI high-detail tiles
pub const IM_MAX: u32 = 768 * 768;
/// Raster MIME types that vision APIs accept, in fastcore's `IMG_MIMES` order. `resize` limits them to `IM_MAX`.
pub const IMG_MIMES: [&str; 3] = ["image/jpeg", "image/png", "image/webp"];
/// The note for media of a type the model can't read, with `{mime}` standing for the type
pub const UNSUPPORTED: &str = "An unsupported media type was included in the context: {mime}. Please instruct the user to switch to a model that supports this media type.";

/// Whether media of type `mime` can go into model input: the raster images in `IMG_MIMES`, PDF, audio and video
pub fn readable(mime: &str) -> bool { IMG_MIMES.contains(&mime) || mime == "application/pdf" || mime.starts_with("audio/") || mime.starts_with("video/") }

/// The MIME type of `d` from its leading bytes: PDF, common audio and video formats, and images.
pub fn detect_mime(d: &[u8]) -> Option<&'static str> {
    let at = |pos: usize, sig: &[u8]| d.get(pos..pos + sig.len()) == Some(sig);
    let sub = |sig: &[u8]| d.get(8..8 + sig.len()) == Some(sig);
    Some(if at(0, b"%PDF") { "application/pdf" }
    else if at(0, b"RIFF") {
        if sub(b"WAVE") { "audio/wav" } else if sub(b"AVI ") { "video/avi" } else if sub(b"WEBP") { "image/webp" } else { return None }
    }
    else if at(0, b"ID3") || at(0, b"\xff\xfb") || at(0, b"\xff\xf3") { "audio/mp3" }
    else if at(0, b"FORM") { if sub(b"AIFF") { "audio/aiff" } else { return None } }
    else if at(0, b"OggS") { "audio/ogg" }
    else if at(0, b"fLaC") { "audio/flac" }
    else if at(4, b"ftyp") { if sub(b"3gp") { "video/3gpp" } else { "video/mp4" } }
    else if at(0, b"\x1a\x45\xdf") { "video/webm" }
    else if at(0, b"FLV") { "video/x-flv" }
    else if at(0, b"\x30\x26\xb2\x75") { "video/wmv" }
    else if at(0, b"\x00\x00\x01\xb3") { "video/mpeg" }
    else if at(0, b"\x89PNG\r\n\x1a\n") { "image/png" }
    else if at(0, b"\xff\xd8\xff") { "image/jpeg" }
    else if at(0, b"GIF87a") || at(0, b"GIF89a") { "image/gif" }
    else if at(0, b"BM") { "image/bmp" }
    else if at(0, b"II*\0") || at(0, b"MM\0*") { "image/tiff" }
    else { return None })
}

/// Can a model with `caps` read media of type `mime`? `caps.input_modalities` lists the kinds it reads besides text:
/// `image`, `pdf`, `audio` or `video`.
pub fn supported(mime: &str, caps: &Caps) -> bool {
    let kind = match mime.split('/').next() { _ if mime == "application/pdf" => "pdf", Some(k @ ("image" | "audio" | "video")) => k, _ => return false };
    caps.input_modalities.iter().any(|k| k == kind)
}

/// `data` with its area limited to `max` pixels, in the same format. Other media are unchanged.
pub fn resize(data: Vec<u8>, mime: &str, max: u32) -> Result<Vec<u8>, String> {
    if !IMG_MIMES.contains(&mime) { return Ok(data) }
    let img = image::load_from_memory(&data).map_err(|e| e.to_string())?;
    let (w, h) = (img.width() as f64, img.height() as f64);
    if w * h <= max as f64 { return Ok(data) }
    let s = (max as f64 / (w * h)).sqrt();
    let fmt = image::ImageFormat::from_mime_type(mime).ok_or_else(|| format!("unknown image format {mime}"))?;
    let mut out = std::io::Cursor::new(Vec::new());
    img.resize_exact((w * s) as u32, (h * s) as u32, image::imageops::FilterType::Lanczos3).write_to(&mut out, fmt).map_err(|e| e.to_string())?;
    Ok(out.into_inner())
}

/// `s` with `&`, `<` and `>` escaped for XML
pub fn escape(s: &str) -> String { s.replace('&', "&amp;").replace('<', "&lt;").replace('>', "&gt;") }

/// ` name="value"`, quoted and escaped as fastcore's `to_xml` does: single quotes when `value` has a `"`
pub fn xml_attr(name: &str, value: &str) -> String {
    let v = escape(value);
    if v.contains('"') { format!(" {name}='{}'", v.replace('\'', "&#39;")) } else { format!(" {name}=\"{v}\"") }
}

/// The `<media>` tag that names the media part after it
pub fn tag(typ: &str, id: &str, mime: &str) -> String {
    format!("<media{}{}{}></media>", xml_attr("id", id), xml_attr("type", typ), xml_attr("mime", mime))
}

/// A `<media-unavailable>` tag, telling the model it can't see this media
pub fn unavailable(typ: &str, id: &str, reason: &str, mime: Option<&str>) -> String {
    let mime = mime.map(|m| xml_attr("mime", m)).unwrap_or_default();
    format!("<media-unavailable{}{}{}{mime}>The user included this media, but it was not available to the AI. Tell the user that you cannot see it.</media-unavailable>",
        xml_attr("id", id), xml_attr("type", typ), xml_attr("reason", reason))
}

/// A short id for output image `data`: the first 8 characters of its base32 MD5
pub fn img_id(data: &[u8]) -> String {
    use md5::{Digest, Md5};
    data_encoding::BASE32.encode(&Md5::digest(data))[..8].into()
}

/// Model-input parts for an output image given as notebook base64. The image's id is its `img_id`, or "unknown" when the
/// base64 doesn't decode.
pub fn output_item(mime: &str, b64: &str, caps: &Caps) -> Vec<Value> {
    let data = decode(b64);
    let id = data.as_deref().map_or_else(|_| "unknown".into(), img_id);
    item("output", &id, data, Some(mime), caps)
}

/// A Responses `input_text` part
pub fn text(s: impl Into<String>) -> Value { json!({"type": "input_text", "text": s.into()}) }

/// The text that stands for Responses content part `p` in a stored tool result, as aidialog's `ctext` writes it:
/// a text part's text, or a compact tag such as `<media input_image image/png>`
pub fn ctext(p: &Value) -> String {
    let typ = p["type"].as_str().unwrap_or_default();
    if typ == "input_text" { return p["text"].as_str().unwrap_or_default().into() }
    if let Some(mime) = p["mime"].as_str() { return format!("<media {typ} {mime}>") }
    let url = ["image_url", "video_url", "file_data"].iter().find_map(|k| p[*k].as_str()).unwrap_or_default();
    match url.strip_prefix("data:").and_then(|u| u.split_once(';')) { Some((mime, _)) => format!("<media {typ} {mime}>"), None => format!("<media {typ}>") }
}

/// The Responses content part holding media `data` of type `mime`
pub fn part(mime: &str, data: &[u8]) -> Value {
    let url = format!("data:{mime};base64,{}", base64::engine::general_purpose::STANDARD.encode(data));
    match mime.split('/').next() {
        Some("image") => json!({"type": "input_image", "image_url": url}),
        Some("video") => json!({"type": "input_video", "video_url": url}),
        Some("audio") => json!({"type": "input_audio", "text": url, "mime": mime}),
        _ => json!({"type": "input_file", "file_data": url, "filename": format!("upload.{}", mime.rsplit('/').next().unwrap_or_default())}),
    }
}

/// Bytes from notebook base64 text, which may be wrapped across lines
pub fn decode(s: &str) -> Result<Vec<u8>, String> {
    let s: String = s.split_whitespace().collect();
    base64::engine::general_purpose::STANDARD.decode(s).map_err(|e| e.to_string())
}

/// Model-input parts for one media item: the item's tag, then its data.
/// Missing, unrecognised or unsupported data gives one `<media-unavailable>` tag instead. `mime` overrides detection.
pub fn item(typ: &str, id: &str, data: Result<Vec<u8>, String>, mime: Option<&str>, caps: &Caps) -> Vec<Value> {
    let mut known = mime;
    let res = data.and_then(|data| {
        let mime = mime.or_else(|| detect_mime(&data)).ok_or("unsupported or unknown media type")?;
        known = Some(mime);
        if !supported(mime, caps) {
            return Err(UNSUPPORTED.replace("{mime}", mime));
        }
        let data = resize(data, mime, IM_MAX)?;
        Ok(vec![text(tag(typ, id, mime)), part(mime, &data)])
    });
    res.unwrap_or_else(|e| vec![text(unavailable(typ, id, &e, known))])
}

