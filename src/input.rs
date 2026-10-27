//! A prompt's model input as Responses input items: the examples, CRAFTs and `$` values that the host supplies, the
//! dialog's path, the cells above the prompt, any warnings, then the prompt. Cells that aren't pinned are dropped,
//! oldest first, to fit 80% of the model's input window.

use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::Caps;
use crate::cells::{ai_fmt, envelope, msg, parts, shown, stored_atts, text_str, truthy};
use crate::media::{self, escape, text, xml_attr};
use crate::{reply, tokens};

/// How many more tokens than `tokens::count` other providers' tokenizers may find
const ALLOWANCE: f64 = 1.4;
/// The share of the model's input window that the input may fill
const FILL: f64 = 0.8;
/// The input window assumed for a model whose `Caps` don't give one
const WINDOW: u64 = 128_000;

/// What a prompt's model input is built from
#[derive(Debug, Default, Deserialize)]
#[serde(default)]
pub struct Prompt {
    /// The dialog's path, which the model sees in a `<dialog path>` tag; empty leaves the tag out
    pub path: String,
    /// The dialog's notebook
    pub nb: Value,
    /// The prompt cell's id. The cells above it are its context.
    pub prompt: String,
    /// Dialogs whose cells go first as CRAFTs, in order
    pub crafts: Vec<Craft>,
    /// Each `$` expression with its `eval` record, in the format of a Jupyter execute reply's `user_expressions`
    pub vars: Map<String, Value>,
    /// An error for each tool whose schema couldn't be read
    pub tool_errors: Vec<String>,
    /// Errors from reading or running CRAFTs
    pub craft_errors: Vec<String>,
    /// What the model can read
    pub caps: Caps,
    /// The system prompt that the host sends with the input. Only its size counts here.
    pub instructions: Option<String>,
    /// Example notebooks, which go before everything else. Their last message ends a prefix that a provider can cache.
    pub examples: Vec<Value>,
    /// Each cell's time, by cell id, in the user's time zone
    pub times: Map<String, Value>,
}

/// A dialog whose cells go into the context as a CRAFT
#[derive(Debug, Default, Deserialize)]
pub struct Craft {
    /// The dialog's path, which the model sees in its `<craft-block path>` tag, after a `/`
    pub path: String,
    /// The dialog's notebook
    pub nb: Value,
}

/// `n` tokens with the allowance for other tokenizers
fn scaled(n: u64) -> u64 { (n as f64 * ALLOWANCE) as u64 }

/// One message of model input: its content parts and, for a prompt, its stored reply. `cache` ends a stable prefix
/// after the reply, which a provider can cache.
struct Entry { parts: Vec<Value>, reply: Option<String>, pinned: bool, tokens: u64, cache: bool }

impl Entry {
    /// A message of one text part, such as a tag that opens a CRAFT
    fn text(s: String) -> Entry { Entry { tokens: scaled(tokens::count(&s)), parts: vec![text(s)], reply: None, pinned: true, cache: false } }
}

/// Cell `cell` of notebook `nb` as a message. `time` is the cell's time in the user's time zone.
fn entry(nb: &Value, cell: &Value, caps: &Caps, time: Option<&str>) -> Entry {
    let m = msg(cell);
    let reply = (m.typ == "prompt").then(|| ai_fmt(&m.out));
    Entry { parts: parts(nb, cell, caps, time), reply, pinned: truthy(&cell["metadata"]["pinned"]), tokens: scaled(tokens::total(cell, stored_atts(nb, cell))), cache: false }
}

/// The messages of every cell in `nb` that the model sees
fn shown_entries(nb: &Value, caps: &Caps) -> Vec<Entry> {
    nb["cells"].as_array().into_iter().flatten().filter(|c| shown(c)).map(|c| entry(nb, c, caps, None)).collect()
}

/// A note to the model that the user may need to know about
pub fn reminder(s: &str) -> String { format!("<system-reminder>**NB**: {s}</system-reminder>") }

/// The warning about undefined `$` names, unreadable tool schemas and CRAFT errors, if there are any
pub fn warning(undef: &[String], tool_errors: &[String], craft_errors: &[String]) -> Option<String> {
    let list = |head: &str, xs: &[String]| format!("{head}\n\n- {}", xs.join("\n\n- "));
    let mut tags = vec![];
    if !undef.is_empty() { tags.push(format!("<variable-errors>{}</variable-errors>", escape(&list("The following vars are undefined:", undef)))) }
    if !tool_errors.is_empty() { tags.push(format!("<tool-errors>{}</tool-errors>", escape(&list("Errors in tools:", tool_errors)))) }
    if !craft_errors.is_empty() { tags.push(format!("<context-warning>{}</context-warning>", escape(&craft_errors.join("\n")))) }
    (!tags.is_empty()).then(|| reminder(&format!("Notify the user of this if it might be relevant to their context: {}", tags.join("\n\n"))))
}

/// Whether `eval` record `r` failed because its expression names nothing
pub fn is_nameerr(r: &Value) -> bool { r["status"] != "ok" && r["ename"] == "NameError" }

/// The content of the turn that gives the model `$` values: a `<variables>` element, then any media values. Also
/// returns the expressions that name nothing. The content is empty when every expression names nothing.
pub fn vars(records: &Map<String, Value>, caps: &Caps) -> (Vec<Value>, Vec<String>) {
    let (mut names, mut vs, mut imgs, mut undef) = (vec![], String::new(), vec![], vec![]);
    for (expr, r) in records {
        if is_nameerr(r) { undef.push(expr.clone()); continue }
        names.push(format!("$`{expr}`"));
        let data = r["data"].as_object();
        if let Some((mime, v)) = data.and_then(|d| d.iter().find(|(m, _)| media::readable(m))) {
            imgs.extend(media::item("variable", expr, media::decode(&text_str(v)), Some(mime), caps));
            continue;
        }
        let val = if r["status"] == "ok" {
            data.and_then(|d| tokens::first_mime(d, &tokens::TEXT_MIMES)).map(|(_, v)| text_str(v)).unwrap_or_default()
        } else { format!("{}: {}", text_str(&r["ename"]), text_str(&r["evalue"])) };
        let typ = r["metadata"]["type"].as_str().map(|t| xml_attr("type", t)).unwrap_or_default();
        vs += &format!("<variable{}{typ}>{val}</variable>", xml_attr("name", expr));
    }
    let mut parts = vec![];
    if !vs.is_empty() { parts.push(text(format!("<variables>The 'variables' section provides values of: {}{vs}</variables>", names.join(", ")))) }
    parts.extend(imgs);
    (parts, undef)
}

/// The reply that acknowledges the `$` values
pub const VARS_REPLY: &str = "Understood, I will use these variable values.";

/// `hist` within `budget` tokens: every pinned message, then the newest others while they fit. Also returns how
/// many were dropped.
fn fit(hist: Vec<Entry>, budget: u64) -> (Vec<Entry>, usize) {
    let mut used: u64 = hist.iter().filter(|e| e.pinned).map(|e| e.tokens).sum();
    let mut keep: Vec<bool> = hist.iter().map(|e| e.pinned).collect();
    for (i, e) in hist.iter().enumerate().rev().filter(|(_, e)| !e.pinned) {
        used += e.tokens;
        if used > budget { break }
        keep[i] = true;
    }
    let n = hist.len();
    let kept: Vec<Entry> = hist.into_iter().zip(keep).filter_map(|(e, k)| k.then_some(e)).collect();
    let dropped = n - kept.len();
    (kept, dropped)
}

/// Responses items for `entries`. Each prompt ends a user turn and adds its stored reply, except the last entry,
/// which is the prompt being answered.
fn turns(entries: Vec<Entry>) -> Vec<Value> {
    let (mut input, mut user) = (vec![], vec![]);
    let n = entries.len();
    for (i, e) in entries.into_iter().enumerate() {
        user.extend(e.parts);
        let Some(r) = e.reply.filter(|_| i + 1 < n) else { continue };
        input.push(json!({"type": "message", "role": "user", "content": std::mem::take(&mut user)}));
        input.extend(reply::items(&reply::parse(&r)));
        if e.cache { input.last_mut().expect("a reply's items")["content"][0]["cache_control"] = json!({"type": "ephemeral"}) }
    }
    input.push(json!({"type": "message", "role": "user", "content": user}));
    input
}

/// The model input for prompt `p`: the examples, the CRAFTs, the `$` values, the dialog's path, the cells above the
/// prompt that fit, any warnings, then the prompt
pub fn build(p: &Prompt) -> Result<Vec<Value>, String> {
    let all = p.nb["cells"].as_array().ok_or("the notebook has no cells")?;
    let at = all.iter().position(|c| c["id"] == p.prompt.as_str()).ok_or_else(|| format!("no prompt cell {}", p.prompt))?;
    let caps = &p.caps;
    let mut pre: Vec<Entry> = p.examples.iter().flat_map(|nb| shown_entries(nb, caps)).collect();
    if let Some(e) = pre.last_mut() { e.cache = true }
    if !p.crafts.is_empty() {
        pre.push(Entry::text("# Loaded CRAFTs.".into()));
        for c in &p.crafts {
            pre.push(Entry::text(format!("<craft-block{}>", xml_attr("path", &format!("/{}", c.path)))));
            pre.extend(shown_entries(&c.nb, caps));
            pre.push(Entry::text("</craft-block>".into()));
        }
        pre.push(Entry { reply: Some("Ok!".into()), cache: true, ..Entry::text(envelope("End of CRAFTs.", "crafts", None)) });
    }
    let (vparts, undef) = vars(&p.vars, caps);
    if !vparts.is_empty() {
        let tokens = vparts.iter().map(|p| p["text"].as_str().map_or(tokens::IMG_TOKENS, |t| scaled(tokens::count(t)))).sum();
        pre.push(Entry { parts: vparts, reply: Some(VARS_REPLY.into()), pinned: true, tokens, cache: false });
    }
    if !p.path.is_empty() { pre.push(Entry::text(format!("<dialog{}/>", xml_attr("path", &p.path)))) }
    let time = |c: &Value| p.times.get(c["id"].as_str().unwrap_or_default()).and_then(Value::as_str);
    let mut last: Vec<Entry> = warning(&undef, &p.tool_errors, &p.craft_errors).map(Entry::text).into_iter().collect();
    last.push(entry(&p.nb, &all[at], caps, time(&all[at])));
    let hist = all[..at].iter().filter(|c| shown(c)).map(|c| entry(&p.nb, c, caps, time(c))).collect();
    let fixed: u64 = pre.iter().chain(&last).map(|e| e.tokens).sum::<u64>() + p.instructions.as_deref().map_or(0, |s| scaled(tokens::count(s)));
    let budget = (caps.max_input_tokens.unwrap_or(WINDOW) as f64 * FILL) as u64;
    let (hist, dropped) = fit(hist, budget.saturating_sub(fixed));
    if dropped > 0 {
        last.insert(0, Entry::text(reminder(&format!("Notify the user if relevant: {dropped} older non-pinned cell(s) were omitted to fit the model context."))));
    }
    Ok(turns(pre.into_iter().chain(hist).chain(last).collect()))
}
