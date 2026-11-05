//! A prompt's model input as Responses input items: the examples, CRAFTs and `$` values that the host supplies, the
//! dialog's path, the cells above the prompt, any warnings, then the prompt. Cells that aren't pinned are dropped,
//! oldest first, to fit 80% of the model's input window.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};
use std::path::Path;

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
#[derive(Debug, Default, Deserialize, Serialize)]
#[serde(default)]
pub struct Turn { pub parts: Vec<Value>, pub reply: Option<String>, pub cache: bool }

struct Entry { turn: Turn, tokens: u64 }

struct Candidate { index: usize, tokens: u64, pinned: bool, boundary: bool }

impl Entry {
    /// A message of one text part, such as a tag that opens a CRAFT
    fn text(s: String) -> Entry { Entry { tokens: scaled(tokens::count(&s)), turn: Turn { parts: vec![text(s)], ..Default::default() } } }
}

/// One cell's model parts and its normalized saved reply, with complete call IDs.
pub async fn turn(nb: &Value, cell: &Value, caps: &Caps, time: Option<&str>, ctx: &media::Context<'_>) -> Turn {
    Turn { parts: parts(nb, cell, caps, time, ctx).await, reply: crate::cells::is_prompt(cell).then(|| ai_fmt(&msg(cell).out)), cache: false }
}

/// Cell `cell` of notebook `nb` as a message. `time` is the cell's time in the user's time zone.
async fn entry(nb: &Value, cell: &Value, caps: &Caps, time: Option<&str>, ctx: &media::Context<'_>) -> Entry {
    Entry { turn: turn(nb, cell, caps, time, ctx).await, tokens: scaled(tokens::total(cell, stored_atts(nb, cell))) }
}

/// The messages of every cell in `nb` that the model sees
async fn shown_entries(nb: &Value, caps: &Caps, ctx: &media::Context<'_>) -> Vec<Entry> {
    let mut res = vec![];
    for c in nb["cells"].as_array().into_iter().flatten().filter(|c| shown(c)) { res.push(entry(nb, c, caps, None, ctx).await); }
    res
}

/// Sorted unique references from shown preceding cells, the addressed prompt, and every CRAFT cell, before fitting.
pub fn references(nb: &Value, prompt: &str, crafts: &[Craft]) -> Result<rgapi::CellRefs, String> {
    let cells = nb["cells"].as_array().ok_or("the notebook has no cells")?;
    let at = cells.iter().position(|c| c["id"] == prompt).ok_or_else(|| format!("no prompt cell {prompt}"))?;
    let mut refs = rgapi::CellRefs::default();
    for c in cells[..at].iter().filter(|c| shown(c)).chain(cells.get(at))
        .chain(crafts.iter().flat_map(|c| c.nb["cells"].as_array().into_iter().flatten())) {
        let r = rgapi::cell_refs(c);
        refs.vars.extend(r.vars);
        refs.cmds.extend(r.cmds);
        refs.tools.extend(r.tools);
    }
    for names in [&mut refs.vars, &mut refs.cmds, &mut refs.tools] { names.sort(); names.dedup(); }
    Ok(refs)
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

/// `hist` within `budget` tokens: every pinned message and the newest suffix starting at a chunk boundary.
/// Also returns how many were dropped.
fn fit(hist: Vec<Candidate>, budget: u64) -> (Vec<usize>, usize) {
    let mut used: u64 = hist.iter().filter(|e| e.pinned).map(|e| e.tokens).sum();
    let mut start = hist.len();
    for (i, e) in hist.iter().enumerate().rev() {
        if !e.pinned { used += e.tokens }
        if used > budget { break }
        if e.boundary { start = i }
    }
    let n = hist.len();
    let kept: Vec<usize> = hist.into_iter().enumerate().filter_map(|(i, e)| (e.pinned || i >= start).then_some(e.index)).collect();
    let dropped = n - kept.len();
    (kept, dropped)
}

/// Group cell parts through each supplied reply; a final unanswered turn retains its parts without a reply.
pub fn group_turns(entries: Vec<Turn>) -> Vec<Turn> {
    let (mut res, mut user) = (vec![], vec![]);
    for e in entries {
        user.extend(e.parts);
        if e.reply.is_some() { res.push(Turn { parts: std::mem::take(&mut user), reply: e.reply, cache: e.cache }); }
    }
    if !user.is_empty() { res.push(Turn { parts: user, ..Default::default() }); }
    res
}

/// Responses input items for cell contributions. All supplied replies and complete call IDs are retained.
pub fn turns(entries: Vec<Turn>) -> Vec<Value> {
    let mut input = vec![];
    for e in group_turns(entries) {
        input.push(json!({"type": "message", "role": "user", "content": e.parts}));
        let Some(r) = e.reply else { continue };
        input.extend(reply::items(&reply::parse(&r)));
        if e.cache { input.last_mut().expect("a reply's items")["content"][0]["cache_control"] = json!({"type": "ephemeral"}) }
    }
    input
}

fn model_call_ids(items: &mut [Value]) {
    for item in items {
        if matches!(item["type"].as_str(), Some("function_call" | "function_call_output")) {
            if let Some(id) = item["call_id"].as_str() { item["call_id"] = id.chars().take(64).collect::<String>().into(); }
        }
    }
}

/// The model input for prompt `p`: the examples, the CRAFTs, the `$` values, the dialog's path, the cells above the
/// prompt that fit, any warnings, then the prompt
pub async fn build(p: &Prompt, ctx: &media::Context<'_>) -> Result<Vec<Value>, String> {
    let all = p.nb["cells"].as_array().ok_or("the notebook has no cells")?;
    let at = all.iter().position(|c| c["id"] == p.prompt.as_str()).ok_or_else(|| format!("no prompt cell {}", p.prompt))?;
    let caps = &p.caps;
    let mut pre = vec![];
    let example_ctx = media::Context { path: Path::new(""), ..*ctx };
    for nb in &p.examples { pre.extend(shown_entries(nb, caps, &example_ctx).await); }
    if let Some(e) = pre.last_mut() { e.turn.cache = true }
    if !p.crafts.is_empty() {
        pre.push(Entry::text("# Loaded CRAFTs.".into()));
        for c in &p.crafts {
            pre.push(Entry::text(format!("<craft-block{}>", xml_attr("path", &format!("/{}", c.path)))));
            let craft_ctx = media::Context { path: Path::new(&c.path), ..*ctx };
            pre.extend(shown_entries(&c.nb, caps, &craft_ctx).await);
            pre.push(Entry::text("</craft-block>".into()));
        }
        let mut end = Entry::text(envelope("End of CRAFTs.", "crafts", None));
        end.turn.reply = Some("Ok!".into());
        end.turn.cache = true;
        pre.push(end);
    }
    let (vparts, undef) = vars(&p.vars, caps);
    if !vparts.is_empty() {
        let tokens = vparts.iter().map(|p| p["text"].as_str().map_or(tokens::IMG_TOKENS, |t| scaled(tokens::count(t)))).sum();
        pre.push(Entry { turn: Turn { parts: vparts, reply: Some(VARS_REPLY.into()), cache: false }, tokens });
    }
    if !p.path.is_empty() { pre.push(Entry::text(format!("<dialog{}/>", xml_attr("path", &p.path)))) }
    let time = |c: &Value| p.times.get(c["id"].as_str().unwrap_or_default()).and_then(Value::as_str);
    let mut last: Vec<Entry> = warning(&undef, &p.tool_errors, &p.craft_errors).map(Entry::text).into_iter().collect();
    let mut target = entry(&p.nb, &all[at], caps, time(&all[at]), ctx).await;
    target.turn.reply = None;
    last.push(target);
    let (mut boundary, mut after_large) = (false, false);
    let hist = all[..at].iter().enumerate().filter_map(|(i, c)| {
        let m = msg(c);
        let large = m.content.chars().count() + m.out.chars().count() > 20_000;
        boundary |= i % 10 == 0 || large || after_large;
        after_large = large;
        if !shown(c) { return None }
        let e = Candidate { index: i, boundary, pinned: truthy(&c["metadata"]["pinned"]), tokens: scaled(tokens::total(c, stored_atts(&p.nb, c))) };
        boundary = false;
        Some(e)
    }).collect();
    let fixed: u64 = pre.iter().chain(&last).map(|e| e.tokens).sum::<u64>() + p.instructions.as_deref().map_or(0, |s| scaled(tokens::count(s)));
    let budget = (caps.max_input_tokens.unwrap_or(WINDOW) as f64 * FILL) as u64;
    let (selected, dropped) = fit(hist, budget.saturating_sub(fixed));
    let mut hist = vec![];
    for i in selected { hist.push(entry(&p.nb, &all[i], caps, time(&all[i]), ctx).await); }
    if dropped > 0 {
        last.insert(0, Entry::text(reminder(&format!("Notify the user if relevant: {dropped} older non-pinned dialog message(s) were omitted to fit the model context."))));
    }
    let mut input = turns(pre.into_iter().chain(hist).chain(last).map(|e| e.turn).collect());
    model_call_ids(&mut input);
    Ok(input)
}
