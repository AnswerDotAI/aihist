//! The stored reply format: a prompt's reply as Markdown, with each tool call in a fenced `json {.tool}` block
//! holding its `id`, `name`, `args` and `result`, and usage in a `json {.usage}` block.
//! `parse` reads a stored reply into blocks, `items` turns blocks into Responses input items for the model, and `write` writes a turn as a stored reply.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

/// Fence info string of a tool block
pub const TOOL_INFO: &str = "json {.tool}";
/// Fence info string of a usage block
pub const USAGE_INFO: &str = "json {.usage}";
/// Markers around thinking that a reply shows
pub const THINK: (&str, &str) = ("<!--think_start-->", "<!--think_end-->");
/// The note that ends a reply cut off by an interrupt, as aidialog's `INTERRUPTED`
pub const INTERRUPTED: &str = "*[Response interrupted]*";

/// One tool call and its result, as a tool block stores them
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Tool {
    pub id: String,
    pub name: String,
    pub args: Value,
    pub result: Value,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub server: bool,
    #[serde(default, skip_serializing_if = "std::ops::Not::not")]
    pub md: bool,
}

/// A piece of a stored reply: prose, or a tool call with its result
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Block { Text(String), Tool(Tool) }

/// Top-level fenced blocks in `s` as `(info, body, start, end)`, where `start..end` spans the whole block.
/// A fence closes only at a line of at least as many backticks, so shorter fences nest inside the body.
pub fn fenced_blocks(s: &str) -> Vec<(&str, &str, usize, usize)> {
    let (mut res, mut open, mut pos) = (vec![], None, 0);
    for line in s.split_inclusive('\n') {
        let t = line.strip_suffix('\n').unwrap_or(line);
        let run = t.len() - t.trim_start_matches('`').len();
        match open {
            None if run >= 3 => open = Some((run, t[run..].trim(), pos, pos + line.len())),
            Some((fence, info, start, body)) if run >= fence && t[run..].trim().is_empty() => {
                res.push((info, &s[body..pos], start, pos + line.len()));
                open = None;
            }
            _ => {}
        }
        pos += line.len();
    }
    res
}

/// `s` without its fenced blocks whose info string is in `infos`
pub fn strip_blocks(s: &str, infos: &[&str]) -> String {
    let (mut out, mut pos) = (String::new(), 0);
    for (i, _, start, end) in fenced_blocks(s) {
        if !infos.contains(&i) { continue }
        out.push_str(&s[pos..start]);
        pos = end;
    }
    out + &s[pos..]
}

/// `s` without thinking between `THINK` markers, or the newline after it
fn strip_think(mut s: String) -> String {
    while let Some(a) = s.find(THINK.0) && let Some(b) = s[a..].find(THINK.1) {
        let end = a + b + THINK.1.len();
        s.replace_range(a..end + usize::from(s[end..].starts_with('\n')), "");
    }
    s
}

/// `s` split at its tool blocks: the text before each block, with the block's JSON, then the text after the last block,
/// with `None`. A tool block that isn't valid JSON stays in the text.
pub fn segments(s: &str) -> Vec<(&str, Option<Value>)> {
    let (mut res, mut pos) = (vec![], 0);
    for (info, body, start, end) in fenced_blocks(s) {
        if info != TOOL_INFO { continue }
        let Ok(d) = serde_json::from_str::<Value>(body) else { continue };
        res.push((&s[pos..start], Some(d)));
        pos = end;
    }
    res.push((&s[pos..], None));
    res
}

/// `reply` with call `id`'s stored result replaced, subject to the tool-text ceiling. Every other byte stays as it was.
pub fn set_result(reply: &str, id: &str, result: &str) -> String {
    for (info, body, start, end) in fenced_blocks(reply) {
        if info != TOOL_INFO { continue }
        let Ok(mut d) = serde_json::from_str::<Value>(body) else { continue };
        if d["id"] != id { continue }
        d["result"] = limit_text(result).into();
        let nl = if reply[..end].ends_with('\n') { "\n" } else { "" };
        return format!("{}{}{nl}{}", &reply[..start], fenced(&serde_json::to_string_pretty(&d).expect("JSON serializes"), TOOL_INFO), &reply[end..]);
    }
    reply.into()
}

/// The tool call in a tool block's JSON. A block without a string `id` holds none.
fn tool(d: &Value) -> Option<Tool> {
    let flag = |k: &str| d[k].as_bool() == Some(true);
    Some(Tool {
        id: d["id"].as_str()?.into(),
        name: d["name"].as_str().unwrap_or_default().into(),
        args: if d["args"].is_object() { d["args"].clone() } else { json!({}) },
        result: d["result"].clone(),
        server: flag("server"),
        md: flag("md"),
    })
}

/// `t` as a `Text` block, unless it's only whitespace
fn push_text(res: &mut Vec<Block>, t: &str) { if !t.trim().is_empty() { res.push(Block::Text(t.trim().into())) } }

/// The prose and tool calls of stored reply `reply`. Usage and thinking are dropped, and so are
/// tool blocks without an `id`. A tool block that isn't valid JSON stays in the prose.
pub fn parse(reply: &str) -> Vec<Block> {
    let s = strip_think(strip_blocks(reply, &[USAGE_INFO]));
    let mut res = vec![];
    for (text, d) in segments(s.trim()) {
        push_text(&mut res, text);
        res.extend(d.as_ref().and_then(tool).map(Block::Tool));
    }
    res
}

/// A Responses input item for assistant prose
fn assistant(text: &str) -> Value { json!({"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}) }

/// Responses input items for a reply's blocks, grouped as a round of a chat: prose and the calls after it, then the
/// calls' outputs. A reply that is empty or ends with outputs gets a final "." message.
pub fn items(blocks: &[Block]) -> Vec<Value> {
    let (mut res, mut calls, mut outs) = (vec![], vec![], vec![]);
    for b in blocks {
        match b {
            Block::Text(t) => {
                if !outs.is_empty() { res.append(&mut calls); res.append(&mut outs); }
                calls.push(assistant(t));
            }
            Block::Tool(t) => {
                let id: String = t.id.chars().take(64).collect();
                calls.push(json!({"type": "function_call", "call_id": id, "name": t.name, "arguments": t.args.to_string()}));
                let output = match &t.result { Value::String(s) => s.clone(), v => v.to_string() };
                outs.push(json!({"type": "function_call_output", "call_id": id, "output": output}));
            }
        }
    }
    let ends_with_outputs = !outs.is_empty();
    res.append(&mut calls);
    res.append(&mut outs);
    if res.is_empty() || ends_with_outputs { res.push(assistant(".")) }
    res
}

/// Longest tool result a tool block stores, in characters. String arguments can be five times as long.
const MX: usize = 5000;

fn limit_text(text: &str) -> String { crate::output::keep_ends(text.into(), 0, crate::output::TEXT_MAX) }

/// One part of a turn, in the order the model produced it, tagged as aidialog's `msg2dict` tags parts.
/// Other part types, such as thinking, aren't stored.
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Part {
    Text { text: Option<String> },
    Refusal { text: Option<String> },
    ToolUse(Call),
    ToolResult(Call),
    #[serde(other)]
    Other,
}

/// A tool call or its result. A server call carries its result in `text`.
#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Call {
    pub id: Option<String>,
    pub name: String,
    pub arguments: Map<String, Value>,
    pub server: bool,
    pub text: Option<String>,
    /// The result is Markdown, as aidialog's `MdStr` marks it
    pub md: bool,
    /// Bypass saved-history shortening, not the tool-text ceiling
    pub full: bool,
}

/// Token use and billed cost of a turn's responses, as fastllm's `UsageStats` adds them up
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Usage {
    pub model: String,
    pub prompt_tokens: u64,
    pub completion_tokens: u64,
    pub total_tokens: u64,
    pub cached_tokens: u64,
    pub cache_creation_tokens: u64,
    pub reasoning_tokens: u64,
    pub web_search_requests: u64,
    pub cost: f64,
    /// The user's balance after the latest response, which the stored block leaves out
    #[serde(skip)]
    pub balance: Option<f64>,
}

impl std::ops::AddAssign for Usage {
    fn add_assign(&mut self, o: Usage) {
        if !o.model.is_empty() { self.model = o.model }
        self.prompt_tokens += o.prompt_tokens;
        self.completion_tokens += o.completion_tokens;
        self.total_tokens += o.total_tokens;
        self.cached_tokens += o.cached_tokens;
        self.cache_creation_tokens += o.cache_creation_tokens;
        self.reasoning_tokens += o.reasoning_tokens;
        self.web_search_requests += o.web_search_requests;
        self.cost += o.cost;
        if o.balance.is_some() { self.balance = o.balance }
    }
}

impl Usage {
    /// The `json {.usage}` block that ends a stored reply: the fields that aren't zero or empty, and no block at all
    /// without tokens
    pub fn block(&self) -> String {
        if self.total_tokens == 0 { return String::new() }
        let Value::Object(d) = json!(self) else { unreachable!() };
        let d: Map<String, Value> = d.into_iter().filter(|(_, v)| !(*v == "" || *v == 0.0)).collect();
        format!("\n\n{}\n", fenced(&serde_json::to_string(&d).expect("JSON serializes"), USAGE_INFO))
    }
}

/// A turn's reply as it streams in. Each method returns the text to show on the stream, and `finish` gives the reply
/// to store.
#[derive(Debug, Default)]
pub struct Reply {
    pub parts: Vec<Part>,
    pub usage: Usage,
    last_thinking: Option<bool>,
}

impl Reply {
    /// A blank line when the stream crosses between thinking and other text
    fn sep(&mut self, thinking: bool) -> &'static str {
        let crossed = self.last_thinking.is_some_and(|t| t != thinking);
        self.last_thinking = Some(thinking);
        if crossed { "\n\n" } else { "" }
    }

    /// Add text the model wrote
    pub fn text(&mut self, delta: &str) -> String {
        match self.parts.last_mut() {
            Some(Part::Text { text: Some(t) }) => t.push_str(delta),
            _ => self.parts.push(Part::Text { text: Some(delta.into()) }),
        }
        format!("{}{delta}", self.sep(false))
    }

    /// Show that the model is thinking. Thinking isn't stored.
    pub fn thinking(&mut self) -> String { format!("{}🧠", self.sep(true)) }

    /// Add a tool call: a pending client row, or a completed server block
    pub fn call(&mut self, c: Call) -> String {
        let row = if c.server { write(&[Part::ToolUse(c.clone())], None) } else { pending(&c) };
        let shown = format!("{}\n{row}\n", self.sep(false));
        self.parts.push(Part::ToolUse(c));
        shown
    }

    /// Add a client call's result, returning its completed block for the stream
    pub fn result(&mut self, r: Call) -> String {
        let block = self.parts.iter().rev().find_map(|p| match p {
            Part::ToolUse(c) if c.id == r.id && !c.server => Some(write(&[p.clone(), Part::ToolResult(r.clone())], None)),
            _ => None,
        }).unwrap_or_default();
        self.parts.push(Part::ToolResult(r));
        if block.is_empty() { block } else { format!("\n\n{block}\n") }
    }

    /// The reply to store, with its usage block when `usage`. When `note` explains why the turn ended early, the reply
    /// so far and the note, without usage.
    pub fn finish(&self, note: Option<&str>, usage: bool) -> String {
        let Some(note) = note else { return write(&self.parts, usage.then_some(&self.usage)) };
        let partial = write(&self.parts, None);
        if partial.is_empty() { note.into() } else { format!("{partial}\n\n{note}") }
    }
}

/// A turn as a stored reply, then `usage`'s block. Each result is written with its call's arguments, and a server
/// call, which holds its own result, where it happened. A client call without a result is left out.
pub fn write(parts: &[Part], usage: Option<&Usage>) -> String {
    let (mut open, mut out): (Vec<&Call>, Vec<String>) = (vec![], vec![]);
    for p in parts {
        match p {
            Part::Text { text } | Part::Refusal { text } => out.push(text.as_deref().unwrap_or_default().trim().into()),
            Part::ToolUse(c) if c.server => out.push(tool_block(c, &c.arguments, c.text.as_deref().filter(|t| !t.is_empty()).unwrap_or("Server tool call executed."))),
            Part::ToolUse(c) => match open.iter().position(|o| o.id == c.id) { Some(i) => open[i] = c, None => open.push(c) },
            Part::ToolResult(r) => {
                let args = open.iter().position(|o| o.id == r.id).map(|i| &open.remove(i).arguments);
                if r.id.is_some() { out.push(tool_block(r, args.unwrap_or(&Map::new()), r.text.as_deref().unwrap_or_default())) }
            }
            Part::Other => {}
        }
    }
    let mut res = out.into_iter().filter(|o| !o.is_empty()).collect::<Vec<_>>().join("\n\n");
    res.push_str(&usage.map(Usage::block).unwrap_or_default());
    res
}

/// A prompt cell's outputs holding `reply`, as aidialog's `prompt_output` makes them: none for an empty reply
pub fn output(reply: &str) -> Vec<Value> {
    if reply.is_empty() { return vec![] }
    vec![json!({"output_type": "display_data", "metadata": {"is_ai_res": true}, "data": {"text/markdown": reply}})]
}

/// The `json {.tool}` block for call `c` with `args` and result `text`, as aidialog's `mk_tr_details` writes it
fn tool_block(c: &Call, args: &Map<String, Value>, text: &str) -> String {
    let args: Map<String, Value> = args.iter()
        .map(|(k, v)| (k.clone(), match v { Value::String(s) => limit_text(&trunc_str(s, MX * 5, 10, true)).into(), v => v.clone() })).collect();
    let text = if c.full { text.to_string() } else { trunc_str(text, MX, 10, true) };
    let mut d = json!({"id": c.id, "name": c.name, "args": args, "result": limit_text(&text)});
    if c.server { d["server"] = true.into() }
    if c.md { d["md"] = true.into() }
    fenced(&serde_json::to_string_pretty(&d).expect("JSON serializes"), TOOL_INFO)
}

/// The row a client call shows while it runs: a list item with the call and its arguments, each cut to 40 characters,
/// between hourglasses
fn pending(c: &Call) -> String {
    let params: Vec<String> = c.arguments.iter().map(|(k, v)| format!("{k}={}", short(v))).collect();
    let txt = format!("{}({})", c.name, params.join(", "));
    let ticks = fence(&txt, 1);
    let pad = if txt.contains('`') { " " } else { "" };
    format!("- ⏳ {ticks}{pad}{txt}{pad}{ticks} ⏳")
}

/// An argument as a pending row shows it: a string's text or the JSON of anything else, cut to 40 characters, as a
/// JSON string
fn short(v: &Value) -> String {
    let s = match v { Value::String(s) => s.clone(), v => v.to_string() }.replace('`', "\\`");
    Value::String(trunc_str(&s, 40, 0, false)).to_string()
}

/// `s` cut as aidialog's `trunc_str` cuts it. Trailing whitespace and any `𝍁` markers around `s` go, and with `tag`
/// a marked string is never cut. A string over `mx` characters keeps characters `skip` to `mx - skip`, a final word
/// over 150 characters keeps 5, and ellipses mark the cut: after it, and before it too when `skip` is set. With `tag`,
/// the result goes inside a `<TRUNCATED>` tag.
fn trunc_str(s: &str, mx: usize, skip: usize, tag: bool) -> String {
    let mut cs: Vec<char> = s.trim_end().chars().collect();
    if cs.len() > 2 && cs[0] == '𝍁' && cs[cs.len() - 1] == '𝍁' {
        cs = cs[1..cs.len() - 1].to_vec();
        if tag { return cs.into_iter().collect() }
    }
    if cs.len() <= mx { return cs.into_iter().collect() }
    let mut cut: String = cs[skip..mx - skip].iter().collect();
    let i = cut.rfind(' ').map_or(0, |i| i + 1);
    if cut[i..].chars().count() > 150 { cut = format!("{}{}", &cut[..i], cut[i..].chars().take(5).collect::<String>()) }
    let cut = format!("{}{cut}…", if skip > 0 { "…" } else { "" });
    if tag { format!("<TRUNCATED>{cut}</TRUNCATED>") } else { cut }
}

/// A run of backticks one longer than any run inside `text`, and at least `min` long
fn fence(text: &str, min: usize) -> String { "`".repeat(text.split(|c| c != '`').map(str::len).max().unwrap_or(0).max(min - 1) + 1) }

/// `text` in a fence of at least three backticks, as fastcore's `fenced` writes it
fn fenced(text: &str, info: &str) -> String {
    let fence = fence(text, 3);
    format!("{fence}{info}\n{}\n{fence}", text.trim_end_matches('\n'))
}

