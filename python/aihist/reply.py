"Read saved replies as blocks or chat messages, write streamed or completed replies, and show tool and usage blocks as folded Markdown"
import json, re
from fastcore.xtras import fenced
from aidialog.msg_parts import (Msg, Text, Thinking, ToolUse, ToolResult, MdStr, FullResponse, split_fence_msgs, think_start, think_end,
    tool_info, tool_text, dict2msg)
from . import _core

__all__ = ['usage_info', 're_think', 'parse_tools', 'set_result', 'strip_tools', 'conv_tools', 'fmt2hist', 'Reply', 'write_reply', 'tool_md', 'usage_md', 'fmt_tools']

usage_info = _core.USAGE_INFO   # fence info string of a usage block: a turn's token counts and cost

re_think = re.compile(rf'{re.escape(think_start)}.*?{re.escape(think_end)}\n?', re.DOTALL)

def parse_tools(s):
    "Split `s` into `(text, data)` segments: `data` is a parsed `{.tool}` block dict, `None` for the final segment"
    return [tuple(o) for o in json.loads(_core.tool_segments(s))]

set_result = _core.set_result

def strip_tools(s, tools=True, usage=True):
    "Remove `{.tool}` (and `{.usage}`) blocks from `s`"
    return _core.strip_blocks(s, [i for i,keep in ((tool_info, tools), (usage_info, usage)) if keep])

# Frozen legacy envelope recognition, used only by `conv_tools`: the `<details>`
# envelopes fastllm shipped, with and without `markdown='1'`.
_lg_tool_tag = "<details class='tool-usage-details'(?: markdown='1')?>"
_lg_token_tag = "<details class='token-usage-details'(?: markdown='1')?>"
_lg_tools = re.compile(fr"^{_lg_tool_tag}\n*(?:<summary>.*?</summary>\n*)?\n*```json\n+(?P<json>.*?)\n+```\n+</details>", flags=re.DOTALL|re.MULTILINE)
_lg_token = re.compile(fr"^{_lg_token_tag}\n*<summary>(?P<summ>.*?)</summary>\n*\n*`(?P<det>.*?)`\n*\n*</details>\n?", flags=re.DOTALL|re.MULTILINE)

def conv_tools(s):
    "Convert legacy tool/usage envelopes in `s` to the fenced JSON wire format. Idempotent."
    def _tool(m):
        try: d = json.loads(m['json'].strip())
        except Exception: return m[0]
        call = d.get('call') or {}
        res = dict(id=d.get('id'), name=call.get('function'), args=call.get('arguments') or {}, result=d.get('result'))
        if d.get('server'): res['server'] = True
        return fenced(json.dumps(res, indent=2, ensure_ascii=False), tool_info)
    def _tok(m): return fenced(json.dumps(dict(summary=m['summ'], detail=m['det']), ensure_ascii=False), usage_info)
    return _lg_token.sub(_tok, _lg_tools.sub(_tool, s))

def fmt2hist(outp:str)->list[Msg]:
    "Saved reply `outp` as chat messages: assistant text and tool calls, then a tool message with each round's results"
    hist = [dict2msg(m) for m in json.loads(_core.reply_messages(outp))]
    for m in hist:
        for p in m.content:
            if isinstance(p, ToolResult) and p.md: p.text,p.md = MdStr(p.text),False
    return split_fence_msgs(hist)

def _part(p):
    "Part `p` for the reply writer, preserving tool text's `md` and `full` flags"
    if isinstance(p, Thinking): return dict(type=p.type)
    text = getattr(p, 'text', None)
    if not isinstance(p, (ToolUse, ToolResult)): return dict(type='text', text=text)
    return dict(type=p.type, id=p.id, name=p.name or '', arguments=p.arguments or {}, server=p.server,
        text=tool_text(text) if text is not None else None, md=p.md or isinstance(text, MdStr), full=p.full or isinstance(text, FullResponse))

def write_reply(msgs):
    "A completed turn's assistant and tool messages as a saved reply"
    return _core.write_reply(json.dumps([_part(p) for m in msgs for p in m.content]))

class Reply:
    "A turn's reply as it streams in: `add` takes each stream item and returns the text to show"
    def __init__(self): self.r = _core.Reply()
    def add(self, o)->str:
        "Add a stream item, returning text, a thinking glyph, a pending row or a completed tool block"
        if isinstance(o, Thinking): return self.r.thinking()
        if isinstance(o, Text): return self.r.text(o.text or '')
        if isinstance(o, ToolUse): return self.r.call(json.dumps(_part(o)))
        if isinstance(o, ToolResult): return self.r.result(json.dumps(_part(o)))
        return ''
    def finish(self, note=None)->str:
        "The reply to save for a turn cut short, with `note` after the reply so far"
        return self.r.finish(note)

def tool_md(d):
    "Display markdown for one parsed `{.tool}` block: a folded details div labeled `func(params)→result`, code shown as code, the result fenced unless the block marks it `md`"
    return _core.tool_md(json.dumps(d, ensure_ascii=False))

def usage_md(d):
    "Display markdown for one parsed `{.usage}` block: a folded details div labeled with cost or tokens"
    return _core.usage_md(json.dumps(d, ensure_ascii=False))

def fmt_tools(s):
    "Rewrite fenced JSON tool/usage wire blocks in `s` into folded `::: details` display markdown"
    return _core.fmt_tools(s)
