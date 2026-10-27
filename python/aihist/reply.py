"Read saved replies as blocks or chat messages, write streamed or completed replies, and show tool and usage blocks as folded Markdown"
import json, re
from fastcore.xtras import fenced, fenced_blocks
from aidialog.msg_parts import (Msg, Text, Thinking, ToolUse, ToolResult, MdStr, FullResponse, split_fence_msgs, think_start, think_end,
    tool_info, tool_text)
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

def _tool_parts(t):
    "A saved tool block as an ordinary matched `(ToolUse, ToolResult)` pair"
    text = tool_text(t['result'])
    return (ToolUse(id=t['id'], name=t['name'], arguments=t['args']),
            ToolResult(id=t['id'], name=t['name'], text=MdStr(text) if t.get('md') else text))

def fmt2hist(outp:str)->list[Msg]:
    "Saved reply `outp` as chat messages: assistant text and tool calls, then a tool message with each round's results"
    hist,asst,res = [],[],[]
    for b in json.loads(_core.parse_reply(outp)):
        if 'text' in b:
            if res: hist,asst,res = hist+[Msg('assistant', asst), Msg('tool', res)],[],[]
            asst.append(Text(b['text']))
        else:
            tu,tr = _tool_parts(b['tool'])
            asst.append(tu)
            res.append(tr)
    if res: hist += [Msg('assistant', asst), Msg('tool', res)]
    elif asst: hist.append(Msg('assistant', asst))
    if not hist or hist[-1].role == 'tool': hist.append(Msg('assistant', [Text('.')]))
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

def _fmt_param(v, mx=40):
    "Compact display form of one tool arg or result value"
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
    s = re.sub(r'\n(?:\s*\n)*', '¶', s)
    if len(s) > mx: s = s[:mx] + '…'
    return json.dumps(s, ensure_ascii=False)

def _code_span(txt):
    "Wrap `txt` in a backtick span longer than any backtick run inside it"
    ticks = '`'*(max(map(len, re.findall('`+', txt)), default=0)+1)
    pad = ' ' if '`' in txt else ''
    return f"{ticks}{pad}{txt}{pad}{ticks}"

_code_args = dict(py='code', python='code', bash='cmd')

def tool_md(d):
    "Display markdown for one parsed `{.tool}` block: a folded details div labeled `func(params)→result`, code shown as code, the result fenced unless the block marks it `md`"
    params = ', '.join(f"{k}={_fmt_param(v)}" for k,v in (d.get('args') or {}).items())
    res = d.get('result')
    tail = f"→{_fmt_param(res)}" if res not in (None, '') else ''
    label = _code_span(f"{d.get('name')}({params}){tail}")
    code = (d.get('args') or {}).get(_code_args.get(d.get('name'), ''))
    if code is not None:
        body = f"Code:\n{fenced(str(code), d.get('name'))}"
        if str(res or '').strip(): body += f"\n\nOutput:\n\n{fenced(str(res))}"
    else:
        body = f"Args:\n{fenced(json.dumps(d.get('args') or {}, indent=2, ensure_ascii=False), 'json')}"
        if str(res or '').strip(): body += f"\n\n{res}" if d.get('md') else f"\n\nOutput:\n\n{fenced(str(res))}"
    return fenced(f"## {label}\n\n{body}", ' {.details .tool-usage-details}', ch=':')

def usage_md(d):
    "Display markdown for one parsed `{.usage}` block: a folded details div labeled with cost or tokens"
    summ = d.get('summary') or (f"${d['cost']:.4f}" if d.get('cost') else f"{d.get('total_tokens', 0):,} tokens")
    det = d.get('detail') or ' | '.join(f"{k}={v:,}" if isinstance(v, int) else f"{k}={v}" for k,v in d.items() if v and k != 'model')
    body = f"## {summ}\n\n{_code_span(det)}"
    return fenced(body, ' {.details .token-usage-details}', ch=':')

def fmt_tools(s):
    "Rewrite fenced JSON tool/usage wire blocks in `s` into folded `::: details` display markdown"
    out, pos = [], 0
    for info,body,start,end in fenced_blocks(s):
        if info not in (tool_info, usage_info): continue
        try: d = json.loads(body)
        except Exception: continue
        out.append(s[pos:start])
        out.append((tool_md(d) if info == tool_info else usage_md(d)) + '\n')
        pos = end
    out.append(s[pos:])
    return ''.join(out)
