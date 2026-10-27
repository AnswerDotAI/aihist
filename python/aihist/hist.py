"Dialogs as model input, chat messages and Markdown documents, and chat messages back as dialogs"
import ast, base64, binascii, hashlib, json, re
from ast import literal_eval
from datetime import datetime
from zoneinfo import ZoneInfo
from fastcore.utils import first, listify, loop_first, rtoken_hex
from fastcore.nbio import render_text
from fastcore.xtras import fenced
from aidialog.dialog import Dialog, Attachment, code_output, render_md, render_outputs_ai, export_filter, sprompt, scode, snote, sraw
from aidialog.ipynb import get_ipynb, msgs2ipynb
from aidialog.msg_parts import Msg, Text, ToolUse, ToolResult, data_url, hist2fmt, mk_content, mk_part, mk_tr_details, resp_part
from . import _core
from .reply import fmt2hist, fmt_tools

__all__ = ['UNSUPPORTED_MSG', 'im_max', 'IMG_TOKS', 'jwrap', 'to_local_time', 'media_item', 'resize_img', 'prep_img', 'pdf_pages', 'count_tokens',
    'output_parts', 'merge_media', 'tool_result', 'get_refs', 'get_exprs', 'is_nameerr', 'vars_tag', 'vars_hist', 'task_tags', 'warning_tag',
    'local_time', 'hist_xml', 'to_media', 'to_parts', 'tool_call', 'build', 'dlg2hist', 'reply2chat', 'dlg2chat', 'chat2dlg', 'reply2dlg',
    'dlg2reply', 'msg2md', 'dlg2md']

UNSUPPORTED_MSG,im_max,IMG_TOKS = _core.UNSUPPORTED,_core.IM_MAX,_core.IMG_TOKENS

def _cdict(aim_info):
    "What a model can read, from fastllm model info or a model catalog entry; every kind of media when `aim_info` is None"
    return dict(input_modalities=['text', 'image', 'pdf', 'audio', 'video']) if aim_info is None else aim_info

def _caps(aim_info): return json.dumps(_cdict(aim_info), default=str)

def _data(p):
    "The bytes in Responses media part `p`'s data URL"
    return base64.b64decode(data_url(p.get('image_url') or p.get('video_url') or p['file_data'])[1])

def _content(parts):
    "Core parts as `mk_msgs` content: text, media bytes, or canonical audio parts"
    return [mk_part(**p) if p['type'] == 'input_audio' else p['text'] if p['type'] == 'input_text' else _data(p) for p in parts]

def _parts(parts):
    "Responses content parts from `_core` as aidialog `Part`s"
    return [mk_content(o) for o in _content(parts)]

def jwrap(txts:list=['']):
    "Wrap plain `txt` in a minimal Jupyter-style cell output"
    return [dict(output_type='stream', name='stdout', text=txt) for txt in listify(txts)]

def to_local_time(time_str, tz='UTC'):
    "Convert ISO time string to local timezone"
    if not time_str: return ''
    try: dt = datetime.fromisoformat(time_str)
    except ValueError: return ''
    return dt.astimezone(ZoneInfo(tz)).isoformat()

def media_item(media_type, media_id, data, aim_info, mime=None):
    "Media `data`, bytes or a function that loads them, as `[xml_tag, data]`, or `[media-unavailable tag]` when the model can't read it"
    try: data = data() if callable(data) else data
    except (OSError, binascii.Error): data = None
    if callable(media_id): media_id = media_id(data) if isinstance(data, bytes) else 'unknown'
    return _content(json.loads(_core.media_item(media_type, media_id, data if isinstance(data, bytes) else None, _caps(aim_info), mime)))

def resize_img(data:bytes, max_im_sz=im_max):
    "`data` with its image area limited to `max_im_sz` pixels, in the same format. Other data is unchanged."
    mime = _core.detect_mime(data)
    return _core.resize(data, mime, max_im_sz) if mime else data

def prep_img(data, mime, max_im_sz=None):
    "Limit raster image area before adding it to LLM context."
    return _core.resize(data, mime, max_im_sz or im_max)

def pdf_pages(b:bytes):
    "Page count of PDF `b`: the largest `/Count` on a `/Pages` node, in the clear or inside an inflated object stream"
    return _core.pdf_pages(b)

def count_tokens(text:str):
    "Approximate token count of `text`: a character-class estimate calibrated against OpenAI's o200k tokenizer"
    return _core.count_tokens(text)

def output_parts(m, aim_info=None):
    "Message `m`'s output images as `Part`s: each image's `<media>` tag and the image, or a note when the model can't read it"
    return _parts(json.loads(_core.image_parts(json.dumps(m.output or []), _caps(aim_info))))

def merge_media(text, parts):
    "Output `text` and media `parts` as one tool result: a string when no media remains, else the parts then the text"
    out = json.loads(_core.merge(text, json.dumps([resp_part(p) for p in parts])))['output']
    return out if isinstance(out, str) else _parts(out)

def tool_result(outputs, aim_info=None, code=False):
    "A canonical `ToolResult` from notebook `outputs`, retaining Markdown/full flags. `code` is true when the tool ran code."
    r = json.loads(_core.tool_result(json.dumps(outputs), _caps(aim_info), code))
    text = r['output']
    return ToolResult(text=text if isinstance(text, str) else _parts(text), md=r['md'], full=r['full'])

def get_refs(cts):
    "Sorted unique names that `cts` refers to as &`name` or &`[name, name]`, or None when there are none"
    return sorted(set(json.loads(_core.cell_refs(json.dumps(dict(cell_type='markdown', source=cts, metadata={}))))['tools'])) or None

def get_exprs(msgs):
    "Sorted unique `$` expressions in the prompt messages of `msgs`"
    cells = [dict(cell_type='code', source=m.content or '', metadata=dict(solveit_ai=True)) for m in msgs if m.msg_type == sprompt]
    return sorted({o for c in cells for o in json.loads(_core.cell_refs(json.dumps(c)))['vars']})

def is_nameerr(r):
    "Does `eval` record `r` mark an undefined name?"
    return _core.is_nameerr(json.dumps(r, default=str))

def _vars(aim_info, recs):
    "The content of the variables turn for `eval` records `recs`, with media as bytes"
    parts,_ = json.loads(_core.vars(json.dumps(recs or {}, default=str), _caps(aim_info)))
    return _content(parts)

def vars_tag(aim_info, recs=None):
    "The `<variables>` element and media for `eval` records `recs`, as `dict(vars_tag, vars_bytes)`; `{}` when every name is undefined"
    if not (c := _vars(aim_info, recs)): return {}
    tag = c[0] if c[0].startswith('<variables>') else None
    return dict(vars_tag=tag, vars_bytes=c[1:] if tag else c)

def vars_hist(aim_info, recs=None):
    "One history turn holding the values of `eval` records `recs`, then its acknowledgement; `[]` when every name is undefined"
    return [c, _core.VARS_REPLY] if (c := _vars(aim_info, recs)) else []

def task_tags(task:str, sid=None, time=None, tz='UTC'):
    "A prompt as the model reads it: the instruction to answer, then the `<prompt>` with its `sid` and local run time"
    return _core.envelope(task, sid or '', to_local_time(time, tz) or None)

def warning_tag(warning:str):
    "`warning` in a `<system-reminder>`, or None when it's empty"
    return _core.reminder(warning) if warning else None

def local_time(m):
    "Message `m`'s `time_run` in its dialog's time zone; '' when it has none"
    if not (t := getattr(m, 'time_run', '')): return ''
    return to_local_time(t, getattr(m.dlg, 'timezone', 'UTC') if m.dlg else 'UTC')

def _cell(m):
    "Message `m` as JSON for `_core`: the notebook holding it, which homes a code cell's attachments, and its cell"
    nb = msgs2ipynb([m])
    return json.dumps(nb), json.dumps(nb['cells'][0])

def hist_xml(m):
    "Message `m` as the XML a model reads, with its local run time"
    return _core.cell_xml(_cell(m)[1], local_time(m) or None)

def to_media(m, aim_info):
    "The media message `m` shows a model: the attachments it refers to, then a code message's output images"
    return _content(json.loads(_core.cell_media(*_cell(m), _caps(aim_info))))

def to_parts(m, aim_info):
    "Message `m` as history parts: its media, then its XML"
    return _content(json.loads(_core.cell_parts(*_cell(m), _caps(aim_info), local_time(m) or None)))

def _dotted_name(s): return bool(s) and all(p.isidentifier() for p in s.split('.'))

def _parse_call(src):
    "`(name, arguments)` parsed from literal call source `src`, or None if it isn't one"
    try: call = ast.parse(src.strip(), mode='eval').body
    except SyntaxError: return None
    if not (isinstance(call, ast.Call) and not call.args and all(kw.arg for kw in call.keywords)): return None
    name = ast.unparse(call.func)
    if not _dotted_name(name): return None
    return name, {kw.arg: literal_eval(kw.value) for kw in call.keywords}

def tool_call(m):
    "Parsed `dict(name=..., arguments=...)` when code message `m`'s source is a literal tool call, else None"
    if m.msg_type != scode or (p := _parse_call(m.content)) is None: return None
    return dict(name=p[0], arguments=p[1])

def build(nb, prompt, aim_info=None, **kw):
    "Responses input items for prompt cell `prompt` of `nb`, a `Dialog` or notebook dict, with the extra context `kw` that `aihist::input::Prompt` lists"
    if isinstance(nb, Dialog): nb = get_ipynb(nb)
    return json.loads(_core.build(json.dumps(dict(nb=nb, prompt=prompt, caps=_cdict(aim_info), **kw), default=str)))

def dlg2hist(
    dlg, # A `Dialog`, or iterable of messages, ending with a prompt
    aim_info:dict, # What the model can read, such as fastllm's model info
    plain:bool=False, # Render prompts as bare content, without the serving envelope?
):
    "`dlg` as model history: each user turn's parts, then its reply"
    msgs = [(m, *_cell(m)) for m in (dlg.messages if isinstance(dlg, Dialog) else dlg)]
    msgs = [o for o in msgs if _core.shown(o[2])]
    if not msgs or msgs[-1][0].msg_type != sprompt: raise ValueError("dlg2hist requires the messages to end with a prompt")
    caps,res = _caps(aim_info),[]
    for m,nb,cell in reversed(msgs):
        parts = _content(json.loads(_core.cell_parts(nb, cell, caps, local_time(m) or None)))
        if m.msg_type != sprompt: res[-1] = parts + res[-1]
        else: res += [_core.cell_reply(cell), _content(json.loads(_core.cell_media(nb, cell, caps))) + [m.content] if plain else parts]
    return res[::-1]

def _explode(a, t):
    "One call/result pair per tool use in a batched assistant/tool pair"
    tus = [p for p in a.content if isinstance(p, ToolUse)]
    if len(tus)<2: return [a,t]
    pre,trs = [p for p in a.content if not isinstance(p, ToolUse)],{p.id:p for p in t.content}
    res = []
    for is_first,tu in loop_first(tus): res += [Msg(role='assistant', content=(pre if is_first else [])+[tu]), Msg(role='tool', content=[trs[tu.id]])]
    return res

def _seq_tools(msgs):
    "Explode batched tool calls into strict call/result alternation"
    out = []
    for m in msgs:
        if m.role=='tool' and out and out[-1].role=='assistant': out += _explode(out.pop(), m)
        else: out.append(m)
    return out

def reply2chat(outp:str)->list[Msg]:
    "Chat messages for a stored reply: `fmt2hist`-parsed parts in strict call/result alternation"
    return _seq_tools(fmt2hist(outp))

def dlg2chat(
    dlg, # A `Dialog`, ending with a prompt
    aim_info=None, # What the model can read; every kind of media when None
    plain=False, # Render prompts as bare content, without the serving envelope?
):
    "Canonical chat messages for `dlg`, with each reply's tool calls recovered as real parts"
    msgs = []
    for i,t in enumerate(dlg2hist(dlg, aim_info, plain=plain)):
        if i%2==0: msgs.append(Msg('user', [mk_content(o) for o in t]))
        else: msgs += reply2chat(t)
    ids = [p.id for m in msgs for p in m.content if isinstance(p, ToolUse)]
    if dups := {i for i in ids if ids.count(i)>1}: raise ValueError(f"duplicate tool call id(s): {', '.join(sorted(dups))}")
    return msgs

_env_pat = re.compile(r'<instructions>.*?</instructions>\n<prompt(?: sid="(?P<sid>[^"]*)")?(?: time="[^"]*")?>(?P<task>.*)</prompt>$', re.S)
def chat2dlg(
    msgs, # Canonical messages, e.g. from `dlg2chat`
    name, # Dialog name
    cls=Dialog, # Dialog class to create
    mx=2000, # Maximum characters per rendered tool input/output string; None disables truncation (see `hist2fmt`)
):
    "Convert canonical messages to a dialog with one prompt per user turn."
    dlg,turns = cls(name=name),[]
    for m in msgs:
        if m.role=='user': turns.append((m,[]))
        elif turns: turns[-1][1].append(m)
        else: turns.append((Msg('user', []), [m]))
    for i,(u,replies) in enumerate(turns):
        segs,atts = [],[]
        for p in u.content:
            if isinstance(p, Text): segs.append(p.text)
            elif p.type == 'input_image':
                mime,data = data_url(p.text)
                atts.append(Attachment(base64.b64decode(data), mime))
                segs.append(f'![](attachment:{atts[-1].id})')
            else: raise ValueError(f'unsupported user part: {p.type}')
        meta = getattr(u, 'meta', None)
        uid = (meta or {}).get('uid') or f'{name}\x1f{i}'
        kw = dict(id=hashlib.md5(uid.encode()).hexdigest()[:8])
        if meta: kw['meta'] = meta
        if segs and (mt := _env_pat.search(segs[-1])):
            segs[-1] = segs[-1][:mt.start()] + mt['task']
            if mt['sid']: kw['id'] = mt['sid']
        dlg.mk_message('\n\n'.join(segs), msg_type=sprompt, output=hist2fmt(replies, mx=mx), attachments=atts, **kw)
    return dlg

def reply2dlg(pmsg):
    "Explode `pmsg`'s AI reply into a `Dialog` of note and code messages"
    sub = Dialog(name=pmsg.id)
    sub.msg_cls = type(pmsg)
    msgs = fmt2hist(pmsg.ai_res)
    trs = {p.id:p for m in msgs if m.role=='tool' for p in m.content}
    for m in msgs:
        if m.role=='tool': continue
        for p in m.content:
            if isinstance(p, ToolUse):
                if not _dotted_name(p.name): raise ValueError(f"not a dotted Python identifier: {p.name}")
                if bad := first(k for k in p.arguments if not k.isidentifier()): raise ValueError(f"not a Python identifier: {bad}")
                args = ', '.join(f'{k}={v!r}' for k,v in p.arguments.items())
                sub.mk_message(f"{p.name}({args})", output=code_output(trs[p.id].text), meta=dict(tool_id=p.id))
            elif isinstance(p, Text) and p.text: sub.mk_message(p.text, msg_type=snote)
    return sub

def _msg2tr(m):
    "Tool-result `Part` recovered from code message `m`, whose source is the literal call"
    if (p := _parse_call(m.content)) is None: raise ValueError(f"not a keyword-only tool call: {m.content!r}")
    tid = m.meta.setdefault('tool_id', 'call_'+rtoken_hex(8))
    return ToolResult(id=tid, name=p[0], arguments=p[1], text=render_outputs_ai(m.output or []))

def dlg2reply(sub):
    "Implode a sub-dialog of note and code messages back into reply markdown"
    parts = [mk_tr_details(_msg2tr(m), mx=None).strip() if m.msg_type==scode else m.content.strip() for m in sub.messages]
    return '\n\n'.join(p for p in parts if p)

def _inline_atts(m, text):
    "Rewrite `attachment:` refs in `text` as data URIs from `m`'s attachments, making the Markdown self-contained"
    for a in m.attachments: text = text.replace(f'attachment:{a.id}', f'data:{a.content_type};base64,{base64.b64encode(a.data).decode()}')
    return text

def msg2md(m,
    weave:bool=False, # Code messages contribute only `render_text` over their outputs, as document prose?
):
    "One message as Markdown: notes verbatim with attachments inlined as data URIs, code in a ```python fence with outputs in a `::: output` div, prompts and replies in `::: prompt`/`::: reply` divs; None for empty code"
    if m.msg_type == snote: return _inline_atts(m, m.content)
    if m.msg_type == sraw: return fenced(m.content)
    if m.msg_type == sprompt:
        parts = [fenced(_inline_atts(m, m.content), ' prompt', ch=':')]
        if (ai := (m.ai_res or '')).strip(): parts.append(fenced(fmt_tools(ai), ' reply', ch=':'))
        return '\n\n'.join(parts)
    if weave: return render_text(m.output or []) or None
    if not m.content.strip(): return None
    parts = [fenced(m.content, 'python')]
    if (outs := render_md(m.output or [])): parts.append(fenced(outs, ' output', ch=':'))
    return '\n\n'.join(parts)

def dlg2md(d,
    exportfilter:bool=False, # Keep only the messages `export_filter` selects?
    weave:bool=False, # Code messages contribute only `render_text` over their outputs, as document prose?
):
    "The dialog as one Markdown document"
    ms = export_filter(d) if exportfilter else d
    return '\n\n'.join(filter(None, (msg2md(m, weave=weave) for m in ms))) + '\n'
