"aihist turns dialogs into model history, chat messages and Markdown documents, chat messages back into dialogs, and one message into its XML and media"
import base64, json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from aidialog.dialog import Dialog, Message, Attachment, prompt_output, code_output, displayobj
from aidialog.msg_parts import Msg, Text, ToolUse, ToolResult, hist2fmt
from aihist import (build, dlg2hist, dlg2chat, chat2dlg, reply2dlg, dlg2reply, tool_call, hist_xml, to_parts, to_media, media_item, resize_img,
    pdf_pages, output_parts, merge_media, vars_hist, is_nameerr, get_exprs, get_refs, references, cell_turn, _core, task_tags, warning_tag, msg2md, dlg2md)

png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGP4DwABAQEAsTj2FAAAAABJRU5ErkJggg==')
images = dict(input_modalities=['text', 'image'])
def tool(tid, code, res): return f"```json {{.tool}}\n{json.dumps(dict(id=tid, name='python', args=dict(code=code), result=res))}\n```"

def test_history():
    d = Dialog(name='h')
    d.mk_message('a note', msg_type='note')
    d.mk_message('x = 1', output=[displayobj('1', 'markdown')])
    d.mk_message('What is x?', msg_type='prompt', output=prompt_output('x is 1.'))
    d.mk_message('And $`x`?', msg_type='prompt')
    # Notes and code before a prompt join its user turn, and each prompt is followed by its reply, pending when it has none.
    hist = dlg2hist(d, {})
    assert [len(t) if isinstance(t, list) else t for t in hist] == [3, 'x is 1.', 1, '<output result="pending" reason="incomplete"/>']
    assert hist[0][:2] == [f'<markdown id="{d.messages[0].id}">a note</markdown>', f'<code id="{d.messages[1].id}">x = 1<out>1</out></code>']
    # A prompt reads as an instruction and the prompt in an envelope. `plain` keeps the bare text.
    assert hist[2][0] == task_tags('And $`x`?', d.messages[3].id) and dlg2hist(d, {}, plain=True)[2] == ['And $`x`?']
    assert get_exprs(d.messages) == ['x'] and get_refs('use &`bash` and &`[python, bash]`') == ['bash', 'python']
    # `$` values make their own exchange, without the names that are undefined.
    recs = dict(x=dict(status='ok', data={'text/plain': '1'}, metadata=dict(type='int')), y=dict(status='error', ename='NameError', evalue='no y'))
    assert is_nameerr(recs['y']) and not is_nameerr(recs['x']) and vars_hist({}, dict(y=recs['y'])) == []
    assert vars_hist({}, recs) == [["""<variables>The 'variables' section provides values of: $`x`<variable name="x" type="int">1</variable></variables>"""],
        'Understood, I will use these variable values.']
    assert warning_tag('') is None and warning_tag('x is undefined') == '<system-reminder>**NB**: x is undefined</system-reminder>'
    # `build` takes a `Dialog` as well as a notebook, and gives the Responses input items that end with the prompt.
    items = build(d, d.messages[3].id)
    assert [i['role'] for i in items] == ['user', 'assistant', 'user'] and items[-1]['content'][-1]['text'] == task_tags('And $`x`?', d.messages[3].id)
    # An explicitly addressed skipped prompt contributes vars; earlier skipped prompts and non-prompt vars don't.
    d.messages[0].content += r' $`note_var` &`helper`'
    d.messages[1].content += r' $`code_var`'
    d.messages[2].content,d.messages[2].skipped = r'$`hidden_var`',True
    d.messages[3].skipped = True
    assert references(d, d.messages[3].id) == dict(vars=['x'], cmds=[], tools=['helper'])
    craft = dict(path='CRAFT.ipynb', nb=dict(cells=[dict(cell_type='code', source=r'$`craft_var`', metadata=dict(solveit_ai=True, skipped=True))]))
    assert references(d, d.messages[3].id, [craft])['vars'] == ['craft_var', 'x']

def test_chats_and_dialogs():
    d = Dialog(name='seq')
    tid = 'call_'+'x'*70
    p = d.mk_message('Two steps.', msg_type='prompt', output=f"{tool(tid, '1+1', '2')}\n\n{tool('x2', '2+2', '4')}\n\nDone.")
    # `dlg2chat` recovers each reply's tool calls, one call and its result at a time, and `chat2dlg` turns the chat back into the dialog.
    chat = dlg2chat(d)
    assert [(m.role, len(m.content)) for m in chat] == [('user', 1), ('assistant', 1), ('tool', 1), ('assistant', 1), ('tool', 1), ('assistant', 1)]
    back = chat2dlg(chat, 'seq round-trip')
    assert back.messages[0].content == 'Two steps.' and back.messages[0].id == d.messages[0].id and dlg2chat(back) == chat
    # Full exports keep IDs and the final reply; model input alone applies its existing 64-character normalization.
    items = json.loads(_core.turns(json.dumps([cell_turn(p)])))
    assert next(i['call_id'] for i in items if i['type']=='function_call') == tid and items[-1]['content'][0]['text']=='Done.'
    target = d.mk_message('Continue.', msg_type='prompt')
    assert next(i['call_id'] for i in build(d, target.id) if i['type']=='function_call') == tid[:64]
    # `reply2dlg` turns one reply into notes and code that call the tools, which `dlg2reply` turns back into the reply.
    pm = Message('What is 1+1?', msg_type='prompt', output=hist2fmt([Msg('assistant', [Text('Check:'), ToolUse(id='x1', name='python', arguments=dict(code='1+1'))]),
        Msg('tool', [ToolResult(id='x1', name='python', text='2')]), Msg('assistant', [Text('It is 2.')])], mx=None))
    sub = reply2dlg(pm)
    assert [m.msg_type for m in sub.messages] == ['note', 'code', 'note'] and tool_call(sub.messages[1]) == dict(name='python', arguments=dict(code='1+1'))
    assert dlg2reply(sub) == pm.ai_res
    sub.messages[1].output = code_output("'hello'")
    assert json.loads(_core.parse_reply(dlg2reply(sub)))[1]['tool']['result'] == 'hello'

def test_messages_and_media(tmp_path):
    att = Attachment(png, 'image/png')
    m = Message(f'What is in ![](attachment:{att.id})?', msg_type='prompt', attachments=[att], id='_t1')
    # A message's parts are the media it refers to, each after its tag, then its XML.
    parts = to_parts(m, images)
    assert parts == [f'<media id="{att.id}" type="content" mime="image/png"></media>', png, hist_xml(m)]
    assert hist_xml(m).endswith('<prompt sid="_t1">What is in ![](attachment:' + att.id + ')?</prompt>')
    assert '<media-unavailable' in to_media(m, {})[0]
    m.content = m.content.replace(att.id, att.id+'#ai')
    assert to_media(m, images)[1]==png
    assert media_item('variable', 'img', png, images) == ['<media id="img" type="variable" mime="image/png"></media>', png]
    assert 'unsupported media type' in media_item('variable', 'img', png, {})[0] and resize_img(png) == png
    audio = media_item('variable', 'clip', b'fLaCtest', dict(input_modalities=['audio']))[-1]
    assert audio.type == 'input_audio' and audio.mime == 'audio/flac' and base64.b64decode(audio.text.split(',', 1)[1]) == b'fLaCtest'
    assert 'unsupported media type' in media_item('variable', 'clip', b'fLaCtest', {})[0]
    assert pdf_pages(b'%PDF-1.4\n1 0 obj << /Type /Pages /Kids [] /Count 3 >> endobj') == 3
    # A code message's images become `Part`s, which `merge_media` puts before the output text.
    code = Message('plot()', msg_type='code', output=[dict(output_type='display_data', metadata={}, data={'image/png': base64.b64encode(png).decode()})])
    assert [p.type for p in merge_media('42', output_parts(code, images))] == ['text', 'input_image', 'text']
    assert 'unsupported media type' in merge_media('42', output_parts(code, {}))

    # Marked embeds load relative to their dialog, while normal links and saved replies remain text.
    (tmp_path/'photo.bin').write_bytes(png)
    d = Dialog(name='linked')
    d.path_ = tmp_path/'linked.ipynb'
    note = d.mk_message(r'![Photo](photo.bin#ai) ![](missing.png#ai) ![](photo.bin) [link](photo.bin#ai)', msg_type='note')
    parts = to_media(note, images)
    assert len(parts)==3 and parts[1]==png and '<media-unavailable' in parts[2]
    source = f'![](data:image/png;base64,{base64.b64encode(png).decode()}#ai)'
    p = d.mk_message(source, msg_type='prompt', output=prompt_output(r'![](missing-reply.png#ai)'))
    assert to_media(p, images)[1:]==[png] and to_parts(p, images)[-1]==hist_xml(p)
    assert to_media(d.mk_message(source, msg_type='code'), images)==[]
    assert note.content.startswith(r'![Photo](photo.bin#ai)') and not hasattr(note, 'context_media')

    # HTTP media uses the same loader. XML and omitted history never fetch it; a direct skipped cell still shows it.
    requests = []
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(png)
        def log_message(self, *args): pass
    with HTTPServer(('127.0.0.1', 0), Handler) as server:
        thread = Thread(target=server.serve_forever, kwargs=dict(poll_interval=.01), daemon=True)
        thread.start()
        try:
            remote = d.mk_message(f'![](http://127.0.0.1:{server.server_port}/image#ai)', msg_type='note', before=p, meta=dict(skipped=True))
            assert 'http://' in hist_xml(remote) and not requests
            assert to_media(remote)[1]==png and requests==['/image']
            remote.skipped = False
            build(d, p.id, dict(images, max_input_tokens=1))
            assert requests==['/image']
        finally: server.shutdown(); thread.join()


def test_markdown_documents():
    d = Dialog(name='doc')
    d.mk_message('# Results ![plot](attachment:img1.png)', msg_type='note').mk_attachment(png, 'image/png', 'img1.png')
    d.mk_message('8*8', output=code_output('64'))
    d.mk_message('Why 64?', msg_type='prompt', output=f"{tool('x1', '8*8', '64')}\n\nBecause $8^2=64$.")
    # `dlg2md` writes a note as it is, with its attachments as data URIs, code with its outputs, and a prompt with its reply, whose tool blocks `fmt_tools` folds.
    doc = dlg2md(d)
    assert doc.startswith(f'# Results ![plot](data:image/png;base64,{base64.b64encode(png).decode()})\n\n```python\n8*8\n```\n\n::: output')
    assert '::: prompt' in doc and '::: reply' in doc and 'tool-usage-details' in doc and msg2md(Message('')) is None
    # `weave` keeps only a code message's output text, and `exportfilter` keeps the messages aidialog's `export_filter` selects.
    assert dlg2md(d, weave=True).split('\n\n')[1] == '64'
    d.messages[2].pinned = True
    assert '::: prompt' not in dlg2md(d, exportfilter=True)
