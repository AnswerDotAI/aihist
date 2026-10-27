"aihist builds a prompt's model input from dialog cells, with the examples, CRAFTs, `$` values and warnings a host supplies, fitted to the model's window"
import base64, json, re
from aidialog.dialog import Dialog, Attachment, prompt_output
from aidialog.ipynb import write_ipynb
from aidialog.msg_parts import InputAudio, Text, InputImage, ToolResponse, tool_text
from aihist import _core, build, tool_result

png = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGP4DwABAQEAsTj2FAAAAABJRU5ErkJggg==')
vision = dict(input_modalities=['text', 'image'], max_input_tokens=10**6)

def out(typ, **kw): return dict(output_type=typ, metadata={}, **kw)
def tool(tid, code, res): return f"```json {{.tool}}\n{json.dumps(dict(id=tid, name='py', args=dict(code=code), result=res))}\n```"
def nb(d): return json.loads(write_ipynb(d))
def texts(item): return [p['text'] for p in item.get('content', []) if 'text' in p]
def last_cell(nb): return nb['cells'][-1]['id']

def fixture():
    "A dialog holding most of what a model reads: notes, code outputs, an attachment, pinned and skipped cells, and tool calls"
    d = Dialog(name='fixture')
    d.mk_message('# Units\nUse metres.', msg_type='note', meta=dict(pinned=True))
    d.mk_message('plot()\nprint("10%\\r100%")\n"all\\ndone"', output=[
        out('stream', name='stdout', text='10%\r100%\n'),
        out('display_data', data={'image/png': base64.b64encode(png).decode(), 'text/plain': '<Figure>'}),
        out('execute_result', execution_count=1, data={'text/plain': "'all\\ndone'"})])
    d.mk_message('<b>raw</b>', msg_type='raw')
    d.mk_message('A secret', msg_type='note', meta=dict(skipped=True))
    att = Attachment(png, 'image/png')
    d.mk_message(f'What is in ![](attachment:{att.id})?', msg_type='prompt', output=prompt_output('A dot.')).attachments = [att]
    reply = f"Let me check.\n\n{tool('t1', 'w=1', '1')}\n\n{tool('t2', 'h=2', '2')}\n\nIt is 1 by 2.\n\n```json {{.usage}}\n{{}}\n```"
    d.mk_message('Measure it.', msg_type='prompt', output=prompt_output(reply))
    d.mk_message('1/0', meta={'nbdev': {'export': 'true'}}, output=[
        out('error', ename='ZeroDivisionError', evalue='division by zero', traceback=['\x1b[31mZeroDivisionError\x1b[0m: division by zero']),
        out('display_data', data={'text/latex': '$x^2$'})])
    d.mk_message('And the area?', msg_type='prompt')
    return nb(d)

def test_dialog_cells():
    dlg = fixture()
    items = build(dlg, last_cell(dlg), vision)
    sent = json.dumps(items)
    # Each prompt ends a user turn and is followed by its saved reply, read back into messages and tool calls.
    assert [i.get('role') or i['type'] for i in items] == ['user', 'assistant', 'user', 'assistant', 'function_call', 'function_call',
        'function_call_output', 'function_call_output', 'assistant', 'user']
    # A skipped cell is hidden, and a model that reads images gets the attachment and the plot.
    assert 'A secret' not in sent and sent.count('"input_image"') == 2
    assert '<media-unavailable' in json.dumps(build(dlg, last_cell(dlg), dict(max_input_tokens=10**6)))
    # Code outputs read as a tool result reads: each output tagged, a progress bar in its final state, and a string result as the string itself.
    code = next(t for it in items for t in texts(it) if 'plot()' in t)
    assert '<stdout>\n100%\n</stdout>' in code and "<execute_result>\nall\ndone\n</execute_result>" in code
    # The last turn ends with the prompt, in its envelope, with its cell id.
    assert texts(items[-1])[-1].endswith('<prompt sid="' + last_cell(dlg) + '">And the area?</prompt>')
    # A small window keeps the pinned note and the prompt, and tells the model how many cells were dropped.
    small = build(dlg, last_cell(dlg), dict(max_input_tokens=1))
    assert len(small) == 1 and 'Use metres.' in texts(small[0])[0]
    assert re.search(r'5 older non-pinned cell\(s\) were omitted', texts(small[0])[1])

def test_host_context():
    dlg = fixture()
    example = Dialog(name='example')
    example.mk_message('Hi there', msg_type='prompt', output=prompt_output('Hello!'))
    craft = Dialog(name='craft')
    craft.mk_message('Distances are in metres.', msg_type='note')
    vars = {'x': dict(status='ok', data={'text/plain': '3'}, metadata=dict(type='int')), 'y': dict(status='error', ename='NameError', evalue="name 'y' is not defined")}
    items = build(dlg, last_cell(dlg), vision, path='garden/plan.ipynb', crafts=[dict(path='CRAFT.ipynb', nb=nb(craft))], vars=vars,
        examples=[nb(example)])
    # The examples come first. Their last reply ends a prefix that a provider can cache.
    assert 'Hi there' in texts(items[0])[-1] and texts(items[1]) == ['Hello!']
    # A heading opens the CRAFTs, each CRAFT is wrapped in a block, and an "End of CRAFTs." exchange closes them, also cached.
    craft_turn = texts(items[2])
    assert craft_turn[:2] == ['# Loaded CRAFTs.', '<craft-block path="/CRAFT.ipynb">'] and craft_turn[-2] == '</craft-block>'
    assert 'End of CRAFTs.' in craft_turn[-1] and texts(items[3]) == ['Ok!']
    assert [i for i, it in enumerate(items) for p in it.get('content', []) if 'cache_control' in p] == [1, 3]
    # `$` values get their own exchange, and an undefined name becomes a warning before the prompt.
    assert texts(items[4])[0] == """<variables>The 'variables' section provides values of: $`x`<variable name="x" type="int">3</variable></variables>"""
    assert texts(items[5]) == ['Understood, I will use these variable values.']
    assert '<dialog path="garden/plan.ipynb"/>' in texts(items[6])
    last = texts(items[-1])
    assert '<variable-errors>' in last[-2] and '- y' in last[-2]

def test_tool_results():
    "A tool's outputs as the model reads them: a lone `str` repr reads as the string only for code, since a `call` result already holds the text. A `ToolResponse` gives its parts unchanged, even to a model that reads no images"
    res = lambda s: [out('execute_result', data={'text/plain': s})]
    assert tool_result(res('"abc"')).text == '"abc"'
    assert tool_result(res("'a\\nb'"), code=True).text == 'a\nb'
    image = out('display_data', data={'image/png': base64.b64encode(png).decode()}) | dict(metadata=dict(full=True))
    shown = tool_result([out('stream', name='stdout', text='hi\n'), image], vision)
    assert [p.type for p in shown.text] == ['text', 'input_image', 'text'] and shown.text[0].text.startswith('<media id=')
    assert shown.full and not shown.md
    assert tool_result(res('42'), dict(input_modalities=[])).text == '42'
    marked = tool_result([out('execute_result', data={'text/markdown': '# Report'}) | dict(metadata=dict(full=True))])
    assert (marked.text, marked.md, marked.full) == ('# Report', True, True)
    audio = InputAudio('https://example.com/clip.flac', mime='audio/flac')
    given = [Text('caption'), InputImage('data:image/png;base64,'+base64.b64encode(png).decode()), audio]
    outs = [out('execute_result', data=ToolResponse(given)._repr_mimebundle_())]
    tr = tool_result(outs, dict(input_modalities=[]))
    assert [p.type for p in tr.text] == ['text', 'input_image', 'input_audio'] and tr.text[-1] == audio
    assert json.loads(_core.tool_result(json.dumps(outs), '{}', False))['text'] == tool_text(tr.text)
