"aihist reads saved replies as blocks and chat messages, writes a reply as it streams in, writes the blocks aidialog's writer does, and shows a reply's blocks as Markdown"
import json
from aidialog.msg_parts import Msg, Text, Thinking, ToolUse, ToolResult, InputImage, MdStr, FullResponse, msg2dict, hist2fmt
from aihist import _core, parse_tools, set_result, strip_tools, conv_tools, fmt2hist, reply2chat, Reply, write_reply, fmt_tools

def tool(tid, code, res): return f"```json {{.tool}}\n{json.dumps(dict(id=tid, name='py', args=dict(code=code), result=res))}\n```"
def blocks(reply): return json.loads(_core.parse_reply(reply))

def flagged(msgs):
    "The parts of `msgs` as `msg2dict` writes them, including canonical flags and public string markers"
    return [p | dict(md=p.get('md', False) or isinstance(o.text, MdStr), full=p.get('full', False) or isinstance(o.text, FullResponse))
        for m in msgs for p,o in zip(msg2dict(m)['content'], m.content)]

def test_saved_replies():
    reply = f"Let me check.\n\n{tool('t1', 'w=1', '1')}\n\n{tool('t2', 'h=2', '2')}\n\nIt is 1 by 2.\n\n```json {{.usage}}\n{{}}\n```"
    # `parse_tools` splits a reply at its tool blocks, and `strip_tools` removes them, with the usage block.
    segs = parse_tools(reply)
    assert [d and d['id'] for _,d in segs] == ['t1', 't2', None] and segs[0][0] == 'Let me check.\n\n'
    assert strip_tools(reply).split() == 'Let me check. It is 1 by 2.'.split()
    # `set_result` replaces one call's stored result and leaves the text and the other blocks alone. Setting the same result again changes nothing.
    new = set_result(reply, 't2', '3')
    assert [d['result'] for _,d in parse_tools(new) if d] == ['1', '3'] and tool('t1', 'w=1', '1') in new and strip_tools(new) == strip_tools(reply)
    assert set_result(new, 't2', '3') == new
    # `fmt2hist` reads the reply as chat messages: both calls in one round, then their results, then the closing text.
    msgs = fmt2hist(reply)
    assert [m.role for m in msgs] == ['assistant', 'tool', 'assistant']
    assert [p.id for p in msgs[1].content] == ['t1', 't2'] and msgs[2].content[0].text == 'It is 1 by 2.'
    # `reply2chat` splits the round into one call and its result at a time.
    assert [m.role for m in reply2chat(reply)] == ['assistant', 'tool', 'assistant', 'tool', 'assistant']
    # `conv_tools` turns the `<details>` envelopes that fastllm once wrote into tool blocks.
    legacy = """<details class='tool-usage-details'>\n<summary>f(a=1)</summary>\n\n```json\n{"id": "x1", "call": {"function": "f", "arguments": {"a": 1}}, "result": "2"}\n```\n\n</details>"""
    tu = fmt2hist(conv_tools(legacy))[0].content[0]
    assert (tu.id, tu.name, tu.arguments) == ('x1', 'f', {'a': 1}) and conv_tools(conv_tools(legacy)) == conv_tools(legacy)

def test_writers_agree():
    "aihist's writer, which rustygate uses, writes the blocks aidialog's `hist2fmt` writes: a long result cut the same way, and a call without a result left out"
    turn = [Msg('assistant', [Text('Let me measure.'), ToolUse(id='t1', name='py', arguments=dict(code='w = 1'))]),
            Msg('tool', [ToolResult(id='t1', name='py', text='1 ' * 3500)]),
            Msg('assistant', [ToolUse(id='s1', name='web_search', arguments=dict(query='dot size'), server=True, text=MdStr('Dots: https://example.com')),
                              Text('It is 1 wide.'), ToolUse(id='t2', name='show', arguments=dict(path='a`b.png', scale=0.5))]),
            Msg('tool', [ToolResult(id='t2', name='show', text='full ' * 1200, md=True, full=True)]),
            Msg('assistant', [ToolUse(id='t3', name='py', arguments=dict(code='2'))])]
    ours = _core.write_reply(json.dumps(flagged(turn)), json.dumps(dict(model='m', prompt_tokens=90, completion_tokens=10, total_tokens=100, cost=0.0123)))
    assert blocks(ours) == blocks(hist2fmt(turn, mx=5000)) == blocks(write_reply(turn))
    full = next(b['tool'] for b in blocks(ours) if b.get('tool', {}).get('id') == 't2')
    assert full['md'] and full['result'] == 'full ' * 1200
    assert '```json {.usage}' in ours and '"cost":0.0123' in ours
    hist = [p for m in fmt2hist(ours) for p in m.content]
    assert all(not p.server for p in hist if isinstance(p, (ToolUse, ToolResult)))
    assert [(p.id, p.arguments) for p in hist if isinstance(p, ToolUse)] == [('t1', dict(code='w = 1')), ('s1', dict(query='dot size')), ('t2', dict(path='a`b.png', scale=0.5))]
    turn[1].content[0].text = [Text('Measured 1'), InputImage(text='data:image/png;base64,AAAA')]
    r = Reply()
    for m in turn:
        for p in m.content: r.add(p)
    assert blocks(r.finish()) == blocks(write_reply(turn))
    turn[2].content[0].text,turn[2].content[1].text = MdStr('Final-only server result.'),'Updated by callback.'
    turn[-1].content += [InputImage(text='https://example.com/answer.png'), Thinking('private reasoning')]
    final = write_reply(turn)
    assert 'Measured 1' in final and '<media input_image' in final and 'Updated by callback.' in final
    assert 'Final-only server result.' in final and 'Final-only server result.' not in r.finish()
    assert 'https://example.com/answer.png' in final and 'private reasoning' not in final

def test_reply_streams():
    "A `Reply` shows each stream item as it arrives: a glyph per thinking delta, the text, and a row while a client call runs"
    r = Reply()
    shown = [r.add(o) for o in (Thinking(text='hmm'), Thinking(text='more'), Text('Let me '), Text('check.'),
        ToolUse(id='t1', name='py', arguments=dict(code='1+1')), ToolResult(id='t1', name='py', text='2'))]
    assert shown[:4] == ['🧠', '🧠', '\n\nLet me ', 'check.'] and '⏳' in shown[4] and 'py(code=' in shown[4]
    assert blocks(shown[5])[0]['tool']['result'] == '2' and len(blocks(r.finish())) == 2
    server = r.add(ToolUse(id='s1', name='search', server=True, text='Found the answer.'))
    assert blocks(server)[0]['tool']['result'] == 'Found the answer.'
    # A turn cut short saves the reply so far, its tool blocks included, then the note.
    msgs = fmt2hist(r.finish('*[Response interrupted]*'))
    assert [m.role for m in msgs] == ['assistant', 'tool', 'assistant']
    assert msgs[0].content[0].text == 'Let me check.' and msgs[1].content[0].text == '2' and msgs[2].content[0].text == '*[Response interrupted]*'

def test_reply_limits():
    "Full or marked results bypass history shortening, not the practical 1MiB ceiling; string arguments retain 25k characters"
    text = 'é'*600_000+'tail'
    call = Msg('assistant', [ToolUse(id='big', name='measure', arguments=dict(code='word '*6000))])
    for value in (FullResponse(text), f'𝍁{text}𝍁'):
        saved = blocks(write_reply([call, Msg('tool', [ToolResult(id='big', name='measure', text=value)])]))[0]['tool']
        assert 24_000 < len(saved['args']['code']) < 25_100
        assert saved['result'].endswith('tail') and len(saved['result'].encode()) < (1<<20)+100
    changed = blocks(set_result(tool('big', '1', 'old'), 'big', text))[0]['tool']['result']
    assert changed.endswith('tail') and len(changed.encode()) < (1<<20)+100 and text not in changed

def test_reply_display():
    "`fmt_tools` shows a reply's tool and usage blocks as folded `::: details` divs, and leaves other text and unparseable blocks alone"
    disp = fmt_tools(f"Before.\n\n{tool('t1', '1+1', '2')}\n\nAfter.")
    assert '::: {.details .tool-usage-details}' in disp and '## `py(code="1+1")→"2"`' in disp and 'Before.' in disp and 'After.' in disp
    # A code tool shows its code and output in fences. Another tool shows its arguments as JSON, and a result marked `md` as Markdown.
    assert 'Code:\n```py\n1+1\n```' in disp and 'Output:' in disp
    other = fmt_tools('```json {.tool}\n{"id": "t3", "name": "web", "args": {"url": "x"}, "result": "ok"}\n```')
    assert '```json\n{\n  "url": "x"\n}\n```' in other and 'Output:\n\n```\nok\n```' in other
    md = fmt_tools('```json {.tool}\n{"id": "t5", "name": "read_url", "args": {"url": "x"}, "result": "# Title\\n\\nBody", "md": true}\n```')
    assert '\n\n# Title\n\nBody\n' in md and '```\n# Title' not in md
    assert '## $0.0123' in fmt_tools('```json {.usage}\n{"model": "m", "total_tokens": 150, "cost": 0.0123}\n```')
    # The display form has no blocks left, so formatting it again changes nothing. The div's fence is longer than any colon run inside it.
    bad = '```json {.tool}\nnot json\n```'
    assert fmt_tools(disp) == disp and fmt_tools(bad) == bad
    assert fmt_tools('```json {.tool}\n{"id": "t2", "name": "sh", "args": {}, "result": ":::"}\n```').startswith('::::')
