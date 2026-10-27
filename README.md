# aihist

aihist turns dialog notebooks into input for a language model, and saved replies back into messages. A dialog is a Jupyter notebook in [aidialog](https://github.com/AnswerDotAI/aidialog)'s format: notes, code, raw cells, and prompts whose replies are saved in the prompt cell's output. Solveit uses aihist through its gateway, rustygate, and Python programs use it directly.

aihist is a Rust crate with Python bindings. The crate needs no Python. The `aihist` Python package works with aidialog's `Dialog`, `Message` and `Msg` objects.

## Install

```bash
pip install aihist
```

## Model input

`build` gives a prompt's model input as Responses input items: the cells above the prompt cell, each as a model reads it, then the prompt. Each earlier prompt is followed by its saved reply, read back into text, tool calls and tool results. A host can add example notebooks, CRAFT dialogs, `$` values and warnings. The input fits 80% of the model's input window. Pinned cells always stay, and other cells are dropped oldest first.

```python
from aidialog.dialog import Dialog
from aihist import build

dlg = Dialog(name='demo')
dlg.mk_message('Distances are in metres.', msg_type='note')
p = dlg.mk_message('How far is 3 km?', msg_type='prompt')
items = build(dlg, p.id, dict(input_modalities=['text', 'image'], max_input_tokens=200_000))
```

The third argument says what the model reads, as fastllm's `get_model_info` or a model catalog entry gives it. fastllm's `response_input` turns the items into fastllm messages.

`build` estimates tokens by character class, calibrated against OpenAI's o200k tokenizer, and adds a margin for other providers' tokenizers. `count_tokens` gives the estimate for any text: `count_tokens('def fetch_daily(lat, lon): pass')` is 9.

## Saved replies

A saved reply is Markdown, with each tool call and its result in a fenced `json {.tool}` block, and the turn's token use in a `json {.usage}` block. `fmt2hist` reads a reply into chat messages, `parse_tools` splits it at its tool blocks, and `strip_tools` removes them. `set_result` replaces one call's stored result and leaves the rest of the reply unchanged. `Reply` builds a reply as a model's stream arrives, and shows the text to display at each step.

`Reply.add` appends completed tool blocks after their pending rows. `write_reply` formats a completed turn's canonical assistant and tool messages through the same Rust writer. Saved history normally shortens tool results to 5,000 characters and string arguments to 25,000. Full or marked text bypasses this shortening, but not the practical 1MiB text ceiling, which also applies to `set_result`.

`fmt_tools` rewrites each tool and usage block in a reply as a folded `::: details` div for display, using `tool_md` and `usage_md`.

## Dialogs and chats

`dlg2chat` turns a dialog into chat messages, with each reply's tool calls as real parts, and `chat2dlg` turns chat messages back into a dialog. `reply2dlg` turns one reply into a dialog of notes and code that calls the tools, which `dlg2reply` turns back into the reply. `hist_xml`, `to_media` and `to_parts` give one message as a model reads it.

`dlg2md` writes a dialog as one Markdown document: notes as they are, code in `python` fences with its outputs in `::: output` divs, and each prompt and its reply in `::: prompt` and `::: reply` divs. `msg2md` writes one message.
