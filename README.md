# aihist

aihist turns dialog notebooks into input for a language model, and saved replies back into messages. A dialog is a Jupyter notebook in [aidialog](https://github.com/AnswerDotAI/aidialog)'s format: notes, code, raw cells, and prompts whose replies are saved in the prompt cell's output. Solveit uses aihist through its gateway, rustygate, and Python programs use it directly.

aihist is a Rust crate with Python bindings. The crate needs no Python. The `aihist` Python package works with aidialog's `Dialog`, `Message` and `Msg` objects.

## Install

```bash
pip install aihist
```

## Model input

`build` gives a prompt's model input as Responses input items: the dialog messages above the prompt, each as a model reads it, then the prompt. Each earlier prompt is followed by its saved reply, read back into text, tool calls and tool results. A host can add example notebooks, CRAFT dialogs, `$` values and warnings. The input fits 80% of the model's input window. Pinned messages always stay. Other messages are dropped oldest first, with cutoffs every ten messages in the original dialog and before and after any message whose input and text output exceed 20,000 characters combined. Pinned and skipped messages count toward those positions.

```python
from aidialog.dialog import Dialog
from aihist import build

dlg = Dialog(name='demo')
dlg.mk_message('Distances are in metres.', msg_type='note')
p = dlg.mk_message('How far is 3 km?', msg_type='prompt')
items = build(dlg, p.id, dict(input_modalities=['text', 'image'], max_input_tokens=200_000))
```

The third argument says what the model reads, as fastllm's `get_model_info` or a model catalog entry gives it. fastllm's `response_input` turns the items into fastllm messages.


Markdown embeds such as `![chart](chart.png#ai)` in notes and prompts include linked images, PDFs, audio or video when the model supports them. Known text files, including Markdown, Quarto, plain text, JSON and TOML, supply UTF-8 text context instead of media parts. Text files exceeding the existing 1MiB text limit produce an unavailable notice; originals remain accessible to tools. Unmarked external embeds and saved replies are not expanded. Notebook attachments and code outputs retain their normal media handling. Python loads HTTP(S) URLs, base64 data URLs and ordinary filesystem paths relative to the dialog; absolute paths keep their normal meaning. Rustygate maps `/static/` to its served root and applies its file restrictions. Linked files are not included in the token estimate.

The Rust core loads linked media while building the retained context. Rust hosts pass `media::Context { http, root, path }` to async `input::build`. `path` is the notebook filename; `root` is the gateway's served root, or `None` for ordinary filesystem paths. Python uses the same core through synchronous bindings with a shared I/O runtime. Media paths accept strings or PathLike objects. Neither path changes the notebook.

`references` returns sorted unique vars, commands and tools from the prompt's prefit context. An explicitly addressed prompt contributes even when skipped. The host evaluates variables and supplies tool schemas. `cell_turn` returns one message's canonical parts and normalized saved reply; `input::group_turns` groups these contributions through each reply without fitting or shortening call IDs. `dlg2hist`, `dlg2chat` and `chat2dlg` remain Python APIs for full transcripts, including the final saved reply.

`build` estimates tokens by character class, calibrated against OpenAI's o200k tokenizer, and adds a margin for other providers' tokenizers. `count_tokens` gives the estimate for any text: `count_tokens('def fetch_daily(lat, lon): pass')` is 9.

## Saved replies

A saved reply is Markdown, with each tool call and its result in a fenced `json {.tool}` block, and the turn's token use in a `json {.usage}` block. `fmt2hist` reads a reply into chat messages, `parse_tools` splits it at its tool blocks, and `strip_tools` removes them. `set_result` replaces one call's stored result and leaves the rest of the reply unchanged. `Reply` builds a reply as a model's stream arrives, and shows the text to display at each step.

`Reply.add` appends completed tool blocks after their pending rows. `write_reply` formats a completed turn's canonical assistant and tool messages through the same Rust writer. Saved history normally shortens tool results to 5,000 characters and string arguments to 25,000. Full or marked text bypasses this shortening, but not the practical 1MiB text ceiling, which also applies to `set_result`.

`fmt_tools` rewrites each tool and usage block in a reply as a folded `::: details` div for display, using `tool_md` and `usage_md`.

## Dialogs and chats

`dlg2chat` turns a dialog into chat messages, with each reply's tool calls as real parts, and `chat2dlg` turns chat messages back into a dialog. `reply2dlg` turns one reply into a dialog of notes and code that calls the tools, which `dlg2reply` turns back into the reply. `hist_xml`, `to_media` and `to_parts` give one message as a model reads it.

`dlg2md` writes a dialog as one Markdown document, and `msg2md` writes one message. Both call the functions of the same names in [mdhtml](https://github.com/AnswerDotAI/mdhtml#notebooks-and-dialogs) and fold the tool calls in each reply with `fmt_tools`. Notes are written as they are, code in language fences with its outputs in `::: output` divs, and each prompt and its reply in `::: prompt` and `::: reply` divs. Stored attachments referenced by notes, prompts and replies are inlined, and ordinary file links remain links.

Pass `cells=True` to write each message as an `mdhtml` notebook cell with its notebook metadata and the id `_<message id>`. The other options are mdhtml's: `language`, `inline_atts`, `weave` and, for `dlg2md`, `exportfilter`.

`msg2text(m, part='input')` gives the source, or `part='output'` gives text output with prompt tools removed. `msgs2py(msgs)` writes nbdev-marked code; `full=True` includes all code, commenting notes, prompts, replies and outputs. These conversions read saved messages without fetching linked files or changing the dialog.

`reply2dlg` and `dlg2reply` retain Python code-cell editing and use the native reply parser, output renderer and writer. `dlg2reply` applies the same saved-history shortening and text ceiling as `write_reply`.

## Viewing a dialog

`viewdlg` needs the `view` extra, which adds mdhtml's highlighter and the packages that `--fill` runs code with:

```bash
pip install 'aihist[view]'
```

```bash
viewdlg examples/nbsample.ipynb
viewdlg examples/nbsample.ipynb --fill
```

`viewdlg` opens a dialog or notebook in the browser, in [mdhtml](https://github.com/AnswerDotAI/mdhtml#browser-viewer)'s viewer. Each message is one `div` element with its notebook metadata. The element's id is the message's id after an underscore, such as `_a1b2c3d4`, so a Solveit message link such as `#_a1b2c3d4` scrolls to it. Note messages are read as one combined document, rather than one document per cell. That way, a numbered list can continue across a code message, and a footnote defined in one message resolves from another.

Code appears as a highlighted block with its stored outputs below it. Each prompt appears in a bordered section with its reply indented below it, and the tool calls in the reply are folded.

`--fill` views a notebook template filled from its `formdata`. This executes the template's participating code, as mdhtml's `fillmd` does. A filled notebook uses its exported messages when any exist, otherwise its non-pinned messages. The other options are `viewmd`'s.

`examples/nbsample.ipynb` is a small notebook to try both commands on. It has a table, an image, footnotes, cross-references, math and code with stored outputs. Its frontmatter has `formdata` for the template tokens in its text, which `--fill` fills.
