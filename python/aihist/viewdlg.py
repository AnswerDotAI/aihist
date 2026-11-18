"View a dialog or notebook in the browser, in mdhtml's viewer"
from pathlib import Path

from fastcore.meta import delegates
from fastcore.nbio import nb_frontmatter
from fastcore.script import call_parse
from aidialog.ipynb import read_ipynb
from mdhtml import parse_dlg, parse_md
from mdhtml.fill import instantiate_nb
from mdhtml.mustache import MUSTACHE
from mdhtml.viewmd import view, MathMode

from .reply import fmt_tools

@call_parse(pos=['file'])
@delegates(view, but=['src', 'name', 'base'])
def main(
    file:str, # Dialog or notebook `.ipynb` to view
    fill:bool=False, # Instantiate the template from formdata, executing its participating code
    math:MathMode=MathMode.brackets, # Math delimiters to recognize
    bare_autolinks:bool=True, # Autolink bare URLs and email addresses
    implicit_figures:bool=True, # Promote image-only paragraphs to figures
    frontmatter:bool=True, # Strip leading frontmatter into document metadata
    **kwargs
):
    "Render a dialog or notebook to a page with one element per message, and open it in a browser"
    d = read_ipynb(file)
    meta = nb_frontmatter(d, strvals=True)
    for k in ('number_headings', 'cite_style'):
        if kwargs.get(k) is None: kwargs[k] = meta.get(k)
    options = dict(math=math, bare_autolinks=bare_autolinks, implicit_figures=implicit_figures, frontmatter=frontmatter, templates=MUSTACHE)
    if fill:
        text = instantiate_nb(file, strict=False)
        for w in text.warnings: print(w)
        document = parse_md(text, **options)
    else: document = parse_dlg(d, reply=fmt_tools, **options)
    view(document, name=Path(file).stem, base=Path(file).resolve().parent, **kwargs)
