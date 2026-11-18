"View a dialog or notebook in the browser, in mdhtml's viewer"
from functools import partial
from pathlib import Path

from fastcore.meta import delegates
from fastcore.nbio import nb_frontmatter
from fastcore.script import call_parse
from aidialog.ipynb import read_ipynb
from mdhtml import dlg2mdhtml
from mdhtml.fill import instantiate_nb
from mdhtml.viewmd import view

from .reply import fmt_tools

@call_parse(pos=['file'])
@delegates(view, but=['src', 'name', 'base', 'parse'])
def main(
    file:str, # Dialog or notebook `.ipynb` to view
    fill:bool=False, # Instantiate the template from formdata, executing its participating code
    **kwargs
):
    "Render a dialog or notebook to a page with one element per message, and open it in a browser"
    d = read_ipynb(file)
    meta = nb_frontmatter(d, strvals=True)
    for k in ('number_headings', 'cite_style'):
        if kwargs.get(k) is None: kwargs[k] = meta.get(k)
    kwargs.update(name=Path(file).stem, base=Path(file).resolve().parent)
    if fill: view(instantiate_nb(file, strict=False), **kwargs)
    else: view(d, parse=partial(dlg2mdhtml, reply=fmt_tools), **kwargs)
