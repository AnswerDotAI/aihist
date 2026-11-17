"View a dialog or notebook in the browser, in mdhtml's viewer"
from pathlib import Path

from fastcore.meta import delegates
from fastcore.nbio import nb_frontmatter
from fastcore.script import call_parse
from aidialog.ipynb import read_ipynb
from mdhtml.fill import instantiate_nb
from mdhtml.viewmd import view

from .hist import dlg2md

@call_parse(pos=['file'])
@delegates(view, but=['text', 'name', 'base'])
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
    text = instantiate_nb(file, strict=False) if fill else dlg2md(d, cells=True)
    view(text, name=Path(file).stem, base=Path(file).resolve().parent, **kwargs)
