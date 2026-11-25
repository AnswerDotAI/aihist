#[cfg(feature = "full")] use base64::{Engine, engine::general_purpose::STANDARD};
#[cfg(feature = "full")] use serde_json::{Map, Value, json};
#[cfg(feature = "full")] use std::io::Write;

use aihist::output::keep_ends;
#[cfg(feature = "full")] use aihist::tokens::{IMG_TOKENS, PDF_PAGE_TOKENS, bundle, cell, count, pdf_pages};
#[cfg(feature = "full")] use aihist::{dlg2md_nb, msg2md_in, msg2text, msgs2py};

#[test]
fn keep_ends_cuts_the_middle_at_char_boundaries() {
    assert_eq!(keep_ends("short".into(), 5, 5), "short");
    let kept = keep_ends("€ab€cd€".into(), 4, 5);
    assert_eq!(kept, "€a\n[4 bytes of output removed]\ncd€");
}

#[cfg(feature = "full")]
#[test]
fn counts_by_character_class() {
    assert_eq!(count("def fetch_daily(lat, lon): pass"), 9);
    assert_eq!(count("**======turn: 8======**"), 7);
    assert_eq!(count("2024-01-01"), 6);
    assert_eq!(count("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nG"), 34);
    assert_eq!(count("  \n\t"), 0);
}

#[cfg(feature = "full")]
#[test]
fn pdf_pages_reads_clear_and_object_streams() {
    assert_eq!(pdf_pages(b"%PDF-1.4\n1 0 obj << /Type /Pages /Kids [] /Count 3 >> endobj"), 3);
    let mut z = flate2::write::ZlibEncoder::new(vec![], flate2::Compression::default());
    z.write_all(b"1 0 << /Type /Pages /Count 7 >>").unwrap();
    let mut pdf = b"%PDF-1.5\n2 0 obj << /Type /ObjStm /N 1 /First 4 >> stream\n".to_vec();
    pdf.extend(z.finish().unwrap());
    pdf.extend(b"\nendstream endobj");
    assert_eq!(pdf_pages(&pdf), 7);
    assert_eq!(pdf_pages(b"not a pdf"), 0);
}

#[cfg(feature = "full")]
#[test]
fn media_charge_by_constant() {
    let data: Map<String, Value> = serde_json::from_value(json!({"text/plain": "<Figure>", "image/png": "aGk="})).unwrap();
    assert_eq!(bundle(&data), IMG_TOKENS + count("<Figure>"));
    let pdf = STANDARD.encode(b"%PDF-1.4 << /Type /Pages /Count 2 >>");
    let data: Map<String, Value> = serde_json::from_value(json!({"application/pdf": pdf})).unwrap();
    assert_eq!(bundle(&data), 2 * PDF_PAGE_TOKENS);
    let cell_v = json!({"id": "c", "cell_type": "markdown", "source": "# hi"});
    assert_eq!(cell(&cell_v, None), json!({"source": 2}));
}

#[cfg(feature = "full")]
#[test]
fn markdown_text_and_python_exports() {
    let reply = "Before.\n\n```json {.tool}\n{\"id\":\"t1\",\"name\":\"py\",\"args\":{\"code\":\"1+1\"},\"result\":\"2\"}\n```\n\nDone. ![](attachment:reply%20plot.png)";
    let nb = json!({
        "cells": [
            {
                "id": "n1", "cell_type": "markdown", "metadata": {},
                "source": "# Results ![p](attachment:a.png)",
                "attachments": {"a.png": {"image/png": "AAAA"}}
            }, {
                "id": "c1", "cell_type": "code", "metadata": {"nbdev": {"export": "true"}},
                "source": "x = 1",
                "outputs": [{"output_type": "stream", "name": "stdout", "text": "ok\n"}]
            }, {
                "id": "p1", "cell_type": "code", "metadata": {"solveit_ai": true, "pinned": true},
                "source": "%%prompt\nCheck",
                "outputs": [{"output_type": "display_data", "metadata": {}, "data": {"text/markdown": reply}}]
            }
        ],
        "metadata": {"attachments": {"p1": {"reply plot.png": {"image/png": "BBBB"}}}}
    });
    let doc = dlg2md_nb(&nb, "python", false, false);
    assert!(doc.contains("data:image/png;base64,AAAA"));
    assert!(doc.contains("data:image/png;base64,BBBB"));
    assert!(doc.contains("::: {.details .tool-usage-details}"));
    assert!(doc.contains("```python\nx = 1\n```"));
    assert!(!dlg2md_nb(&nb, "python", true, false).contains("::: prompt"));
    assert_eq!(msg2md_in(Some(&nb), &nb["cells"][1], "python", true).as_deref(), Some("ok\n"));
    assert_eq!(msg2text(&nb["cells"][1], "output").unwrap(), "ok\n");
    assert_eq!(msg2text(&nb["cells"][2], "output").unwrap(), "Before.\n\n\nDone. ![](attachment:reply%20plot.png)");
    assert!(msg2text(&nb["cells"][1], "bad").is_err());
    let cells = nb["cells"].as_array().unwrap();
    assert_eq!(msgs2py(cells, false), "x = 1");
    let full = msgs2py(cells, true);
    assert!(full.contains("# --- Cell: p1 (prompt) ---") && full.contains("#> ok") && full.contains("# --- Assistant Response ---"));
}
