import base64
import io
import os

from dotenv import load_dotenv
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()


def _describe_image_with_openai(img_b64: str) -> str:
    """Call an OpenAI vision model to generate a rich, searchable description.

    Used at ingestion time so image chunks carry meaningful text content that
    can be found by natural-language queries — not just sparse caption words.
    Because OpenAI embeddings cannot read pixels, this description IS what gets
    embedded for the image chunk, so it must be detailed.

    Returns an empty string on any error so the caller can fall back to the
    Docling-extracted caption or a placeholder.
    """
    vision_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    vision_llm = ChatOpenAI(
        model=vision_model,
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    msg = HumanMessage(
        content=[
            {
                "type": "text",
                "text": (
                    "Describe this image in detail for document search indexing. "
                    "Include chart titles, axis labels, legend entries, key data "
                    "points, trends, numbers, and any visible text. Be specific."
                ),
            },
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{img_b64}"},
            },
        ]
    )
    try:
        response = vision_llm.invoke([msg])
        content = response.content
        if isinstance(content, list):
            return " ".join(
                p.get("text", "")
                for p in content
                if isinstance(p, dict) and p.get("type") == "text"
            ).strip()
        return str(content).strip()
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# Docling label taxonomy (DocItemLabel enum values we care about):
#
#   section_header  — numbered or unnumbered section headings
#   title           — document-level title
#   text / paragraph— body paragraphs
#   list_item       — bullet / numbered list items
#   caption         — figure / table captions (emitted as separate nodes)
#   footnote        — footnotes at the bottom of a page
#   table           — tabular data (Docling reconstructs cell structure)
#   picture         — embedded raster / vector images
#   chart           — chart/graph images (rendered image, no raw data)
#   page_header     — running header printed on every page  ← NOISE, skipped
#   page_footer     — running footer printed on every page  ← NOISE, skipped
# ---------------------------------------------------------------------------


def parse_document(file_path: str) -> list[dict]:
    """Parse a PDF into a flat list of typed content chunks using Docling.

    Each chunk is a dict with three keys:
      content      — text or markdown representation of the element
      content_type — one of: "text", "table", "image"
      metadata     — dict with: content_type, element_type, section,
                     page_number, source_file, image_base64

    The metadata is passed to PGVector, so every
    retrieved chunk tells the query layer what kind of content it is
    and where in the document it came from.
    """

    # ── Step 1: Configure Docling pipeline ───────────────────────────────────
    # do_ocr=True          — run OCR on scanned/rasterised pages so text is
    #                        extractable even when not embedded in the PDF
    # do_table_structure   — detect table grid lines and reconstruct rows/cols
    # generate_picture_images — render each picture element to a PIL Image so
    #                           we can base64-encode it for storage
    #
    # accelerator_options — pin inference to CPU. On Apple Silicon the default
    #   (AUTO) selects the MPS/Metal backend, but Docling's layout model runs in
    #   float64 and MPS rejects float64 ("Cannot convert a MPS Tensor to float64
    #   dtype"), crashing the layout stage. CPU supports float64, so forcing it
    #   here keeps ingestion working on Macs. CUDA/CPU machines are unaffected.
    pipeline_options = PdfPipelineOptions(
        do_ocr=True,
        do_table_structure=True,
        generate_picture_images=True,
        accelerator_options=AcceleratorOptions(device=AcceleratorDevice.CPU),
    )

    # Disable torch.compile for Docling's layout model.
    # Avoids MSVC cl.exe requirement on Windows.
    pipeline_options.layout_options.engine_options.compile_model = False

    converter = DocumentConverter(
        allowed_formats=[InputFormat.PDF],
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        },
    )

    # ── Step 2: Convert the PDF ───────────────────────────────────────────────
    # converter.convert() runs the full Docling pipeline:
    #   layout analysis → OCR → table structure → picture rendering
    # result.document is a DoclingDocument with a typed element tree.
    result = converter.convert(file_path)
    doc = result.document

    parsed_chunks: list[dict] = []
    # Tracks the most recently seen section heading so every chunk carries
    # the section name it belongs to — useful for filtered retrieval.
    current_section: str | None = None
    source_file = os.path.basename(file_path)

    # ── Step 3: Walk the document element tree ────────────────────────────────
    # iterate_items() yields (level, node) tuples in Docling >= 2.x.
    # level is the heading depth (1 = top-level); node is the DocItem.
    for item in doc.iterate_items():
        if isinstance(item, tuple):
            node, _ = item  # iterate_items() yields (node, level); discard level
        else:
            node = item  # older Docling versions yield bare nodes

        # label is a DocItemLabel enum value — convert to lowercase string
        # for pattern matching (e.g. "section_header", "table", "picture")
        label = str(getattr(node, "label", "")).lower()

        # ── Skip page headers/footers ─────────────────────────────────────────
        # These repeat on every page (document title, page number, date stamp)
        # and would pollute retrieval results with irrelevant noise.
        if label in ("page_header", "page_footer"):
            continue

        # ── Extract page number and bounding box from provenance ──────────────
        # prov is a list of ProvenanceItem; prov[0] covers the first (usually
        # only) occurrence of the element. bbox gives the element's position
        # on the page as left/top/right/bottom coordinates (0–1 normalised or
        # absolute, depending on Docling version).
        prov = getattr(node, "prov", None)
        page_no = prov[0].page_no if prov else None
        position: dict | None = None
        if prov and hasattr(prov[0], "bbox") and prov[0].bbox is not None:
            b = prov[0].bbox
            position = {"l": b.l, "t": b.t, "r": b.r, "b": b.b}

        def _make_metadata(content_type: str, element_type: str, img_b64=None):
            """Build a metadata dict that is stored alongside every chunk.

            content_type  — "text" | "table" | "image"  (used by the query
                            layer to decide how to render retrieved content)
            element_type  — raw Docling label ("section_header", "table", …)
            img_b64       — base64-encoded PNG string for image elements;
                            None for text and table elements
            """
            return {
                "content_type": content_type,
                "element_type": element_type,
                "section": current_section,
                "page_number": page_no,
                "source_file": source_file,
                "position": position,  # bounding box stored in JSONB position column
                "image_base64": img_b64,  # decoded to BYTEA by db.store_chunks()
            }

        # ── Section headings & document title ─────────────────────────────────
        # Update current_section so all subsequent chunks carry the correct
        # section name until the next heading is encountered.
        if "section_header" in label or label == "title":
            text = getattr(node, "text", "").strip()
            if text:
                current_section = text
                parsed_chunks.append(
                    {
                        "content": text,
                        "content_type": "text",
                        "metadata": _make_metadata("text", label),
                    }
                )

        # ── Tables ────────────────────────────────────────────────────────────
        # Convert table cells to clean "Header: value" plain text rows so
        # that no markdown pipe/dash symbols pollute the vector store.
        # Strategy:
        #   1. export_to_dataframe() — preferred; yields a pandas DataFrame
        #      with header row and typed cell values from Docling's grid.
        #   2. Fallback: export_to_html() stripped of tags, then plain text.
        # Each table row is serialised as "Col1: val1 | Col2: val2" so the
        # column context travels with every value and embeddings are meaningful.
        elif "table" in label:
            table_text = ""
            if hasattr(node, "export_to_dataframe"):
                try:
                    df = node.export_to_dataframe(doc=doc)
                    if df is not None and not df.empty:
                        rows_text: list[str] = []
                        headers = [str(c).strip() for c in df.columns]
                        for _, row in df.iterrows():
                            pairs = [
                                f"{h}: {str(v).strip()}"
                                for h, v in zip(headers, row)
                                if str(v).strip() not in ("", "nan", "None")
                            ]
                            if pairs:
                                rows_text.append("  |  ".join(pairs))
                        table_text = "\n".join(rows_text)
                except Exception:
                    pass

            # Fallback: strip HTML tags from export_to_html()
            if not table_text and hasattr(node, "export_to_html"):
                try:
                    import re as _re

                    raw_html = node.export_to_html(doc)
                    table_text = _re.sub(r"<[^>]+>", " ", raw_html or "")
                    table_text = _re.sub(r"\s+", " ", table_text).strip()
                except Exception:
                    pass

            # Last resort: raw text attribute
            if not table_text:
                table_text = getattr(node, "text", "")

            if table_text and table_text.strip():
                parsed_chunks.append(
                    {
                        "content": table_text.strip(),
                        "content_type": "table",
                        "metadata": _make_metadata("table", "table"),
                    }
                )

        # ── Pictures, figures, and charts ─────────────────────────────────────
        # Charts are rendered images in Docling (no structured data is
        # extracted), so they are handled identically to pictures.
        # Extraction strategy:
        #   1. get_image(doc) — preferred; uses pre-rendered PIL Images
        #      produced when generate_picture_images=True
        #   2. .image.pil_image — fallback attribute on some Docling versions
        # The PIL Image is encoded as a base64 PNG and stored in metadata so
        # the OpenAI vision model can receive it directly during generation.
        elif "picture" in label or "figure" in label or label == "chart":
            img_b64 = None
            # .text on a PictureItem is the inline caption, if any
            caption = getattr(node, "text", "") or ""

            try:
                if hasattr(node, "get_image"):
                    pil_img = node.get_image(doc)
                    if pil_img:
                        buf = io.BytesIO()
                        pil_img.save(buf, format="PNG")
                        img_b64 = base64.b64encode(buf.getvalue()).decode()

                # Fallback path for older Docling versions
                if img_b64 is None and hasattr(node, "image") and node.image:
                    pil_img = getattr(node.image, "pil_image", None)
                    if pil_img:
                        buf = io.BytesIO()
                        pil_img.save(buf, format="PNG")
                        img_b64 = base64.b64encode(buf.getvalue()).decode()
            except Exception:
                # Image extraction is best-effort; a missing image is not
                # fatal — the caption / placeholder text is still indexed.
                pass

            # Use an OpenAI vision model to generate a rich description for this
            # image. This becomes the chunk's searchable text content — far more
            # useful than a sparse caption like "Figure 3" for embedding and
            # retrieval. Falls back to Docling caption → placeholder on failure.
            if img_b64:
                description = _describe_image_with_openai(img_b64)
                content = description or caption.strip() or f"[Image on page {page_no}]"
            else:
                content = caption.strip() or f"[Image on page {page_no}]"
            parsed_chunks.append(
                {
                    "content": content,
                    "content_type": "image",
                    "metadata": _make_metadata("image", "picture", img_b64),
                }
            )

        # ── Plain text: paragraphs, list items, captions, footnotes, etc. ─────
        # Everything that is not a heading, table, or image falls here.
        # Empty nodes (layout artefacts with no text) are silently dropped.
        else:
            text = getattr(node, "text", "")
            if text and text.strip():
                parsed_chunks.append(
                    {
                        "content": text.strip(),
                        "content_type": "text",
                        "metadata": _make_metadata("text", label),
                    }
                )

    return parsed_chunks
