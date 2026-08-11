import base64
import io
import os
import re

from dotenv import load_dotenv

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    AcceleratorDevice,
    AcceleratorOptions,
    PdfPipelineOptions,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

load_dotenv()


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class DocumentParserError(Exception):
    """
    Base exception for document parsing failures.
    """

    pass


class DocumentConversionError(DocumentParserError):
    """
    Raised when Docling cannot convert the source document.
    """

    pass


# ---------------------------------------------------------------------------
# Vision model
# ---------------------------------------------------------------------------


def _describe_image_with_openai(img_b64: str) -> str:
    """
    Generate a searchable description for an extracted image.

    Image description is best-effort.

    If the vision model fails for any reason, return an empty string so
    ingestion can continue using the Docling caption or a placeholder.

    This function intentionally does NOT fail the entire PDF ingestion.
    """

    if not img_b64:
        return ""

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        print(
            "[docling_parser] OPENAI_API_KEY is not configured. "
            "Skipping image description."
        )
        return ""

    vision_model = os.getenv(
        "OPENAI_CHAT_MODEL",
        "gpt-4o-mini",
    )

    try:
        vision_llm = ChatOpenAI(
            model=vision_model,
            api_key=api_key,
        )

        msg = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": (
                        "Describe this image in detail for document "
                        "search indexing. Include chart titles, axis "
                        "labels, legend entries, key data points, "
                        "trends, numbers, and visible text. "
                        "Be specific and factual."
                    ),
                },
                {
                    "type": "image_url",
                    "image_url": {"url": ("data:image/png;base64," f"{img_b64}")},
                },
            ]
        )

        response = vision_llm.invoke([msg])

        content = response.content

        if isinstance(content, list):
            text_parts = []

            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text = part.get("text", "")

                    if text:
                        text_parts.append(text)

            return " ".join(text_parts).strip()

        return str(content).strip()

    except Exception as exc:
        print("[docling_parser] Vision description failed: " f"{exc}")

        # Vision enrichment is optional.
        return ""


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _build_metadata(
    *,
    content_type: str,
    element_type: str,
    section: str | None,
    page_number: int | None,
    source_file: str,
    position: dict | None,
    image_base64: str | None = None,
) -> dict:
    """
    Build consistent metadata for every parsed element.
    """

    return {
        "content_type": content_type,
        "element_type": element_type,
        "section": section,
        "page_number": page_number,
        "source_file": source_file,
        "position": position,
        "image_base64": image_base64,
    }


# ---------------------------------------------------------------------------
# Page / position extraction
# ---------------------------------------------------------------------------


def _get_provenance(node):
    """
    Safely retrieve the first provenance record from a Docling node.
    """

    prov = getattr(node, "prov", None)

    if not prov:
        return None

    try:
        return prov[0]
    except (IndexError, TypeError):
        return None


def _extract_page_number(node) -> int | None:
    """
    Extract the page number from a Docling node.
    """

    provenance = _get_provenance(node)

    if provenance is None:
        return None

    try:
        return provenance.page_no
    except AttributeError:
        return None


def _extract_position(node) -> dict | None:
    """
    Extract bounding-box coordinates from a Docling node.
    """

    provenance = _get_provenance(node)

    if provenance is None:
        return None

    bbox = getattr(provenance, "bbox", None)

    if bbox is None:
        return None

    try:
        return {
            "l": bbox.l,
            "t": bbox.t,
            "r": bbox.r,
            "b": bbox.b,
        }

    except AttributeError:
        return None


# ---------------------------------------------------------------------------
# Table extraction
# ---------------------------------------------------------------------------


def _extract_table_text(node, doc) -> str:
    """
    Convert a Docling table into searchable plain text.

    Extraction strategy:

        1. DataFrame
        2. HTML fallback
        3. Raw text fallback

    Returns an empty string if no usable table content can be extracted.
    """

    # -----------------------------------------------------------------------
    # Strategy 1: DataFrame
    # -----------------------------------------------------------------------

    if hasattr(node, "export_to_dataframe"):

        try:
            dataframe = node.export_to_dataframe(doc=doc)

            if dataframe is not None and not dataframe.empty:

                rows_text: list[str] = []

                headers = [str(column).strip() for column in dataframe.columns]

                for _, row in dataframe.iterrows():

                    pairs = []

                    for header, value in zip(headers, row):

                        value_text = str(value).strip()

                        if value_text in (
                            "",
                            "nan",
                            "None",
                        ):
                            continue

                        pairs.append(f"{header}: {value_text}")

                    if pairs:
                        rows_text.append(" | ".join(pairs))

                if rows_text:
                    return "\n".join(rows_text).strip()

        except Exception as exc:
            print("[docling_parser] DataFrame table extraction " f"failed: {exc}")

    # -----------------------------------------------------------------------
    # Strategy 2: HTML fallback
    # -----------------------------------------------------------------------

    if hasattr(node, "export_to_html"):

        try:
            raw_html = node.export_to_html(doc)

            if raw_html:

                table_text = re.sub(
                    r"<[^>]+>",
                    " ",
                    raw_html,
                )

                table_text = re.sub(
                    r"\s+",
                    " ",
                    table_text,
                ).strip()

                if table_text:
                    return table_text

        except Exception as exc:
            print("[docling_parser] HTML table extraction " f"failed: {exc}")

    # -----------------------------------------------------------------------
    # Strategy 3: Raw node text
    # -----------------------------------------------------------------------

    try:
        raw_text = getattr(node, "text", "") or ""

        if raw_text.strip():
            return raw_text.strip()

    except Exception as exc:
        print("[docling_parser] Raw table text extraction " f"failed: {exc}")

    return ""


# ---------------------------------------------------------------------------
# Image extraction
# ---------------------------------------------------------------------------


def _extract_image_base64(node, doc) -> str | None:
    """
    Extract a Docling image as base64 PNG.

    Returns None when image extraction is unavailable.
    """

    # -----------------------------------------------------------------------
    # Preferred Docling API
    # -----------------------------------------------------------------------

    try:

        if hasattr(node, "get_image"):

            pil_image = node.get_image(doc)

            if pil_image:

                buffer = io.BytesIO()

                pil_image.save(
                    buffer,
                    format="PNG",
                )

                return base64.b64encode(buffer.getvalue()).decode()

    except Exception as exc:
        print("[docling_parser] Primary image extraction " f"failed: {exc}")

    # -----------------------------------------------------------------------
    # Fallback for older Docling versions
    # -----------------------------------------------------------------------

    try:

        image = getattr(node, "image", None)

        if image:

            pil_image = getattr(
                image,
                "pil_image",
                None,
            )

            if pil_image:

                buffer = io.BytesIO()

                pil_image.save(
                    buffer,
                    format="PNG",
                )

                return base64.b64encode(buffer.getvalue()).decode()

    except Exception as exc:
        print("[docling_parser] Fallback image extraction " f"failed: {exc}")

    return None


# ---------------------------------------------------------------------------
# Parse document
# ---------------------------------------------------------------------------


def parse_document(file_path: str) -> list[dict]:
    """
    Parse a PDF into typed content elements using Docling.

    Returns a list of dictionaries containing:

        content
        content_type
        metadata

    Content types:

        text
        table
        image

    Important error-handling behaviour:

        - Document conversion failure -> ingestion failure.
        - Individual table extraction failure -> fallback / skip.
        - Individual image extraction failure -> continue.
        - Vision model failure -> continue.
        - Empty elements -> ignored.
    """

    if not file_path:
        raise DocumentParserError("A document path is required.")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Document not found: {file_path}")

    if not os.path.isfile(file_path):
        raise DocumentParserError(f"Document path is not a file: {file_path}")

    if os.path.getsize(file_path) == 0:
        raise DocumentParserError(f"Document is empty: {file_path}")

    source_file = os.path.basename(file_path)

    print(f"[docling_parser] Starting parsing: " f"{source_file}")

    # -----------------------------------------------------------------------
    # Step 1: Configure Docling
    # -----------------------------------------------------------------------

    try:

        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            do_table_structure=True,
            generate_picture_images=True,
            accelerator_options=AcceleratorOptions(
                device=AcceleratorDevice.CPU,
            ),
        )

        # Disable torch.compile.
        #
        # This avoids the MSVC cl.exe requirement on Windows and preserves
        # the existing behaviour of your current parser.
        pipeline_options.layout_options.engine_options.compile_model = False

        converter = DocumentConverter(
            allowed_formats=[
                InputFormat.PDF,
            ],
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
            },
        )

    except Exception as exc:

        print("[docling_parser] Failed to initialize " f"Docling: {exc}")

        raise DocumentConversionError(
            "Unable to initialize the document parser."
        ) from exc

    # -----------------------------------------------------------------------
    # Step 2: Convert document
    # -----------------------------------------------------------------------

    try:

        print(f"[docling_parser] Converting: " f"{source_file}")

        result = converter.convert(file_path)

        if result is None:
            raise DocumentConversionError("Docling returned no conversion result.")

        doc = getattr(result, "document", None)

        if doc is None:
            raise DocumentConversionError("Docling conversion returned no document.")

    except DocumentConversionError:
        raise

    except Exception as exc:

        print("[docling_parser] Document conversion failed: " f"{exc}")

        raise DocumentConversionError(
            f"Unable to parse document '{source_file}'."
        ) from exc

    # -----------------------------------------------------------------------
    # Step 3: Walk document elements
    # -----------------------------------------------------------------------

    parsed_chunks: list[dict] = []

    current_section: str | None = None

    try:
        items = doc.iterate_items()

    except Exception as exc:

        print("[docling_parser] Unable to iterate document " f"elements: {exc}")

        raise DocumentConversionError(
            "Unable to read parsed document elements."
        ) from exc

    try:

        for item_index, item in enumerate(items):

            # Docling versions differ in what iterate_items() returns.
            #
            # Current versions return:
            #
            #     (node, level)
            #
            # Older versions may return:
            #
            #     node

            if isinstance(item, tuple):

                if not item:
                    continue

                node = item[0]

            else:
                node = item

            if node is None:
                continue

            try:
                label = str(
                    getattr(
                        node,
                        "label",
                        "",
                    )
                ).lower()

                page_number = _extract_page_number(node)

                position = _extract_position(node)

            except Exception as exc:

                print(
                    f"[docling_parser] Failed to read "
                    f"metadata for element {item_index}: "
                    f"{exc}"
                )

                continue

            # ----------------------------------------------------------------
            # Skip repeated page headers/footers
            # ----------------------------------------------------------------

            if label in (
                "page_header",
                "page_footer",
            ):
                continue

            # ----------------------------------------------------------------
            # Section headings / document title
            # ----------------------------------------------------------------

            if "section_header" in label or label == "title":

                try:

                    text = (
                        getattr(
                            node,
                            "text",
                            "",
                        )
                        or ""
                    ).strip()

                    if not text:
                        continue

                    current_section = text

                    parsed_chunks.append(
                        {
                            "content": text,
                            "content_type": "text",
                            "metadata": _build_metadata(
                                content_type="text",
                                element_type=label,
                                section=current_section,
                                page_number=page_number,
                                source_file=source_file,
                                position=position,
                            ),
                        }
                    )

                except Exception as exc:

                    print(
                        f"[docling_parser] Failed to process "
                        f"heading element {item_index}: "
                        f"{exc}"
                    )

                continue

            # ----------------------------------------------------------------
            # Tables
            # ----------------------------------------------------------------

            if "table" in label:

                try:

                    table_text = _extract_table_text(
                        node,
                        doc,
                    )

                    if not table_text:
                        print(
                            f"[docling_parser] Skipping empty "
                            f"table at element {item_index}."
                        )
                        continue

                    parsed_chunks.append(
                        {
                            "content": table_text,
                            "content_type": "table",
                            "metadata": _build_metadata(
                                content_type="table",
                                element_type="table",
                                section=current_section,
                                page_number=page_number,
                                source_file=source_file,
                                position=position,
                            ),
                        }
                    )

                except Exception as exc:

                    print(
                        f"[docling_parser] Failed to process "
                        f"table element {item_index}: {exc}"
                    )

                continue

            # ----------------------------------------------------------------
            # Pictures / figures / charts
            # ----------------------------------------------------------------

            if "picture" in label or "figure" in label or label == "chart":

                try:

                    caption = (
                        getattr(
                            node,
                            "text",
                            "",
                        )
                        or ""
                    ).strip()

                    img_b64 = _extract_image_base64(
                        node,
                        doc,
                    )

                    # --------------------------------------------------------
                    # Vision enrichment
                    # --------------------------------------------------------

                    if img_b64:

                        description = _describe_image_with_openai(img_b64)

                        content = (
                            description or caption or f"[Image on page {page_number}]"
                        )

                    else:

                        content = caption or f"[Image on page {page_number}]"

                    parsed_chunks.append(
                        {
                            "content": content,
                            "content_type": "image",
                            "metadata": _build_metadata(
                                content_type="image",
                                element_type="picture",
                                section=current_section,
                                page_number=page_number,
                                source_file=source_file,
                                position=position,
                                image_base64=img_b64,
                            ),
                        }
                    )

                except Exception as exc:

                    # --------------------------------------------------------
                    # Image processing is intentionally best-effort.
                    # --------------------------------------------------------

                    print(
                        f"[docling_parser] Failed to process "
                        f"image element {item_index}: {exc}"
                    )

                continue

            # ----------------------------------------------------------------
            # Plain text
            #
            # Paragraphs, list items, captions, footnotes, etc.
            # ----------------------------------------------------------------

            try:

                text = (
                    getattr(
                        node,
                        "text",
                        "",
                    )
                    or ""
                ).strip()

                if not text:
                    continue

                parsed_chunks.append(
                    {
                        "content": text,
                        "content_type": "text",
                        "metadata": _build_metadata(
                            content_type="text",
                            element_type=label,
                            section=current_section,
                            page_number=page_number,
                            source_file=source_file,
                            position=position,
                        ),
                    }
                )

            except Exception as exc:

                print(
                    f"[docling_parser] Failed to process "
                    f"text element {item_index}: {exc}"
                )

    except Exception as exc:

        # This means the overall iteration itself failed rather than
        # an individual element failing.
        print("[docling_parser] Failed while iterating " f"document elements: {exc}")

        raise DocumentConversionError(
            f"Unable to process elements from '{source_file}'."
        ) from exc

    # -----------------------------------------------------------------------
    # Final validation
    # -----------------------------------------------------------------------

    if not parsed_chunks:

        raise DocumentConversionError(
            f"No usable content was extracted from " f"'{source_file}'."
        )

    print(
        f"[docling_parser] Successfully extracted "
        f"{len(parsed_chunks)} elements from "
        f"{source_file}"
    )

    return parsed_chunks
