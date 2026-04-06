from __future__ import annotations

import gradio as gr

from ragforge.demo import (
    DEMO_PIPELINE_LABELS,
    DEFAULT_HF_GENERATION_MODEL,
    build_run_summary,
    format_documents_table,
    format_results_table,
    format_trace_payload,
    run_demo_query,
)

PIPELINE_LABEL_TO_KEY = {label: key for key, label in DEMO_PIPELINE_LABELS.items()}
GENERATION_LABEL_TO_KEY = {
    "Fallback Extractive": "fallback",
    "Hugging Face LLM": "huggingface",
}

DOCUMENT_HEADERS = ["Filename", "Type", "Document ID", "Characters", "Pages"]
RESULT_HEADERS = ["Rank", "Filename", "Document ID", "Chunk ID", "Retriever", "Score", "Preview"]


def _empty_outputs(message: str) -> tuple[str, str, list[list[object]], list[list[object]], str, dict[str, object]]:
    """Return a consistent empty UI state with an explanatory message."""
    return (
        f"### Demo Status\n{message}",
        "No answer generated.",
        [],
        [],
        "",
        {},
    )


def run_demo(
    uploaded_files: list[str] | None,
    query: str,
    pipeline_label: str,
    generation_label: str,
    top_k: int,
    chunk_size: int,
    overlap: int,
) -> tuple[str, str, list[list[object]], list[list[object]], str, dict[str, object]]:
    """Bridge Gradio inputs to the ragforge demo service."""
    try:
        result = run_demo_query(
            uploaded_files=uploaded_files,
            query=query,
            pipeline_key=PIPELINE_LABEL_TO_KEY[pipeline_label],
            generation_mode=GENERATION_LABEL_TO_KEY[generation_label],
            top_k=top_k,
            chunk_size=chunk_size,
            overlap=overlap,
        )
    except Exception as exc:
        return _empty_outputs(f"Error: `{exc}`")

    return (
        build_run_summary(result),
        result.generation.answer,
        format_documents_table(result.documents),
        format_results_table(result.generation.retrieval_results),
        result.generation.context.prompt_context,
        format_trace_payload(result.generation.trace),
    )


with gr.Blocks(title="Hybrid RAG Lab Demo 01") as demo:
    gr.Markdown(
        """
        # Hybrid RAG Lab Demo 01

        Upload a small `.txt`, `.pdf`, or `.docx` document, choose a retrieval technique, ask a query,
        and inspect both the ranked evidence and the generated grounded answer.

        Demo 1 is intentionally lightweight:
        - best with 1 short document
        - supports up to 3 files
        - retrieval works without external keys
        - generation can use the local fallback mode or a Hugging Face LLM if `HF_TOKEN` is configured
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            uploaded_files = gr.File(
                label="Upload Documents",
                file_count="multiple",
                file_types=[".txt", ".pdf", ".docx"],
                type="filepath",
            )
            query = gr.Textbox(
                label="Query",
                placeholder="Ask a question about the uploaded document...",
                lines=3,
            )
            pipeline_label = gr.Radio(
                choices=list(PIPELINE_LABEL_TO_KEY.keys()),
                value="Hybrid",
                label="Retrieval Technique",
            )
            generation_label = gr.Radio(
                choices=list(GENERATION_LABEL_TO_KEY.keys()),
                value="Fallback Extractive",
                label="Generation Mode",
                info=f"Hugging Face mode uses `{DEFAULT_HF_GENERATION_MODEL}` and requires HF_TOKEN.",
            )
            top_k = gr.Slider(minimum=1, maximum=5, value=3, step=1, label="Top K Results")

            with gr.Accordion("Advanced Retrieval Settings", open=False):
                chunk_size = gr.Slider(minimum=150, maximum=500, value=300, step=25, label="Chunk Size")
                overlap = gr.Slider(minimum=0, maximum=100, value=50, step=10, label="Chunk Overlap")

            run_button = gr.Button("Run Demo", variant="primary")
            gr.ClearButton([uploaded_files, query, pipeline_label, generation_label, top_k, chunk_size, overlap])

        with gr.Column(scale=1):
            summary = gr.Markdown()
            answer = gr.Markdown()

    loaded_documents = gr.Dataframe(
        headers=DOCUMENT_HEADERS,
        datatype=["str", "str", "str", "number", "str"],
        label="Loaded Documents",
        wrap=True,
    )
    ranked_results = gr.Dataframe(
        headers=RESULT_HEADERS,
        datatype=["number", "str", "str", "str", "str", "number", "str"],
        label="Ranked Retrieval Results",
        wrap=True,
    )

    with gr.Accordion("Constructed Context", open=False):
        context_block = gr.Textbox(label="Prompt Context", lines=14)

    with gr.Accordion("Pipeline Trace", open=False):
        trace_payload = gr.JSON(label="Latency and Metadata")

    run_button.click(
        fn=run_demo,
        inputs=[uploaded_files, query, pipeline_label, generation_label, top_k, chunk_size, overlap],
        outputs=[summary, answer, loaded_documents, ranked_results, context_block, trace_payload],
    )

demo.queue(default_concurrency_limit=1)


if __name__ == "__main__":
    demo.launch()
