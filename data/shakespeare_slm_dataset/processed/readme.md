# Data Processing Documentation

This document outlines the technical breakdown of the data processing pipeline designed for the Shakespeare dataset. 

The processed directory contains two types of datasets to support different training and retrieval strategies.

---

## 1. Dataset Formats & Use Cases

### A. Scene-Level Datasets (`*_cleaned.jsonl`)
*   **Granularity**: One record per complete scene.
*   **Characteristics**: Provides the full narrative arc of a scene in a single block.
*   **Ideal Use Case**: 
    *   **Long-context training**: Teaching models to understand entire plot movements.
    *   **Summarization tasks**: Generating or verifying scene summaries.
    *   **Baseline RAG**: When you want to retrieve the entire scene context for a question.

### B. Chunk-Level Datasets (`*_chunks.jsonl`)
*   **Granularity**: Smaller segments of 12 utterances with a 3-utterance overlap.
*   **Characteristics**: Uses a **Sliding Window** approach to maintain local conversation flow while reducing memory footprint.
*   **Ideal Use Case**:
    *   **Fine-tuning (SFT)**: Teaching models to mimic specific character dialogue styles in short exchanges.
    *   **Precision RAG**: Retrieving only the most relevant snippets of a conversation to save tokens and improve answer focus.
    *   **Sliding Window Context**: Ensuring that even if a key event happens at a boundary, it is captured in at least one (and often two) chunks.

---

## 2. Processing Methodology

### A. Corpus Acquisition & Chunking
Instead of treating every line in isolation, we use two aggregation strategies:
1.  **Scene Aggregation**: Grouping by `scene_id`.
2.  **Utterance Batching**: Using a sliding window (`chunk_size=12`, `overlap=3`) to create overlapping dialogue segments. This ensures that every dialogue turn has sufficient surrounding context.

### B. Data Cleaning
We applied several transformation layers using Regular Expressions (Regex):
*   **Whitespace Normalization**: Removed redundant tabs, newlines, and double spaces.
*   **Punctuation Correction**: Fixed "Gutenberg-style" spacing (e.g., removing spaces before commas or periods).
*   **Speaker Standardization**: Converted speaker names from `ALL CAPS` to `Title Case` (e.g., "MACBETH" -> "Macbeth") for better alignment with modern LLM training distributions.
*   **Semantic Tagging**: Stage directions are wrapped in square brackets `[]` to distinguish action from dialogue.

### C. Metadata Structure Design
Every record (Scene or Chunk) is self-describing via a "Flat Context" schema:
*   **Structural Metadata**: `play_title`, `act`, and `scene`.
*   **Semantic Metadata**: `scene_summary` and `keywords` (inherited from the source scene).
*   **Dynamic Character Mapping**: 
    *   In `_cleaned` files: Lists all characters in the scene.
    *   In `_chunks` files: Lists **only** the characters who actually speak within that specific window (`characters_in_chunk`).

---

## 3. Why This Approach?
*   **JSONL Format**: Standard format for stream-loading during training (compatible with HuggingFace `datasets`).
*   **Stateless Records**: Every chunk contains enough metadata to be understood in isolation, allowing for randomized shuffling during the training process without losing track of the source material.

---

## 4. Characteristics of the Processed Dataset
*   **High Signal-to-Noise Ratio**: Project Gutenberg boilerplate and OCR artifacts have been stripped.
*   **Richly Annotated**: Beyond raw text, the metadata enables complex filtering (e.g., "Train only on Macbeth's scenes in Act 1").
*   **Contextually Aware**: Overlapping chunks prevent the "Context Clipping" problem where a conversation is cut off mid-sentence.
