# Shakespeare Knowledge Base

This directory contains two JSON Lines (`.jsonl`) files designed to provide beginner-friendly, plain-English summaries for the Retrieval-Augmented Generation (RAG) system.

## Files Overview

### 1. `shakespeare_knowledge_base.jsonl`
This is the comprehensive knowledge base file. It contains short, easy-to-understand summaries for:
- **Each Play:** High-level overviews of the main plot.
- **Each Character:** Brief descriptions of key characters, their motivations, and their roles in the story.
- **Every Scene of Each Play:** Scene-by-scene breakdowns detailing the main events and their significance to the overall plot.

### 2. `scene_summaries_only.jsonl`
This file is a more focused dataset. It exclusively contains the plain-English, scene-by-scene summaries for every act and scene of each play. It does not include the overarching play or character summaries.

---
*Note: Both datasets are formatted as strict JSONL (JSON Lines), where every single line is a valid, standalone JSON object, making them ideal for line-by-line parsing and ingestion.*

_Data source: Wikipedia & Gemini_