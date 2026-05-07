import json
import re
from pathlib import Path
from typing import List, Dict, Any

class ShakespeareProcessor:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.output_dir = self.data_dir / "processed"
        self.output_dir.mkdir(exist_ok=True)
        
    def clean_text(self, text: str) -> str:
        """
        Basic text cleaning: remove redundant whitespace, fix punctuation spacing, etc.
        """
        if not text:
            return ""
        # Remove extra whitespace characters
        text = re.sub(r'\s+', ' ', text).strip()
        # Fix common punctuation spacing issues (e.g., those common in Gutenberg texts)
        text = re.sub(r'\s+([,.?!;:])', r'\1', text)
        return text

    def format_utterance(self, utterance: Dict[str, Any]) -> str:
        """
        Format a single utterance into 'Speaker: Text' and handle stage directions.
        """
        speaker = utterance.get("speaker", "UNKNOWN")
        text = self.clean_text(utterance.get("text", ""))
        
        if speaker == "STAGE_DIRECTION":
            # Stage directions are typically enclosed in square brackets
            return f"[{text}]"
        else:
            # Standardize speaker name (Title Case)
            speaker_fmt = speaker.title()
            return f"{speaker_fmt}: {text}"

    def process_play(self, play_name: str, chunk_size: int = 10, overlap: int = 2) -> List[Dict[str, Any]]:
        """
        Process the JSON file of a single play with utterance-based chunking.
        chunk_size: Number of utterance turns per chunk.
        overlap: Number of overlapping utterance turns between adjacent chunks (maintains context).
        """
        file_path = self.data_dir / f"{play_name}.json"
        if not file_path.exists():
            print(f"Warning: {file_path} not found.")
            return []

        print(f"Processing {play_name} with chunk_size={chunk_size}, overlap={overlap}...")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        processed_chunks = []
        play_metadata = data.get("metadata", {})

        for scene in data.get("scenes", []):
            # Corpus Acquisition: Extract and format all utterances in the scene
            utterances = scene.get("utterances", [])
            formatted_texts = [self.format_utterance(u) for u in utterances]
            
            # Sliding window chunking
            step = chunk_size - overlap
            if step <= 0:
                step = 1 # Fallback to prevent infinite loop
                
            for i in range(0, len(formatted_texts), step):
                chunk_texts = formatted_texts[i:i + chunk_size]
                if not chunk_texts:
                    break
                
                chunk_content = "\n".join(chunk_texts)
                
                # Dynamically extract characters present in this specific chunk
                # We look for 'Name: ' at the start of lines, excluding stage directions starting with '['
                current_chunk_characters = sorted(list(set(
                    re.match(r"^([^:]+):", text).group(1)
                    for text in chunk_texts if ":" in text and not text.startswith("[")
                )))

                # Assign a unique ID for each chunk
                chunk_idx = i // step
                chunk_id = f"{play_name}_{scene.get('act')}_{scene.get('scene')}_chunk_{chunk_idx}"

                # Metadata Structure Design
                chunk_data = {
                    "chunk_id": chunk_id,
                    "content": chunk_content,
                    "metadata": {
                        "play_title": play_metadata.get("title", play_name.replace("_", " ").title()),
                        "act": scene.get("act"),
                        "scene": scene.get("scene"),
                        "location": scene.get("location"),
                        "scene_summary": scene.get("scene_summary"),
                        "characters_in_chunk": current_chunk_characters,
                        "keywords": scene.get("keywords", []),
                        "source": play_metadata.get("source", "Project Gutenberg")
                    }
                }
                processed_chunks.append(chunk_data)

        return processed_chunks

    def run(self, chunk_size: int = 10, overlap: int = 2):
        plays = ["hamlet", "macbeth", "romeo_and_juliet"]
        all_data = []

        for play in plays:
            chunks = self.process_play(play, chunk_size=chunk_size, overlap=overlap)
            all_data.extend(chunks)
            
            # Save a separate file for each play
            output_file = self.output_dir / f"{play}_chunks.jsonl"
            with open(output_file, "w", encoding="utf-8") as f:
                for chunk in chunks:
                    f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
            print(f"Saved {len(chunks)} chunks to {output_file}")

        # Save the combined dataset
        combined_file = self.output_dir / "shakespeare_all_chunks.jsonl"
        with open(combined_file, "w", encoding="utf-8") as f:
            for item in all_data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Total processed chunks: {len(all_data)}")
        print(f"Combined dataset saved to {combined_file}")

if __name__ == "__main__":
    # Assume the script is run from the project root directory
    processor = ShakespeareProcessor("shakespeare_slm_dataset")
    # Using the suggested chunking parameters
    processor.run(chunk_size=12, overlap=3)
