# TheChooser 🤖🧪

TheChooser is a "dumb robot" framework designed to identify which AI models in your 12GB VRAM "swarm" are actually capable of conducting science.

It "sips" recipes (prompts and workflows) from the AI Scientist pipeline but runs as a deterministic Python engine.

## Directory Structure
- `models/`: Drop your model JSON configs here.
- `tasks/`: Define your benchmark tasks in `tasks.json`.
- `results/`: Where the leaderboard and detailed logs are saved.
- `unsuitable/`: Models that fail health checks or interrogation are moved here.

## The Robot's "Recipe Book" (Extracted from AI Scientist)
- **Idea Generation:** Prompts that force creative constraints.
- **Peer Review:** A multi-dimensional scoring system (Originality, Quality, Clarity).
- **Code Repair:** The "Linear Fix" and "MCTS-lite" escalation strategies.

## CLI Tools the Robot Uses
To fully utilize these recipes, the following system tools are recommended:
- `pdflatex` & `bibtex`: For paper compilation.
- `chktex`: For LaTeX linting.
- `pymupdf4llm`: For converting PDFs to LLM-friendly Markdown.
