# Creating the AI Scientist Environment

Follow these steps to set up your AI Scientist environment:

1. Install Homebrew (if not already installed)
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```

2. Prepare a clean environment
   ```bash
   rm -rf ai_scientist
   ```Only and only if you have an existing environment. with the same name.
   otherwise dont do it.

3. Create and activate a new virtual environment
   ```bash
   uv venv ai_scientist
   source ai_scientist/bin/activate
   ```

4. Install required dependencies
   ```bash
   uv pip install -r requirements.txt
   ```

5. Set up LaTeX
   - For basic version:
     ```bash
     brew install texlive
     ```
   - For full version (requires more space):
     ```bash
     brew install --cask mactex
     ```

Note: The full MacTeX version offers more features but requires more storage space. Choose based on your needs and available disk space.

> 🔮 Pro Tip: Choose your LaTeX package wisely. The full MacTeX version is more powerful but requires more space in your digital lab.
