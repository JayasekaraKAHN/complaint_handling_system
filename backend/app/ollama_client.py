import subprocess
import json

def query_ollama(prompt: str, model: str = "llama3.2:1b") -> str:
    """Send prompt to Ollama LLaMA model and return response."""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt.encode("utf-8"),
            capture_output=True,
            check=True
        )
        return result.stdout.decode("utf-8").strip()
    except Exception as e:
        return f"Error: {e}"
