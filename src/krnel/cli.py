from krnel import graph
import warnings

try:
    from cyclopts import App
except ImportError:
    raise ImportError("You must install the 'cli' extra to use the CLI features of Krnel. Run: pip install krnel[cli]")

try:
    import krnel_private.implementations
except ImportError:
    warnings.warn("No private implementations for krnel functions found. Some features may be limited.")

app = App(
    name="krnel",
)

@app.default
def run():
    print("Welcome to Krnel! Use the CLI to run data processing and LLM operations.")