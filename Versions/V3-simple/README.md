# Directory Code Analyzer

A Python-based tool that leverages GPT to analyze codebases and generate detailed insights and summaries.

## Features
- Fast async file processing
- Natural language file type filtering
- Customizable analysis parameters
- Per-file and codebase-wide analysis
- Markdown report generation
- Color-coded logging

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install openai pydantic colorlog aiofiles
```

## Configuration

Set the following environment variable:
```bash
export OPENAI_API_KEY=your_api_key_here
```

Key configuration variables in `codebase_analyzer.py`:
```python
OPENAI_MODEL = "gpt-4o-mini"
PROJECT_DIR = "path/to/your/project"
OUTPUT_FILE = "path/to/output/report.md"
PERFORM_FINAL_ANALYSIS = True
```

## Usage

Basic usage:
```python
from codebase_analyzer import SimpleCodeAnalyzer

analyzer = SimpleCodeAnalyzer(
    file_types="python and javascript files",
    analysis_prompt="""
    For each file:
    1. List all imports and external dependencies
    2. Summarize the main purpose of the file
    3. List key functions and classes
    4. Identify any potential issues or improvements
    """
)

analysis = analyzer.analyze_directory("path/to/project")
```

Run from command line:
```bash
python codebase_analyzer.py
```

## Customization

### Analysis Prompts

Modify `ANALYSIS_PROMPT` for per-file analysis:
```python
ANALYSIS_PROMPT = """
Your custom per-file analysis instructions here
"""
```

Modify `FINAL_ANALYSIS_PROMPT` for the final summary:
```python
FINAL_ANALYSIS_PROMPT = """
Your custom final analysis instructions here
"""
```

### File Types

Specify file types using natural language:
- "python and javascript files"
- "configuration files"
- "typescript, react files, and styles"

## Output

The tool generates a markdown report containing:
1. Project overview
2. Individual file analyses
3. Final codebase summary (if enabled)

## Error Handling

- Multiple encoding attempts for file reading
- Rate limiting for API calls
- Comprehensive error logging
- Fallback mechanisms for file type processing

## Requirements

- Python 3.7+
- OpenAI API key
- Required packages: openai, pydantic, colorlog, aiofiles

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
```

This README provides a comprehensive overview of the project, including installation instructions, configuration options, usage examples, and customization possibilities. You can adjust the content based on your specific needs or preferences.