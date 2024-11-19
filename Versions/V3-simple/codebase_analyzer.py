import logging
from datetime import datetime
from typing import Optional, List
from pathlib import Path
from pydantic import BaseModel, field_validator
from openai import OpenAI
import colorlog
import asyncio
from asyncio import Semaphore
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import os

# Global Configuration
OPENAI_MODEL = "gpt-4o-mini"  # Change this to use a different model
PROJECT_DIR = r"C:\Users\wkraf\Documents\Coding\Directory_Summarizer\Versions\V3-simple\Sample_Directories\Local-File-Organizer"  # Change this to your project directory
OUTPUT_FILE = r"C:\Users\wkraf\Documents\Coding\Directory_Summarizer\Versions\V3-simple\Sample_Outputs\code_analysis_report.md"  # Change this to your desired output file
ANALYSIS_PROMPT = """
For each file:
1. List all imports and external dependencies
2. Summarize the main purpose of the file
3. List key functions and classes
4. Identify any potential issues or improvements
"""

# Configure logging with colors
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'red,bg_white',
    }
))

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set default logging level to INFO
logger.handlers = []  # Remove any existing handlers
logger.addHandler(handler)

class FileExtensions(BaseModel):
    extensions: List[str]

    @field_validator('extensions')  # Using the imported validator
    @classmethod  # Add @classmethod decorator
    def process_extensions(cls, extensions: List[str]) -> List[str]:
        processed = []
        for ext in extensions:
            # Strip any spaces and dots
            ext = ext.strip().strip('.')
            # Ensure extension starts with a dot
            if not ext.startswith('.'):
                ext = f'.{ext}'
            processed.append(ext.lower())
        return processed

    @classmethod
    def from_string(cls, extension_string: str, client: Optional[OpenAI] = None, model: str = "gpt-4o-mini") -> 'FileExtensions':
        """
        Create from a string description of file types, using GPT to interpret
        natural language descriptions when client is provided.
        """
        if client is None:
            # Fallback to simple comma splitting if no OpenAI client
            extensions = [ext.strip() for ext in extension_string.split(',')]
            extensions = [ext for ext in extensions if ext]
            return cls(extensions=extensions)

        try:
            # Use GPT to interpret the file types
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": """
                    Convert the user's description of file types into a list of file extensions.
                    Examples:
                    - "python and javascript files" -> [".py", ".js"]
                    - "typescript, react files, and styles" -> [".ts", ".tsx", ".jsx", ".css", ".scss"]
                    - "configuration files" -> [".json", ".yaml", ".yml", ".toml", ".ini"]
                    Return only the extensions, one per line.
                    """},
                    {"role": "user", "content": f"Convert this description to file extensions: {extension_string}"}
                ],
                temperature=0
            )

            # Process GPT response into list of extensions
            extensions = [
                ext.strip() 
                for ext in response.choices[0].message.content.split('\n')
                if ext.strip()
            ]
            logger.info(f"GPT interpreted '{extension_string}' as extensions: {extensions}")
            return cls(extensions=extensions)
            
        except Exception as e:
            logger.error(f"Error using GPT to process file types: {str(e)}")
            logger.info("Falling back to simple extension processing")
            return cls.from_string(extension_string, client=None)

class FileContent(BaseModel):
    path: str
    content: str

class AnalysisResult(BaseModel):
    path: str
    analysis: str

class DirectoryAnalysis(BaseModel):
    directory: str
    file_count: int
    results: List[AnalysisResult]
    timestamp: str
    file_types: List[str]

class RateLimiter:
    """Simple rate limiter for API calls"""
    def __init__(self, calls_per_second: float = 1.0):
        self.calls_per_second = calls_per_second
        self.semaphore = Semaphore(1)
        self.last_call = 0.0

    async def __aenter__(self):
        async with self.semaphore:
            now = asyncio.get_event_loop().time()
            time_passed = now - self.last_call
            if time_passed < 1.0 / self.calls_per_second:
                await asyncio.sleep(1.0 / self.calls_per_second - time_passed)
            self.last_call = asyncio.get_event_loop().time()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

class SimpleCodeAnalyzer:
    def __init__(
        self,
        file_types: str,
        analysis_prompt: str,
        api_key: Optional[str] = None,
        max_concurrent: int = 3,
        model: str = OPENAI_MODEL  # Use global model config
    ):
        # Use environment variable if api_key not provided
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.model = model
        # Now passing the OpenAI client and model to process file types
        self.file_extensions = FileExtensions.from_string(file_types, client=self.client, model=self.model)
        self.analysis_prompt = analysis_prompt
        self.max_concurrent = max_concurrent
        self.rate_limiter = RateLimiter()
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)

    async def read_file_content_async(self, file_path: Path) -> Optional[FileContent]:
        """Asynchronously read file content with multiple encoding attempts"""
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                    content = await f.read()
                    return FileContent(path=str(file_path), content=content)
            except UnicodeDecodeError:
                continue
        logger.error(f"Failed to read file: {file_path}")
        return None

    async def analyze_file_content_async(self, file_content: FileContent) -> AnalysisResult:
        """Analyze file content using GPT"""
        async with self.rate_limiter:
            try:
                # Enhanced system prompt
                system_prompt = f"""You are a code analysis expert. Analyze the provided file according to these instructions:
                {self.analysis_prompt}
                
                Focus on providing clear, actionable insights. If the file is empty or cannot be analyzed,
                indicate this clearly in your response.
                """
                
                user_prompt = f"""File: {file_content.path}
                Content:
                {file_content.content}
                """
                
                loop = asyncio.get_event_loop()
                completion = await loop.run_in_executor(
                    self.executor,
                    lambda: self.client.chat.completions.create(
                        model=self.model,  # Use the specified model
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    )
                )
                
                analysis = completion.choices[0].message.content
                return AnalysisResult(path=file_content.path, analysis=analysis)
                
            except Exception as e:
                logger.error(f"Error analyzing {file_content.path}: {str(e)}")
                return AnalysisResult(
                    path=file_content.path,
                    analysis=f"Error analyzing file: {str(e)}"
                )

    def find_files(self, directory_path: str) -> List[Path]:
        """Find all files with the specified extensions"""
        dir_path = Path(directory_path)
        if not dir_path.is_dir():
            raise ValueError(f"Directory not found: {directory_path}")

        matching_files = []
        for ext in self.file_extensions.extensions:
            matching_files.extend(dir_path.rglob(f"*{ext}"))
        
        return sorted(set(matching_files))  # Remove duplicates and sort

    async def analyze_directory_async(self, directory_path: str) -> DirectoryAnalysis:
        """Analyze all matching files in the directory"""
        logger.info(f"Beginning analysis of directory: {directory_path}")
        logger.info(f"Looking for files with extensions: {self.file_extensions.extensions}")
        
        # Find all matching files
        files = self.find_files(directory_path)
        if not files:
            logger.warning(f"No matching files found in {directory_path}")
            return DirectoryAnalysis(
                directory=directory_path,
                file_count=0,
                results=[],
                timestamp=datetime.now().isoformat(),
                file_types=self.file_extensions.extensions
            )

        # Read files concurrently
        logger.info(f"Found {len(files)} files to analyze")
        file_contents = await asyncio.gather(
            *[self.read_file_content_async(f) for f in files]
        )
        file_contents = [fc for fc in file_contents if fc is not None]

        # Analyze files concurrently
        logger.info("Analyzing files...")
        analyses = await asyncio.gather(
            *[self.analyze_file_content_async(fc) for fc in file_contents]
        )

        return DirectoryAnalysis(
            directory=directory_path,
            file_count=len(analyses),
            results=analyses,
            timestamp=datetime.now().isoformat(),
            file_types=self.file_extensions.extensions
        )

    def analyze_directory(self, directory_path: str) -> DirectoryAnalysis:
        """Synchronous wrapper for analyze_directory_async"""
        return asyncio.run(self.analyze_directory_async(directory_path))

def save_analysis_to_file(analysis: DirectoryAnalysis, output_path: str):
    """Save analysis results to a markdown file"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(f"# Code Analysis Report\n\n")
        f.write(f"- **Directory:** {analysis.directory}\n")
        f.write(f"- **Files Analyzed:** {analysis.file_count}\n")
        f.write(f"- **File Types:** {', '.join(analysis.file_types)}\n")
        f.write(f"- **Analysis Date:** {analysis.timestamp}\n\n")

        for result in analysis.results:
            f.write(f"## {Path(result.path).name}\n\n")
            f.write(f"**Path:** {result.path}\n\n")
            f.write(f"**Analysis:**\n\n{result.analysis}\n\n")
            f.write("---\n\n")

async def async_main():
    """Example usage"""
    analyzer = SimpleCodeAnalyzer(
        file_types="python and configuration files",
        analysis_prompt=ANALYSIS_PROMPT,  # Use global analysis prompt
    )

    try:
        analysis = await analyzer.analyze_directory_async(PROJECT_DIR)  # Use global project directory
        
        # Save results to file using global output file path
        save_analysis_to_file(analysis, OUTPUT_FILE)
        
        # Print summary to console
        print(f"\nAnalysis completed at {analysis.timestamp}")
        print(f"Analyzed {analysis.file_count} files in {analysis.directory}")
        print(f"File types analyzed: {', '.join(analysis.file_types)}")
        print(f"\nResults saved to {OUTPUT_FILE}")

    except Exception as e:
        logger.error("Error during analysis", exc_info=True)

def main():
    """Entry point"""
    asyncio.run(async_main())

if __name__ == "__main__":
    main()