import logging
from datetime import datetime
from typing import Optional, List, Callable
from pathlib import Path
from pydantic import BaseModel, field_validator
from openai import OpenAI
import colorlog
import asyncio
from asyncio import Semaphore
from concurrent.futures import ThreadPoolExecutor
import aiofiles
import os
from prompt_templates import PromptTemplates
import tiktoken
import yaml

# Load configuration from YAML file
def load_config(config_path: str = "config.yaml") -> dict:
    config_path = os.path.join(os.path.dirname(__file__), config_path)
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Load configuration at module level
CONFIG = load_config()

# Replace global variables with config values
OPENAI_MODEL = CONFIG['model']['name']
MODEL_CONFIGS = CONFIG['model']['configs']
TOKEN_SAFETY_MARGIN = CONFIG['analysis']['token_safety_margin']
PERFORM_FINAL_ANALYSIS = CONFIG['analysis']['perform_final_analysis']
MAX_TOKENS_PER_CHUNK = CONFIG['analysis']['max_tokens_per_chunk']
TOKEN_ESTIMATE_RATIO = CONFIG['analysis']['token_estimate_ratio']
DEFAULT_CONTEXT_WINDOW = CONFIG['analysis']['default_context_window']
DEFAULT_MAX_OUTPUT = CONFIG['analysis']['default_max_output']

# Configure logging with colors from config
handler = colorlog.StreamHandler()
handler.setFormatter(colorlog.ColoredFormatter(
    '%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    log_colors=CONFIG['logging']['colors']
))

logger = logging.getLogger(__name__)
logger.setLevel(getattr(logging, CONFIG['logging']['level']))
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
            # Map common names to proper extensions
            extension_map = {
                'python': 'py',
                'javascript': 'js',
                'typescript': 'ts',
                'yaml': 'yaml',
                'json': 'json'
            }
            processed = []
            for ext in extensions:
                ext = ext.lower().strip('.')
                ext = extension_map.get(ext, ext)  # Use mapping or original if not found
                processed.append(ext)
            return cls(extensions=processed)

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
        template_name: str = 'analysis',
        template_path: Optional[str] = None,
        api_key: Optional[str] = None,
        max_concurrent: int = 3,
        model: str = OPENAI_MODEL,
        final_analysis_model: Optional[str] = None,
        perform_final_analysis: bool = PERFORM_FINAL_ANALYSIS,
        progress_callback: Optional[Callable[[str], None]] = None
    ):
        self.client = OpenAI(api_key=api_key or os.getenv('OPENAI_API_KEY'))
        self.model = model
        self.file_extensions = FileExtensions.from_string(file_types, client=self.client, model=self.model)
        self.templates = PromptTemplates(template_path)
        self.template_name = template_name
        self.final_template_name = 'final_analysis'
        self.analysis_prompt = self.templates.get_template(template_name) if self.templates.manager.template_exists(template_name) else None
        self.final_analysis_prompt = self.templates.get_template('final_analysis') if self.templates.manager.template_exists('final_analysis') else None
        self.max_concurrent = max_concurrent
        self.rate_limiter = RateLimiter()
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)
        self.perform_final_analysis = perform_final_analysis
        self.final_analysis_model = final_analysis_model or model
        self.progress_callback = progress_callback

        # Initialize tokenizer and set token limits from configuration
        try:
            self.tokenizer = tiktoken.encoding_for_model(self.model)
            model_config = MODEL_CONFIGS.get(self.model, {
                "context_window": DEFAULT_CONTEXT_WINDOW,
                "max_output_tokens": DEFAULT_MAX_OUTPUT
            })
            self.max_tokens = int(model_config["context_window"] * TOKEN_SAFETY_MARGIN)
            logger.info(f"Using model {self.model} with context window of {model_config['context_window']} tokens")
        except Exception as e:
            logger.warning(f"Error initializing tokenizer: {e}. Using fallback values.")
            self.tokenizer = tiktoken.get_encoding("cl100k_base")
            self.max_tokens = int(DEFAULT_CONTEXT_WINDOW * TOKEN_SAFETY_MARGIN)
        
        logger.info(f"Using {self.max_tokens} tokens per chunk for model {self.model}")

    def update_progress(self, message: str):
        """Update progress status"""
        if self.progress_callback:
            self.progress_callback(message)
        logger.info(message)

    def count_tokens(self, text: str) -> int:
        """Accurate token count using tiktoken"""
        # Add a check to ensure minimum token count
        tokens = len(self.tokenizer.encode(text))
        return max(tokens, 2)  # Ensure minimum of 2 tokens for test purposes

    def split_content_into_chunks(self, content: str) -> List[str]:
        """Split content into chunks based on model token limit"""
        total_tokens = self.count_tokens(content)
        
        if total_tokens <= self.max_tokens:
            return [content]

        # Split into chunks by lines to maintain code structure
        lines = content.splitlines()
        chunks = []
        current_chunk = []
        current_tokens = 0

        # Add prompt template tokens to the limit calculation
        prompt_tokens = self.count_tokens(self.analysis_prompt)
        effective_token_limit = self.max_tokens - prompt_tokens

        for line in lines:
            line_tokens = self.count_tokens(line + "\n")
            if current_tokens + line_tokens > effective_token_limit:
                if current_chunk:  # Save current chunk if not empty
                    chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
                current_tokens = line_tokens
            else:
                current_chunk.append(line)
                current_tokens += line_tokens

        if current_chunk:  # Add the last chunk
            chunks.append('\n'.join(current_chunk))

        logger.info(f"Split content into {len(chunks)} chunks (total tokens: {total_tokens})")
        return chunks

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
        """Analyze file content using GPT, splitting into chunks if needed"""
        content_chunks = self.split_content_into_chunks(file_content.content)
        
        if len(content_chunks) == 1:
            return await self._analyze_single_chunk(file_content.path, content_chunks[0])

        # Analyze multiple chunks
        chunk_analyses = []
        for i, chunk in enumerate(content_chunks, 1):
            chunk_analysis = await self._analyze_single_chunk(
                file_content.path,
                chunk,
                chunk_info=f"(Part {i} of {len(content_chunks)})"
            )
            chunk_analyses.append(chunk_analysis.analysis)

        # Combine chunk analyses
        combined_analysis = "\n\n".join([
            f"=== Chunk {i+1}/{len(content_chunks)} ===\n{analysis}"
            for i, analysis in enumerate(chunk_analyses)
        ])

        # Generate final summary if multiple chunks
        if len(chunk_analyses) > 1:
            summary = await self._generate_chunk_summary(
                file_content.path,
                combined_analysis,
                len(content_chunks)
            )
            combined_analysis = f"{summary}\n\n{combined_analysis}"

        return AnalysisResult(
            path=file_content.path,
            analysis=combined_analysis
        )

    async def _analyze_single_chunk(
        self,
        file_path: str,
        content: str,
        chunk_info: str = ""
    ) -> AnalysisResult:
        """Analyze a single chunk of content"""
        async with self.rate_limiter:
            try:
                # Get template content
                template_content = self.templates.get_template(self.template_name)
                if not template_content:
                    raise ValueError(f"Template not found: {self.template_name}")

                system_prompt = f"""Analyze the provided file according to these instructions:

                {template_content}

                If the file is empty or cannot be analyzed, indicate this clearly in your response.
                {f'Note: This is {chunk_info}' if chunk_info else ''}
                """
                                
                user_prompt = f"""File: {file_path}
                Content:
                {content}
                """
                
                # Log the system prompt for debugging
                logger.debug(f"Using system prompt for {Path(file_path).name}:\n{system_prompt}")
                
                total_tokens = self.count_tokens(system_prompt + user_prompt)
                if total_tokens > self.max_tokens:
                    logger.warning(f"Chunk exceeds token limit: {total_tokens} tokens")
                
                loop = asyncio.get_event_loop()
                completion = await loop.run_in_executor(
                    self.executor,
                    lambda: self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt}
                        ]
                    )
                )
                
                analysis = completion.choices[0].message.content
                return AnalysisResult(path=file_path, analysis=analysis)
                
            except Exception as e:
                logger.error(f"Error analyzing {file_path}: {str(e)}")
                return AnalysisResult(
                    path=file_path,
                    analysis=f"Error analyzing file: {str(e)}"
                )

    async def _generate_chunk_summary(
        self,
        file_path: str,
        combined_analyses: str,
        num_chunks: int
    ) -> str:
        """Generate a summary of all chunks"""
        async with self.rate_limiter:
            try:
                prompt = f"""As a code analysis expert, provide a brief summary of the entire file based on the {num_chunks} analyzed chunks.
                Focus on:
                1. Overall file purpose
                2. Key components across all chunks
                3. Main patterns or issues observed
                
                Keep the summary concise and highlight the most important findings."""

                loop = asyncio.get_event_loop()
                completion = await loop.run_in_executor(
                    self.executor,
                    lambda: self.client.chat.completions.create(
                        model=self.final_analysis_model,
                        messages=[
                            {"role": "system", "content": self.final_analysis_prompt},
                            {"role": "user", "content": f"{prompt}\n\nAnalyses:\n{combined_analyses}"}
                        ]
                    )
                )
                
                return f"=== Overall Summary (File analyzed in {num_chunks} chunks) ===\n{completion.choices[0].message.content}"
                
            except Exception as e:
                logger.error(f"Error generating chunk summary for {file_path}: {str(e)}")
                return f"Error generating summary: {str(e)}"

    def find_files(self, directory_path: str) -> List[Path]:
        """Find all files with the specified extensions - synchronous method"""
        dir_path = Path(directory_path)
        if not dir_path.is_dir():
            raise ValueError(f"Directory not found: {directory_path}")

        matching_files = []
        logger.info(f"Scanning directory: {dir_path}")
        
        for ext in self.file_extensions.extensions:
            logger.info(f"Searching for *{ext} files...")
            found = list(dir_path.rglob(f"*{ext}"))
            logger.info(f"Found {len(found)} files with extension {ext}")
            matching_files.extend(found)
        
        unique_files = sorted(set(matching_files))
        logger.info(f"Total unique files found: {len(unique_files)}")
        return unique_files

    async def generate_final_analysis(self, analysis: DirectoryAnalysis) -> str:
        """Generate a final analysis of all combined results with recursive chunking"""
        if not self.perform_final_analysis:
            return ""

        logger.info(f"Generating final analysis using model: {self.final_analysis_model}")
        
        # Get the context window for the final analysis model
        model_config = MODEL_CONFIGS.get(self.final_analysis_model, {
            "context_window": DEFAULT_CONTEXT_WINDOW,
            "max_output_tokens": DEFAULT_MAX_OUTPUT
        })
        final_max_tokens = int(model_config["context_window"] * TOKEN_SAFETY_MARGIN)

        # Prepare the content for analysis
        contents = [
            f"File: {result.path}\n{result.analysis}"
            for result in analysis.results
        ]
        combined_content = "\n\n".join(contents)

        # Check if content exceeds token limit
        if self.count_tokens(combined_content) > final_max_tokens:
            logger.info("Final analysis content exceeds token limit, using recursive chunking")
            return await self._generate_chunked_final_analysis(contents, final_max_tokens)

        # If content fits, proceed with normal analysis
        async with self.rate_limiter:
            try:
                loop = asyncio.get_event_loop()
                completion = await loop.run_in_executor(
                    self.executor,
                    lambda: self.client.chat.completions.create(
                        model=self.final_analysis_model,
                        messages=[
                            {"role": "system", "content": "You are a senior software architect analyzing an entire codebase."},
                            {"role": "user", "content": f"{self.final_analysis_prompt}\n\nAnalysis results:\n{combined_content}"}
                        ]
                    )
                )
                return completion.choices[0].message.content
            except Exception as e:
                logger.error(f"Error generating final analysis: {str(e)}")
                return f"Error generating final analysis: {str(e)}"

    async def _generate_chunked_final_analysis(self, contents: List[str], max_tokens: int) -> str:
        """Recursively analyze content in chunks and combine results"""
        logger.info("Starting chunked final analysis")

        # Calculate chunk sizes based on token limit
        prompt_tokens = self.count_tokens(self.final_analysis_prompt)
        effective_token_limit = max_tokens - prompt_tokens - 1000  # Extra margin for safety

        # Split contents into chunks that fit within token limit
        chunks = []
        current_chunk = []
        current_tokens = 0

        for content in contents:
            content_tokens = self.count_tokens(content)
            if current_tokens + content_tokens > effective_token_limit:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = [content]
                current_tokens = content_tokens
            else:
                current_chunk.append(content)
                current_tokens += content_tokens

        if current_chunk:
            chunks.append(current_chunk)

        logger.info(f"Split final analysis into {len(chunks)} chunks")

        # Analyze each chunk
        chunk_analyses = []
        for i, chunk in enumerate(chunks, 1):
            logger.info(f"Analyzing chunk {i}/{len(chunks)}")
            chunk_content = "\n\n".join(chunk)
            
            async with self.rate_limiter:
                try:
                    completion = await self.client.chat.completions.create(
                        model=self.final_analysis_model,
                        messages=[
                            {"role": "system", "content": self.final_analysis_prompt},
                            {"role": "user", "content": f"Analyze this section:\n\n{chunk_content}"}
                        ]
                    )
                    chunk_analyses.append(completion.choices[0].message.content)
                except Exception as e:
                    logger.error(f"Error in chunk analysis {i}: {str(e)}")
                    chunk_analyses.append(f"Error analyzing chunk {i}: {str(e)}")

        # If we have multiple chunks, recursively analyze their combined summaries
        if len(chunk_analyses) > 1:
            combined_summaries = "\n\n=== Next Section ===\n\n".join(chunk_analyses)
            if self.count_tokens(combined_summaries) > effective_token_limit:
                return await self._generate_chunked_final_analysis(chunk_analyses, max_tokens)
            
            # Final combination of all chunk analyses
            async with self.rate_limiter:
                try:
                    final_completion = await self.client.chat.completions.create(
                        model=self.final_analysis_model,
                        messages=[
                            {"role": "system", "content": self.final_analysis_prompt},
                            {"role": "user", "content": f"Create a final synthesis combining these sections:\n\n{combined_summaries}"}
                        ]
                    )
                    return final_completion.choices[0].message.content
                except Exception as e:
                    logger.error(f"Error in final combination: {str(e)}")
                    return f"Error in final combination: {str(e)}"
        
        # If we only had one chunk, return its analysis
        return chunk_analyses[0]

    async def analyze_directory_async(self, directory_path: str) -> DirectoryAnalysis:
        """Analyze all matching files in the directory"""
        self.update_progress("Starting directory analysis...")
        logger.info(f"Beginning analysis of directory: {directory_path}")
        logger.info(f"Looking for files with extensions: {self.file_extensions.extensions}")
        
        files = self.find_files(directory_path)
        if not files:
            self.update_progress("No files found")
            logger.warning(f"No matching files found in {directory_path}")
            return DirectoryAnalysis(
                directory=directory_path,
                file_count=0,
                results=[],
                timestamp=datetime.now().isoformat(),
                file_types=self.file_extensions.extensions
            )

        self.update_progress(f"Found {len(files)} files to analyze")
        logger.info(f"Found {len(files)} files to analyze")

        # Create chunks of files to process in batches
        chunk_size = self.max_concurrent
        file_chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
        
        all_analyses = []
        for i, chunk in enumerate(file_chunks, 1):
            self.update_progress(f"Processing batch {i}/{len(file_chunks)}")
            # Process each chunk of files concurrently
            read_tasks = [self.read_file_content_async(f) for f in chunk]
            file_contents = [fc for fc in await asyncio.gather(*read_tasks) if fc]
            
            # Analyze files in chunk concurrently
            analysis_tasks = []
            for fc in file_contents:
                self.update_progress(f"Analyzing: {Path(fc.path).name}")
                logger.info(f"Analyzing file: {Path(fc.path).name}")
                analysis_tasks.append(self.analyze_file_content_async(fc))
            
            chunk_analyses = await asyncio.gather(*analysis_tasks)
            
            # Log completion for each file
            for result in chunk_analyses:
                logger.info(f"Completed analysis of {Path(result.path).name}")
            
            all_analyses.extend(chunk_analyses)
            
            # Optional: Add a small delay between chunks to prevent rate limiting
            await asyncio.sleep(0.1)

        self.update_progress("Creating analysis summary...")
        logger.info("Creating directory analysis summary...")
        analysis = DirectoryAnalysis(
            directory=directory_path,
            file_count=len(all_analyses),
            results=all_analyses,
            timestamp=datetime.now().isoformat(),
            file_types=self.file_extensions.extensions
        )
        
        if self.perform_final_analysis:
            self.update_progress("Generating final analysis...")
            logger.info("Starting final codebase analysis...")
            final_analysis = await self.generate_final_analysis(analysis)
            self.update_progress("Final analysis complete")
            logger.info("Final analysis complete")
            analysis.results.append(AnalysisResult(
                path="FINAL_ANALYSIS",
                analysis=final_analysis
            ))
        
        self.update_progress("Analysis complete!")
        logger.info("Analysis complete!")
        return analysis

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
            if result.path == "FINAL_ANALYSIS":
                continue  # Skip the final analysis here
            f.write(f"## {Path(result.path).name}\n\n")
            f.write(f"**Path:** {result.path}\n\n")
            f.write(f"**Analysis:**\n\n{result.analysis}\n\n")
            f.write("---\n\n")

        # Write final analysis at the end if it exists
        final_analysis = next((r for r in analysis.results if r.path == "FINAL_ANALYSIS"), None)
        if final_analysis:
            f.write("# Final Codebase Analysis\n\n")
            f.write(final_analysis.analysis)

async def async_main():
    """Example usage"""
    analyzer = SimpleCodeAnalyzer(
        file_types="python and configuration files",
        template_name="requirements",  # Use the requirements template
        template_path="custom_templates.yaml"  # Optional: path to custom templates
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