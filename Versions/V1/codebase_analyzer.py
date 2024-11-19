import logging
from datetime import datetime
from typing import Optional, Literal, List
from pathlib import Path
from pydantic import BaseModel
from openai import OpenAI
import time
import json
import colorlog  # Add this import
import asyncio
from asyncio import Semaphore
import aiohttp
from concurrent.futures import ThreadPoolExecutor

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
logger.setLevel(logging.DEBUG)
logger.handlers = []  # Remove any existing handlers
logger.addHandler(handler)

# System-determined metadata models
class FileSystemMetadata(BaseModel):
    """Metadata that can be determined directly from the filesystem"""
    name: str
    path: str
    size_bytes: int
    last_modified: str  # ISO format timestamp

# GPT-analyzed metadata models
# Update FileAnalysis model to be more focused
class FileAnalysis(BaseModel):
    """Metadata that requires analysis/inference"""
    file_type: Literal["config", "core", "utility", "test", "documentation", "unknown"]
    key_functions: List[str]
    unused_functions: List[str]  # Added
    public_interfaces: List[str]
    description: str
    detailed_description: str
    complexity_analysis: str
    external_dependencies: List[str]
    # Removed: key_classes, internal_interfaces, coding_patterns, potential_issues

class DependencyReference(BaseModel):
    """Represents a dependency relationship between files"""
    target_path: str  # Path to the dependency
    dependency_type: Literal["imports", "inherits", "uses", "configures", "tests"]
    importance: int  # 1-10 scale
    relevant_sections: List[str]

class DependencyList(BaseModel):
    """Wrapper for list of dependency references"""
    dependencies: List[DependencyReference]

class ProcessingOrderEntry(BaseModel):
    """Entry in the processing order determined by GPT"""
    file_path: str
    dependencies: List[str]
    processing_priority: int
    processing_notes: Optional[str] = None

class ProcessingOrder(BaseModel):
    """Complete processing order with reasoning"""
    ordered_entries: List[ProcessingOrderEntry]
    overall_reasoning: Optional[str] = None

# Combined models for complete file information
class FileNode(BaseModel):
    """Complete information about a file, combining system and GPT-analyzed data"""
    system_metadata: FileSystemMetadata
    analysis: Optional[FileAnalysis] = None
    required_dependencies: List[DependencyReference] = []
    provides_context_for: List[DependencyReference] = []

class CodebaseAnalysis(BaseModel):
    """Complete analysis of the codebase"""
    root_directory: str
    total_files: int
    analysis_timestamp: str  # ISO format timestamp
    files: List[FileNode]
    processing_order: ProcessingOrder

# Update FileSummary model to include analysis details
class FileSummary(BaseModel):
    """Simplified summary of a single file"""
    name: str
    type: str
    purpose: str
    key_features: List[str]
    dependencies: List[str]
    importance_score: int  # 1-10 based on how many other files depend on it
    detailed_description: Optional[str] = None
    complexity: Optional[str] = None
    unused_functions: List[str] = []
    external_deps: List[str] = []

class DirectoryStructure(BaseModel):
    """Represents the logical structure of the codebase"""
    core_files: List[str]
    utilities: List[str]
    config_files: List[str]
    test_files: List[str]
    documentation: List[str]

class DirectorySummary(BaseModel):
    """High-level summary of the entire codebase"""
    project_name: str
    total_files: int
    main_purpose: str
    key_components: List[FileSummary]
    structure: DirectoryStructure
    suggested_reading_order: List[str]
    entry_points: List[str]  # Files that serve as main entry points
    recommendations: List[str]  # Suggestions for understanding the codebase

class CodebaseAnalyzer:
    def __init__(self, api_key: Optional[str] = None, max_concurrent: int = 3):
        self.client = OpenAI(api_key=api_key)
        self.logger = logging.getLogger(__name__)
        self.max_concurrent = max_concurrent
        self.semaphore = Semaphore(max_concurrent)
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)

    async def analyze_file_content_async(self, content: str, file_path: str) -> FileAnalysis:
        """Async version of analyze_file_content"""
        async with self.semaphore:
            try:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    self.executor,
                    lambda: self.analyze_file_content(content, file_path)
                )
            except Exception as e:
                self.logger.error(f"Error analyzing {Path(file_path).name}: {str(e)}")
                return FileAnalysis(
                    file_type="unknown",
                    key_functions=[],
                    unused_functions=[],
                    public_interfaces=[],
                    description="Error analyzing file",
                    detailed_description="Error occurred during analysis",
                    complexity_analysis="Unknown",
                    external_dependencies=[]
                )

    async def analyze_dependencies_async(self, content: str, file_path: str, all_files: List[str]) -> List[DependencyReference]:
        """Async version of analyze_dependencies"""
        async with self.semaphore:
            try:
                loop = asyncio.get_event_loop()
                return await loop.run_in_executor(
                    self.executor,
                    lambda: self.analyze_dependencies(content, file_path, all_files)
                )
            except Exception as e:
                self.logger.error(f"Error analyzing dependencies for {Path(file_path).name}: {str(e)}")
                return []

    async def process_file_async(self, file_path: Path, all_file_paths: List[str]) -> Optional[FileNode]:
        """Process a single file asynchronously"""
        try:
            content = self.read_file_content(file_path)
            if content is None:
                return None

            system_metadata = self.get_system_metadata(file_path)
            
            # Run analysis and dependency check concurrently
            analysis, dependencies = await asyncio.gather(
                self.analyze_file_content_async(content, str(file_path)),
                self.analyze_dependencies_async(content, str(file_path), all_file_paths)
            )

            return FileNode(
                system_metadata=system_metadata,
                analysis=analysis,
                required_dependencies=dependencies,
                provides_context_for=[]
            )
        except Exception as e:
            self.logger.error(f"Error processing {file_path.name}: {str(e)}")
            return None

    async def _analyze_codebase_detailed_async(self, directory_path: str) -> CodebaseAnalysis:
        """Async version of _analyze_codebase_detailed"""
        start_time = time.time()
        self.logger.info(f"Beginning analysis of codebase at: {directory_path}")
        dir_path = Path(directory_path)
        
        if not dir_path.is_dir():
            raise ValueError(f"Directory not found: {directory_path}")

        python_files = list(dir_path.rglob("*.py"))
        all_file_paths = [str(f) for f in python_files]
        
        # Process files concurrently
        tasks = [self.process_file_async(f, all_file_paths) for f in python_files]
        file_nodes = [node for node in await asyncio.gather(*tasks) if node is not None]

        # Determine processing order and update relationships
        processing_order = self.determine_processing_order(file_nodes)
        self._update_context_relationships(file_nodes)

        analysis = CodebaseAnalysis(
            root_directory=str(dir_path),
            total_files=len(file_nodes),
            analysis_timestamp=datetime.now().isoformat(),
            files=file_nodes,
            processing_order=processing_order
        )

        total_time = time.time() - start_time
        self.logger.info(f"Analysis completed in {total_time:.2f}s")
        return analysis

    def analyze_codebase(self, directory_path: str) -> DirectorySummary:
        """Modified main method to use async processing"""
        detailed_analysis = asyncio.run(self._analyze_codebase_detailed_async(directory_path))
        return self.generate_directory_summary(detailed_analysis)

    def get_system_metadata(self, file_path: Path) -> FileSystemMetadata:
        """Get filesystem-determined metadata"""
        stats = file_path.stat()
        return FileSystemMetadata(
            name=file_path.name,
            path=str(file_path),
            size_bytes=stats.st_size,
            last_modified=datetime.fromtimestamp(stats.st_mtime).isoformat()
        )

    def analyze_file_content(self, content: str, file_path: str) -> FileAnalysis:
        """Use GPT to analyze file content with streamlined analysis"""
        try:
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """Analyze this Python file and provide:
                        1. File type (config/core/utility/test/documentation/unknown)
                        2. Key functions used in the codebase
                        3. Functions that appear unused or dead code
                        4. Public interfaces/methods
                        5. Brief high-level description
                        6. Detailed description focusing on main responsibilities
                        7. Basic complexity assessment (Low/Medium/High with brief reason)
                        8. External library dependencies
                        
                        Keep descriptions concise and focus on essential information."""
                    },
                    {
                        "role": "user",
                        "content": f"File path: {file_path}\n\nContent:\n{content}"
                    }
                ],
                response_format=FileAnalysis
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            self.logger.error(f"Error analyzing {Path(file_path).name}: {str(e)}")
            return FileAnalysis(
                file_type="unknown",
                key_functions=[],
                unused_functions=[],
                public_interfaces=[],
                description="Error analyzing file",
                detailed_description="Error occurred during analysis",
                complexity_analysis="Unknown",
                external_dependencies=[]
            )

    def analyze_dependencies(self, content: str, file_path: str, all_files: List[str]) -> List[DependencyReference]:
        """Use GPT to analyze file dependencies"""
        try:
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """Analyze this Python file and identify its dependencies from the provided list of files.
                        For each dependency, determine:
                        1. Type of dependency
                        2. Importance (1-10)
                        3. Relevant sections that are used"""
                    },
                    {
                        "role": "user",
                        "content": f"""File: {file_path}
                        Available files: {json.dumps(all_files)}
                        Content: {content}"""
                    }
                ],
                response_format=DependencyList
            )
            return completion.choices[0].message.parsed.dependencies
        except Exception as e:
            self.logger.error(f"Error analyzing dependencies for {Path(file_path).name}: {str(e)}")
            return []

    def determine_processing_order(self, nodes: List[FileNode]) -> ProcessingOrder:
        """Use GPT to determine optimal processing order"""
        try:
            node_info = [
                {
                    "path": node.system_metadata.path,
                    "type": node.analysis.file_type if node.analysis else "unknown",
                    "dependencies": [dep.target_path for dep in node.required_dependencies]
                }
                for node in nodes
            ]
            
            completion = self.client.beta.chat.completions.parse(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """Determine the optimal processing order for these files.
                        Consider dependencies and file types when ordering."""
                    },
                    {
                        "role": "user",
                        "content": f"Files and dependencies:\n{json.dumps(node_info, indent=2)}"
                    }
                ],
                response_format=ProcessingOrder
            )
            return completion.choices[0].message.parsed
        except Exception as e:
            self.logger.error(f"Error determining processing order: {str(e)}")
            # Fallback to simple alphabetical order
            return ProcessingOrder(
                ordered_entries=[
                    ProcessingOrderEntry(
                        file_path=node.system_metadata.path,
                        dependencies=[],
                        processing_priority=idx
                    )
                    for idx, node in enumerate(sorted(nodes, key=lambda x: x.system_metadata.path))
                ],
                overall_reasoning="Fallback to alphabetical order due to error"
            )

    def _analyze_codebase_detailed(self, directory_path: str) -> CodebaseAnalysis:
        """Renamed original analyze_codebase method"""
        start_time = time.time()
        self.logger.info(f"Beginning analysis of codebase at: {directory_path}")
        dir_path = Path(directory_path)
        if not dir_path.is_dir():
            raise ValueError(f"Directory not found: {directory_path}")

        # First pass: Collect all Python files
        self.logger.info("Starting first pass: Collecting Python files...")
        python_files = list(dir_path.rglob("*.py"))
        file_nodes: List[FileNode] = []
        all_file_paths = [str(f) for f in python_files]
        self.logger.info(f"Found {len(python_files)} Python files to analyze")

        # Second pass: Analyze each file
        self.logger.info("\nStarting second pass: Analyzing individual files...")
        for idx, file_path in enumerate(python_files, 1):
            file_start_time = time.time()
            self.logger.info(f"\nProcessing file {idx}/{len(python_files)}: {file_path.name}")
            try:
                content = self.read_file_content(file_path)
                if content is None:
                    self.logger.warning(f"Skipping {file_path.name} - Unable to read content")
                    continue

                self.logger.debug(f"Getting system metadata for {file_path.name}")
                system_metadata = self.get_system_metadata(file_path)
                
                self.logger.debug(f"Analyzing content of {file_path.name}")
                analysis = self.analyze_file_content(content, str(file_path))
                
                self.logger.debug(f"Analyzing dependencies for {file_path.name}")
                dependencies = self.analyze_dependencies(content, str(file_path), all_file_paths)

                node = FileNode(
                    system_metadata=system_metadata,
                    analysis=analysis,
                    required_dependencies=dependencies,
                    provides_context_for=[]
                )
                file_nodes.append(node)
                self.logger.info(f"""File Analysis Results for {file_path.name}:
                Type: {analysis.file_type}
                Description: {analysis.description}
                Key Functions: {', '.join(analysis.key_functions)}
                Dependencies Found: {len(dependencies)}
                Dependencies: {[d.target_path.split('/')[-1] for d in dependencies]}
                Processing Time: {time.time() - file_start_time:.2f}s""")
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                self.logger.error(f"Error processing {file_path.name}: {str(e)}", exc_info=True)
                continue

        # Third pass: Determine processing order
        self.logger.info("\nStarting third pass: Determining optimal processing order...")
        processing_order = self.determine_processing_order(file_nodes)
        self.logger.info("\nProcessing Order Details:")
        for entry in processing_order.ordered_entries:
            self.logger.info(f"""    {entry.file_path.split('/')[-1]}:
            Priority: {entry.processing_priority}
            Dependencies: {', '.join(d.split('/')[-1] for d in entry.dependencies)}
            Notes: {entry.processing_notes or 'None'}""")

        # Fourth pass: Update relationships
        self.logger.info("\nStarting fourth pass: Updating context relationships...")
        self._update_context_relationships(file_nodes)

        analysis = CodebaseAnalysis(
            root_directory=str(dir_path),
            total_files=len(file_nodes),
            analysis_timestamp=datetime.now().isoformat(),
            files=file_nodes,
            processing_order=processing_order
        )

        # Final Analysis Summary
        total_time = time.time() - start_time
        self.logger.info(f"""\nFinal Analysis Summary:
        Total Files Analyzed: {len(file_nodes)}
        Total Processing Time: {total_time:.2f}s
        Average Time per File: {total_time/len(file_nodes):.2f}s
        
        File Type Distribution:
        {self._get_file_type_distribution(file_nodes)}
        
        Most Referenced Files:
        {self._get_most_referenced_files(file_nodes)}
        
        Files with Most Dependencies:
        {self._get_files_with_most_dependencies(file_nodes)}""")

        return analysis

    def _get_file_type_distribution(self, nodes: List[FileNode]) -> str:
        """Helper to get distribution of file types"""
        distribution = {}
        for node in nodes:
            if node.analysis:
                distribution[node.analysis.file_type] = distribution.get(node.analysis.file_type, 0) + 1
        return '\n    '.join(f"{k}: {v}" for k, v in distribution.items())

    def _get_most_referenced_files(self, nodes: List[FileNode], limit: int = 3) -> str:
        """Helper to get most referenced files"""
        references = {node.system_metadata.path: len(node.provides_context_for) for node in nodes}
        sorted_refs = sorted(references.items(), key=lambda x: x[1], reverse=True)[:limit]
        return '\n    '.join(f"{Path(k).name}: {v} references" for k, v in sorted_refs)

    def _get_files_with_most_dependencies(self, nodes: List[FileNode], limit: int = 3) -> str:
        """Helper to get files with most dependencies"""
        deps = {node.system_metadata.path: len(node.required_dependencies) for node in nodes}
        sorted_deps = sorted(deps.items(), key=lambda x: x[1], reverse=True)[:limit]
        return '\n    '.join(f"{Path(k).name}: {v} dependencies" for k, v in sorted_deps)

    def _update_context_relationships(self, nodes: List[FileNode]):
        """Update provides_context_for relationships based on dependencies"""
        path_to_node = {node.system_metadata.path: node for node in nodes}
        
        for node in nodes:
            for dep in node.required_dependencies:
                if dep.target_path in path_to_node:
                    target_node = path_to_node[dep.target_path]
                    target_node.provides_context_for.append(
                        DependencyReference(
                            target_path=node.system_metadata.path,
                            dependency_type=dep.dependency_type,
                            importance=dep.importance,
                            relevant_sections=dep.relevant_sections
                        )
                    )

    def read_file_content(self, file_path: Path) -> Optional[str]:
        """Read file content with multiple encoding attempts"""
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except UnicodeDecodeError:
                continue
        self.logger.error(f"Failed to read file: {file_path}")
        return None

    def _simplify_path(self, full_path: str, root_dir: str) -> str:
        """Convert absolute path to relative path or filename"""
        try:
            # Convert both paths to Path objects for easier manipulation
            full_path = Path(full_path)
            root_dir = Path(root_dir)
            
            # Try to get relative path
            try:
                relative = full_path.relative_to(root_dir)
                return str(relative)
            except ValueError:
                # If relative path fails, return just the filename
                return full_path.name
        except Exception:
            # Fallback to just the filename if any error occurs
            return Path(full_path).name

    def generate_directory_summary(self, analysis: CodebaseAnalysis) -> DirectorySummary:
        """Convert detailed analysis into a user-friendly summary"""
        # Calculate importance scores based on references
        importance_scores = {
            node.system_metadata.path: len(node.provides_context_for)
            for node in analysis.files
        }

        # Generate file summaries for key components (files with importance > 0)
        key_components = []
        for node in analysis.files:
            if importance_scores[node.system_metadata.path] > 0:
                key_components.append(FileSummary(
                    name=node.system_metadata.name,
                    type=node.analysis.file_type if node.analysis else "unknown",
                    purpose=node.analysis.description if node.analysis else "Unknown purpose",
                    key_features=node.analysis.key_functions if node.analysis else [],
                    dependencies=[self._simplify_path(d.target_path, analysis.root_directory) 
                                for d in node.required_dependencies],
                    importance_score=importance_scores[node.system_metadata.path],
                    detailed_description=node.analysis.detailed_description if node.analysis else None,
                    complexity=node.analysis.complexity_analysis if node.analysis else None,
                    unused_functions=node.analysis.unused_functions if node.analysis else [],
                    external_deps=node.analysis.external_dependencies if node.analysis else []
                ))

        # Group files by type (using simplified paths)
        structure = DirectoryStructure(
            core_files=[n.system_metadata.name for n in analysis.files 
                       if n.analysis and n.analysis.file_type == "core"],
            utilities=[n.system_metadata.name for n in analysis.files 
                      if n.analysis and n.analysis.file_type == "utility"],
            config_files=[n.system_metadata.name for n in analysis.files 
                         if n.analysis and n.analysis.file_type == "config"],
            test_files=[n.system_metadata.name for n in analysis.files 
                       if n.analysis and n.analysis.file_type == "test"],
            documentation=[n.system_metadata.name for n in analysis.files 
                         if n.analysis and n.analysis.file_type == "documentation"]
        )

        # Use simplified paths for reading order
        suggested_reading_order = [
            self._simplify_path(e.file_path, analysis.root_directory)
            for e in analysis.processing_order.ordered_entries
        ]

        # Determine entry points (files with high importance and few dependencies)
        entry_points = [
            node.system_metadata.name for node in analysis.files
            if importance_scores[node.system_metadata.path] > 2 and len(node.required_dependencies) < 2
        ]

        # Generate recommendations
        recommendations = self._generate_recommendations(analysis.files)

        return DirectorySummary(
            project_name=Path(analysis.root_directory).name,
            total_files=analysis.total_files,
            main_purpose=self._infer_main_purpose(analysis.files),
            key_components=sorted(key_components, key=lambda x: x.importance_score, reverse=True),
            structure=structure,
            suggested_reading_order=suggested_reading_order,
            entry_points=entry_points,
            recommendations=recommendations
        )

    def _infer_main_purpose(self, files: List[FileNode]) -> str:
        """Infer the main purpose of the codebase from core files"""
        core_files = [f for f in files if f.analysis and f.analysis.file_type == "core"]
        if not core_files:
            return "Unable to determine main purpose"
        
        descriptions = [f.analysis.description for f in core_files if f.analysis]
        # Use GPT to synthesize a concise main purpose from the descriptions
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "system",
                    "content": "Synthesize a concise (1-2 sentences) main purpose from these component descriptions:"
                }, {
                    "role": "user",
                    "content": "\n".join(descriptions)
                }]
            )
            return completion.choices[0].message.content
        except Exception:
            return descriptions[0] if descriptions else "Unable to determine main purpose"

    def _generate_recommendations(self, files: List[FileNode]) -> List[str]:
        """Generate recommendations for understanding the codebase"""
        recs = []
        core_count = sum(1 for f in files if f.analysis and f.analysis.file_type == "core")
        
        if core_count > 0:
            recs.append(f"Start by examining the {core_count} core files to understand the main functionality")
        
        complex_files = [f for f in files if len(f.required_dependencies) > 3]
        if complex_files:
            recs.append(f"The most complex files ({', '.join(f.system_metadata.name for f in complex_files[:3])}) " +
                       "might require extra attention")
        
        # Add more recommendations based on analysis...
        return recs

def main():
    logger.info("="*50)
    logger.info("Starting codebase analyzer")
    logger.info("="*50)
    
    import argparse
    # parser = argparse.ArgumentParser(description='Analyze a Python codebase with GPT-4O Mini')
    # parser.add_argument('directory', help='Path to the Python codebase directory')
    # parser.add_argument('--output', default='codebase_analysis.json', help='Output JSON file path')
    # args = parser.parse_args()
    
    args = argparse.Namespace(
        directory=r"C:\Users\wkraf\Documents\Coding\Directory_Summarizer\Versions\V1\Sample_Directories\trading_system",
        output=r"C:\Users\wkraf\Documents\Coding\Directory_Summarizer\Versions\V1\Sample_Outputs\codebase_analysis.json"
    )

    analyzer = CodebaseAnalyzer()
    try:
        analysis = analyzer.analyze_codebase(args.directory)
        
        # Save analysis to file with a more intuitive name
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Writing directory summary to: {args.output}")
        with open(args.output, 'w') as f:
            f.write(analysis.model_dump_json(indent=2))
        
        # Print a human-readable summary to console
        logger.info("\n" + "="*80)
        logger.info(f"Detailed Directory Summary for: {analysis.project_name}")
        logger.info("="*80)
        logger.info(f"\nMain Purpose: {analysis.main_purpose}")
        logger.info("\nKey Components:")
        for component in analysis.key_components:
            logger.info(f"\n{'-'*40}")
            logger.info(f"File: {component.name} ({component.type})")
            logger.info(f"Purpose: {component.purpose}")
            
            if component.detailed_description:
                logger.info("\nDetailed Analysis:")
                logger.info(f"  Description: {component.detailed_description}")
            
            logger.info("\n  Key Functions:")
            for func in component.key_features:
                logger.info(f"    • {func}")
            
            if component.unused_functions:
                logger.info("\n  Unused Functions (Potential Dead Code):")
                for func in component.unused_functions:
                    logger.info(f"    • {func}")
            
            if component.complexity:
                logger.info(f"\n  Complexity: {component.complexity}")
            
            if component.external_deps:
                logger.info("\n  External Dependencies:")
                for dep in component.external_deps:
                    logger.info(f"    • {dep}")
            
            logger.info(f"\n  Dependencies:")
            for dep in component.dependencies:
                logger.info(f"    • {dep}")
        logger.info("="*50)
        
    except Exception as e:
        logger.error("Fatal error during analysis", exc_info=True)

if __name__ == "__main__":
    main()
