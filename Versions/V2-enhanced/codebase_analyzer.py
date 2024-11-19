import logging
from datetime import datetime
from typing import Optional, Literal, List
from pathlib import Path
from pydantic import BaseModel
from openai import OpenAI
import time
import json
import colorlog
import argparse
import asyncio
from asyncio import Semaphore
from concurrent.futures import ThreadPoolExecutor
import aiofiles

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

# System-determined metadata models
class FileSystemMetadata(BaseModel):
    """Metadata that can be determined directly from the filesystem"""
    name: str
    path: str
    size_bytes: int
    last_modified: str  # ISO format timestamp

# GPT-analyzed metadata models
class FileAnalysis(BaseModel):
    """Metadata that requires analysis/inference"""
    file_type: Literal["config", "core", "utility", "test", "documentation", "unknown"]
    key_functions: List[str]
    public_interfaces: List[str]
    description: str

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

class FileSummary(BaseModel):
    """Simplified summary of a single file"""
    name: str
    type: str
    purpose: str
    key_features: List[str]
    dependencies: List[str]
    importance_score: int  # 1-10 based on how many other files depend on it

class DetailedFileSummary(BaseModel):
    """Enhanced summary of a single file"""
    name: str
    type: str
    purpose: str
    key_features: List[str]
    dependencies: List[str]
    importance_score: int
    lines_of_code: int
    complexity_score: int  # Based on number of functions/classes and dependencies
    last_modified: str
    provides_context_for: List[str]
    key_classes: List[str]
    key_functions: List[str]
    public_interfaces: List[str]
    notable_patterns: List[str]
    potential_issues: List[str]

class DirectoryStructure(BaseModel):
    """Represents the logical structure of the codebase"""
    core_files: List[str] = []
    utilities: List[str] = []
    config_files: List[str] = []
    test_files: List[str] = []
    documentation: List[str] = []

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
    detailed_components: List[DetailedFileSummary]  # Add this field

class RateLimiter:
    """Rate limiter for API calls"""
    def __init__(self, calls_per_second: float = 1.0):
        self.calls_per_second = calls_per_second
        self.semaphore = Semaphore(1)
        self.last_call = 0

    async def __aenter__(self):
        async with self.semaphore:
            now = time.time()
            time_passed = now - self.last_call
            if time_passed < 1.0 / self.calls_per_second:
                await asyncio.sleep(1.0 / self.calls_per_second - time_passed)
            self.last_call = time.time()

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass

class CodebaseAnalyzer:
    def __init__(self, api_key: Optional[str] = None, max_concurrent: int = 3):
        self.client = OpenAI(api_key=api_key)
        self.logger = logging.getLogger(__name__)
        self.rate_limiter = RateLimiter()
        self.max_concurrent = max_concurrent
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent)

    async def read_file_content_async(self, file_path: Path) -> Optional[str]:
        """Asynchronously read file content"""
        for encoding in ['utf-8', 'latin-1', 'cp1252']:
            try:
                async with aiofiles.open(file_path, 'r', encoding=encoding) as f:
                    return await f.read()
            except UnicodeDecodeError:
                continue
        self.logger.error(f"Failed to read file: {file_path}")
        return None

    async def analyze_file_content_async(self, content: str, file_path: str) -> FileAnalysis:
        """Enhanced asynchronous file content analysis"""
        async with self.rate_limiter:
            try:
                prompt = """Analyze this Python file in detail and determine:
                1. Its logical file type (config/core/utility/test/documentation)
                2. All classes and their purposes
                3. All public methods/functions and their roles
                4. Key design patterns used
                5. Notable implementation details
                6. A comprehensive description of its purpose and role
                7. Any potential areas for improvement"""
                
                loop = asyncio.get_event_loop()
                completion = await loop.run_in_executor(
                    self.executor,
                    lambda: self.client.beta.chat.completions.parse(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": prompt
                            },
                            {
                                "role": "user",
                                "content": f"File path: {file_path}\n\nContent:\n{content}"
                            }
                        ],
                        response_format=FileAnalysis
                    )
                )
                return completion.choices[0].message.parsed
            except Exception as e:
                self.logger.error(f"Error analyzing {Path(file_path).name}: {str(e)}")
                return FileAnalysis(
                    file_type="unknown",
                    key_functions=[],
                    public_interfaces=[],
                    description=f"Error analyzing file: {str(e)}"
                )

    async def analyze_dependencies_async(self, content: str, file_path: str, all_files: List[str]) -> List[DependencyReference]:
        """Asynchronous version of analyze_dependencies"""
        async with self.rate_limiter:
            try:
                loop = asyncio.get_event_loop()
                completion = await loop.run_in_executor(
                    self.executor,
                    lambda: self.client.beta.chat.completions.parse(
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
                )
                dependencies = completion.choices[0].message.parsed.dependencies
                # Remove self-references
                dependencies = [dep for dep in dependencies if dep.target_path != file_path]
                return dependencies
            except Exception as e:
                self.logger.error(f"Error analyzing dependencies for {Path(file_path).name}: {str(e)}")
                return []

    async def process_file(self, file_path: Path, all_file_paths: List[str]) -> Optional[FileNode]:
        """Process a single file asynchronously"""
        try:
            content = await self.read_file_content_async(file_path)
            if content is None:
                return None

            system_metadata = self.get_system_metadata(file_path)
            analysis = await self.analyze_file_content_async(content, str(file_path))
            dependencies = await self.analyze_dependencies_async(content, str(file_path), all_file_paths)

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
        """Asynchronous version of _analyze_codebase_detailed"""
        start_time = time.time()
        self.logger.info(f"Beginning analysis of codebase at: {directory_path}")
        
        dir_path = Path(directory_path)
        if not dir_path.is_dir():
            raise ValueError(f"Directory not found: {directory_path}")

        # Collect Python files
        python_files = list(dir_path.rglob("*.py"))
        all_file_paths = [str(f) for f in python_files]
        
        # Process files concurrently
        tasks = [self.process_file(f, all_file_paths) for f in python_files]
        file_nodes = [node for node in await asyncio.gather(*tasks) if node is not None]

        # Rest of the analysis (processing order, relationships) remains synchronous
        processing_order = self.determine_processing_order(file_nodes)
        self._update_context_relationships(file_nodes)

        return CodebaseAnalysis(
            root_directory=str(dir_path),
            total_files=len(file_nodes),
            analysis_timestamp=datetime.now().isoformat(),
            files=file_nodes,
            processing_order=processing_order
        )

    async def analyze_codebase_async(self, directory_path: str) -> DirectorySummary:
        """Asynchronous version of analyze_codebase"""
        detailed_analysis = await self._analyze_codebase_detailed_async(directory_path)
        return self.generate_directory_summary(detailed_analysis)

    def analyze_codebase(self, directory_path: str) -> DirectorySummary:
        """Synchronous wrapper for analyze_codebase_async"""
        return asyncio.run(self.analyze_codebase_async(directory_path))

    def get_system_metadata(self, file_path: Path) -> FileSystemMetadata:
        """Get filesystem-determined metadata"""
        stats = file_path.stat()
        return FileSystemMetadata(
            name=file_path.name,
            path=str(file_path),
            size_bytes=stats.st_size,
            last_modified=datetime.fromtimestamp(stats.st_mtime).isoformat()
        )

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

    def _analyze_file_complexity(self, content: str) -> int:
        """Calculate complexity score based on code structure"""
        try:
            # Simple complexity calculation
            class_count = content.count('class ')
            function_count = content.count('def ')
            import_count = len([l for l in content.split('\n') if l.strip().startswith('import') or l.strip().startswith('from')])
            
            # Weight different factors
            return (class_count * 3) + (function_count * 2) + import_count
        except Exception as e:
            self.logger.error(f"Error calculating complexity: {str(e)}")
            return 0

    def _identify_patterns_and_issues(self, content: str, file_path: str) -> tuple[List[str], List[str]]:
        """Identify code patterns and potential issues"""
        patterns = []
        issues = []
        
        # Common patterns
        if 'async ' in content:
            patterns.append("Uses asynchronous programming")
        if 'class ' in content:
            patterns.append("Object-oriented design")
        if '@property' in content:
            patterns.append("Uses property decorators")
        
        # Potential issues
        if 'global ' in content:
            issues.append("Uses global variables")
        if 'except:' in content:
            issues.append("Contains bare except clauses")
        if content.count('TODO') > 0:
            issues.append(f"Contains {content.count('TODO')} TODO comments")
        
        return patterns, issues

    def generate_directory_summary(self, analysis: CodebaseAnalysis) -> DirectorySummary:
        """Enhanced directory summary generation"""
        # Calculate importance scores based on references
        importance_scores = {
            node.system_metadata.path: len(node.provides_context_for)
            for node in analysis.files
        }

        # Generate file summaries for key components (files with importance >= 2)
        key_components = []
        detailed_components = []
        for node in analysis.files:
            if importance_scores[node.system_metadata.path] >= 2:
                key_components.append(FileSummary(
                    name=node.system_metadata.name,
                    type=node.analysis.file_type if node.analysis else "unknown",
                    purpose=node.analysis.description if node.analysis else "Unknown purpose",
                    key_features=node.analysis.key_functions if node.analysis else [],
                    dependencies=[Path(d.target_path).name for d in node.required_dependencies],
                    importance_score=importance_scores[node.system_metadata.path]
                ))

            content = self.read_file_content(Path(node.system_metadata.path))
            if not content:
                continue

            complexity_score = self._analyze_file_complexity(content)
            patterns, issues = self._identify_patterns_and_issues(content, node.system_metadata.path)
            
            detailed_components.append(DetailedFileSummary(
                name=node.system_metadata.name,
                type=node.analysis.file_type if node.analysis else "unknown",
                purpose=node.analysis.description if node.analysis else "Unknown purpose",
                key_features=node.analysis.key_functions if node.analysis else [],
                dependencies=[Path(d.target_path).name for d in node.required_dependencies],
                importance_score=importance_scores[node.system_metadata.path],
                lines_of_code=len(content.splitlines()),
                complexity_score=complexity_score,
                last_modified=node.system_metadata.last_modified,
                provides_context_for=[Path(d.target_path).name for d in node.provides_context_for],
                key_classes=[f for f in node.analysis.key_functions if f.startswith('class')] if node.analysis else [],
                key_functions=[f for f in node.analysis.key_functions if not f.startswith('class')] if node.analysis else [],
                public_interfaces=node.analysis.public_interfaces if node.analysis else [],
                notable_patterns=patterns,
                potential_issues=issues
            ))

        # Group files by type
        structure = DirectoryStructure()
        for node in analysis.files:
            if node.analysis and node.analysis.file_type:
                file_list = getattr(structure, f"{node.analysis.file_type}_files", None)
                if file_list is not None:
                    file_list.append(node.system_metadata.name)

        # Determine entry points (files with high importance and few dependencies)
        entry_points = [
            node.system_metadata.name for node in analysis.files
            if importance_scores[node.system_metadata.path] >= 2 and len(node.required_dependencies) <= 1
        ]

        # Generate recommendations
        recommendations = self._generate_recommendations(analysis.files)

        return DirectorySummary(
            project_name=Path(analysis.root_directory).name,
            total_files=analysis.total_files,
            main_purpose=self._infer_main_purpose(analysis.files),
            key_components=sorted(key_components, key=lambda x: x.importance_score, reverse=True),
            structure=structure,
            suggested_reading_order=[
                Path(e.file_path).relative_to(analysis.root_directory).as_posix()
                for e in analysis.processing_order.ordered_entries
            ],
            entry_points=entry_points,
            recommendations=recommendations,
            detailed_components=sorted(detailed_components, 
                                    key=lambda x: (x.importance_score, x.complexity_score), 
                                    reverse=True)
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

async def async_main():
    """Asynchronous version of main"""
    logger.info("="*50)
    logger.info("Starting codebase analyzer")
    logger.info("="*50)
    
    args = argparse.Namespace(
        directory=r"C:\Users\wkraf\Documents\Coding\Event_Trader\versions\v7_concurrent_batches",
        output=r"C:\Users\wkraf\Documents\Coding\Event_Trader\versions\v7_concurrent_batches\codebase_analysis.json"
    )
    
    analyzer = CodebaseAnalyzer()
    try:
        analysis = await analyzer.analyze_codebase_async(args.directory)
        # Save analysis to file with a more intuitive name
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Writing directory summary to: {args.output}")
        with open(args.output, 'w') as f:
            f.write(analysis.model_dump_json(indent=2))
        
        # Print a human-readable summary to console
        logger.info("\n" + "="*50)
        logger.info(f"Directory Summary for: {analysis.project_name}")
        logger.info("="*50)
        logger.info(f"\nMain Purpose: {analysis.main_purpose}")
        logger.info("\nKey Components:")
        for component in analysis.key_components:
            logger.info(f"\n  {component.name} ({component.type}):")
            logger.info(f"    Purpose: {component.purpose}")
            logger.info(f"    Key Features: {', '.join(component.key_features)}")
        logger.info("\nSuggested Reading Order:")
        for idx, file in enumerate(analysis.suggested_reading_order, 1):
            logger.info(f"  {idx}. {file}")
        logger.info("\nRecommendations:")
        for rec in analysis.recommendations:
            logger.info(f"  • {rec}")
        logger.info("="*50)

        # Enhanced console output
        logger.info("\nDetailed Component Analysis:")
        for component in analysis.detailed_components:
            logger.info(f"\n{'-'*40}")
            logger.info(f"File: {component.name} ({component.type})")
            logger.info(f"Purpose: {component.purpose}")
            logger.info(f"Complexity Score: {component.complexity_score}")
            logger.info(f"Lines of Code: {component.lines_of_code}")
            logger.info(f"Last Modified: {component.last_modified}")
            
            if component.key_classes:
                logger.info("\nKey Classes:")
                for cls in component.key_classes:
                    logger.info(f"  • {cls}")
            
            if component.key_functions:
                logger.info("\nKey Functions:")
                for func in component.key_functions:
                    logger.info(f"  • {func}")
            
            if component.notable_patterns:
                logger.info("\nNotable Patterns:")
                for pattern in component.notable_patterns:
                    logger.info(f"  • {pattern}")
            
            if component.potential_issues:
                logger.info("\nPotential Issues:")
                for issue in component.potential_issues:
                    logger.info(f"  • {issue}")
        logger.info("="*50)
        
    except Exception as e:
        logger.error("Fatal error during analysis", exc_info=True)

def main():
    """Entry point that runs the async main"""
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
