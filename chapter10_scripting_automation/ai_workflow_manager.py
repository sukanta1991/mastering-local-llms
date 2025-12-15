#!/usr/bin/env python3
"""
ai_workflow_manager.py - Advanced AI workflow automation

This module provides comprehensive workflow management for AI tasks including:
- Multi-stage processing pipelines
- Async task execution
- Dependency resolution
- Progress tracking and monitoring

Usage:
    python ai_workflow_manager.py

Author: Book Example
License: MIT
"""

import asyncio
import concurrent.futures
import json
import sqlite3
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Callable, Optional, Union
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager
from enum import Enum

try:
    from ollama_client import OllamaClient, ChatMessage, OllamaAPIError
except ImportError:
    print("Error: ollama_client module not found. Ensure it's in the same directory.")
    raise

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class WorkflowTask:
    """Represents a single workflow task"""
    id: str
    name: str
    model: str
    prompt_template: str
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class WorkflowConfig:
    """Configuration for workflow execution"""
    max_concurrent_tasks: int = 3
    task_timeout: int = 300
    retry_delay: int = 5
    save_intermediate_results: bool = True
    results_directory: str = "workflow_results"

class WorkflowDatabase:
    """SQLite database for workflow persistence"""
    
    def __init__(self, db_path: str = "workflow.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS workflows (
                    id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    config TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id TEXT PRIMARY KEY,
                    workflow_id TEXT NOT NULL,
                    name TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_template TEXT NOT NULL,
                    dependencies TEXT NOT NULL,
                    parameters TEXT NOT NULL,
                    status TEXT NOT NULL,
                    result TEXT,
                    error TEXT,
                    start_time TIMESTAMP,
                    end_time TIMESTAMP,
                    retry_count INTEGER DEFAULT 0,
                    FOREIGN KEY (workflow_id) REFERENCES workflows (id)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS executions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    workflow_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    execution_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    duration_ms INTEGER,
                    tokens_used INTEGER,
                    success BOOLEAN,
                    FOREIGN KEY (workflow_id) REFERENCES workflows (id),
                    FOREIGN KEY (task_id) REFERENCES tasks (id)
                )
            """)
    
    @contextmanager
    def get_connection(self):
        """Get database connection with proper cleanup"""
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
        finally:
            conn.close()
    
    def save_workflow(self, workflow_id: str, name: str, config: WorkflowConfig, status: str):
        """Save workflow metadata"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO workflows (id, name, config, status, updated_at)
                VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (workflow_id, name, json.dumps(asdict(config)), status))
            conn.commit()
    
    def save_task(self, workflow_id: str, task: WorkflowTask):
        """Save task data"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO tasks 
                (id, workflow_id, name, model, prompt_template, dependencies, 
                 parameters, status, result, error, start_time, end_time, retry_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task.id, workflow_id, task.name, task.model, task.prompt_template,
                json.dumps(task.dependencies), json.dumps(task.parameters),
                task.status.value, task.result, task.error,
                task.start_time, task.end_time, task.retry_count
            ))
            conn.commit()
    
    def log_execution(self, workflow_id: str, task_id: str, duration_ms: int, 
                     tokens_used: int, success: bool):
        """Log task execution metrics"""
        with self.get_connection() as conn:
            conn.execute("""
                INSERT INTO executions 
                (workflow_id, task_id, duration_ms, tokens_used, success)
                VALUES (?, ?, ?, ?, ?)
            """, (workflow_id, task_id, duration_ms, tokens_used, success))
            conn.commit()

class AIWorkflowManager:
    """
    Advanced workflow manager for AI task automation
    
    Features:
    - Dependency resolution
    - Parallel execution
    - Error handling and retries
    - Progress monitoring
    - Result persistence
    """
    
    def __init__(self, client: OllamaClient, config: WorkflowConfig = None):
        self.client = client
        self.config = config or WorkflowConfig()
        self.db = WorkflowDatabase()
        self.tasks: Dict[str, WorkflowTask] = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=self.config.max_concurrent_tasks
        )
        
        # Create results directory
        Path(self.config.results_directory).mkdir(exist_ok=True)
    
    def add_task(self, task: WorkflowTask) -> None:
        """Add a task to the workflow"""
        self.tasks[task.id] = task
        print(f"Added task: {task.name} (id: {task.id})")
    
    def create_content_pipeline(self) -> List[WorkflowTask]:
        """Create a sample content creation pipeline"""
        tasks = [
            WorkflowTask(
                id="research",
                name="Research Topic",
                model="llama3.2:3b",
                prompt_template="Research and provide key information about: {topic}. Include recent developments and important facts.",
                parameters={"temperature": 0.3}
            ),
            WorkflowTask(
                id="outline",
                name="Create Outline",
                model="llama3.2:3b",
                prompt_template="Based on this research: {research_result}\n\nCreate a detailed outline for a blog post about {topic}.",
                dependencies=["research"],
                parameters={"temperature": 0.5}
            ),
            WorkflowTask(
                id="draft",
                name="Write Draft",
                model="llama3.2:3b",
                prompt_template="Using this outline: {outline_result}\n\nWrite a comprehensive blog post about {topic}. Make it engaging and informative.",
                dependencies=["outline"],
                parameters={"temperature": 0.7}
            ),
            WorkflowTask(
                id="edit",
                name="Edit Content",
                model="llama3.2:3b",
                prompt_template="Edit and improve this blog post: {draft_result}\n\nFocus on clarity, flow, and readability.",
                dependencies=["draft"],
                parameters={"temperature": 0.3}
            ),
            WorkflowTask(
                id="meta",
                name="Generate Metadata",
                model="llama3.2:3b",
                prompt_template="For this blog post: {edit_result}\n\nGenerate SEO metadata including title, description, and tags.",
                dependencies=["edit"],
                parameters={"temperature": 0.2}
            )
        ]
        
        return tasks
    
    def resolve_dependencies(self) -> List[List[str]]:
        """Resolve task dependencies and create execution order"""
        # Simple topological sort
        execution_levels = []
        remaining_tasks = set(self.tasks.keys())
        
        while remaining_tasks:
            current_level = []
            
            for task_id in list(remaining_tasks):
                task = self.tasks[task_id]
                dependencies_met = all(
                    dep not in remaining_tasks for dep in task.dependencies
                )
                
                if dependencies_met:
                    current_level.append(task_id)
            
            if not current_level:
                raise ValueError("Circular dependency detected")
            
            execution_levels.append(current_level)
            remaining_tasks -= set(current_level)
        
        return execution_levels
    
    async def execute_task(self, task: WorkflowTask, context: Dict[str, Any]) -> bool:
        """Execute a single task"""
        task.status = TaskStatus.RUNNING
        task.start_time = datetime.now()
        
        try:
            # Build prompt from template
            prompt = self._build_prompt(task.prompt_template, context)
            
            print(f"🔄 Executing: {task.name}")
            
            # Execute with AI model
            start_time = time.time()
            
            result = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.client.generate(
                    model=task.model,
                    prompt=prompt,
                    **task.parameters
                )
            )
            
            execution_time = int((time.time() - start_time) * 1000)
            
            # Save result
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.end_time = datetime.now()
            
            # Log execution
            self.db.log_execution(
                workflow_id="default",
                task_id=task.id,
                duration_ms=execution_time,
                tokens_used=len(result.split()),  # Approximate
                success=True
            )
            
            # Save intermediate results if configured
            if self.config.save_intermediate_results:
                self._save_task_result(task)
            
            print(f"✅ Completed: {task.name} ({execution_time}ms)")
            return True
            
        except Exception as e:
            task.error = str(e)
            task.status = TaskStatus.FAILED
            task.end_time = datetime.now()
            
            print(f"❌ Failed: {task.name} - {e}")
            
            # Log failure
            self.db.log_execution(
                workflow_id="default",
                task_id=task.id,
                duration_ms=0,
                tokens_used=0,
                success=False
            )
            
            return False
    
    def _build_prompt(self, template: str, context: Dict[str, Any]) -> str:
        """Build prompt from template and context"""
        # Replace dependency results
        for task_id, task in self.tasks.items():
            if task.result:
                context[f"{task_id}_result"] = task.result
        
        try:
            return template.format(**context)
        except KeyError as e:
            raise ValueError(f"Missing context variable: {e}")
    
    def _save_task_result(self, task: WorkflowTask):
        """Save task result to file"""
        result_file = Path(self.config.results_directory) / f"{task.id}_result.txt"
        with open(result_file, 'w', encoding='utf-8') as f:
            f.write(f"Task: {task.name}\n")
            f.write(f"Model: {task.model}\n")
            f.write(f"Completed: {task.end_time}\n")
            f.write("=" * 50 + "\n")
            f.write(task.result)
    
    async def execute_workflow(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the entire workflow"""
        workflow_id = f"workflow_{int(time.time())}"
        
        print(f"🚀 Starting workflow: {workflow_id}")
        
        # Save workflow config
        self.db.save_workflow(workflow_id, "Content Pipeline", self.config, "running")
        
        try:
            # Resolve execution order
            execution_levels = self.resolve_dependencies()
            
            print(f"📋 Execution plan: {len(execution_levels)} levels")
            
            # Execute tasks level by level
            for level_idx, task_ids in enumerate(execution_levels):
                print(f"\n📊 Level {level_idx + 1}: {len(task_ids)} tasks")
                
                # Execute tasks in parallel within each level
                tasks_to_execute = [self.tasks[task_id] for task_id in task_ids]
                
                results = await asyncio.gather(*[
                    self.execute_task(task, context) for task in tasks_to_execute
                ], return_exceptions=True)
                
                # Check for failures
                for task, result in zip(tasks_to_execute, results):
                    if isinstance(result, Exception) or not result:
                        print(f"❌ Workflow failed at task: {task.name}")
                        return {"status": "failed", "failed_task": task.name}
                    
                    # Save task state
                    self.db.save_task(workflow_id, task)
            
            # Workflow completed successfully
            self.db.save_workflow(workflow_id, "Content Pipeline", self.config, "completed")
            
            results = {
                "status": "completed",
                "workflow_id": workflow_id,
                "results": {task.id: task.result for task in self.tasks.values()}
            }
            
            print("🎉 Workflow completed successfully!")
            return results
            
        except Exception as e:
            print(f"💥 Workflow failed: {e}")
            self.db.save_workflow(workflow_id, "Content Pipeline", self.config, "failed")
            return {"status": "failed", "error": str(e)}
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get current workflow status"""
        status = {
            "total_tasks": len(self.tasks),
            "pending": len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING]),
            "running": len([t for t in self.tasks.values() if t.status == TaskStatus.RUNNING]),
            "completed": len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED]),
            "failed": len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED]),
            "tasks": {
                task_id: {
                    "name": task.name,
                    "status": task.status.value,
                    "progress": "100%" if task.status == TaskStatus.COMPLETED else "0%"
                }
                for task_id, task in self.tasks.items()
            }
        }
        return status
    
    def cleanup(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=True)

async def main():
    """Example workflow execution"""
    # Initialize client
    client = OllamaClient()
    
    if not client.health_check():
        print("❌ Ollama server not available")
        return
    
    # Create workflow manager
    config = WorkflowConfig(
        max_concurrent_tasks=2,
        save_intermediate_results=True
    )
    
    workflow = AIWorkflowManager(client, config)
    
    try:
        # Add content pipeline tasks
        tasks = workflow.create_content_pipeline()
        for task in tasks:
            workflow.add_task(task)
        
        # Execute workflow
        context = {"topic": "The Future of Artificial Intelligence"}
        
        results = await workflow.execute_workflow(context)
        
        # Show final results
        if results['status'] == 'completed':
            print("\n=== Final Blog Post ===")
            print(results['results'].get('edit', 'No final content'))
            print("\n=== Metadata ===")
            print(results['results'].get('meta', 'No metadata'))
        
        # Show status
        status = workflow.get_workflow_status()
        print(f"\n📊 Final Status: {status['completed']}/{status['total_tasks']} completed")
        
    finally:
        workflow.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
