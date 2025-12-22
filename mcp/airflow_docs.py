"""
Apache Airflow 3.x Documentation MCP Server

This MCP server provides tools for AI agents to query Apache Airflow 3 documentation.
It helps agents understand Airflow concepts, best practices, and API usage.
"""

from fastmcp import FastMCP

mcp = FastMCP("Apache Airflow 3.x Documentation")

# Airflow 3.x documentation sections
DOCS_SECTIONS = {
    "overview": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/index.html",
        "description": "Overview of Apache Airflow and its capabilities",
    },
    "quick_start": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/start.html",
        "description": "Quick Start guide to get Airflow running",
    },
    "installation": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/installation/",
        "description": "Installation of Airflow - installation methods and requirements",
    },
    "security": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/security/",
        "description": "Security features and best practices",
    },
    "tutorials": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/tutorial/",
        "description": "Step-by-step tutorials for learning Airflow",
    },
    "howto_guides": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/howto/",
        "description": "How-to Guides for common tasks and patterns",
    },
    "ui_overview": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/ui.html",
        "description": "UI Overview - understanding the Airflow web interface",
    },
    "core_concepts": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/",
        "description": "Core Concepts - DAGs, Tasks, Operators, Sensors, etc.",
    },
    "authoring_scheduling": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/",
        "description": "Authoring and Scheduling DAGs - TaskFlow API, dependencies, scheduling",
    },
    "administration": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/administration-and-deployment/",
        "description": "Administration and Deployment - configuration, scaling, monitoring",
    },
    "integration": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/integration.html",
        "description": "Integration with external systems and services",
    },
    "public_interface": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/public-airflow-interface.html",
        "description": "Public Interface for Airflow 3.0+ - stable APIs",
    },
    "best_practices": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/best-practices.html",
        "description": "Best Practices for writing DAGs and managing workflows",
    },
    "faq": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/faq.html",
        "description": "Frequently Asked Questions",
    },
    "troubleshooting": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/troubleshooting.html",
        "description": "Troubleshooting common issues",
    },
    "release_policies": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/release_process.html",
        "description": "Release Policies and versioning",
    },
    "release_notes": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/release_notes.html",
        "description": "Release Notes for all versions",
    },
}

# Reference documentation
REFERENCE_DOCS = {
    "operators_hooks": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/operators-and-hooks-ref.html",
        "description": "Operators and Hooks reference - built-in operators and hooks",
    },
    "cli": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/cli-and-env-variables-ref.html",
        "description": "CLI commands and environment variables",
    },
    "templates": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/templates-ref.html",
        "description": "Templates reference - Jinja templating in Airflow",
    },
    "rest_api": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/stable-rest-api-ref.html",
        "description": "REST API reference for programmatic access",
    },
    "configurations": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/configurations-ref.html",
        "description": "Configuration options reference",
    },
    "extra_packages": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/extra-packages-ref.html",
        "description": "Extra packages and provider packages",
    },
}

# Common topics and their locations
COMMON_TOPICS = {
    "taskflow_api": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/tutorial/taskflow.html",
        "description": "TaskFlow API - modern way to write DAGs using Python decorators (@task, @dag)",
        "category": "authoring",
    },
    "dags": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/dags.html",
        "description": "DAGs - Directed Acyclic Graphs definition and properties",
        "category": "core_concepts",
    },
    "tasks": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/tasks.html",
        "description": "Tasks - units of work in Airflow",
        "category": "core_concepts",
    },
    "operators": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/operators.html",
        "description": "Operators - predefined task templates",
        "category": "core_concepts",
    },
    "xcoms": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/xcoms.html",
        "description": "XComs - cross-communication between tasks",
        "category": "core_concepts",
    },
    "connections": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/connections.html",
        "description": "Connections - managing external system credentials",
        "category": "authoring",
    },
    "variables": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/core-concepts/variables.html",
        "description": "Variables - runtime configuration values",
        "category": "core_concepts",
    },
    "scheduling": {
        "url": "https://airflow.apache.org/docs/apache-airflow/stable/authoring-and-scheduling/timetable.html",
        "description": "Scheduling and Timetables - when DAGs run",
        "category": "authoring",
    },
}


@mcp.tool()
async def get_airflow_overview() -> str:
    """
    Get an overview of Apache Airflow 3.x and key concepts.

    Returns:
        Overview of Airflow, its purpose, and architecture
    """
    return """Apache Airflow 3.x Overview
    Documentation: https://airflow.apache.org/docs/apache-airflow/stable/

    Apache Airflow is an open-source platform for developing, scheduling, and monitoring
    batch-oriented workflows. It allows you to define workflows as Directed Acyclic Graphs (DAGs) of tasks.

    Key Features in Airflow 3.x:
    - TaskFlow API: Modern decorator-based DAG authoring (@task, @dag)
    - Dynamic, extensible, and elegant pipelines
    - Scalable execution with various executors (Local, Celery, Kubernetes)
    - Rich UI for monitoring and troubleshooting
    - Extensive integration with external systems via providers
    - Stable public API guarantee for 3.x series

    Core Concepts:
    - DAGs: Directed Acyclic Graphs that define workflow structure
    - Tasks: Units of work (using @task decorator or operators)
    - Operators: Predefined task templates for common operations
    - XComs: Mechanism for passing small amounts of data between tasks
    - Connections: Managed credentials for external systems
    - Scheduling: Cron-based or custom timetables

    Use get_airflow_docs() to explore specific documentation sections."""


@mcp.tool()
async def get_airflow_docs(section: str) -> str:
    """
    Get documentation for a specific Airflow section.

    Args:
        section: The documentation section - one of:
                 'overview', 'quick_start', 'installation', 'security', 'tutorials',
                 'howto_guides', 'ui_overview', 'core_concepts', 'authoring_scheduling',
                 'administration', 'integration', 'public_interface', 'best_practices',
                 'faq', 'troubleshooting', 'release_policies', 'release_notes'

    Returns:
        URL and description for the requested documentation section

    Examples:
        - get_airflow_docs('taskflow_api') - Learn about modern DAG authoring
        - get_airflow_docs('core_concepts') - Understand DAGs, Tasks, Operators
        - get_airflow_docs('best_practices') - Best practices for Airflow
    """
    section_lower = section.lower()

    # Check if it's a common topic first
    if section_lower in COMMON_TOPICS:
        topic = COMMON_TOPICS[section_lower]
        return f"""Apache Airflow Documentation: {section}

{topic["description"]}

Documentation URL: {topic["url"]}
Category: {topic["category"]}

Visit the URL for detailed documentation including examples and usage patterns."""

    # Check main docs sections
    if section_lower not in DOCS_SECTIONS:
        available = ", ".join(sorted(DOCS_SECTIONS.keys()))
        common = ", ".join(sorted(COMMON_TOPICS.keys()))
        return f"Section '{section}' not found.\n\nMain sections: {available}\n\nCommon topics: {common}"

    doc = DOCS_SECTIONS[section_lower]
    return f"""Apache Airflow Documentation: {section}

{doc["description"]}

Documentation URL: {doc["url"]}

Visit the URL for complete documentation including examples, code samples, and detailed explanations."""


@mcp.tool()
async def get_airflow_reference(reference: str) -> str:
    """
    Get reference documentation for Airflow APIs, CLI, etc.

    Args:
        reference: The reference type - one of:
                   'operators_hooks', 'cli', 'templates', 'rest_api',
                   'configurations', 'extra_packages'

    Returns:
        URL and description for the requested reference documentation

    Examples:
        - get_airflow_reference('operators_hooks') - Built-in operators and hooks
        - get_airflow_reference('cli') - CLI commands
        - get_airflow_reference('rest_api') - REST API for programmatic access
    """
    ref_lower = reference.lower()

    if ref_lower not in REFERENCE_DOCS:
        available = ", ".join(sorted(REFERENCE_DOCS.keys()))
        return f"Reference '{reference}' not found. Available references: {available}"

    ref = REFERENCE_DOCS[ref_lower]
    return f"""Apache Airflow Reference: {reference}

{ref["description"]}

Reference URL: {ref["url"]}

This reference documentation provides detailed technical specifications."""


@mcp.tool()
async def list_airflow_sections() -> str:
    """
    List all available Airflow documentation sections.

    Returns:
        Organized list of all documentation sections, references, and common topics
    """
    result = "Apache Airflow 3.x Documentation Sections:\n\n"
    result += "=== MAIN DOCUMENTATION ===\n\n"

    for section, info in sorted(DOCS_SECTIONS.items()):
        result += f"  {section}:\n    {info['description']}\n\n"

    result += "\n=== REFERENCE DOCUMENTATION ===\n\n"

    for ref, info in sorted(REFERENCE_DOCS.items()):
        result += f"  {ref}:\n    {info['description']}\n\n"

    result += "\n=== COMMON TOPICS ===\n\n"

    for topic, info in sorted(COMMON_TOPICS.items()):
        result += f"  {topic}:\n    {info['description']}\n\n"

    result += "\nUse get_airflow_docs('<section>') or get_airflow_reference('<reference>') for detailed documentation."
    return result


@mcp.tool()
async def search_airflow_docs(query: str) -> str:
    """
    Search for Airflow documentation by keyword.

    Args:
        query: Search term (e.g., 'task', 'dag', 'operator', 'schedule', 'xcom')

    Returns:
        List of matching documentation sections and topics
    """
    query_lower = query.lower()
    matches = []

    # Search in main docs
    for section, info in DOCS_SECTIONS.items():
        if query_lower in section.lower() or query_lower in info["description"].lower():
            matches.append(
                {
                    "name": section,
                    "type": "Documentation",
                    "description": info["description"],
                    "url": info["url"],
                }
            )

    # Search in references
    for ref, info in REFERENCE_DOCS.items():
        if query_lower in ref.lower() or query_lower in info["description"].lower():
            matches.append(
                {
                    "name": ref,
                    "type": "Reference",
                    "description": info["description"],
                    "url": info["url"],
                }
            )

    # Search in common topics
    for topic, info in COMMON_TOPICS.items():
        if query_lower in topic.lower() or query_lower in info["description"].lower():
            matches.append(
                {
                    "name": topic,
                    "type": "Topic",
                    "description": info["description"],
                    "url": info["url"],
                }
            )

    if not matches:
        return f"No documentation found matching '{query}'. Use list_airflow_sections() to see all available sections."

    result = f"Airflow documentation matching '{query}':\n\n"
    for match in matches:
        result += f"  [{match['type']}] {match['name']}:\n"
        result += f"    {match['description']}\n"
        result += f"    URL: {match['url']}\n\n"

    return result


@mcp.tool()
async def get_taskflow_api_info() -> str:
    """
    Get detailed information about the TaskFlow API in Airflow 3.x.

    Returns:
        Overview of TaskFlow API with examples and best practices
    """
    return """TaskFlow API in Apache Airflow 3.x
    Documentation: https://airflow.apache.org/docs/apache-airflow/stable/tutorial/taskflow.html

    The TaskFlow API is the modern, Pythonic way to author DAGs in Airflow 3.x using decorators.

    Key Decorators:
    - @dag: Defines a DAG
    - @task: Defines a task (replaces PythonOperator)
    - @task.virtualenv: Task in isolated virtualenv
    - @task.docker: Task in Docker container

    Basic Example:
    ```python
    from airflow.decorators import dag, task
    from datetime import datetime

    @dag(
        schedule="@daily",
        start_date=datetime(2025, 1, 1),
        catchup=False,
    )
    def my_workflow():
        @task
        def extract():
            return {"data": "value"}

        @task
        def transform(data: dict):
            return data["data"].upper()

        @task
        def load(value: str):
            print(f"Loading: {value}")

        # Define task dependencies through function calls
        data = extract()
        transformed = transform(data)
        load(transformed)

    my_workflow()
    ```

    Benefits:
    - Automatic XCom handling (return values passed between tasks)
    - Type hints for better IDE support and documentation
    - Cleaner, more readable code
    - Automatic task_id generation from function names
    - Easier testing (functions can be tested independently)

    Best Practices:
    - Pass metadata (paths, IDs) not large data through XComs
    - Use type hints for better documentation
    - Keep tasks small and focused
    - Use multiple task dependencies for parallel execution

    For more details, visit: https://airflow.apache.org/docs/apache-airflow/stable/tutorial/taskflow.html"""


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
