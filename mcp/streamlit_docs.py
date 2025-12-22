"""
Streamlit Documentation MCP Server

This MCP server provides tools for AI agents to query Streamlit documentation.
It helps agents understand Streamlit concepts, API usage, and deployment.
"""

from fastmcp import FastMCP

mcp = FastMCP("Streamlit Documentation")

# Streamlit documentation sections
DOCS_SECTIONS = {
    # Get Started
    "installation": {
        "url": "https://docs.streamlit.io/get-started/installation",
        "description": "Installing Streamlit and setting up your environment",
        "category": "get_started",
    },
    "fundamentals": {
        "url": "https://docs.streamlit.io/get-started/fundamentals",
        "description": "Core concepts and fundamentals of Streamlit",
        "category": "get_started",
    },
    "first_steps": {
        "url": "https://docs.streamlit.io/get-started/first-steps",
        "description": "Creating your first Streamlit app",
        "category": "get_started",
    },
    # Develop - Concepts
    "concepts": {
        "url": "https://docs.streamlit.io/develop/concepts",
        "description": "Core concepts for developing Streamlit apps",
        "category": "develop",
    },
    "architecture": {
        "url": "https://docs.streamlit.io/develop/concepts/architecture",
        "description": "Understanding Streamlit's architecture and execution model",
        "category": "develop",
    },
    "app_model": {
        "url": "https://docs.streamlit.io/develop/concepts/architecture/app-model",
        "description": "How Streamlit apps work - reruns, state, and caching",
        "category": "develop",
    },
    "session_state": {
        "url": "https://docs.streamlit.io/develop/concepts/architecture/session-state",
        "description": "Session State - maintaining state across reruns",
        "category": "develop",
    },
    "caching": {
        "url": "https://docs.streamlit.io/develop/concepts/architecture/caching",
        "description": "Caching - @st.cache_data and @st.cache_resource",
        "category": "develop",
    },
    # Develop - API Reference
    "api_reference": {
        "url": "https://docs.streamlit.io/develop/api-reference",
        "description": "Complete API reference for all Streamlit commands",
        "category": "develop",
    },
    "write_magic": {
        "url": "https://docs.streamlit.io/develop/api-reference/write-magic",
        "description": "Write and magic commands - st.write() and magic",
        "category": "develop",
    },
    "text_elements": {
        "url": "https://docs.streamlit.io/develop/api-reference/text",
        "description": "Text elements - st.title, st.header, st.text, st.markdown",
        "category": "develop",
    },
    "data_elements": {
        "url": "https://docs.streamlit.io/develop/api-reference/data",
        "description": "Data display - st.dataframe, st.table, st.json, st.metric",
        "category": "develop",
    },
    "charts": {
        "url": "https://docs.streamlit.io/develop/api-reference/charts",
        "description": "Chart elements - st.line_chart, st.bar_chart, st.pyplot, st.plotly_chart",
        "category": "develop",
    },
    "widgets": {
        "url": "https://docs.streamlit.io/develop/api-reference/widgets",
        "description": "Input widgets - st.button, st.slider, st.text_input, st.selectbox",
        "category": "develop",
    },
    "layout": {
        "url": "https://docs.streamlit.io/develop/api-reference/layout",
        "description": "Layout - st.columns, st.sidebar, st.tabs, st.expander",
        "category": "develop",
    },
    "media": {
        "url": "https://docs.streamlit.io/develop/api-reference/media",
        "description": "Media elements - st.image, st.audio, st.video",
        "category": "develop",
    },
    "status": {
        "url": "https://docs.streamlit.io/develop/api-reference/status",
        "description": "Status elements - st.progress, st.spinner, st.success, st.error",
        "category": "develop",
    },
    # Develop - Tutorials
    "tutorials": {
        "url": "https://docs.streamlit.io/develop/tutorials",
        "description": "Step-by-step tutorials for building Streamlit apps",
        "category": "develop",
    },
    # Develop - Quick Reference
    "quick_reference": {
        "url": "https://docs.streamlit.io/develop/quick-reference",
        "description": "Cheat sheet of commonly used Streamlit commands",
        "category": "develop",
    },
    # Deploy
    "deploy_concepts": {
        "url": "https://docs.streamlit.io/deploy/concepts",
        "description": "Deployment concepts and considerations",
        "category": "deploy",
    },
    "community_cloud": {
        "url": "https://docs.streamlit.io/deploy/streamlit-community-cloud",
        "description": "Deploying to Streamlit Community Cloud (free hosting)",
        "category": "deploy",
    },
    "snowflake": {
        "url": "https://docs.streamlit.io/deploy/snowflake",
        "description": "Deploying Streamlit in Snowflake",
        "category": "deploy",
    },
    "other_platforms": {
        "url": "https://docs.streamlit.io/deploy/other-platforms",
        "description": "Deploying to other platforms (Docker, Kubernetes, cloud providers)",
        "category": "deploy",
    },
    # Knowledge Base
    "faq": {
        "url": "https://docs.streamlit.io/knowledge-base/faq",
        "description": "Frequently Asked Questions",
        "category": "knowledge_base",
    },
    "dependencies": {
        "url": "https://docs.streamlit.io/knowledge-base/dependencies",
        "description": "Installing and managing dependencies",
        "category": "knowledge_base",
    },
    "deployment_issues": {
        "url": "https://docs.streamlit.io/knowledge-base/deploy",
        "description": "Troubleshooting deployment issues",
        "category": "knowledge_base",
    },
}

# Common topics with their direct URLs
COMMON_TOPICS = {
    "dataframe": {
        "url": "https://docs.streamlit.io/develop/api-reference/data/st.dataframe",
        "description": "Display interactive dataframes with st.dataframe()",
    },
    "cache_data": {
        "url": "https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_data",
        "description": "@st.cache_data decorator for caching data computations",
    },
    "cache_resource": {
        "url": "https://docs.streamlit.io/develop/api-reference/caching-and-state/st.cache_resource",
        "description": "@st.cache_resource decorator for caching global resources",
    },
    "button": {
        "url": "https://docs.streamlit.io/develop/api-reference/widgets/st.button",
        "description": "Display a button widget with st.button()",
    },
    "selectbox": {
        "url": "https://docs.streamlit.io/develop/api-reference/widgets/st.selectbox",
        "description": "Display a select dropdown with st.selectbox()",
    },
    "multiselect": {
        "url": "https://docs.streamlit.io/develop/api-reference/widgets/st.multiselect",
        "description": "Display a multiselect dropdown with st.multiselect()",
    },
    "slider": {
        "url": "https://docs.streamlit.io/develop/api-reference/widgets/st.slider",
        "description": "Display a slider widget with st.slider()",
    },
    "text_input": {
        "url": "https://docs.streamlit.io/develop/api-reference/widgets/st.text_input",
        "description": "Display a text input with st.text_input()",
    },
    "columns": {
        "url": "https://docs.streamlit.io/develop/api-reference/layout/st.columns",
        "description": "Create columns layout with st.columns()",
    },
    "sidebar": {
        "url": "https://docs.streamlit.io/develop/api-reference/layout/st.sidebar",
        "description": "Add widgets to sidebar with st.sidebar",
    },
    "tabs": {
        "url": "https://docs.streamlit.io/develop/api-reference/layout/st.tabs",
        "description": "Create tabs with st.tabs()",
    },
    "plotly_chart": {
        "url": "https://docs.streamlit.io/develop/api-reference/charts/st.plotly_chart",
        "description": "Display Plotly charts with st.plotly_chart()",
    },
}


@mcp.tool()
async def get_streamlit_overview() -> str:
    """
    Get an overview of Streamlit and its capabilities.

    Returns:
        Overview of Streamlit, its purpose, and key features
    """
    return """Streamlit Documentation Overview
    Documentation: https://docs.streamlit.io/

    Streamlit is an open-source Python framework for building data apps quickly.
    It turns Python scripts into interactive web apps without requiring frontend experience.

    Key Features:
    - Pure Python - no HTML, CSS, or JavaScript required
    - Instant updates - see changes as you save your script
    - Interactive widgets - buttons, sliders, text inputs, etc.
    - Built-in caching - optimize performance with @st.cache_data
    - Easy deployment - free hosting on Streamlit Community Cloud
    - Rich data visualization - supports Plotly, Matplotlib, Altair, etc.

    Core Concepts:
    - Reruns: App reruns from top to bottom on every interaction
    - Session State: Maintain state across reruns with st.session_state
    - Caching: Use @st.cache_data and @st.cache_resource for performance
    - Widgets: Interactive elements that capture user input
    - Layouts: Organize content with columns, sidebars, tabs, and expanders

    Basic Example:
    ```python
    import streamlit as st
    import pandas as pd

    st.title("My First Streamlit App")

    # Add a slider
    age = st.slider("Select your age", 0, 100, 25)
    st.write(f"You selected: {age}")

    # Display a dataframe
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    st.dataframe(df)
    ```

    Use get_streamlit_docs() to explore specific documentation sections."""


@mcp.tool()
async def get_streamlit_docs(section: str) -> str:
    """
    Get documentation for a specific Streamlit section or topic.

    Args:
        section: The documentation section - examples:
                 'installation', 'fundamentals', 'first_steps', 'concepts',
                 'api_reference', 'session_state', 'caching', 'widgets',
                 'charts', 'layout', 'tutorials', 'community_cloud', 'faq'

    Returns:
        URL and description for the requested documentation section

    Examples:
        - get_streamlit_docs('session_state') - Learn about Session State
        - get_streamlit_docs('caching') - Caching with @st.cache_data
        - get_streamlit_docs('widgets') - Input widgets reference
    """
    section_lower = section.lower()

    # Check if it's a common topic first
    if section_lower in COMMON_TOPICS:
        topic = COMMON_TOPICS[section_lower]
        return f"""Streamlit Documentation: {section}

{topic["description"]}

Documentation URL: {topic["url"]}

Visit the URL for detailed documentation including parameters, examples, and usage patterns."""

    # Check main docs sections
    if section_lower not in DOCS_SECTIONS:
        available_sections = [k for k in DOCS_SECTIONS.keys()]
        available_topics = [k for k in COMMON_TOPICS.keys()]
        sections_str = ", ".join(sorted(available_sections[:15])) + "..."
        topics_str = ", ".join(sorted(available_topics[:10])) + "..."
        return f"Section '{section}' not found.\n\nSections: {sections_str}\n\nCommon topics: {topics_str}\n\nUse list_streamlit_sections() for the full list."

    doc = DOCS_SECTIONS[section_lower]
    return f"""Streamlit Documentation: {section}

{doc["description"]}

Category: {doc["category"]}
Documentation URL: {doc["url"]}

Visit the URL for complete documentation including examples, code samples, and detailed explanations."""


@mcp.tool()
async def list_streamlit_sections() -> str:
    """
    List all available Streamlit documentation sections organized by category.

    Returns:
        Organized list of all documentation sections and common topics
    """
    result = "Streamlit Documentation Sections:\n\n"

    # Group by category
    categories = {
        "get_started": "GET STARTED",
        "develop": "DEVELOP",
        "deploy": "DEPLOY",
        "knowledge_base": "KNOWLEDGE BASE",
    }

    for cat_key, cat_name in categories.items():
        result += f"=== {cat_name} ===\n\n"
        for section, info in sorted(DOCS_SECTIONS.items()):
            if info["category"] == cat_key:
                result += f"  {section}:\n    {info['description']}\n\n"

    result += "\n=== COMMON API TOPICS ===\n\n"
    for topic, info in sorted(COMMON_TOPICS.items()):
        result += f"  {topic}:\n    {info['description']}\n\n"

    result += "\nUse get_streamlit_docs('<section>') for detailed documentation."
    return result


@mcp.tool()
async def search_streamlit_docs(query: str) -> str:
    """
    Search for Streamlit documentation by keyword.

    Args:
        query: Search term (e.g., 'cache', 'widget', 'dataframe', 'deploy', 'chart')

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
                    "category": info["category"],
                    "url": info["url"],
                }
            )

    # Search in common topics
    for topic, info in COMMON_TOPICS.items():
        if query_lower in topic.lower() or query_lower in info["description"].lower():
            matches.append(
                {
                    "name": topic,
                    "type": "API Topic",
                    "description": info["description"],
                    "category": "api",
                    "url": info["url"],
                }
            )

    if not matches:
        return f"No documentation found matching '{query}'. Use list_streamlit_sections() to see all available sections."

    result = f"Streamlit documentation matching '{query}':\n\n"
    for match in matches:
        result += f"  [{match['type']}] {match['name']}:\n"
        result += f"    {match['description']}\n"
        result += f"    URL: {match['url']}\n\n"

    return result


@mcp.tool()
async def get_streamlit_caching_guide() -> str:
    """
    Get detailed guide on caching in Streamlit.

    Returns:
        Comprehensive guide on using @st.cache_data and @st.cache_resource
    """
    return """Streamlit Caching Guide
    Documentation: https://docs.streamlit.io/develop/concepts/architecture/caching

    Streamlit provides two caching decorators for optimizing performance:

    1. @st.cache_data - For caching data computations
    ----------------------------------------------
    Use for functions that return serializable data (DataFrames, lists, dicts, etc.)

    ```python
    import streamlit as st
    import pandas as pd

    @st.cache_data
    def load_data():
        # Expensive data loading
        df = pd.read_csv("large_dataset.csv")
        return df

    df = load_data()  # Cached after first run
    st.dataframe(df)
    ```

    Best for:
    - Loading data from files or databases
    - Data transformations and computations
    - API calls that return data
    - Machine learning predictions (outputs only)

    2. @st.cache_resource - For caching global resources
    ---------------------------------------------------
    Use for functions that return non-serializable objects (connections, models, sessions)

    ```python
    import streamlit as st
    from transformers import pipeline

    @st.cache_resource
    def load_model():
        # Load ML model (only once)
        return pipeline("sentiment-analysis")

    model = load_model()  # Cached, reused across users
    result = model("I love Streamlit!")
    st.write(result)
    ```

    Best for:
    - Database connections
    - ML model objects
    - TensorFlow/PyTorch sessions
    - Any non-serializable global resource

    Key Differences:
    - cache_data: Creates a new copy for each user (isolated)
    - cache_resource: Shares same object across all users (global)

    Cache Parameters:
    - ttl: Time to live (e.g., ttl=3600 for 1 hour)
    - max_entries: Max number of cached entries
    - show_spinner: Show loading spinner (default: True)

    Clearing Cache:
    - st.cache_data.clear() - Clear all data cache
    - st.cache_resource.clear() - Clear all resource cache
    - Add button: st.button("Clear cache", on_click=st.cache_data.clear)

    For more details: https://docs.streamlit.io/develop/concepts/architecture/caching"""


@mcp.tool()
async def get_streamlit_session_state_guide() -> str:
    """
    Get detailed guide on Session State in Streamlit.

    Returns:
        Comprehensive guide on using st.session_state
    """
    return """Streamlit Session State Guide
    Documentation: https://docs.streamlit.io/develop/concepts/architecture/session-state

    Session State allows you to maintain variables across reruns (interactions).

    Basic Usage:
    ------------
    ```python
    import streamlit as st

    # Initialize state
    if "counter" not in st.session_state:
        st.session_state.counter = 0

    # Increment counter
    def increment():
        st.session_state.counter += 1

    st.button("Increment", on_click=increment)
    st.write(f"Counter: {st.session_state.counter}")
    ```

    Access Methods:
    ---------------
    1. Dictionary-style: st.session_state["key"]
    2. Attribute-style: st.session_state.key

    Common Patterns:
    ----------------

    1. Form State Management:
    ```python
    if "form_submitted" not in st.session_state:
        st.session_state.form_submitted = False

    if st.button("Submit"):
        st.session_state.form_submitted = True
    ```

    2. Multi-page Navigation:
    ```python
    if "page" not in st.session_state:
        st.session_state.page = "home"

    if st.button("Go to Settings"):
        st.session_state.page = "settings"
    ```

    3. User Authentication:
    ```python
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if st.session_state.logged_in:
        st.write("Welcome back!")
    else:
        # Show login form
    ```

    Widget Callbacks:
    -----------------
    Associate state changes with widget interactions:

    ```python
    def update_name():
        st.session_state.greeting = f"Hello, {st.session_state.name}!"

    st.text_input("Name", key="name", on_change=update_name)

    if "greeting" in st.session_state:
        st.write(st.session_state.greeting)
    ```

    Key Points:
    - State persists across reruns for the same user session
    - Each user has their own session state
    - State is lost when page is refreshed or browser closed
    - Use callbacks for immediate state updates

    For more details: https://docs.streamlit.io/develop/concepts/architecture/session-state"""


if __name__ == "__main__":
    # Run the MCP server
    mcp.run()
