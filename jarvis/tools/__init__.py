"""Tools the local LLM is allowed to call."""

from .apps import AppLauncher, AppToolError, MediaController
from .browser import BrowserAgentTool, BrowserToolError
from .registry import ToolBox
from .shell import ShellRunner, ShellToolError
from .websearch import SearchResult, WebSearch, WebSearchError

__all__ = [
    "AppLauncher",
    "AppToolError",
    "BrowserAgentTool",
    "BrowserToolError",
    "MediaController",
    "SearchResult",
    "ShellRunner",
    "ShellToolError",
    "ToolBox",
    "WebSearch",
    "WebSearchError",
]
