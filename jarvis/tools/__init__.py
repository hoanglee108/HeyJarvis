"""Tools the local LLM is allowed to call."""

from .apps import AppLauncher, AppToolError, MediaController
from .browser import BrowserAgentTool, BrowserToolError
from .clock import Clock, ClockToolError
from .music import MusicPlayer, MusicToolError
from .registry import ToolBox
from .shell import ShellRunner, ShellToolError
from .websearch import SearchResult, WebSearch, WebSearchError

__all__ = [
    "AppLauncher",
    "AppToolError",
    "BrowserAgentTool",
    "BrowserToolError",
    "Clock",
    "ClockToolError",
    "MediaController",
    "MusicPlayer",
    "MusicToolError",
    "SearchResult",
    "ShellRunner",
    "ShellToolError",
    "ToolBox",
    "WebSearch",
    "WebSearchError",
]
