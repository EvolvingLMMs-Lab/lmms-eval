import asyncio
import json
from typing import Union

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import AudioContent, ImageContent, TextContent
from openai import OpenAI


class MCPClient:
    def __init__(self, server_path: str):
        """
        Initialize the MCPClient with the path to the MCP server.
        """
        self.server_path = server_path

    async def get_function_list(self):
        """
        Connect to the MCP server and retrieve the list of available functions.
        """
        server_params = StdioServerParameters(command="python", args=[self.server_path])
        async with stdio_client(server=server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                tools = (await session.list_tools()).tools

                functions = []
                for tool in tools:
                    functions.append({"type": "function", "function": {"name": tool.name, "description": tool.description or "", "parameters": tool.inputSchema}})
                return functions

    async def run_tool(self, tool_name: str, tool_args: dict):
        """
        Run a specific tool with the given arguments.
        :param tool_name: Name of the tool to run.
        :param tool_args: Arguments for the tool.
        :return: Result of the tool execution.
        """
        server_params = StdioServerParameters(command="python", args=[self.server_path])
        async with stdio_client(server=server_params) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()

                result = await session.call_tool(tool_name, tool_args)
                return result

    def convert_result_to_openai_format(self, result: Union[ImageContent, TextContent, AudioContent]) -> dict:
        """
        Convert the result from the MCP tool to OpenAI compatible format.
        :param result: Result from the MCP tool.
        :return: Converted result.
        """
        if isinstance(result, ImageContent):
            return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{result.data}"}}
        elif isinstance(result, TextContent):
            return {"type": "text", "text": result.data}
        elif isinstance(result, AudioContent):
            return {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{result.data}"}}
        else:
            raise ValueError("Unsupported result type")
