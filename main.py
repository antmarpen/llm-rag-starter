import asyncio
import os

import uvicorn

from integrations.core.manager import IntegrationManager
from servers.api.server import APIServer
from servers.mcp.server import MCPServer
from utils.logger import Logger, LoggerLevel


class Application:

    def __init__(self):
        self.__logger = Logger.get_logger(self.__class__)

        # Printing banner
        banner = [
            "=" * 50,
            "  RAG PoC Server starting",
            "  by antmarpen",
            "  Simple retrieval augmented generation demo",
            "=" * 50,
        ]
        print("\n".join(banner))

        self.__logger.info("Starting RAG PoC Server")

        # API Server
        self.api = APIServer(title="RAG API")

        # MCP Server
        self.mcp = MCPServer()

        # Integration Manager
        self.__logger.info("Initializing integration manager")
        self.integration_manager = IntegrationManager()
        self.integration_manager.register_all()
        self.__logger.info("Integration manager initialized successfully")

    async def __run_integrations(self):
        await self.integration_manager.start_all()

    def run(self):
        asyncio.run(self.__run_integrations())

        host = "0.0.0.0"
        api_port = 5000
        mcp_port = 8000

        use_api = True
        use_mcp = True
        if use_api:
            config = uvicorn.Config(self.api, host=host, port=api_port, log_config=None, reload=debug)
            server = uvicorn.Server(config)
            self.__logger.info("API server started successfully!")
            self.__logger.info(f"API server running on http://{host}:{api_port}")
            asyncio.run(server.serve())

        if use_mcp:
            self.mcp.settings.host = host
            self.mcp.settings.port = mcp_port

            # mcp_transport_protocol = "streamable-http"
            mcp_transport_protocol = "stdio"
            self.__logger.info("MCP server started successfully!")
            if mcp_transport_protocol == "stdio":
                self.__logger.info(f"MCP server running on STDIO")
            elif mcp_transport_protocol in ["streamable-http", "sse"]:
                self.__logger.info(f"MCP server running on http://{host}:{mcp_port}")

            self.mcp.run(transport=mcp_transport_protocol)

debug = os.environ.get("DEBUG", "false").lower() in {"1", "true", "yes"}
if debug:
    Logger.set_level(LoggerLevel.DEBUG)
else:
    Logger.set_level(LoggerLevel.INFO)

_app_instance = Application()
app = _app_instance.api

if __name__ == "__main__":
    _app_instance.run()