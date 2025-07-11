import asyncio
import os
from threading import Thread
from typing import Optional

import uvicorn

from integrations.core.manager import IntegrationManager
from servers.api.server import APIServer
from servers.mcp.server import MCPServer
from servers.mcp.transport import TransportType
from utils.logger import Logger, LoggerLevel


class Application:

    def __init__(self):
        self.debug = os.environ.get("DEBUG", "false").lower() in {"1", "true", "yes"}
        if self.debug:
            Logger.set_level(LoggerLevel.DEBUG)
        else:
            Logger.set_level(LoggerLevel.INFO)

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
        self.api_thread: Optional[Thread] = None
        self.api: Optional[APIServer] = None

        # MCP Server
        self.mcp_thread: Optional[Thread] = None
        self.mcp: Optional[MCPServer] = None

        # Integration Manager
        self.__logger.info("Initializing integration manager")
        self.integration_manager = IntegrationManager()
        self.integration_manager.register_all()
        self.__logger.info("Integration manager initialized successfully")

    def run(self):
        self.integration_manager.start_all()

        host = "0.0.0.0"
        api_port = 5000
        mcp_port = 8000

        use_api = True
        use_mcp = True

        if use_api:
            self.start_api_server(api_port, host)

        if use_mcp:
            self.start_mcp_server(host, mcp_port)

        if self.api_thread is not None and self.api_thread.is_alive():
            self.api_thread.join()
        if self.mcp_thread is not None and self.mcp_thread.is_alive():
            self.mcp_thread.join()

    def start_mcp_server(self, host, mcp_port):
        if self.mcp_thread is not None and self.mcp_thread.is_alive():
            self.__logger.warning("MCP Server thread already started")
            return

        self.__logger.info("Starting MCP Server")

        self.mcp = MCPServer()
        self.mcp.settings.host = host
        self.mcp.settings.port = mcp_port
        mcp_transport = TransportType.HTTP  # STDIO (local) by default

        self.__logger.info("MCP Server started successfully")
        if mcp_transport == TransportType.STDIO:
            self.__logger.info(f"MCP server running on STDIO")
        elif mcp_transport in [TransportType.SSE, TransportType.HTTP]:
            self.__logger.info(f"MCP server running on http://{host}:{mcp_port}")

        self.mcp_thread = Thread(
            target=self.mcp.run,
            kwargs={"transport": mcp_transport.value},
            name="MCP Server",
            daemon=not self.debug,
        )
        self.mcp_thread.start()

    def start_api_server(self, api_port, host):
        if self.api_thread is not None and self.api_thread.is_alive():
            self.__logger.warning("API server thread already started")
            return

        self.__logger.info("Starting API server")

        self.api = APIServer(title="RAG API")
        config = uvicorn.Config(self.api, host=host, port=api_port, log_config=None, reload=self.debug)
        server = uvicorn.Server(config)

        self.__logger.info("API server started successfully!")
        self.__logger.info(f"API server running on http://{host}:{api_port}")

        self.api_thread = Thread(
            target=lambda: asyncio.run(server.serve()),
            name="API Server",
            daemon=not self.debug,
        )
        self.api_thread.start()

_app_instance = Application()
app = _app_instance.api


def monkey_patch_to_prevent_uvicorn_print_logs():
    # 1) Get uvicorn config __init__ method
    _original_uvicorn_config_init = uvicorn.Config.__init__

    # 2) Define a wrapper that sets the log_config value
    def _patched_uvicorn_config_init(self, *args, **kwargs):
        # Set or override log_config value
        kwargs["log_config"] = None
        # Calls the original method with the log_config value set
        _original_uvicorn_config_init(self, *args, **kwargs)

    # 3) Apply the patch
    uvicorn.Config.__init__ = _patched_uvicorn_config_init

monkey_patch_to_prevent_uvicorn_print_logs()
if __name__ == "__main__":
    _app_instance.run()