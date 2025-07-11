from enum import Enum


class TransportType(Enum):

    STDIO = "stdio"
    SSE = "sse"
    HTTP = "streamable-http"