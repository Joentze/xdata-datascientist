import logging

from fastapi import Request


def get_logger(request: Request) -> logging.Logger:
    logger = getattr(request.app.state, "logger", None)
    if logger is None:
        return logging.getLogger("app")
    return logger
