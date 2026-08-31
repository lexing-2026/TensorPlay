from __future__ import annotations

import faulthandler
import json
import logging
import os
import time
import traceback
import warnings
from typing import Any

__all__ = ["ErrorHandler"]

logger = logging.getLogger(__name__)


class ErrorHandler:
    def _get_error_file_path(self) -> str | None:
        return os.environ.get("TORCHELASTIC_ERROR_FILE")

    def initialize(self) -> None:
        try:
            faulthandler.enable(all_threads=True)
        except Exception as exc:
            warnings.warn(f"Unable to enable fault handler: {type(exc).__name__}: {exc}", stacklevel=2)

    def _write_error_file(self, file_path: str, error_msg: str) -> None:
        try:
            os.makedirs(os.path.dirname(file_path) or ".", exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as stream:
                stream.write(error_msg)
        except Exception as exc:
            warnings.warn(f"Unable to write error file: {type(exc).__name__}: {exc}", stacklevel=2)

    def record_exception(self, e: BaseException) -> None:
        payload: dict[str, Any] = {
            "message": {
                "message": f"{type(e).__name__}: {e}",
                "extraInfo": {
                    "py_callstack": traceback.format_exc(),
                    "timestamp": str(int(time.time())),
                },
            }
        }
        path = self._get_error_file_path()
        encoded = json.dumps(payload)
        if path:
            self._write_error_file(path, encoded)
        else:
            logger.error(encoded)

    def maybe_enrich_signal_failure_message(self, message: str, error_file: str) -> str:
        return message

    def override_error_code_in_rootcause_data(
        self, rootcause_error_file: str, rootcause_error: dict[str, Any], error_code: int = 0
    ) -> None:
        message = rootcause_error.get("message")
        if isinstance(message, dict):
            message["errorCode"] = error_code
        else:
            logger.warning("error file %s has no structured message", rootcause_error_file)

    def dump_error_file(self, rootcause_error_file: str, error_code: int = 0):
        with open(rootcause_error_file, encoding="utf-8") as stream:
            rootcause_error = json.load(stream)
        if error_code:
            self.override_error_code_in_rootcause_data(rootcause_error_file, rootcause_error, error_code)
        path = self._get_error_file_path()
        if path:
            self._rm(path)
            self._write_error_file(path, json.dumps(rootcause_error))
        else:
            logger.error("child error data: %s", json.dumps(rootcause_error, indent=2))

    def _rm(self, my_error_file):
        if os.path.isfile(my_error_file):
            os.remove(my_error_file)
