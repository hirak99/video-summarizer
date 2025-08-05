"""Primarily designed to manage llama-cpp instance.

Instead of using python-llama-cpp, this class manages a llama-cpp server
instance directly.

While python-llama-cpp provides a convenient interface, I find directly working
with llama-cpp to be cleaner.

This gives us more control over the llama binary, and avoids setup issues
(especially with CUDA) which can be difficult to diagnose with python-llama-cpp.
"""

import dataclasses
import logging
import os
import signal
import subprocess
import time

_LLAMA_CPP_PATH = os.path.expanduser("~/git/llama.cpp/build/bin/llama-server")

# Seconds to give llama.cpp server to shut down gracefully before SIGKILL is used.
_LLAMA_SHUTDOWN_TIMEOUT = 30


@dataclasses.dataclass
class ModelConfig:
    path: str
    desc: str
    params: str


# Llama Server Documentation: https://github.com/ggml-org/llama.cpp/tree/master/tools/server
# Additional Payload Documentation: https://platform.openai.com/docs/api-reference/responses-streaming/response/incomplete

# Available model configurations.
MODEL_QWEN2_5_CODER_7B_INSTRUCT_Q6_K = ModelConfig(
    path=os.path.expanduser("~/data/models/qwen2.5-coder-7b-instruct-q6_k.gguf"),
    desc="Qwen2.5 Coder 7B Instruct Q6_K",
    # Note: Lower temp makes the output more deterministic.
    params="--gpu-layers 65 --temp 0.2 --top-p 0.9",
)
MODEL_QWEN3_30B_A3B_Q4_K_M = ModelConfig(
    path=os.path.expanduser("~/data/models/Qwen3-30B-A3B-Q4_K_M.gguf"),
    desc="Qwen3 30B A3B Q4_K_M",
    # Offical params from https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html
    params="--reasoning-format deepseek -ngl 99 -fa -sm row --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0 -c 40960 -n 32768 --no-context-shift",
)


class LocalServer:
    def __init__(self, model_config: ModelConfig):
        self.model_config = model_config
        self._llama_process: subprocess.Popen[str] | None = None

    def start(self, port: int) -> None:
        # Command to run
        cmd = [
            _LLAMA_CPP_PATH,
            "-m",
            self.model_config.path,
            "--port",
            str(port),
        ] + self.model_config.params.split()

        start_time = time.time()
        # Start the process in the background.
        self._llama_process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,  # We don't need to monitor stdout.
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,  # So we can kill the whole process group.
            text=True,  # To check the logs.
        )

        if not self._wait_for_server_ready(timeout=120):
            self.terminate()  # Clean up if server fails to start.
            raise RuntimeError("Failed to start server or server.")
        logging.info(
            f"Local server started in {time.time() - start_time:.2f}s with PID: {self._llama_process.pid}"
        )

    def _wait_for_server_ready(self, timeout: float) -> bool:
        if not self._llama_process:
            return False

        start_time = time.time()
        ready_message = "server is listening on"  # The message to look for.

        while True:
            time.sleep(0.2)  # Avoid busy waiting.

            if time.time() - start_time > timeout:
                logging.warning("Timed out waiting for server to be ready.")
                # Log any stderr output if available.
                if self._llama_process.stdout:
                    for err_line in self._llama_process.stdout.readlines():
                        logging.error(
                            f"(server stdout [unmonitored]): {err_line.strip()}"
                        )
                return False

            assert self._llama_process.stderr is not None, "we should capture stderr"
            for line in iter(self._llama_process.stderr.readline, ""):
                logging.info(f"(echoing server stderr): {line.strip()}")
                if ready_message in line:
                    return True

            # Check if process has exited prematurely
            if self._llama_process.poll() is not None:
                logging.error("Llama.cpp server process exited prematurely.")
                return False

    def terminate(self) -> None:
        # Note: If the process gets stuck, nvidia-smi will show the process id.
        if self._llama_process:
            logging.info("Shutting down llama.cpp server...")
            # TODO: Sometimes the process does not respond to SIGTERM. We should
            # check and use SIGKILL if necessary.
            try:
                pgid = os.getpgid(self._llama_process.pid)
                os.killpg(pgid, signal.SIGTERM)
                try:
                    # For some reason, SIGTERM does not work on llama.cpp after
                    # a lot of prompts. Will try SIGKILL after a timeout.
                    self._llama_process.wait(timeout=_LLAMA_SHUTDOWN_TIMEOUT)
                    logging.info("Server stopped.")
                except subprocess.TimeoutExpired:
                    logging.warning(
                        "Llama.cpp server did not terminate gracefully, sending SIGKILL."
                    )
                    os.killpg(pgid, signal.SIGKILL)
                    self._llama_process.wait()  # SIGKILL should terminate it.
                    logging.info("Server stopped with SIGKILL.")
            except ProcessLookupError:
                pass
            self._process = None

    def __del__(self) -> None:
        logging.info("LocalServer __del__(): Ensuring server is terminated.")
        self.terminate()
