import abc
import logging
import pathlib
import time

from typing import Any, Callable

_NUM_RETRIES = 3


class RetriableException(Exception):
    """Raised when an LLM call fails and should be retried.

    Implementations should raise this exception on issues that should be
    retried, for example network failures.
    Parsers can also raise this error to request retries.
    """

    def __init__(self, *, retry_delay_s: int = 0) -> None:
        self.retry_delay_s = retry_delay_s


class AbstractLlm(abc.ABC):
    @abc.abstractmethod
    def model_description(self) -> str:
        """Returns the model ID for the current instance."""
        pass

    @abc.abstractmethod
    def do_prompt(self, prompt: str, **kwargs) -> str:
        """Gets LLM response."""
        pass

    def _log_llm_debug_info(
        self,
        *,
        log_fname: str | pathlib.Path | None,
        prompt: str,
        raw_response: str,
        processed_response: Any,
        additional_info: str | None,
    ):
        if log_fname is None:
            return
        logging.info(f"Writing LLM debug info to {log_fname!r}")
        section_break = "\n===========\n"
        llm_log_lines: list[str] = [
            f"Model: {self.model_description()}",
        ]
        if additional_info is not None:
            llm_log_lines += [section_break, "Additional Info:", additional_info]
        llm_log_lines += [
            section_break,
            "Full Prompt:",
            prompt,
            section_break,
            "LLM Response:",
            raw_response,
            section_break,
            "Processed LLM Response:",
            (
                "(Same as response)"
                if processed_response == raw_response
                else f"{processed_response}"
            ),
        ]
        with open(log_fname, "w") as file:
            file.write("\n".join(llm_log_lines))

    def do_prompt_and_parse(
        self,
        prompt: str,
        max_tokens: int,
        transformers: list[Callable[[Any], Any]],
        log_file: str | pathlib.Path | None = None,
        log_additional_info: str | None = None,
        image_b64: str | None = None,
    ) -> Any:
        """Gets LLM response, then validates and transforms it.

        Args:
            prompt: The prompt to send to the LLM.

            max_tokens: The maximum number of tokens to get from the LLM.

            transformers: A list of functions that will be applied to the
            response. If the input cannot be parsed by any of the transformers,
            it should raise RetriableError().

            log_file: A text file which will be created or overwritten.

            log_additional_info: Information to be added to the log file.

            image_b64: Optional image as part of the query.
        """
        retries_left = _NUM_RETRIES
        logging.info(f"Sending prompt: {prompt}")
        while True:
            retries_left -= 1
            # Pass on the image= arg only if it is given.
            extra_kwargs = {}
            if image_b64 is not None:
                extra_kwargs.update(image_b64=image_b64)

            try:
                response: Any = self.do_prompt(
                    prompt, max_tokens=max_tokens, **extra_kwargs
                )
                processed_response = response
                for parser in transformers:
                    processed_response = parser(processed_response)
            except RetriableException as e:
                logging.warning(f"RetriableException: {e}")
                logging.info(f"Retries left: {retries_left}")
                if retries_left <= 0:
                    logging.warning("Exhausted all retries.")
                    raise
                logging.info(f"Retry delay: {e.retry_delay_s}s")
                time.sleep(e.retry_delay_s)
                continue

            self._log_llm_debug_info(
                log_fname=log_file,
                prompt=prompt,
                raw_response=response,
                processed_response=processed_response,
                additional_info=log_additional_info,
            )
            return processed_response

    def finalize(self):
        """Optionally implement this to release resources."""
        pass
