import hashlib

from ...flow import process_node

from typing import override


class FileChecksummer(process_node.ProcessNode):

    @override
    def process(self, source_file: str) -> dict[str, str]:
        hasher = hashlib.sha256()
        with open(source_file, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                hasher.update(byte_block)
        return {
            "checksum": hasher.hexdigest(),
        }
