"""Named tensors of a Hugging Face safetensors release, read by byte range into one local file.

A converter source can need a few small tensors of a release whose shards it does not otherwise
read, such as DeepSeek-V4.1's routing bias of image positions, one vector per block spread over
43 of its 48 shards. This reads the release's index, each needed shard's header and the matching
tensors' bytes, nothing else, and writes them, as stored, into one safetensors file:

    python -m tools.convert.fetch_tensors deepseek-ai/DeepSeek-V4.1-Flash-1017 \\
        --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 \\
        --match 'layers.*.ffn.gate.bias_vl' --out models/deepseek-v4.1/bias_vl.safetensors

`--match` takes fnmatch patterns over the tensor names (repeatable). HF_TOKEN in the environment
authorizes a gated repository; it is sent to huggingface.co only.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
from pathlib import Path
import struct
import urllib.parse
import urllib.request
from typing import Callable

ENDPOINT = "https://huggingface.co"
INDEX = "model.safetensors.index.json"

# read(file, begin, end) -> the file's bytes [begin, end)
Reader = Callable[[str, int, int], bytes]


class _NoAuthOnRedirect(urllib.request.HTTPRedirectHandler):
    """The hub redirects file reads to its storage hosts: the token stays with the hub."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new is not None and urllib.parse.urlsplit(newurl).hostname != \
                urllib.parse.urlsplit(req.full_url).hostname:
            new.remove_header("Authorization")
        return new


def _hub_get(repo: str, revision: str, file: str, byte_range: tuple[int, int] | None = None):
    url = f"{ENDPOINT}/{repo}/resolve/{urllib.parse.quote(revision)}/{urllib.parse.quote(file)}"
    request = urllib.request.Request(url)
    if byte_range is not None:
        request.add_header("Range", f"bytes={byte_range[0]}-{byte_range[1] - 1}")
    if os.environ.get("HF_TOKEN"):
        request.add_header("Authorization", f"Bearer {os.environ['HF_TOKEN']}")
    with urllib.request.build_opener(_NoAuthOnRedirect).open(request, timeout=120) as response:
        return response.read()


def hub_reader(repo: str, revision: str) -> Reader:
    def read(file: str, begin: int, end: int) -> bytes:
        data = _hub_get(repo, revision, file, (begin, end))
        if len(data) != end - begin:
            raise OSError(f"{file}: asked for bytes {begin}..{end}, got {len(data)} bytes")
        return data

    return read


def _header(read: Reader, file: str) -> tuple[int, dict]:
    size = struct.unpack("<Q", read(file, 0, 8))[0]
    return 8 + size, json.loads(read(file, 8, 8 + size))


def fetch(read: Reader, index: dict, patterns: list[str]) -> dict[str, tuple[dict, bytes]]:
    """The tensors whose names match a pattern: name -> (header entry, bytes), in name order."""

    names = sorted(name for name in index["weight_map"]
                   if any(fnmatch.fnmatchcase(name, p) for p in patterns))
    if not names:
        raise ValueError(f"no tensor of the index matches {patterns}")
    headers: dict[str, tuple[int, dict]] = {}
    out = {}
    for name in names:
        file = index["weight_map"][name]
        if file not in headers:
            headers[file] = _header(read, file)
        base, header = headers[file]
        entry = header[name]
        begin, end = entry["data_offsets"]
        out[name] = (entry, read(file, base + begin, base + end))
    return out


def write_safetensors(path: Path, tensors: dict[str, tuple[dict, bytes]],
                      metadata: dict[str, str]) -> None:
    header, offset = {"__metadata__": metadata}, 0
    for name, (entry, data) in tensors.items():
        header[name] = {"dtype": entry["dtype"], "shape": entry["shape"],
                        "data_offsets": [offset, offset + len(data)]}
        offset += len(data)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    with open(path, "wb") as out:
        out.write(struct.pack("<Q", len(encoded)))
        out.write(encoded)
        for _, data in tensors.values():
            out.write(data)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("repo")
    parser.add_argument("--revision", required=True, help="the release's commit")
    parser.add_argument("--match", action="append", required=True, metavar="PATTERN")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    read = hub_reader(args.repo, args.revision)
    index = json.loads(_hub_get(args.repo, args.revision, INDEX))
    tensors = fetch(read, index, args.match)
    write_safetensors(args.out, tensors, {"repo": args.repo, "revision": args.revision})
    total = sum(len(data) for _, data in tensors.values())
    print(f"wrote {args.out}: {len(tensors)} tensors, {total} bytes", flush=True)


if __name__ == "__main__":
    main()
