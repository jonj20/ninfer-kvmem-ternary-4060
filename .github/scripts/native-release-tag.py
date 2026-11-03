#!/usr/bin/env python3.11
"""Allocate a UTC date/build tag, reusing a release for the same source revision.

With --tag, check instead that the tag the build was given is still this revision's: its release,
or a number no release or Git tag has taken since.
"""
import argparse
from datetime import datetime, timezone
import json
import re
from pathlib import Path


def flatten(pages):
    return [item for page in pages for item in page] if pages and isinstance(pages[0], list) else pages


def existing(releases, revision):
    same = [r for r in releases if r.get('target_commitish') == revision
            and re.fullmatch(r'(?:native-[0-9a-f]{12}|[0-9]{4}-[0-9]{2}-[0-9]{2}\.[1-9][0-9]*)', r['tag_name'])]
    return sorted(same, key=lambda r: (r['draft'], -r['id']))[0] if same else None


def taken(releases, refs):
    return {r['tag_name'] for r in releases} | {r['ref'].removeprefix('refs/tags/') for r in refs}


def select(releases, refs, revision, day):
    if release := existing(releases, revision):
        return release['tag_name'], release['id'], release['draft']
    numbers = [int(m[1]) for tag in taken(releases, refs)
               if (m := re.fullmatch(re.escape(day) + r'\.([1-9][0-9]*)', tag))]
    return f'{day}.{max(numbers, default=0)+1}', 0, True


def check(releases, refs, revision, tag):
    if release := existing(releases, revision):
        if release['tag_name'] != tag:
            raise SystemExit(f"{revision} is released as {release['tag_name']}, but its binaries report {tag}")
        return release['tag_name'], release['id'], release['draft']
    if not re.fullmatch(r'[0-9]{4}-[0-9]{2}-[0-9]{2}\.[1-9][0-9]*', tag):
        raise SystemExit(f'Not a release tag: {tag!r}')
    if tag in taken(releases, refs):
        raise SystemExit(f'{tag}, which the binaries report, was taken by another release')
    return tag, 0, True


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--releases', type=Path, required=True)
    parser.add_argument('--refs', type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--tag', help='the tag the build was given')
    args = parser.parse_args()
    releases = flatten(json.loads(args.releases.read_text()))
    refs = flatten(json.loads(args.refs.read_text()))
    if args.tag is not None:
        tag, release_id, draft = check(releases, refs, args.revision, args.tag)
    else:
        day = datetime.now(timezone.utc).strftime('%Y-%m-%d')
        tag, release_id, draft = select(releases, refs, args.revision, day)
    print(tag, release_id, str(draft).lower())
