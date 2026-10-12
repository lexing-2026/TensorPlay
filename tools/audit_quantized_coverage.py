#!/usr/bin/env python3
"""Inventory quantized operator overloads and report callable public entries."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorplay as tp


def inventory(declarations):
    names = sorted(set(re.findall(r'"((?:_?quantized)::[A-Za-z_0-9.]+)\(', declarations)))
    entries = []
    for qualified in names:
        namespace, name = qualified.split('::', 1)
        packet, separator, overload = name.partition('.')
        try:
            operator = getattr(getattr(tp.ops, namespace), packet)
            available = callable(getattr(operator, overload)) if separator else callable(operator)
            reason = None if available else 'entry is not callable'
        except (AttributeError, RuntimeError) as error:
            available = False
            reason = type(error).__name__
        entries.append(dict(operator=qualified, callable=available,
            validation='pending', performance='pending', reason=reason))
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declarations', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, default=Path('benchmark/quantized_coverage.json'))
    args = parser.parse_args()
    entries = inventory('\n'.join(path.read_text() for path in args.declarations))
    result = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        total=len(entries), callable=sum(entry['callable'] for entry in entries), operators=entries)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'operators'}, indent=2))
    for entry in entries:
        if not entry['callable']:
            print(entry['operator'])


if __name__ == '__main__':
    main()
