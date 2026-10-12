#!/usr/bin/env python3
"""Inventory quantized operator overloads and report callable public entries."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorplay as tp


def declarations_from(path):
    source = path.read_text()
    if path.suffix not in ('.yaml', '.yml'):
        return set(re.findall(r'"((?:_?quantized::|aten::quantized_)[A-Za-z_0-9.]+)\(', source))
    names = set()
    for entry in yaml.safe_load(source):
        dispatch = entry.get('dispatch', {})
        name = entry['func'].split('(', 1)[0]
        if (name.startswith(('quantized_', 'fbgemm_')) or
                any('Quantized' in key for key in dispatch)):
            names.add('aten::' + name)
    return names


def inventory(names):
    entries = []
    for qualified in sorted(names):
        namespace, name = qualified.split('::', 1)
        public_namespace = 'tp' if namespace == 'aten' else namespace
        packet, separator, overload = name.partition('.')
        try:
            operator = getattr(getattr(tp.ops, public_namespace), packet)
            available = callable(getattr(operator, overload)) if separator else callable(operator)
            reason = None if available else 'entry is not callable'
        except (AttributeError, RuntimeError) as error:
            available = False
            reason = type(error).__name__
        entries.append(dict(operator=qualified, public_namespace=public_namespace, callable=available,
            validation='pending', performance='pending', reason=reason))
    return entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declarations', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, default=Path('benchmark/quantized_coverage.json'))
    args = parser.parse_args()
    names = set().union(*(declarations_from(path) for path in args.declarations))
    entries = inventory(names)
    result = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        total=len(entries), callable=sum(entry['callable'] for entry in entries), operators=entries)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'operators'}, indent=2))
    for entry in entries:
        if not entry['callable']:
            print(entry['operator'])


if __name__ == '__main__':
    main()
