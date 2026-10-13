#!/usr/bin/env python3
"""Inventory quantized overloads and attach observed passing test evidence."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import yaml
from collections import defaultdict
from functools import wraps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import tensorplay as tp


def declarations_from(path):
    if path.is_dir():
        return set().union(*(declarations_from(source) for source in sorted(path.rglob('*.cpp'))))
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


class ExecutionAudit:
    def __init__(self, entries):
        self.entries = {entry['operator']: entry for entry in entries}
        self.observed = defaultdict(lambda: defaultdict(lambda: {
            'tests': set(), 'native_entries': set(), 'returned': 0, 'raised': 0,
            'native_returned': 0}))
        self.current = None
        self.events = []
        self.native_stack = []
        self.restore = []
        self.outcomes = defaultdict(dict)
        self.native_names = {}

    def devices(self, values):
        found = set()
        seen = set()
        def visit(value, depth=0):
            if id(value) in seen or depth > 4:
                return
            seen.add(id(value))
            if isinstance(value, tp.Tensor):
                found.add(str(value.device).split(':', 1)[0])
            elif isinstance(value, dict):
                for item in value.values():
                    visit(item, depth + 1)
            elif isinstance(value, (tuple, list)):
                for item in value:
                    visit(item, depth + 1)
            elif type(value).__module__.startswith('tensorplay.ao.nn.quantized'):
                visit(vars(value), depth + 1)
        visit(values)
        return found or {'metadata'}

    def wrap(self, function, qualified=None, native=None):
        native = native or self.native_names.get(id(function))
        @wraps(function)
        def observed(*args, **kwargs):
            if self.current is None:
                return function(*args, **kwargs)
            calls = set()
            if native:
                entry = native + ':' + str(args[0]) if native == '_call_overload' and args else native
                for active in self.native_stack:
                    active.add(entry)
            self.native_stack.append(calls)
            result, raised = None, False
            try:
                result = function(*args, **kwargs)
                return result
            except BaseException:
                raised = True
                raise
            finally:
                self.native_stack.pop()
                if qualified in self.entries:
                    if native:
                        calls.add(native)
                    self.events.append((qualified, self.devices((args, kwargs, result)), calls, raised))
        return observed

    def install(self):
        replacements = {}
        for name in list(vars(tp._C)):
            function = getattr(tp._C, name)
            qualified = 'aten::' + name
            if callable(function) and (qualified in self.entries or 'quantiz' in name or name.startswith('fbgemm_') or name in ('_call_overload', '_wrapped_linear_prepack')):
                self.native_names[id(function)] = name
                replacement = self.wrap(function, qualified, native=name)
                replacements[id(function)] = replacement
                self.restore.append((tp._C, name, function))
                setattr(tp._C, name, replacement)
                if getattr(tp, name, None) is function:
                    self.restore.append((tp, name, function))
                    setattr(tp, name, replacement)
        for module_name, module in list(sys.modules.items()):
            if not module_name.startswith('tensorplay.') or module is None or module is tp._C:
                continue
            for name, function in list(vars(module).items()):
                replacement = replacements.get(id(function))
                if replacement is not None:
                    self.restore.append((module, name, function))
                    setattr(module, name, replacement)
        audit = self
        class Packet:
            def __init__(self, original, prefix):
                self.original, self.prefix = original, prefix
            def __call__(self, *args, **kwargs):
                native = audit.native_names.get(id(getattr(self.original, 'default', self.original)))
                return audit.wrap(self.original, self.prefix, native)(*args, **kwargs)
            def __getattr__(self, name):
                original = getattr(self.original, name)
                qualified = self.prefix if name == 'default' else self.prefix + '.' + name
                return audit.wrap(original, qualified) if qualified in audit.entries and callable(original) else original
        packets = {}
        for qualified, entry in self.entries.items():
            namespace = getattr(tp.ops, entry['public_namespace'])
            name = qualified.split('::', 1)[1].split('.', 1)[0]
            key = (entry['public_namespace'], name)
            if key in packets or not entry['callable']:
                continue
            try:
                original = getattr(namespace, name)
            except AttributeError:
                continue
            prefix = qualified.split('.', 1)[0]
            packets[key] = Packet(original, prefix)
            self.restore.append((namespace, name, original))
            setattr(namespace, name, packets[key])

    def uninstall(self):
        for owner, name, original in reversed(self.restore):
            setattr(owner, name, original)

    def pytest_runtest_setup(self, item):
        self.current, self.events = item.nodeid, []

    def pytest_runtest_logreport(self, report):
        self.outcomes[report.nodeid][report.when] = report.outcome
        if report.when != 'teardown':
            return
        phases = self.outcomes[report.nodeid]
        if phases.get('call') == 'passed' and all(value == 'passed' for value in phases.values()):
            for qualified, devices, calls, raised in self.events:
                for device in devices:
                    evidence = self.observed[qualified][device]
                    evidence['tests'].add(report.nodeid)
                    evidence['native_entries'].update(calls)
                    evidence['raised' if raised else 'returned'] += 1
                    evidence['native_returned'] += int(not raised and bool(calls))
        self.current, self.events = None, []

    def attach(self):
        for qualified, entry in self.entries.items():
            entry['test_evidence'] = {
                device: {**evidence, 'tests': sorted(evidence['tests']),
                         'native_entries': sorted(evidence['native_entries'])}
                for device, evidence in sorted(self.observed[qualified].items())}
            if entry['test_evidence']:
                returned = any(evidence['returned'] for evidence in entry['test_evidence'].values())
                entry['validation'] = 'exercised_in_passed_tests' if returned else 'rejection_tested'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declarations', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, default=Path('benchmark/quantized_coverage.json'))
    parser.add_argument('--run-tests', type=Path, nargs='+')
    args = parser.parse_args()
    names = set().union(*(declarations_from(path) for path in args.declarations))
    entries = inventory(names)
    status = 0
    if args.run_tests:
        import pytest
        audit = ExecutionAudit(entries)
        audit.install()
        try:
            status = pytest.main(['-q', *(str(path) for path in args.run_tests)], plugins=[audit])
        finally:
            audit.uninstall()
        audit.attach()
    result = dict(commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        total=len(entries), callable=sum(entry['callable'] for entry in entries),
        exercised=sum(entry['validation'] == 'exercised_in_passed_tests' for entry in entries),
        rejection_tested=sum(entry['validation'] == 'rejection_tested' for entry in entries),
        test_exit_code=int(status) if args.run_tests else None, operators=entries)
    result['device_evidence'] = {
        device: dict(
            returned=sum(bool(entry.get('test_evidence', {}).get(device, {}).get('returned')) for entry in entries),
            native_returned=sum(bool(entry.get('test_evidence', {}).get(device, {}).get('native_returned')) for entry in entries),
            rejection_tested=sum(bool(entry.get('test_evidence', {}).get(device, {}).get('raised')) for entry in entries))
        for device in ('cpu', 'cuda', 'metadata')}
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items() if key != 'operators'}, indent=2))
    for entry in entries:
        if not entry['callable']:
            print(entry['operator'])
    return int(status)


if __name__ == '__main__':
    raise SystemExit(main())
