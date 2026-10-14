from types import SimpleNamespace

import pytest
import tensorplay as tp
from tensorplay.ao.nn.quantized import functional as F

from tools.audit_quantized_coverage import ExecutionAudit, inventory


def finish(audit, name, outcome='passed', teardown='passed'):
    for phase, result in (('setup', 'passed'), ('call', outcome), ('teardown', teardown)):
        audit.pytest_runtest_logreport(SimpleNamespace(nodeid=name, when=phase, outcome=result))


def test_execution_audit_tracks_native_aliases_and_restores_packets():
    entries = inventory({'quantized::cat', 'quantized::celu', '_quantized::add', 'quantized::add'})
    audit = ExecutionAudit(entries)
    original = tp.ops.quantized.cat
    alias = F._quantized_elu
    audit.install()
    try:
        audit.pytest_runtest_setup(SimpleNamespace(nodeid='successful'))
        x = tp.quantize_per_tensor(tp.tensor([-.5, .5]), .1, 117, tp.quint8)
        result = tp.ops.quantized.cat([x, x], 0, .1, 117)
        assert tuple(result.shape) == (4,)
        result = tp.ops.quantized.celu(x, .1, 117, 1.3)
        assert tuple(result.shape) == (2,)
        tp.ops._quantized.add(x, x, .1, 117)
        finish(audit, 'successful')
    finally:
        audit.uninstall()
    audit.attach()
    by_name = {entry['operator']: entry for entry in entries}
    assert tp.ops.quantized.cat is original and F._quantized_elu is alias
    assert 'quantized_cat' in by_name['quantized::cat']['test_evidence']['cpu']['native_entries']
    assert 'quantized_elu' in by_name['quantized::celu']['test_evidence']['cpu']['native_entries']
    assert by_name['quantized::celu']['test_evidence']['cpu']['native_returned'] == 1
    assert by_name['_quantized::add']['validation'] == 'exercised_in_passed_tests'
    assert by_name['quantized::add']['validation'] == 'pending'


@pytest.mark.parametrize('failure', ['call', 'teardown'])
def test_execution_audit_excludes_failed_test_evidence(failure):
    entries = inventory({'quantized::add'})
    audit = ExecutionAudit(entries)
    audit.install()
    try:
        audit.pytest_runtest_setup(SimpleNamespace(nodeid='failed'))
        x = tp.quantize_per_tensor(tp.tensor([1.]), .1, 117, tp.quint8)
        tp.ops.quantized.add(x, x, .1, 117)
        finish(audit, 'failed', outcome='failed' if failure == 'call' else 'passed',
               teardown='failed' if failure == 'teardown' else 'passed')
    finally:
        audit.uninstall()
    audit.attach()
    assert entries[0]['validation'] == 'pending' and not entries[0]['test_evidence']


def test_execution_audit_distinguishes_rejection_from_return():
    entries = inventory({'quantized::linear_prepack_legacy'})
    audit = ExecutionAudit(entries)
    audit.install()
    try:
        audit.pytest_runtest_setup(SimpleNamespace(nodeid='rejected'))
        with pytest.raises(RuntimeError):
            tp.ops.quantized.linear_prepack_legacy(tp.zeros(2, 3), None)
        finish(audit, 'rejected')
    finally:
        audit.uninstall()
    audit.attach()
    assert entries[0]['validation'] == 'rejection_tested'
    evidence = entries[0]['test_evidence']['cpu']
    assert evidence['returned'] == 0 and evidence['raised'] == 1
    assert evidence['native_returned'] == 0
