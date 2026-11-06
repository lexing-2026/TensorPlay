from pathlib import Path
from unittest.mock import Mock

import pytest

import tensorplay as tp
import tensorplay.audio as audio
from tensorplay.audio.datasets import SPEECHCOMMANDS, YESNO
from tensorplay.audio.datasets.utils import _load_waveform


@pytest.fixture
def mock_audio_load(monkeypatch):
    waveform = tp.tensor([[0.25, -0.5, 0.0]], dtype=tp.float32, device="cpu")
    load = Mock(return_value=(waveform, 16000))
    monkeypatch.setattr(audio, "load", load)
    return load


@pytest.mark.parametrize("path_type", [str, Path])
def test_load_waveform_uses_audio_load(tmp_path, mock_audio_load, path_type):
    filename = "speaker/chapter/sample.flac"
    waveform = _load_waveform(path_type(tmp_path), filename, 16000)

    assert waveform is mock_audio_load.return_value[0]
    mock_audio_load.assert_called_once_with(str(tmp_path / filename))


def test_load_waveform_rejects_unexpected_sample_rate(tmp_path, mock_audio_load):
    mock_audio_load.return_value = (mock_audio_load.return_value[0], 8000)

    with pytest.raises(ValueError, match="sample rate should be 16000, but got 8000"):
        _load_waveform(str(tmp_path), "sample.wav", 16000)

    mock_audio_load.assert_called_once_with(str(tmp_path / "sample.wav"))


def test_yesno_getitem_uses_audio_load(tmp_path, mock_audio_load):
    audio_path = tmp_path / "waves_yesno" / "0_1_0_1_1_0_0_1.wav"
    audio_path.parent.mkdir()
    audio_path.touch()
    mock_audio_load.return_value = (mock_audio_load.return_value[0], 8000)
    dataset = YESNO(tmp_path, download=False)

    waveform, sample_rate, labels = dataset[0]

    assert len(dataset) == 1
    assert waveform is mock_audio_load.return_value[0]
    assert sample_rate == 8000
    assert labels == [0, 1, 0, 1, 1, 0, 0, 1]
    mock_audio_load.assert_called_once_with(str(audio_path))


def test_speechcommands_getitem_uses_shared_waveform_loader(tmp_path, mock_audio_load):
    audio_path = (
        tmp_path / "SpeechCommands" / "speech_commands_v0.02" / "yes" / "speaker_nohash_2.wav"
    )
    audio_path.parent.mkdir(parents=True)
    audio_path.touch()
    dataset = SPEECHCOMMANDS(tmp_path, download=False)

    waveform, sample_rate, label, speaker_id, utterance_number = dataset[0]

    assert len(dataset) == 1
    assert waveform is mock_audio_load.return_value[0]
    assert (sample_rate, label, speaker_id, utterance_number) == (16000, "yes", "speaker", 2)
    mock_audio_load.assert_called_once_with(str(audio_path))
