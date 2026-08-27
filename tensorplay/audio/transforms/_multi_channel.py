# -*- coding: utf-8 -*-

import warnings
from typing import Optional, Union

import tensorplay as tensorplay
from tensorplay import Tensor
from tensorplay.audio import functional as F


__all__ = []


def _get_mvdr_vector(
    psd_s: tensorplay.Tensor,
    psd_n: tensorplay.Tensor,
    reference_vector: tensorplay.Tensor,
    solution: str = "ref_channel",
    diagonal_loading: bool = True,
    diag_eps: float = 1e-7,
    eps: float = 1e-8,
) -> tensorplay.Tensor:
    r"""Compute the MVDR beamforming weights with ``solution`` argument.

    Args:
        psd_s (tensorplay.Tensor): The complex-valued power spectral density (PSD) matrix of target speech.
            Tensor with dimensions `(..., freq, channel, channel)`.
        psd_n (tensorplay.Tensor): The complex-valued power spectral density (PSD) matrix of noise.
            Tensor with dimensions `(..., freq, channel, channel)`.
        reference_vector (tensorplay.Tensor): one-hot reference channel matrix.
        solution (str, optional): Solution to compute the MVDR beamforming weights.
            Options: [``ref_channel``, ``stv_evd``, ``stv_power``]. (Default: ``ref_channel``)
        diagonal_loading (bool, optional): If ``True``, enables applying diagonal loading to ``psd_n``.
            (Default: ``True``)
        diag_eps (float, optional): The coefficient multiplied to the identity matrix for diagonal loading.
            It is only effective when ``diagonal_loading`` is set to ``True``. (Default: ``1e-7``)
        eps (float, optional): Value to add to the denominator in the beamforming weight formula.
            (Default: ``1e-8``)

    Returns:
        tensorplay.Tensor: the mvdr beamforming weight matrix
    """
    if solution == "ref_channel":
        beamform_vector = F.mvdr_weights_souden(psd_s, psd_n, reference_vector, diagonal_loading, diag_eps, eps)
    else:
        if solution == "stv_evd":
            stv = F.rtf_evd(psd_s)
        else:
            stv = F.rtf_power(psd_s, psd_n, reference_vector, diagonal_loading=diagonal_loading, diag_eps=diag_eps)
        beamform_vector = F.mvdr_weights_rtf(stv, psd_n, reference_vector, diagonal_loading, diag_eps, eps)

    return beamform_vector


class PSD(tensorplay.nn.Module):
    r"""Compute cross-channel power spectral density (PSD) matrix.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Args:
        multi_mask (bool, optional): If ``True``, only accepts multi-channel Time-Frequency masks. (Default: ``False``)
        normalize (bool, optional): If ``True``, normalize the mask along the time dimension. (Default: ``True``)
        eps (float, optional): Value to add to the denominator in mask normalization. (Default: ``1e-15``)
    """

    def __init__(self, multi_mask: bool = False, normalize: bool = True, eps: float = 1e-15):
        super().__init__()
        self.multi_mask = multi_mask
        self.normalize = normalize
        self.eps = eps

    def forward(self, specgram: tensorplay.Tensor, mask: Optional[tensorplay.Tensor] = None):
        """
        Args:
            specgram (tensorplay.Tensor): Multi-channel complex-valued spectrum.
                Tensor with dimensions `(..., channel, freq, time)`.
            mask (tensorplay.Tensor or None, optional): Time-Frequency mask for normalization.
                Tensor with dimensions `(..., freq, time)` if multi_mask is ``False`` or
                with dimensions `(..., channel, freq, time)` if multi_mask is ``True``.
                (Default: ``None``)

        Returns:
            tensorplay.Tensor: The complex-valued PSD matrix of the input spectrum.
                Tensor with dimensions `(..., freq, channel, channel)`
        """
        if mask is not None:
            if self.multi_mask:
                # Averaging mask along channel dimension
                mask = mask.mean(dim=-3)  # (..., freq, time)
        psd = F.psd(specgram, mask, self.normalize, self.eps)

        return psd


class MVDR(tensorplay.nn.Module):
    """Minimum Variance Distortionless Response (MVDR) module that performs MVDR beamforming with Time-Frequency masks.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Based on https://github.com/espnet/espnet/blob/master/espnet2/enh/layers/beamformer.py

    We provide three solutions of MVDR beamforming. One is based on *reference channel selection*
    :cite:`souden2009optimal` (``solution=ref_channel``).

    .. math::
        \\textbf{w}_{\\text{MVDR}}(f) =\
        \\frac{{{\\bf{\\Phi}_{\\textbf{NN}}^{-1}}(f){\\bf{\\Phi}_{\\textbf{SS}}}}(f)}\
        {\\text{Trace}({{{\\bf{\\Phi}_{\\textbf{NN}}^{-1}}(f) \\bf{\\Phi}_{\\textbf{SS}}}(f))}}\\bm{u}

    where :math:`\\bf{\\Phi}_{\\textbf{SS}}` and :math:`\\bf{\\Phi}_{\\textbf{NN}}` are the covariance\
        matrices of speech and noise, respectively. :math:`\\bf{u}` is an one-hot vector to determine the\
         reference channel.

    The other two solutions are based on the steering vector (``solution=stv_evd`` or ``solution=stv_power``).

    .. math::
        \\textbf{w}_{\\text{MVDR}}(f) =\
        \\frac{{{\\bf{\\Phi}_{\\textbf{NN}}^{-1}}(f){\\bm{v}}(f)}}\
        {{\\bm{v}^{\\mathsf{H}}}(f){\\bf{\\Phi}_{\\textbf{NN}}^{-1}}(f){\\bm{v}}(f)}

    where :math:`\\bm{v}` is the acoustic transfer function or the steering vector.\
        :math:`.^{\\mathsf{H}}` denotes the Hermitian Conjugate operation.

    We apply either *eigenvalue decomposition*
    :cite:`higuchi2016robust` or the *power method* :cite:`mises1929praktische` to get the
    steering vector from the PSD matrix of speech.

    After estimating the beamforming weight, the enhanced Short-time Fourier Transform (STFT) is obtained by

    .. math::
        \\hat{\\bf{S}} = {\\bf{w}^\\mathsf{H}}{\\bf{Y}}, {\\bf{w}} \\in \\mathbb{C}^{M \\times F}

    where :math:`\\bf{Y}` and :math:`\\hat{\\bf{S}}` are the STFT of the multi-channel noisy speech and\
        the single-channel enhanced speech, respectively.

    For online streaming audio, we provide a *recursive method* :cite:`higuchi2017online` to update the
    PSD matrices of speech and noise, respectively.

    Args:
        ref_channel (int, optional): Reference channel for beamforming. (Default: ``0``)
        solution (str, optional): Solution to compute the MVDR beamforming weights.
            Options: [``ref_channel``, ``stv_evd``, ``stv_power``]. (Default: ``ref_channel``)
        multi_mask (bool, optional): If ``True``, only accepts multi-channel Time-Frequency masks. (Default: ``False``)
        diagonal_loading (bool, optional): If ``True``, enables applying diagonal loading to the covariance matrix
            of the noise. (Default: ``True``)
        diag_eps (float, optional): The coefficient multiplied to the identity matrix for diagonal loading.
            It is only effective when ``diagonal_loading`` is set to ``True``. (Default: ``1e-7``)
        online (bool, optional): If ``True``, updates the MVDR beamforming weights based on
            the previous covarience matrices. (Default: ``False``)

    Note:
        To improve the numerical stability, the input spectrogram will be converted to double precision
        (``tensorplay.complex128`` or ``tensorplay.cdouble``) dtype for internal computation. The output spectrogram
        is converted to the dtype of the input spectrogram to be compatible with other modules.

    Note:
        If you use ``stv_evd`` solution, the gradient of the same input may not be identical if the
        eigenvalues of the PSD matrix are not distinct (i.e. some eigenvalues are close or identical).
    """

    def __init__(
        self,
        ref_channel: int = 0,
        solution: str = "ref_channel",
        multi_mask: bool = False,
        diag_loading: bool = True,
        diag_eps: float = 1e-7,
        online: bool = False,
    ):
        super().__init__()
        if solution not in [
            "ref_channel",
            "stv_evd",
            "stv_power",
        ]:
            raise ValueError(
                '`solution` must be one of ["ref_channel", "stv_evd", "stv_power"]. Given {}'.format(solution)
            )
        self.ref_channel = ref_channel
        self.solution = solution
        self.multi_mask = multi_mask
        self.diag_loading = diag_loading
        self.diag_eps = diag_eps
        self.online = online
        self.psd = PSD(multi_mask)

        psd_s: tensorplay.Tensor = tensorplay.zeros(1)
        psd_n: tensorplay.Tensor = tensorplay.zeros(1)
        mask_sum_s: tensorplay.Tensor = tensorplay.zeros(1)
        mask_sum_n: tensorplay.Tensor = tensorplay.zeros(1)
        self.register_buffer("psd_s", psd_s)
        self.register_buffer("psd_n", psd_n)
        self.register_buffer("mask_sum_s", mask_sum_s)
        self.register_buffer("mask_sum_n", mask_sum_n)

    def _get_updated_mvdr_vector(
        self,
        psd_s: tensorplay.Tensor,
        psd_n: tensorplay.Tensor,
        mask_s: tensorplay.Tensor,
        mask_n: tensorplay.Tensor,
        reference_vector: tensorplay.Tensor,
        solution: str = "ref_channel",
        diagonal_loading: bool = True,
        diag_eps: float = 1e-7,
        eps: float = 1e-8,
    ) -> tensorplay.Tensor:
        r"""Recursively update the MVDR beamforming vector.

        Args:
            psd_s (tensorplay.Tensor): The complex-valued power spectral density (PSD) matrix of target speech.
                Tensor with dimensions `(..., freq, channel, channel)`.
            psd_n (tensorplay.Tensor): The complex-valued power spectral density (PSD) matrix of noise.
                Tensor with dimensions `(..., freq, channel, channel)`.
            mask_s (tensorplay.Tensor): Time-Frequency mask of the target speech.
                Tensor with dimensions `(..., freq, time)` if multi_mask is ``False``
                or with dimensions `(..., channel, freq, time)` if multi_mask is ``True``.
            mask_n (tensorplay.Tensor or None, optional): Time-Frequency mask of the noise.
                Tensor with dimensions `(..., freq, time)` if multi_mask is ``False``
                or with dimensions `(..., channel, freq, time)` if multi_mask is ``True``.
            reference_vector (tensorplay.Tensor): One-hot reference channel matrix.
            solution (str, optional): Solution to compute the MVDR beamforming weights.
                Options: [``ref_channel``, ``stv_evd``, ``stv_power``]. (Default: ``ref_channel``)
            diagonal_loading (bool, optional): If ``True``, enables applying diagonal loading to ``psd_n``.
                (Default: ``True``)
            diag_eps (float, optional): The coefficient multiplied to the identity matrix for diagonal loading.
                It is only effective when ``diagonal_loading`` is set to ``True``. (Default: ``1e-7``)
            eps (float, optional): Value to add to the denominator in the beamforming weight formula.
                (Default: ``1e-8``)

        Returns:
            tensorplay.Tensor: The MVDR beamforming weight matrix.
        """
        if self.multi_mask:
            # Averaging mask along channel dimension
            mask_s = mask_s.mean(dim=-3)  # (..., freq, time)
            mask_n = mask_n.mean(dim=-3)  # (..., freq, time)
        if self.psd_s.ndim == 1:
            self.psd_s = psd_s
            self.psd_n = psd_n
            self.mask_sum_s = mask_s.sum(dim=-1)
            self.mask_sum_n = mask_n.sum(dim=-1)
            return _get_mvdr_vector(psd_s, psd_n, reference_vector, solution, diagonal_loading, diag_eps, eps)
        else:
            psd_s = self._get_updated_psd_speech(psd_s, mask_s)
            psd_n = self._get_updated_psd_noise(psd_n, mask_n)
            self.psd_s = psd_s
            self.psd_n = psd_n
            self.mask_sum_s = self.mask_sum_s + mask_s.sum(dim=-1)
            self.mask_sum_n = self.mask_sum_n + mask_n.sum(dim=-1)
            return _get_mvdr_vector(psd_s, psd_n, reference_vector, solution, diagonal_loading, diag_eps, eps)

    def _get_updated_psd_speech(self, psd_s: tensorplay.Tensor, mask_s: tensorplay.Tensor) -> tensorplay.Tensor:
        r"""Update psd of speech recursively.

        Args:
            psd_s (tensorplay.Tensor): The complex-valued power spectral density (PSD) matrix of target speech.
                Tensor with dimensions `(..., freq, channel, channel)`.
            mask_s (tensorplay.Tensor): Time-Frequency mask of the target speech.
                Tensor with dimensions `(..., freq, time)`.

        Returns:
            tensorplay.Tensor: The updated PSD matrix of target speech.
        """
        numerator = self.mask_sum_s / (self.mask_sum_s + mask_s.sum(dim=-1))
        denominator = 1 / (self.mask_sum_s + mask_s.sum(dim=-1))
        psd_s = self.psd_s * numerator[..., None, None] + psd_s * denominator[..., None, None]
        return psd_s

    def _get_updated_psd_noise(self, psd_n: tensorplay.Tensor, mask_n: tensorplay.Tensor) -> tensorplay.Tensor:
        r"""Update psd of noise recursively.

        Args:
            psd_n (tensorplay.Tensor): The complex-valued power spectral density (PSD) matrix of noise.
                Tensor with dimensions `(..., freq, channel, channel)`.
            mask_n (tensorplay.Tensor or None, optional): Time-Frequency mask of the noise.
                Tensor with dimensions `(..., freq, time)`.

        Returns:
            tensorplay.Tensor:  The updated PSD matrix of noise.
        """
        numerator = self.mask_sum_n / (self.mask_sum_n + mask_n.sum(dim=-1))
        denominator = 1 / (self.mask_sum_n + mask_n.sum(dim=-1))
        psd_n = self.psd_n * numerator[..., None, None] + psd_n * denominator[..., None, None]
        return psd_n

    def forward(
        self, specgram: tensorplay.Tensor, mask_s: tensorplay.Tensor, mask_n: Optional[tensorplay.Tensor] = None
    ) -> tensorplay.Tensor:
        """Perform MVDR beamforming.

        Args:
            specgram (tensorplay.Tensor): Multi-channel complex-valued spectrum.
                Tensor with dimensions `(..., channel, freq, time)`
            mask_s (tensorplay.Tensor): Time-Frequency mask of target speech.
                Tensor with dimensions `(..., freq, time)` if multi_mask is ``False``
                or with dimensions `(..., channel, freq, time)` if multi_mask is ``True``.
            mask_n (tensorplay.Tensor or None, optional): Time-Frequency mask of noise.
                Tensor with dimensions `(..., freq, time)` if multi_mask is ``False``
                or with dimensions `(..., channel, freq, time)` if multi_mask is ``True``.
                (Default: None)

        Returns:
            tensorplay.Tensor: Single-channel complex-valued enhanced spectrum with dimensions `(..., freq, time)`.
        """
        dtype = specgram.dtype
        if specgram.ndim < 3:
            raise ValueError(f"Expected at least 3D tensor (..., channel, freq, time). Found: {specgram.shape}")
        if not specgram.is_complex():
            raise ValueError(
                f"The type of ``specgram`` tensor must be ``tensorplay.cfloat`` or ``tensorplay.cdouble``.\
                    Found: {specgram.dtype}"
            )
        if specgram.dtype == tensorplay.cfloat:
            specgram = specgram.cdouble()  # Convert specgram to ``torch.cdouble``.

        if mask_n is None:
            warnings.warn("``mask_n`` is not provided, use ``1 - mask_s`` as ``mask_n``.")
            mask_n = 1 - mask_s

        psd_s = self.psd(specgram, mask_s)  # (..., freq, time, channel, channel)
        psd_n = self.psd(specgram, mask_n)  # (..., freq, time, channel, channel)

        u = tensorplay.zeros(specgram.size()[:-2], device=specgram.device, dtype=tensorplay.cdouble)  # (..., channel)
        u[..., self.ref_channel].fill_(1)

        if self.online:
            w_mvdr = self._get_updated_mvdr_vector(
                psd_s, psd_n, mask_s, mask_n, u, self.solution, self.diag_loading, self.diag_eps
            )
        else:
            w_mvdr = _get_mvdr_vector(psd_s, psd_n, u, self.solution, self.diag_loading, self.diag_eps)

        specgram_enhanced = F.apply_beamforming(w_mvdr, specgram)

        return specgram_enhanced.to(dtype)


class RTFMVDR(tensorplay.nn.Module):
    r"""Minimum Variance Distortionless Response (*MVDR* :cite:`capon1969high`) module
    based on the relative transfer function (RTF) and power spectral density (PSD) matrix of noise.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Given the multi-channel complex-valued spectrum :math:`\textbf{Y}`, the relative transfer function (RTF) matrix
    or the steering vector of target speech :math:`\bm{v}`, the PSD matrix of noise :math:`\bf{\Phi}_{\textbf{NN}}`, and
    a one-hot vector that represents the reference channel :math:`\bf{u}`, the module computes the single-channel
    complex-valued spectrum of the enhanced speech :math:`\hat{\textbf{S}}`. The formula is defined as:

    .. math::
        \hat{\textbf{S}}(f) = \textbf{w}_{\text{bf}}(f)^{\mathsf{H}} \textbf{Y}(f)

    where :math:`\textbf{w}_{\text{bf}}(f)` is the MVDR beamforming weight for the :math:`f`-th frequency bin,
    :math:`(.)^{\mathsf{H}}` denotes the Hermitian Conjugate operation.

    The beamforming weight is computed by:

    .. math::
        \textbf{w}_{\text{MVDR}}(f) =
        \frac{{{\bf{\Phi}_{\textbf{NN}}^{-1}}(f){\bm{v}}(f)}}
        {{\bm{v}^{\mathsf{H}}}(f){\bf{\Phi}_{\textbf{NN}}^{-1}}(f){\bm{v}}(f)}
    """

    def forward(
        self,
        specgram: Tensor,
        rtf: Tensor,
        psd_n: Tensor,
        reference_channel: Union[int, Tensor],
        diagonal_loading: bool = True,
        diag_eps: float = 1e-7,
        eps: float = 1e-8,
    ) -> Tensor:
        """
        Args:
            specgram (tensorplay.Tensor): Multi-channel complex-valued spectrum.
                Tensor with dimensions `(..., channel, freq, time)`
            rtf (tensorplay.Tensor): The complex-valued RTF vector of target speech.
                Tensor with dimensions `(..., freq, channel)`.
            psd_n (tensorplay.Tensor): The complex-valued power spectral density (PSD) matrix of noise.
                Tensor with dimensions `(..., freq, channel, channel)`.
            reference_channel (int or tensorplay.Tensor): Specifies the reference channel.
                If the dtype is ``int``, it represents the reference channel index.
                If the dtype is ``tensorplay.Tensor``, its shape is `(..., channel)`, where the ``channel`` dimension
                is one-hot.
            diagonal_loading (bool, optional): If ``True``, enables applying diagonal loading to ``psd_n``.
                (Default: ``True``)
            diag_eps (float, optional): The coefficient multiplied to the identity matrix for diagonal loading.
                It is only effective when ``diagonal_loading`` is set to ``True``. (Default: ``1e-7``)
            eps (float, optional): Value to add to the denominator in the beamforming weight formula.
                (Default: ``1e-8``)

        Returns:
            tensorplay.Tensor: Single-channel complex-valued enhanced spectrum with dimensions `(..., freq, time)`.
        """
        w_mvdr = F.mvdr_weights_rtf(rtf, psd_n, reference_channel, diagonal_loading, diag_eps, eps)
        spectrum_enhanced = F.apply_beamforming(w_mvdr, specgram)
        return spectrum_enhanced


class SoudenMVDR(tensorplay.nn.Module):
    r"""Minimum Variance Distortionless Response (*MVDR* :cite:`capon1969high`) module
    based on the method proposed by *Souden et, al.* :cite:`souden2009optimal`.

    .. devices:: CPU CUDA

    .. properties:: Autograd TorchScript

    Given the multi-channel complex-valued spectrum :math:`\textbf{Y}`, the power spectral density (PSD) matrix
    of target speech :math:`\bf{\Phi}_{\textbf{SS}}`, the PSD matrix of noise :math:`\bf{\Phi}_{\textbf{NN}}`, and
    a one-hot vector that represents the reference channel :math:`\bf{u}`, the module computes the single-channel
    complex-valued spectrum of the enhanced speech :math:`\hat{\textbf{S}}`. The formula is defined as:

    .. math::
        \hat{\textbf{S}}(f) = \textbf{w}_{\text{bf}}(f)^{\mathsf{H}} \textbf{Y}(f)

    where :math:`\textbf{w}_{\text{bf}}(f)` is the MVDR beamforming weight for the :math:`f`-th frequency bin.

    The beamforming weight is computed by:

    .. math::
        \textbf{w}_{\text{MVDR}}(f) =
        \frac{{{\bf{\Phi}_{\textbf{NN}}^{-1}}(f){\bf{\Phi}_{\textbf{SS}}}}(f)}
        {\text{Trace}({{{\bf{\Phi}_{\textbf{NN}}^{-1}}(f) \bf{\Phi}_{\textbf{SS}}}(f))}}\bm{u}
    """

    def forward(
        self,
        specgram: Tensor,
        psd_s: Tensor,
        psd_n: Tensor,
        reference_channel: Union[int, Tensor],
        diagonal_loading: bool = True,
        diag_eps: float = 1e-7,
        eps: float = 1e-8,
    ) -> tensorplay.Tensor:
        """
        Args:
            specgram (tensorplay.Tensor): Multi-channel complex-valued spectrum.
                Tensor with dimensions `(..., channel, freq, time)`.
            psd_s (tensorplay.Tensor): The complex-valued power spectral density (PSD) matrix of target speech.
                Tensor with dimensions `(..., freq, channel, channel)`.
            psd_n (tensorplay.Tensor): The complex-valued power spectral density (PSD) matrix of noise.
                Tensor with dimensions `(..., freq, channel, channel)`.
            reference_channel (int or tensorplay.Tensor): Specifies the reference channel.
                If the dtype is ``int``, it represents the reference channel index.
                If the dtype is ``tensorplay.Tensor``, its shape is `(..., channel)`, where the ``channel`` dimension
                is one-hot.
            diagonal_loading (bool, optional): If ``True``, enables applying diagonal loading to ``psd_n``.
                (Default: ``True``)
            diag_eps (float, optional): The coefficient multiplied to the identity matrix for diagonal loading.
                It is only effective when ``diagonal_loading`` is set to ``True``. (Default: ``1e-7``)
            eps (float, optional): Value to add to the denominator in the beamforming weight formula.
                (Default: ``1e-8``)

        Returns:
            tensorplay.Tensor: Single-channel complex-valued enhanced spectrum with dimensions `(..., freq, time)`.
        """
        w_mvdr = F.mvdr_weights_souden(psd_s, psd_n, reference_channel, diagonal_loading, diag_eps, eps)
        spectrum_enhanced = F.apply_beamforming(w_mvdr, specgram)
        return spectrum_enhanced
