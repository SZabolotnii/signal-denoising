"""
Unit-тести для DSGEFeatureExtractor.

Запуск:
    cd /Users/serhiizabolotnii/Projects/signal-denoising
    source .venv/bin/activate
    python -m pytest tests/test_dsge_layer.py -v
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from models.dsge_layer import DSGEFeatureExtractor, fractional_basis, polynomial_basis, trigonometric_basis, robust_basis


# ──────────────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────────────

@pytest.fixture
def synthetic_signals():
    """100 синтетичних сигналів довжиною 512 (чисті + зашумлені)."""
    np.random.seed(0)
    N, T = 100, 512
    t = np.linspace(0, 1, T, endpoint=False)
    clean = np.array([np.sin(2 * np.pi * 5 * t + np.random.uniform(0, 2 * np.pi))
                      for _ in range(N)])
    noisy = clean + np.random.normal(0, 0.5, (N, T))
    return clean, noisy


@pytest.fixture
def fitted_dsge(synthetic_signals):
    """Fitted DSGEFeatureExtractor на synthetic_signals."""
    clean, noisy = synthetic_signals
    dsge = DSGEFeatureExtractor(
        basis_type='fractional',
        powers=[0.5, 1.5, 2.0],
        tikhonov_lambda=0.01,
        stft_params={'nperseg': 64, 'noverlap': 48, 'fs': 512},
    )
    dsge.fit(clean, noisy)
    return dsge, clean, noisy


# ──────────────────────────────────────────────────────
#  Test 1: basis shapes
# ──────────────────────────────────────────────────────

class TestBasisFunctions:
    def test_fractional_shape(self):
        x = np.random.randn(512)
        phi = fractional_basis(x, [0.5, 1.5, 2.0])
        assert phi.shape == (3, 512)

    def test_polynomial_shape(self):
        x = np.random.randn(512)
        phi = polynomial_basis(x, [2.0, 3.0, 4.0])
        assert phi.shape == (3, 512)

    def test_trigonometric_shape(self):
        x = np.random.randn(512)
        phi = trigonometric_basis(x, [1.0, 2.0, 3.0])
        assert phi.shape == (3, 512)

    def test_robust_shape(self):
        x = np.random.randn(512)
        phi = robust_basis(x)
        assert phi.shape == (3, 512)

    def test_fractional_sign_preserved(self):
        """φᵢ(x) має той самий знак, що й x."""
        x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        phi = fractional_basis(x, [0.5, 1.5])
        for i in range(2):
            assert np.all(np.sign(phi[i][x != 0]) == np.sign(x[x != 0])), \
                "Знак не збережений у fractional_basis"

    def test_no_nan_inf(self):
        """Базисні функції не повинні давати NaN або inf для стандартних입력."""
        x = np.random.randn(1000)
        for fn, args in [
            (fractional_basis, ([0.5, 1.5, 2.0],)),
            (polynomial_basis, ([2.0, 3.0, 4.0],)),
            (trigonometric_basis, ([1.0, 2.0, 3.0],)),
        ]:
            phi = fn(x, *args)
            assert not np.any(np.isnan(phi)) and not np.any(np.isinf(phi))


# ──────────────────────────────────────────────────────
#  Test 2: DSGEFeatureExtractor API
# ──────────────────────────────────────────────────────

class TestDSGEFeatureExtractor:
    def test_not_fitted_raises(self):
        dsge = DSGEFeatureExtractor()
        with pytest.raises(RuntimeError, match="not fitted"):
            dsge.reconstruct(np.zeros(512))

    def test_fit_sets_state(self, fitted_dsge):
        dsge, _, _ = fitted_dsge
        assert dsge._fitted
        assert dsge.K is not None and dsge.K.shape == (3,)
        assert dsge.psi is not None and dsge.psi.shape == (3,)
        assert dsge.psi_0 is not None
        assert dsge.k0 is not None
        assert dsge.gen_element_norm is not None

    def test_compute_basis_shape(self, fitted_dsge):
        dsge, _, noisy = fitted_dsge
        phi = dsge.compute_basis(noisy[0])
        assert phi.shape == (3, 512), f"Expected (3, 512), got {phi.shape}"

    def test_compute_dsge_spectrograms_shape(self, fitted_dsge):
        dsge, _, noisy = fitted_dsge
        specs = dsge.compute_dsge_spectrograms(noisy[0])
        # F = nperseg//2 + 1 = 33
        assert specs.ndim == 3, "Expected 3D tensor [S, F, T']"
        assert specs.shape[0] == 3, f"Expected S=3 channels, got {specs.shape[0]}"
        assert specs.shape[1] == 33, f"Expected F=33, got {specs.shape[1]}"

    def test_reconstruct_shape(self, fitted_dsge):
        dsge, _, noisy = fitted_dsge
        Y = dsge.reconstruct(noisy[0])
        assert Y.shape == (512,), f"Expected (512,), got {Y.shape}"

    def test_reconstruct_no_nan(self, fitted_dsge):
        dsge, _, noisy = fitted_dsge
        for i in range(5):
            Y = dsge.reconstruct(noisy[i])
            assert not np.any(np.isnan(Y))

    def test_reconstruct_better_than_noisy(self, fitted_dsge):
        """DSGE-наближення повинно бути кращим за передачу зашумленого як є."""
        dsge, clean, noisy = fitted_dsge
        mse_noisy = np.mean((clean[0] - noisy[0]) ** 2)
        Y = dsge.reconstruct(noisy[0])
        mse_dsge = np.mean((clean[0] - Y) ** 2)
        assert mse_dsge < mse_noisy, \
            f"DSGE MSE ({mse_dsge:.4f}) >= noisy MSE ({mse_noisy:.4f})"


# ──────────────────────────────────────────────────────
#  Test 3: Tikhonov stability
# ──────────────────────────────────────────────────────

class TestTikhonovStability:
    def test_large_lambda_no_nan(self, synthetic_signals):
        """Великий λ (сильна регуляризація) не повинен давати NaN."""
        clean, noisy = synthetic_signals
        for lam in [0.0, 0.1, 1.0, 10.0, 100.0]:
            dsge = DSGEFeatureExtractor(tikhonov_lambda=lam)
            dsge.fit(clean, noisy)
            Y = dsge.reconstruct(noisy[0])
            assert not np.any(np.isnan(Y)), f"NaN at lambda={lam}"

    def test_zero_lambda_with_regularization(self, synthetic_signals):
        """λ=0 може дати нестабільний результат, але не повинно кидати виняток."""
        clean, noisy = synthetic_signals
        # λ=0 → F може бути виродженою, але solve_with_rcond=None не кидає
        dsge = DSGEFeatureExtractor(tikhonov_lambda=0.0)
        try:
            dsge.fit(clean, noisy)
            _ = dsge.reconstruct(noisy[0])
        except np.linalg.LinAlgError:
            pass  # Це допустимо при λ=0


# ──────────────────────────────────────────────────────
#  Test 4: Generating element norm
# ──────────────────────────────────────────────────────

class TestGeneratingElementNorm:
    def test_norm_positive(self, fitted_dsge):
        dsge, _, _ = fitted_dsge
        assert dsge.gen_element_norm > 0

    def test_check_returns_bool(self, fitted_dsge):
        dsge, _, _ = fitted_dsge
        result = dsge.check_generating_element_norm(threshold=0.01)
        assert isinstance(result, bool)

    def test_high_threshold_returns_false(self, fitted_dsge):
        dsge, _, _ = fitted_dsge
        result = dsge.check_generating_element_norm(threshold=1e6)
        assert result is False


# ──────────────────────────────────────────────────────
#  Test 5: Save / Load state
# ──────────────────────────────────────────────────────

class TestSaveLoadState:
    def test_save_load_roundtrip(self, fitted_dsge, tmp_path):
        dsge, _, noisy = fitted_dsge
        path = str(tmp_path / 'dsge_test.npz')
        dsge.save_state(path)

        dsge2 = DSGEFeatureExtractor.load_state(
            path, basis_type='fractional',
            stft_params={'nperseg': 64, 'noverlap': 48, 'fs': 512},
        )
        Y1 = dsge.reconstruct(noisy[0])
        Y2 = dsge2.reconstruct(noisy[0])
        np.testing.assert_allclose(Y1, Y2, rtol=1e-5,
                                   err_msg="Reconstruct after load/save should be identical")

    def test_load_preserves_k(self, fitted_dsge, tmp_path):
        dsge, _, _ = fitted_dsge
        path = str(tmp_path / 'dsge_k.npz')
        dsge.save_state(path)
        dsge2 = DSGEFeatureExtractor.load_state(path, basis_type='fractional')
        np.testing.assert_allclose(dsge.K, dsge2.K, rtol=1e-6)


# ──────────────────────────────────────────────────────
#  Test 6: unsupported basis_type
# ──────────────────────────────────────────────────────

def test_unknown_basis_raises():
    with pytest.raises(ValueError, match="Unknown basis_type"):
        DSGEFeatureExtractor(basis_type='unknown_xyz')


# ──────────────────────────────────────────────────────
#  Test 7: STFT compatibility
# ──────────────────────────────────────────────────────

def test_dsge_stft_same_shape_as_signal_stft():
    """DSGE-спектрограма повинна мати той самий F×T', що й звичайний STFT."""
    from scipy.signal import stft
    np.random.seed(42)
    noisy = np.random.randn(512)
    clean = noisy + np.random.randn(512) * 0.1
    stft_params = {'nperseg': 64, 'noverlap': 48, 'fs': 512}

    dsge = DSGEFeatureExtractor(stft_params=stft_params)
    dsge.fit(clean[np.newaxis], noisy[np.newaxis])

    _, _, Zxx = stft(noisy, **stft_params)
    ref_shape = np.abs(Zxx).shape

    dsge_specs = dsge.compute_dsge_spectrograms(noisy)
    for i in range(dsge.S):
        assert dsge_specs[i].shape == ref_shape, \
            f"DSGE spec[{i}] shape {dsge_specs[i].shape} != STFT shape {ref_shape}"
