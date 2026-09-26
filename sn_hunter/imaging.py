from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from scipy.ndimage import map_coordinates


@dataclass(slots=True)
class FitsImage:
    data: np.ndarray
    header: fits.Header
    wcs: WCS
    path: Path | None = None


def load_fits(path: str | Path) -> FitsImage:
    path = Path(path)
    with fits.open(path, memmap=False) as hdul:
        selected = next(
            (hdu for hdu in hdul if hdu.data is not None and hdu.data.ndim >= 2),
            None,
        )
        if selected is None:
            raise ValueError("FITS soubor neobsahuje obrazová data.")
        data = np.asarray(selected.data, dtype=np.float64)
        while data.ndim > 2:
            data = data[0]
        header = selected.header.copy()

    wcs = WCS(header).celestial
    if not wcs.has_celestial:
        raise ValueError("FITS hlavička neobsahuje platnou nebeskou WCS astrometrii.")
    if not np.isfinite(data).any():
        raise ValueError("FITS obraz neobsahuje žádné konečné hodnoty.")
    return FitsImage(data=data, header=header, wcs=wcs, path=path)


def stretch_image(
    data: np.ndarray,
    mode: str = "asinh",
    low_percentile: float = 0.5,
    high_percentile: float = 99.5,
    parameter: float = 10.0,
    statistics_data: np.ndarray | None = None,
) -> np.ndarray:
    finite = np.asarray(data, dtype=np.float64)
    mask = np.isfinite(finite)
    out = np.zeros(finite.shape, dtype=np.float32)
    if not mask.any():
        return out

    statistics = (
        finite
        if statistics_data is None
        else np.asarray(statistics_data, dtype=np.float64)
    )
    statistics_mask = np.isfinite(statistics)
    if not statistics_mask.any():
        statistics = finite
        statistics_mask = mask
    lo, hi = np.nanpercentile(
        statistics[statistics_mask], [low_percentile, high_percentile]
    )
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        hi = lo + 1.0
    x = np.clip((finite[mask] - lo) / (hi - lo), 0.0, 1.0)

    if mode == "sqrt":
        x = np.sqrt(x)
    elif mode == "log":
        strength = max(parameter, 1e-3)
        x = np.log1p(strength * x) / np.log1p(strength)
    elif mode == "asinh":
        strength = max(parameter, 1e-3)
        x = np.arcsinh(strength * x) / np.arcsinh(strength)
    elif mode == "power":
        x = np.power(x, max(parameter, 1e-3))
    elif mode != "linear":
        raise ValueError(f"Neznámý stretch: {mode}")

    out[mask] = x.astype(np.float32, copy=False)
    return out


def reproject_to_image(source: FitsImage, target: FitsImage) -> np.ndarray:
    """Reproject source data to the exact pixel grid of target."""
    try:
        from reproject import reproject_interp

        result, _ = reproject_interp(
            (source.data, source.wcs),
            target.wcs,
            shape_out=target.data.shape,
            order="bilinear",
        )
        return np.asarray(result, dtype=np.float64)
    except ImportError:
        return _fallback_reproject(source, target)


def _fallback_reproject(source: FitsImage, target: FitsImage) -> np.ndarray:
    height, width = target.data.shape
    result = np.empty((height, width), dtype=np.float64)
    # Work in strips so a 2000–2500 px high-resolution comparison grid does
    # not require several full-size coordinate arrays at once.
    for row_start in range(0, height, 256):
        row_end = min(row_start + 256, height)
        yy, xx = np.indices((row_end - row_start, width), dtype=np.float64)
        yy += row_start
        ra, dec = target.wcs.pixel_to_world_values(xx, yy)
        sx, sy = source.wcs.world_to_pixel_values(ra, dec)
        coordinates = np.vstack((sy.ravel(), sx.ravel()))
        sampled = map_coordinates(
            source.data,
            coordinates,
            order=1,
            mode="constant",
            cval=np.nan,
            prefilter=False,
        )
        result[row_start:row_end] = sampled.reshape((row_end - row_start, width))
    return result


def make_comparison_grid(
    image: FitsImage,
    ra: float,
    dec: float,
    fov_arcmin: float,
    preferred_scale_arcsec: float,
    max_pixels: int = 2200,
) -> FitsImage:
    """Create a high-resolution WCS grid with the input image orientation."""
    requested_pixels = int(np.ceil(fov_arcmin * 60.0 / preferred_scale_arcsec))
    size = max(128, min(requested_pixels, max_pixels))
    actual_scale_deg = (fov_arcmin / 60.0) / size

    original_matrix = np.asarray(image.wcs.pixel_scale_matrix, dtype=float)
    original_scale = float(np.mean(np.sqrt(np.sum(original_matrix**2, axis=0))))
    if not np.isfinite(original_scale) or original_scale <= 0:
        raise ValueError("Z WCS nelze určit pixelové měřítko.")

    comparison_wcs = WCS(naxis=2)
    comparison_wcs.wcs.ctype = list(image.wcs.wcs.ctype)
    comparison_wcs.wcs.crval = [ra, dec]
    comparison_wcs.wcs.crpix = [(size + 1.0) / 2.0, (size + 1.0) / 2.0]
    comparison_wcs.wcs.cd = original_matrix * (actual_scale_deg / original_scale)
    comparison_wcs.wcs.radesys = image.wcs.wcs.radesys
    comparison_wcs.wcs.equinox = image.wcs.wcs.equinox
    header = comparison_wcs.to_header(relax=True)
    return FitsImage(
        data=np.zeros((size, size), dtype=np.float32),
        header=header,
        wcs=comparison_wcs,
    )


def field_center_radius(image: FitsImage) -> tuple[float, float, float]:
    height, width = image.data.shape
    center = image.wcs.pixel_to_world(width / 2.0, height / 2.0)
    corners = image.wcs.pixel_to_world(
        np.array([0.0, width - 1.0, 0.0, width - 1.0]),
        np.array([0.0, 0.0, height - 1.0, height - 1.0]),
    )
    radius = float(np.max(center.separation(corners).degree))
    return float(center.ra.degree), float(center.dec.degree), radius


def field_size_deg(image: FitsImage) -> float:
    """Return the larger angular side of the detector footprint in degrees."""
    height, width = image.data.shape
    x_mid = (width - 1.0) / 2.0
    y_mid = (height - 1.0) / 2.0
    left, right = image.wcs.pixel_to_world(
        np.array([0.0, width - 1.0]), np.array([y_mid, y_mid])
    )
    bottom, top = image.wcs.pixel_to_world(
        np.array([x_mid, x_mid]), np.array([0.0, height - 1.0])
    )
    return float(max(left.separation(right).degree, bottom.separation(top).degree))

