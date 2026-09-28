from __future__ import annotations

import hashlib
import re
import tempfile
import threading
from io import BytesIO
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from astropy import units as u
from astropy.coordinates import FK4, SkyCoord
from astropy.io import fits
from astropy.io.votable import parse_single_table
from astropy.time import Time
from astropy.wcs import WCS

from .imaging import FitsImage


HIPS2FITS_URL = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
VIZIER_TAP_URL = "https://tapvizier.cds.unistra.fr/TAPVizieR/tap/sync"
SURVEYS = {
    "DSS2 Red": "CDS/P/DSS2/red",
    "DSS2 Blue": "CDS/P/DSS2/blue",
    "DSS2 IR": "CDS/P/DSS2/IR",
    "Pan-STARRS DR1 r": "CDS/P/PanSTARRS/DR1/r",
}
SURVEY_PIXEL_SCALE_ARCSEC = {
    "DSS2 Red": 1.0,
    "DSS2 Blue": 1.0,
    "DSS2 IR": 1.0,
    "Pan-STARRS DR1 r": 0.25,
}
REFERENCE_CACHE_MAX_BYTES = 1 * 1024**3
_REFERENCE_CACHE_LOCK = threading.RLock()


def _write_cache_atomic(path: Path, content: bytes):
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=path.parent, prefix="catalog-", suffix=".tmp", delete=False
        ) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


@dataclass(slots=True)
class Galaxy:
    name: str
    ra: float
    dec: float
    magnitude: float | None = None
    # Full angular diameters. HECATE's MajAxis/MinAxis values are semi-axes
    # and are converted while parsing; HyperLEDA logD25 is already a diameter.
    major_arcmin: float | None = None
    minor_arcmin: float | None = None
    position_angle_deg: float | None = None
    pgc: int | None = None
    magnitude_band: str = "V"
    magnitude_approximate: bool = False

    @property
    def label(self) -> str:
        if self.magnitude is None:
            mag = ""
        else:
            relation = "≈" if self.magnitude_approximate else "="
            mag = f"  {self.magnitude_band}{relation}{self.magnitude:.1f} mag"
        return f"{self.name}{mag}"


@dataclass(slots=True)
class DeepSkyObject:
    name: str
    ra: float
    dec: float
    object_type: str
    category: str
    magnitude: float | None = None
    size_arcmin: float | None = None
    photographic_magnitude: bool = False

    @property
    def label(self) -> str:
        type_names = {
            "OC": "otevřená hvězdokupa",
            "Gb": "kulová hvězdokupa",
            "Nb": "mlhovina",
            "Pl": "planetární mlhovina",
            "C+N": "hvězdokupa s mlhovinou",
            "Ast": "asterismus",
            "Kt": "nebulózní oblast",
        }
        parts = [self.name, type_names.get(self.object_type, self.object_type)]
        if self.magnitude is not None:
            if self.photographic_magnitude:
                parts.append(f"B≈{self.magnitude:.1f} mag")
            else:
                parts.append(f"{self.magnitude:.1f} mag")
        if self.size_arcmin is not None:
            parts.append(f"velikost {self.size_arcmin:g}′")
        return " — ".join(parts)


def query_galaxies(
    ra: float,
    dec: float,
    radius_deg: float,
    magnitude_limit: float = 16.0,
    cache_dir: Path | None = None,
) -> list[Galaxy]:
    """Query nearby galaxies through the HECATE catalogue in VizieR TAP.

    HECATE provides homogenized total V-band apparent magnitudes and galaxy
    dimensions. The constraints are evaluated by the database, which is
    essential for very wide fields.
    """
    query = f"""
        SELECT TOP 20000
            "PGC", "HyperLEDA", "NED", "RAJ2000", "DEJ2000",
            "Vtmag", "MajAxis", "MinAxis", "PA"
        FROM "J/MNRAS/506/1896/hecate"
        WHERE CONTAINS(
                POINT('ICRS', "RAJ2000", "DEJ2000"),
                CIRCLE('ICRS', {ra:.10f}, {dec:.10f}, {radius_deg:.10f})
              ) = 1
          AND "Vtmag" IS NOT NULL
          AND "Vtmag" <= {magnitude_limit:.3f}
    """
    cache_path = None
    if cache_dir is not None:
        catalog_cache = cache_dir / "catalogs"
        catalog_cache.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(query.encode("utf-8")).hexdigest()[:24]
        cache_path = catalog_cache / f"hecate-{key}.vot"

    if cache_path is not None and cache_path.exists():
        content = cache_path.read_bytes()
    else:
        try:
            response = requests.post(
                VIZIER_TAP_URL,
                data={
                    "request": "doQuery",
                    "lang": "adql",
                    "format": "votable",
                    "query": query,
                },
                headers={"User-Agent": "SN-Hunter/0.1 (visual supernova search)"},
                timeout=(15, 180),
            )
            response.raise_for_status()
        except requests.Timeout as exc:
            raise RuntimeError(
                "VizieR neodpověděl v časovém limitu. Zkuste dotaz zopakovat později."
            ) from exc
        except requests.HTTPError as exc:
            detail = exc.response.text[:800].strip() if exc.response is not None else ""
            raise RuntimeError(f"VizieR odmítl katalogový dotaz: {detail or exc}") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Spojení s VizieR selhalo: {exc}") from exc
        content = response.content
        if cache_path is not None:
            _write_cache_atomic(cache_path, content)

    try:
        table = parse_single_table(BytesIO(content)).to_table()
    except Exception as exc:
        detail = content[:500].decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"VizieR vrátil nečitelnou odpověď. {detail}") from exc

    names = {name.lower(): name for name in table.colnames}

    def col(*candidates: str):
        for candidate in candidates:
            found = names.get(candidate.lower())
            if found:
                return table[found]
        return None

    ra_col = col("RAJ2000")
    dec_col = col("DEJ2000")
    mag_col = col("Vtmag")
    name_cols = [col("HyperLEDA"), col("NED")]
    pgc_col = col("PGC")
    major_col = col("MajAxis")
    minor_col = col("MinAxis")
    pa_col = col("PA")
    if ra_col is None or dec_col is None or mag_col is None:
        raise RuntimeError("VizieR vrátil tabulku bez souřadnic nebo V magnitudy.")

    result: list[Galaxy] = []
    for index in range(len(table)):
        coord = SkyCoord(float(ra_col[index]) * u.deg, float(dec_col[index]) * u.deg)
        magnitude = float(mag_col[index])
        name = ""
        for values in name_cols:
            if values is not None and not np.ma.is_masked(values[index]):
                candidate = str(values[index]).strip()
                if candidate:
                    name = candidate
                    break
        pgc = None
        if pgc_col is not None and not np.ma.is_masked(pgc_col[index]):
            pgc = int(pgc_col[index])
        if not name and pgc is not None:
            name = f"PGC {pgc}"
        # VizieR's Axis convention and HECATE use angular semi-axes. Keep one
        # unambiguous representation in the application: full diameters.
        major_axis = _optional_float(major_col, index)
        minor_axis = _optional_float(minor_col, index)
        major = None if major_axis is None else 2.0 * major_axis
        minor = None if minor_axis is None else 2.0 * minor_axis
        pa = _optional_float(pa_col, index)
        name = re.sub(r"^(NGC|IC|UGC|PGC|M)(\d)", r"\1 \2", name)
        result.append(
            Galaxy(
                name, coord.ra.degree, coord.dec.degree, magnitude,
                major, minor, pa, pgc=pgc,
            )
        )

    # HECATE contains high-quality V photometry, but only for the nearby
    # Universe and only where V is available. Add bright PGC galaxies from
    # GLADE 2, with dimensions from HyperLEDA, so conspicuous objects with a
    # missing HECATE Vtmag (for example PGC 27158) are not silently omitted.
    # GLADE supplies B, which is deliberately labelled as such rather than
    # being presented as measured V. The extra one-magnitude margin admits red
    # galaxies that can still satisfy the user's visual-band limit.
    supplemental = _query_bright_pgc(
        ra, dec, radius_deg, magnitude_limit, cache_dir
    )
    known_pgc = {galaxy.pgc for galaxy in result if galaxy.pgc is not None}
    result.extend(galaxy for galaxy in supplemental if galaxy.pgc not in known_pgc)
    result.sort(key=lambda galaxy: (galaxy.magnitude is None, galaxy.magnitude or 99.0, galaxy.name))
    return result


def _query_bright_pgc(
    ra: float,
    dec: float,
    radius_deg: float,
    visual_magnitude_limit: float,
    cache_dir: Path | None,
) -> list[Galaxy]:
    blue_magnitude_limit = visual_magnitude_limit + 1.0
    query = f"""
        SELECT TOP 20000
            g."PGC", g."RAJ2000", g."DEJ2000", g."Bmag", g."Flag1",
            p."logD25", p."logR25", p."PA", p."ANames"
        FROM "VII/281/glade2" AS g
        LEFT OUTER JOIN "VII/237/pgc" AS p ON g."PGC" = p."PGC"
        WHERE CONTAINS(
                POINT('ICRS', g."RAJ2000", g."DEJ2000"),
                CIRCLE('ICRS', {ra:.10f}, {dec:.10f}, {radius_deg:.10f})
              ) = 1
          AND g."PGC" IS NOT NULL
          AND g."Bmag" IS NOT NULL
          AND g."Bmag" <= {blue_magnitude_limit:.3f}
          AND g."Flag1" = 'G'
    """
    cache_path = None
    if cache_dir is not None:
        catalog_cache = cache_dir / "catalogs"
        catalog_cache.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(query.encode("utf-8")).hexdigest()[:24]
        cache_path = catalog_cache / f"pgc-{key}.vot"

    if cache_path is not None and cache_path.exists():
        content = cache_path.read_bytes()
    else:
        try:
            response = requests.post(
                VIZIER_TAP_URL,
                data={
                    "request": "doQuery",
                    "lang": "adql",
                    "format": "votable",
                    "query": query,
                },
                headers={"User-Agent": "SN-Hunter/0.1 (visual supernova search)"},
                timeout=(15, 180),
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(
                "Doplňkový katalog jasných PGC (GLADE) se nepodařilo načíst."
            ) from exc
        content = response.content
        if cache_path is not None:
            _write_cache_atomic(cache_path, content)

    table = parse_single_table(BytesIO(content)).to_table()
    names = {name.lower(): name for name in table.colnames}

    def col(name: str):
        actual = names.get(name.lower())
        return table[actual] if actual else None

    pgc_col = col("PGC")
    ra_col = col("RAJ2000")
    dec_col = col("DEJ2000")
    mag_col = col("Bmag")
    if pgc_col is None or ra_col is None or dec_col is None or mag_col is None:
        raise RuntimeError("PGC katalog neobsahuje očekávané sloupce.")

    log_major_col = col("logD25")
    log_ratio_col = col("logR25")
    pa_col = col("PA")
    # GLADE 1 contains homogenized HyperLEDA B and B-V for a subset of the
    # objects.  Those values provide a much better visual-band estimate than B
    # alone: V = B - (B-V).  Match by position because GLADE 1 does not expose
    # its PGC identifier through VizieR TAP.
    visual_photometry = _query_glade_visual_photometry(
        ra, dec, radius_deg, blue_magnitude_limit, cache_dir
    )
    visual_coordinates = None
    if visual_photometry:
        visual_coordinates = SkyCoord(
            [item[0] for item in visual_photometry] * u.deg,
            [item[1] for item in visual_photometry] * u.deg,
        )

    galaxies: list[Galaxy] = []
    seen_pgc: set[int] = set()
    for index in range(len(table)):
        pgc = int(pgc_col[index])
        if pgc in seen_pgc:
            continue
        seen_pgc.add(pgc)
        major_log = _optional_float(log_major_col, index)
        ratio_log = _optional_float(log_ratio_col, index)
        major = None if major_log is None else 0.1 * (10.0 ** major_log)
        minor = (
            None
            if major is None or ratio_log is None
            else major / (10.0 ** ratio_log)
        )
        magnitude = float(mag_col[index])
        magnitude_band = "B"
        if visual_coordinates is not None:
            coordinate = SkyCoord(float(ra_col[index]) * u.deg, float(dec_col[index]) * u.deg)
            nearest, separation, _ = coordinate.match_to_catalog_sky(
                visual_coordinates
            )
            if separation.arcsec <= 3.0:
                magnitude = visual_photometry[int(nearest)][2]
                magnitude_band = "V"
        if magnitude_band == "V" and magnitude > visual_magnitude_limit:
            continue
        galaxies.append(
            Galaxy(
                name=f"PGC {pgc}",
                ra=float(ra_col[index]),
                dec=float(dec_col[index]),
                magnitude=magnitude,
                major_arcmin=major,
                minor_arcmin=minor,
                position_angle_deg=_optional_float(pa_col, index),
                pgc=pgc,
                magnitude_band=magnitude_band,
                magnitude_approximate=True,
            )
        )
    return galaxies


def _query_glade_visual_photometry(
    ra: float,
    dec: float,
    radius_deg: float,
    blue_magnitude_limit: float,
    cache_dir: Path | None,
) -> list[tuple[float, float, float]]:
    query = f"""
        SELECT TOP 20000
            "RAJ2000", "DEJ2000", "BmagHypC", "B-VHypC"
        FROM "VII/275/glade1"
        WHERE CONTAINS(
                POINT('ICRS', "RAJ2000", "DEJ2000"),
                CIRCLE('ICRS', {ra:.10f}, {dec:.10f}, {radius_deg:.10f})
              ) = 1
          AND "BmagHypC" IS NOT NULL
          AND "B-VHypC" IS NOT NULL
          AND "BmagHypC" <= {blue_magnitude_limit:.3f}
    """
    cache_path = None
    if cache_dir is not None:
        catalog_cache = cache_dir / "catalogs"
        catalog_cache.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(query.encode("utf-8")).hexdigest()[:24]
        cache_path = catalog_cache / f"glade-v-{key}.vot"

    try:
        if cache_path is not None and cache_path.exists():
            content = cache_path.read_bytes()
        else:
            response = requests.post(
                VIZIER_TAP_URL,
                data={
                    "request": "doQuery",
                    "lang": "adql",
                    "format": "votable",
                    "query": query,
                },
                headers={"User-Agent": "SN-Hunter/0.1 (visual supernova search)"},
                timeout=(15, 180),
            )
            response.raise_for_status()
            content = response.content
            if cache_path is not None:
                _write_cache_atomic(cache_path, content)
        table = parse_single_table(BytesIO(content)).to_table()
    except Exception:
        # This is an optional photometric refinement. The complete GLADE 2 PGC
        # layer remains available with its explicitly labelled B magnitude.
        return []

    names = {name.lower(): name for name in table.colnames}
    required = ["raj2000", "dej2000", "bmaghypc", "b-vhypc"]
    if any(name not in names for name in required):
        return []
    ra_col = table[names["raj2000"]]
    dec_col = table[names["dej2000"]]
    blue_col = table[names["bmaghypc"]]
    color_col = table[names["b-vhypc"]]
    result = []
    for index in range(len(table)):
        if np.ma.is_masked(blue_col[index]) or np.ma.is_masked(color_col[index]):
            continue
        visual_magnitude = float(blue_col[index]) - float(color_col[index])
        if np.isfinite(visual_magnitude):
            result.append(
                (float(ra_col[index]), float(dec_col[index]), visual_magnitude)
            )
    return result


def query_deep_sky_objects(
    ra: float,
    dec: float,
    radius_deg: float,
    cache_dir: Path | None = None,
) -> list[DeepSkyObject]:
    """Return non-galaxy NGC/IC objects from VizieR's NGC 2000.0."""
    # NGC 2000.0 stores FK4 coordinates at equinox B2000 rather than ICRS.
    # Transform the query centre into the catalogue frame and transform the
    # returned positions back to ICRS below.
    center_b2000 = SkyCoord(ra * u.deg, dec * u.deg, frame="icrs").transform_to(
        FK4(equinox=Time("B2000"))
    )
    catalog_radius = radius_deg + 0.5
    query = f"""
        SELECT TOP 20000
            "Name", "Type", "RAB2000", "DEB2000", "size", "mag", "n_mag"
        FROM "VII/118/ngc2000"
        WHERE CONTAINS(
                POINT('FK4', "RAB2000", "DEB2000"),
                CIRCLE('FK4', {center_b2000.ra.degree:.10f},
                              {center_b2000.dec.degree:.10f}, {catalog_radius:.10f})
              ) = 1
    """
    cache_path = None
    if cache_dir is not None:
        catalog_cache = cache_dir / "catalogs"
        catalog_cache.mkdir(parents=True, exist_ok=True)
        key = hashlib.sha256(query.encode("utf-8")).hexdigest()[:24]
        cache_path = catalog_cache / f"ngc2000-{key}.vot"

    if cache_path is not None and cache_path.exists():
        content = cache_path.read_bytes()
    else:
        try:
            response = requests.post(
                VIZIER_TAP_URL,
                data={
                    "request": "doQuery",
                    "lang": "adql",
                    "format": "votable",
                    "query": query,
                },
                headers={"User-Agent": "SN-Hunter/0.1 (visual supernova search)"},
                timeout=(15, 180),
            )
            response.raise_for_status()
        except requests.Timeout as exc:
            raise RuntimeError("NGC/IC katalog VizieR neodpověděl včas.") from exc
        except requests.RequestException as exc:
            raise RuntimeError(f"Stažení NGC/IC katalogu selhalo: {exc}") from exc
        content = response.content
        if cache_path is not None:
            _write_cache_atomic(cache_path, content)

    try:
        table = parse_single_table(BytesIO(content)).to_table()
    except Exception as exc:
        detail = content[:500].decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"VizieR vrátil nečitelný NGC/IC katalog. {detail}") from exc

    names = {name.lower(): name for name in table.colnames}

    def column(name: str):
        actual = names.get(name.lower())
        return table[actual] if actual else None

    name_col = column("Name")
    type_col = column("Type")
    ra_col = column("RAB2000")
    dec_col = column("DEB2000")
    size_col = column("size")
    mag_col = column("mag")
    mag_note_col = column("n_mag")
    if name_col is None or type_col is None or ra_col is None or dec_col is None:
        raise RuntimeError("NGC/IC katalog neobsahuje očekávané souřadnice a typy.")

    # NGC 2000.0 positions are often rounded to arcminutes.  Overlay markers
    # need better positions, so match names against Corwin's accurate J2000
    # position catalogue.  Falling back to the older position still leaves a
    # useful layer if the auxiliary query is temporarily unavailable.
    position_query = f"""
        SELECT TOP 20000
            "Cat", "NGC/IC", "RAJ2000", "DEJ2000"
        FROM "VII/239A/icpos"
        WHERE CONTAINS(
                POINT('ICRS', "RAJ2000", "DEJ2000"),
                CIRCLE('ICRS', {ra:.10f}, {dec:.10f}, {catalog_radius:.10f})
              ) = 1
    """
    precise_positions: dict[str, list[tuple[float, float]]] = {}
    try:
        position_cache = None
        if cache_dir is not None:
            key = hashlib.sha256(position_query.encode("utf-8")).hexdigest()[:24]
            position_cache = cache_dir / "catalogs" / f"ngcic-pos-{key}.vot"
        if position_cache is not None and position_cache.exists():
            position_content = position_cache.read_bytes()
        else:
            position_response = requests.post(
                VIZIER_TAP_URL,
                data={
                    "request": "doQuery",
                    "lang": "adql",
                    "format": "votable",
                    "query": position_query,
                },
                headers={"User-Agent": "SN-Hunter/0.1 (visual supernova search)"},
                timeout=(15, 180),
            )
            position_response.raise_for_status()
            position_content = position_response.content
            if position_cache is not None:
                _write_cache_atomic(position_cache, position_content)
        position_table = parse_single_table(BytesIO(position_content)).to_table()
        position_names = {name.lower(): name for name in position_table.colnames}
        cat_values = position_table[position_names["cat"]]
        number_name = position_names.get("ngc/ic") or position_names.get("ngc_ic")
        if number_name is None:
            raise KeyError("NGC/IC")
        number_values = position_table[number_name]
        position_ra = position_table[position_names["raj2000"]]
        position_dec = position_table[position_names["dej2000"]]
        for index in range(len(position_table)):
            prefix = "IC" if str(cat_values[index]).strip().upper() == "I" else "NGC"
            key = f"{prefix} {int(number_values[index])}"
            precise_positions.setdefault(key, []).append(
                (float(position_ra[index]), float(position_dec[index]))
            )
    except Exception:
        precise_positions = {}

    categories = {
        "OC": "cluster",
        "Gb": "cluster",
        "Nb": "nebula",
        "Pl": "nebula",
        "C+N": "nebula",
        "Kt": "nebula",
        "Ast": "other",
    }
    result: list[DeepSkyObject] = []
    for index in range(len(table)):
        object_type = str(type_col[index]).strip()
        category = categories.get(object_type)
        if category is None:
            continue
        raw_name = str(name_col[index]).strip()
        if not raw_name:
            continue
        if raw_name.startswith("I"):
            name = f"IC {raw_name[1:].strip()}"
        else:
            name = f"NGC {raw_name}"
        magnitude = _optional_float(mag_col, index)
        size = _optional_float(size_col, index)
        photographic = (
            mag_note_col is not None
            and not np.ma.is_masked(mag_note_col[index])
            and str(mag_note_col[index]).strip().lower() == "p"
        )
        coordinate = SkyCoord(
            float(ra_col[index]) * u.deg,
            float(dec_col[index]) * u.deg,
            frame=FK4(equinox=Time("B2000")),
        ).icrs
        precise = precise_positions.get(name)
        if precise:
            coordinate = SkyCoord(
                np.mean([value[0] for value in precise]) * u.deg,
                np.mean([value[1] for value in precise]) * u.deg,
                frame="icrs",
            )
        result.append(
            DeepSkyObject(
                name=name,
                ra=coordinate.ra.degree,
                dec=coordinate.dec.degree,
                object_type=object_type,
                category=category,
                magnitude=magnitude,
                size_arcmin=size,
                photographic_magnitude=photographic,
            )
        )
    result.sort(key=lambda item: (item.category, item.name))
    return result


def _optional_float(column, index: int) -> float | None:
    if column is None or np.ma.is_masked(column[index]):
        return None
    try:
        value = float(column[index])
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def download_reference(
    galaxy: Galaxy,
    survey_id: str,
    cache_dir: Path,
    fov_arcmin: float,
    pixels: int = 900,
) -> FitsImage:
    cache_dir.mkdir(parents=True, exist_ok=True)
    params = _reference_params(galaxy, survey_id, fov_arcmin, pixels)
    path = reference_cache_path(galaxy, survey_id, cache_dir, fov_arcmin, pixels)
    with _REFERENCE_CACHE_LOCK:
        prune_reference_cache(cache_dir, protected=path)
        if path.exists():
            try:
                image = _read_reference_fits(path)
                # mtime acts as a simple last-used timestamp for cache eviction.
                path.touch()
                return image
            except Exception:
                # A previous interrupted download may have left an invalid
                # cache entry. It is safe to replace this derived file.
                path.unlink(missing_ok=True)

    response = requests.get(HIPS2FITS_URL, params=params, timeout=90)
    response.raise_for_status()
    content_type = response.headers.get("content-type", "")
    if "fits" not in content_type.lower() and not response.content.startswith(b"SIMPLE"):
        raise RuntimeError("Archivní služba nevrátila FITS snímek.")

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=cache_dir, prefix="download-", suffix=".fits", delete=False
        ) as temporary:
            temporary.write(response.content)
            temporary_path = Path(temporary.name)
        # Keep replacement, pruning and the first read in one critical section.
        # Otherwise the other prefetch worker could evict this file between
        # replacement and reading it.
        with _REFERENCE_CACHE_LOCK:
            temporary_path.replace(path)
            temporary_path = None
            prune_reference_cache(cache_dir, protected=path)
            return _read_reference_fits(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink(missing_ok=True)


def prune_reference_cache(
    cache_dir: Path,
    max_bytes: int = REFERENCE_CACHE_MAX_BYTES,
    protected: Path | None = None,
) -> tuple[int, int]:
    """Keep derived archive FITS files within a bounded least-recently-used cache."""
    with _REFERENCE_CACHE_LOCK:
        if not cache_dir.exists():
            return 0, 0
        entries: list[tuple[float, int, Path]] = []
        total = 0
        for path in cache_dir.glob("*.fits"):
            if path.name.startswith("download-"):
                continue
            try:
                stat = path.stat()
            except OSError:
                continue
            total += stat.st_size
            entries.append((stat.st_mtime, stat.st_size, path))
        removed = 0
        protected_resolved = protected.resolve() if protected is not None else None
        for _mtime, size, path in sorted(entries):
            if total <= max_bytes:
                break
            if protected_resolved is not None and path.resolve() == protected_resolved:
                continue
            try:
                path.unlink()
            except OSError:
                continue
            total -= size
            removed += 1
        return removed, total


def _reference_params(
    galaxy: Galaxy, survey_id: str, fov_arcmin: float, pixels: int
) -> dict[str, str]:
    return {
        "hips": survey_id,
        "width": str(pixels),
        "height": str(pixels),
        "fov": str(fov_arcmin / 60.0),
        "projection": "TAN",
        "coordsys": "icrs",
        "ra": f"{galaxy.ra:.8f}",
        "dec": f"{galaxy.dec:.8f}",
        "format": "fits",
    }


def reference_cache_path(
    galaxy: Galaxy,
    survey_id: str,
    cache_dir: Path,
    fov_arcmin: float,
    pixels: int,
) -> Path:
    params = _reference_params(galaxy, survey_id, fov_arcmin, pixels)
    key = hashlib.sha256(repr(sorted(params.items())).encode("utf-8")).hexdigest()[:20]
    return cache_dir / f"{key}.fits"


def _read_reference_fits(path: Path) -> FitsImage:
    with fits.open(path, memmap=False) as hdul:
        hdu = next(h for h in hdul if h.data is not None and h.data.ndim >= 2)
        data = np.asarray(hdu.data, dtype=np.float64).squeeze()
        header = hdu.header.copy()
    return FitsImage(data=data, header=header, wcs=WCS(header).celestial, path=path)

