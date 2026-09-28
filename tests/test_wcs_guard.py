import json
import os
import tempfile
import time
import warnings
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from astropy.io import fits
from astropy.io.votable import from_table, writeto
from astropy.table import Table
from astropy.wcs import WCS
from PySide6.QtCore import QSettings, QSignalBlocker
from PySide6.QtWidgets import QApplication

from sn_hunter.app import MainWindow
import sn_hunter.app as app_module
import sn_hunter.services as services_module
from sn_hunter.imaging import FitsImage, make_comparison_grid, stretch_image
from sn_hunter.services import DeepSkyObject, Galaxy


def _votable_bytes(table: Table) -> bytes:
    output = BytesIO()
    writeto(from_table(table), output)
    return output.getvalue()


def test_blink_uses_independent_archive_and_current_stretches():
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    assert window.windowTitle() == "SN Hunter 1.0"
    assert "Page Up" in window.previous_button.text()
    assert "Page Down" in window.next_button.text()
    assert window.blink_button.shortcut().toString() == "End"
    assert window.current_marker_button.isChecked()
    assert not window.marker_button.isChecked()
    assert window.current_panel.markers.isVisible()
    assert not window.reference_panel.markers.isVisible()
    window.current_marker_button.setChecked(False)
    assert not window.current_panel.markers.isVisible()
    assert not window.reference_panel.markers.isVisible()
    window.current_marker_button.setChecked(True)
    window.marker_button.setChecked(True)
    assert window.current_panel.markers.isVisible()
    assert window.reference_panel.markers.isVisible()
    window.marker_button.setChecked(False)
    assert window.current_panel.markers.isVisible()
    assert not window.reference_panel.markers.isVisible()
    current_raw = np.arange(10000, dtype=float).reshape(100, 100)
    current_cutout = np.linspace(2000.0, 3000.0, 400).reshape(20, 20)
    archive = np.linspace(-5.0, 5.0, 400).reshape(20, 20)

    window.current_panel.set_data(current_raw)
    window.current_panel.stretch.setCurrentText("asinh")
    window.current_panel.low.setValue(5.0)
    window.current_panel.high.setValue(95.0)
    window.current_panel.parameter.setValue(7.0)
    window.current_panel.invert.setChecked(True)

    window.reference_panel.set_data(archive)
    window.reference_panel.stretch.setCurrentText("sqrt")
    window.reference_panel.low.setValue(10.0)
    window.reference_panel.high.setValue(90.0)
    window.reference_panel.parameter.setValue(0.8)
    window.reference_panel.invert.setChecked(False)
    window.reference_panel.set_blink_data(
        archive, current_cutout, window.current_panel
    )

    window.reference_panel.showing_alternate = False
    window.reference_panel.refresh()
    expected_archive = stretch_image(
        archive, "sqrt", 10.0, 90.0, 0.8, statistics_data=archive
    )
    assert np.allclose(window.reference_panel.image_item.image, expected_archive)

    window.reference_panel.toggle_blink_frame()
    expected_current = 1.0 - stretch_image(
        current_cutout,
        "asinh",
        5.0,
        95.0,
        7.0,
        statistics_data=current_raw,
    )
    assert np.allclose(window.reference_panel.image_item.image, expected_current)

    with tempfile.TemporaryDirectory() as directory:
        window.settings = QSettings(
            str(Path(directory) / "blink.ini"), QSettings.IniFormat
        )
        window.blink_button.setEnabled(True)
        window.blink_button.setChecked(True)
        window._set_blink(True)
        assert window.blink_timer.isActive()
        assert window.blink_button.text() == "Zastavit blink (End)"
        window.blink_button.setChecked(False)
        assert not window.blink_timer.isActive()
        assert window.blink_button.text() == "Spustit blink (End)"
    window.close()
    app.processEvents()


def test_bright_pgc_supplements_hecate_without_duplicates():
    hecate = Table(
        {
            "PGC": [100],
            "HyperLEDA": ["NGC123"],
            "NED": [""],
            "RAJ2000": [180.0],
            "DEJ2000": [20.0],
            "Vtmag": [12.3],
            "MajAxis": [2.0],
            "MinAxis": [1.0],
            "PA": [45.0],
        }
    )
    pgc = Table(
        {
            "PGC": [100, 101, 102],
            "RAJ2000": [180.0, 180.2, 180.4],
            "DEJ2000": [20.0, 20.1, 20.2],
            "Bmag": [13.1, 14.2, 15.2],
            "Flag1": ["G", "G", "G"],
            "logD25": [np.log10(20.0), np.log10(12.0), np.log10(8.0)],
            "logR25": [np.log10(2.0), np.log10(2.0), np.log10(2.0)],
            "PA": [45, 70, 20],
            "ANames": ["NGC123", "", ""],
        }
    )
    visual = Table(
        {
            "RAJ2000": [180.2],
            "DEJ2000": [20.1],
            "BmagHypC": [14.2],
            "B-VHypC": [0.7],
        }
    )
    payloads = iter(
        [_votable_bytes(hecate), _votable_bytes(pgc), _votable_bytes(visual)]
    )

    class Response:
        def __init__(self, content):
            self.content = content

        def raise_for_status(self):
            return None

    original_post = services_module.requests.post
    services_module.requests.post = lambda *args, **kwargs: Response(next(payloads))
    try:
        galaxies = services_module.query_galaxies(180.0, 20.0, 1.0, 16.0)
    finally:
        services_module.requests.post = original_post

    assert len(galaxies) == 3
    assert sum(galaxy.pgc == 100 for galaxy in galaxies) == 1
    primary = next(galaxy for galaxy in galaxies if galaxy.pgc == 100)
    supplemental = next(galaxy for galaxy in galaxies if galaxy.pgc == 101)
    blue_fallback = next(galaxy for galaxy in galaxies if galaxy.pgc == 102)
    assert primary.label == "NGC 123  V=12.3 mag"
    assert primary.major_arcmin == 4.0
    assert primary.minor_arcmin == 2.0
    assert supplemental.label == "PGC 101  V≈13.5 mag"
    assert blue_fallback.label == "PGC 102  B≈15.2 mag"
    assert abs(supplemental.major_arcmin - 1.2) < 1e-10


def test_wide_field_objects_are_not_transformed_into_small_reference_cutout():
    app = QApplication.instance() or QApplication([])
    wide_wcs = WCS(naxis=2)
    wide_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wide_wcs.wcs.crval = [180.0, 20.0]
    wide_wcs.wcs.crpix = [500.5, 500.5]
    wide_wcs.wcs.cd = np.array([[-0.015, 0.0], [0.0, 0.015]])
    wide = FitsImage(np.zeros((1000, 1000)), fits.Header(), wide_wcs)
    reference = make_comparison_grid(wide, 180.0, 20.0, 8.0, 0.25)

    window = MainWindow()
    assert window._default_magnitude_limit(wide) == 14.5
    window.current = wide
    window.current_panel.set_wcs(wide.wcs)
    window.current_panel.set_data(wide.data)
    window._copy_coordinates(window.current_panel, 499.5, 499.5)
    assert QApplication.clipboard().text() == "12 00 00 +20 00 00"
    window.current_panel._update_zoom_label([[0.0, 100.0], [0.0, 100.0]])
    expected_zoom = window.current_panel.view.getViewBox().width()
    assert expected_zoom > 0
    assert abs(window.current_panel.zoom_percent - expected_zoom) < 1e-9
    assert window.current_panel.zoom_label.text().startswith("Zoom ")
    assert window.current_panel.zoom_label.text().endswith(" %")
    window.current_panel.set_zoom_percent(200.0)
    assert abs(window.current_panel.zoom_percent - 200.0) < 1.0
    window.current_panel.zoom_selector.setEditText("125 %")
    window.current_panel._zoom_text_entered()
    assert abs(window.current_panel.zoom_percent - 125.0) < 1.0
    window.reference_panel.set_wcs(reference.wcs)
    window.reference_panel.set_data(reference.data)
    window.galaxies = [
        Galaxy("CENTER", 180.0, 20.0, 12.0),
        Galaxy("FAR-1", 186.0, 24.0, 13.0),
        Galaxy("FAR-2", 174.0, 16.0, 14.0),
    ]
    window.deep_sky_objects = [
        DeepSkyObject(
            "NGC 0001",
            180.01,
            20.01,
            "OC",
            "cluster",
            magnitude=7.3,
            size_arcmin=30.0,
        ),
        DeepSkyObject("NGC 0002", 186.0, 24.0, "Nb", "nebula"),
    ]
    window.galaxy_list.addItems([galaxy.label for galaxy in window.galaxies])

    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=r".*WCS\.all_world2pix.*")
        window._refresh_markers()
        window._galaxy_selected(1)
        window._cursor_moved(window.current_panel, 100.0, 100.0)
        window._sync_range(
            window.current_panel,
            window.reference_panel,
            [[50.0, 150.0], [50.0, 150.0]],
        )

    reference_points = window.reference_panel.markers.points()
    assert len(reference_points) == 1
    assert reference_points[0].data() == 0
    dso_points = window.reference_panel.dso_markers.points()
    assert len(dso_points) == 1
    tooltip = window.reference_panel.dso_markers.opts["tip"](
        x=0.0, y=0.0, data=dso_points[0].data()
    )
    assert "NGC 0001" in tooltip
    assert "otevřená hvězdokupa" in tooltip
    assert "7.3 mag" in tooltip
    assert dso_points[0].size() > 100
    window.dso_button.setChecked(False)
    assert not window.reference_panel.dso_markers.isVisible()
    window.close()
    app.processEvents()


def test_archive_is_centered_when_galaxy_is_near_current_image_edge():
    app = QApplication.instance() or QApplication([])
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [180.0, 20.0]
    wcs.wcs.crpix = [500.5, 500.5]
    wcs.wcs.cd = np.array([[-0.001, 0.0], [0.0, 0.001]])
    current = FitsImage(np.zeros((1000, 1000)), fits.Header(), wcs)
    coordinate = current.wcs.pixel_to_world(930.0, 510.0)
    reference = make_comparison_grid(
        current, coordinate.ra.degree, coordinate.dec.degree, 8.0, 1.0
    )

    window = MainWindow()
    window.current = current
    window.current_panel.set_wcs(current.wcs)
    window.current_panel.set_data(current.data)
    window.reference_panel.set_wcs(reference.wcs)
    window.reference_panel.set_data(reference.data)

    assert window._focus_coordinate(coordinate, 8.0)
    # Simulate the archive retaining a previous cutout's view. The current
    # panel is already centered, so setting its same range again may emit no
    # signal; explicit two-panel focusing must still repair the archive.
    window.syncing = True
    window.reference_panel.view.setRange(
        xRange=[-800.0, -300.0], yRange=[-700.0, -200.0], padding=0
    )
    window.syncing = False
    assert window._focus_coordinate(coordinate, 8.0)

    for panel in (window.current_panel, window.reference_panel):
        expected_x, expected_y = panel.wcs.world_to_pixel(coordinate)
        ranges = panel.view.viewRange()
        actual_x = (ranges[0][0] + ranges[0][1]) / 2.0
        actual_y = (ranges[1][0] + ranges[1][1]) / 2.0
        assert abs(actual_x - expected_x) < 1e-6
        assert abs(actual_y - expected_y) < 1e-6
    window.close()
    app.processEvents()


def test_prefetch_runs_for_every_row_and_keeps_displayed_reference():
    app = QApplication.instance() or QApplication([])
    wide_wcs = WCS(naxis=2)
    wide_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wide_wcs.wcs.crval = [180.0, 20.0]
    wide_wcs.wcs.crpix = [50.5, 50.5]
    wide_wcs.wcs.cd = np.array([[-0.01, 0.0], [0.0, 0.01]])
    wide = FitsImage(np.zeros((100, 100)), fits.Header(), wide_wcs)
    shown_reference = np.arange(10000, dtype=float).reshape(100, 100)
    calls = []

    def fake_download(galaxy, survey_id, cache_dir, fov_arcmin, pixels=900):
        calls.append((galaxy.name, fov_arcmin, pixels))
        return wide

    original_download = app_module.download_reference
    app_module.download_reference = fake_download
    window = MainWindow()
    try:
        window.current = wide
        window.current_panel.set_wcs(wide.wcs)
        window.current_panel.set_data(wide.data)
        window.reference_panel.set_wcs(wide.wcs)
        window.reference_panel.set_data(shown_reference)
        window.galaxies = [
            Galaxy("ONE", 180.0, 20.0, 12.0, major_arcmin=1.0),
            Galaxy("TWO", 180.1, 20.0, 13.0, major_arcmin=8.0),
        ]
        window.galaxy_list.addItems([galaxy.label for galaxy in window.galaxies])
        window.galaxy_list.setCurrentRow(0)
        window.galaxy_list.item(0).setSelected(True)
        window.galaxy_list.item(1).setSelected(True)
        window.prefetch_selected()

        deadline = time.monotonic() + 5.0
        while window.prefetch_in_progress and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)

        by_name = {name: (fov, pixels) for name, fov, pixels in calls}
        assert set(by_name) == {"ONE", "TWO"}
        assert abs(by_name["ONE"][0] - 8.0 * np.sqrt(2.0)) < 1e-9
        assert abs(by_name["TWO"][0] - 12.0 * np.sqrt(2.0)) < 1e-9
        assert by_name["TWO"][1] > by_name["ONE"][1]
        assert not window.prefetch_in_progress
        assert window.reference_panel.raw_data is shown_reference
        window.resize(1400, 900)
        window.show()
        app.processEvents()
        window.syncing = True
        window.current_panel.view.setRange(
            xRange=[20.0, 80.0], yRange=[20.0, 80.0], padding=0
        )
        window.reference_panel.view.setRange(
            xRange=[15.0, 85.0], yRange=[15.0, 85.0], padding=0
        )
        window.syncing = False
        for _ in range(2):
            window.reference_panel_button.setChecked(False)
            app.processEvents()
            assert window.reference_panel.isHidden()
            assert window.reference_panel.raw_data is shown_reference
            window.reference_panel_button.setChecked(True)
            app.processEvents()
            assert not window.reference_panel.isHidden()
        assert window.reference_panel.raw_data is shown_reference
        for panel in (window.current_panel, window.reference_panel):
            ranges = np.asarray(panel.view.viewRange())
            assert np.all(np.isfinite(ranges))
            assert 0 < np.ptp(ranges[0]) < 1000
            assert panel.zoom_percent is not None
            assert panel.zoom_percent > 0.1

        center_x, center_y = wide.wcs.world_to_pixel_values(180.12, 20.08)
        window.current_panel.view.setRange(
            xRange=[center_x - 10.0, center_x + 10.0],
            yRange=[center_y - 10.0, center_y + 10.0],
            padding=0,
        )
        window.load_reference_at_view_center()
        deadline = time.monotonic() + 5.0
        while window.loading_reference_key is not None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        assert window.loading_reference_key is None
        assert "pozice" in window.reference_panel.title.text()
        assert abs(window.reference_panel.wcs.wcs.crval[0] - 180.12) < 1e-6
        assert abs(window.reference_panel.wcs.wcs.crval[1] - 20.08) < 1e-6
    finally:
        app_module.download_reference = original_download
        window.close()
        app.processEvents()


def test_adaptive_reference_size_and_bounded_cache():
    assert MainWindow._effective_reference_fov(
        Galaxy("SMALL", 1.0, 2.0, major_arcmin=1.0), 8.0
    ) == 8.0
    assert MainWindow._effective_reference_fov(
        Galaxy("LARGE", 1.0, 2.0, major_arcmin=8.0), 8.0
    ) == 12.0
    assert MainWindow._effective_reference_fov(
        Galaxy("HUGE", 1.0, 2.0, major_arcmin=20.0), 8.0
    ) == 15.0
    # NGC 891 has an optical diameter around 12.3', so it must receive the
    # full automatic 15' cutout rather than the formerly produced ~10'.
    assert MainWindow._effective_reference_fov(
        Galaxy("NGC 891", 35.6, 42.35, major_arcmin=12.3), 8.0
    ) == 15.0
    assert MainWindow._effective_reference_fov(
        Galaxy("MANUAL", 1.0, 2.0, major_arcmin=20.0), 20.0
    ) == 20.0
    assert MainWindow._effective_reference_fov(
        Galaxy("UNKNOWN", 1.0, 2.0), 8.0
    ) == 8.0

    with tempfile.TemporaryDirectory() as directory:
        cache = Path(directory)
        oldest = cache / "oldest.fits"
        newer = cache / "newer.fits"
        protected = cache / "protected.fits"
        for path in (oldest, newer, protected):
            path.write_bytes(b"123456")
        os.utime(oldest, (1, 1))
        os.utime(newer, (2, 2))
        os.utime(protected, (3, 3))

        removed, remaining = services_module.prune_reference_cache(
            cache, max_bytes=12, protected=protected
        )
        assert removed == 1
        assert remaining == 12
        assert not oldest.exists()
        assert newer.exists()
        assert protected.exists()


def test_catalog_result_does_not_zoom_to_first_galaxy():
    app = QApplication.instance() or QApplication([])
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [180.0, 20.0]
    wcs.wcs.crpix = [100.5, 100.5]
    wcs.wcs.cd = np.array([[-0.02, 0.0], [0.0, 0.02]])
    image = FitsImage(np.zeros((200, 200)), fits.Header(), wcs)
    window = MainWindow()
    assert window._default_magnitude_limit(image) == 16.0
    window.current = image
    window.current_panel.set_wcs(wcs)
    window.current_panel.set_data(image.data)
    window.current_panel.view.setRange(
        xRange=[20.0, 180.0], yRange=[30.0, 170.0], padding=0
    )
    before = np.asarray(window.current_panel.view.viewRange())

    window._galaxies_ready(
        [
            Galaxy("TEST", 180.3, 20.2, 12.0, major_arcmin=1.0),
            Galaxy("NEAR", 180.32, 20.2, 13.0, major_arcmin=1.0),
            Galaxy("FAR", 180.5, 20.2, 14.0, major_arcmin=1.0),
        ]
    )
    after = np.asarray(window.current_panel.view.viewRange())

    assert window.galaxy_list.currentRow() == -1
    assert np.allclose(after, before)
    assert window.review_groups == [[0, 1], [2]]
    window._step_galaxy(1)
    assert window.galaxy_list.currentRow() == 0
    assert "Výřez 1 / 2" in window.review_position_label.text()
    assert "2 galaxií" in window.review_position_label.text()
    window._step_galaxy(1)
    assert window.galaxy_list.currentRow() == 2
    assert "Výřez 2 / 2" in window.review_position_label.text()
    window.close()
    app.processEvents()


def test_late_manual_catalog_result_cannot_overwrite_a_new_field():
    app = QApplication.instance() or QApplication([])

    def image_at(ra: float, path: Path) -> FitsImage:
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [ra, 20.0]
        wcs.wcs.crpix = [50.5, 50.5]
        wcs.wcs.cd = np.array([[-0.01, 0.0], [0.0, 0.01]])
        return FitsImage(np.zeros((100, 100)), fits.Header(), wcs, path=path)

    window = MainWindow()
    pending = []
    original_submit = window._submit
    window._submit = lambda function, callback, error_callback=None, **_kwargs: pending.append(
        (function, callback, error_callback)
    )
    try:
        first = image_at(180.0, Path("first.fit"))
        second = image_at(190.0, Path("second.fit"))
        window._display_fits_image(first, first.path)
        window.find_galaxies()
        assert len(pending) == 2

        window._display_fits_image(second, second.path)
        entry = app_module.NightField(path=second.path, image=second)
        window.night_fields = [entry]
        blocker = QSignalBlocker(window.night_list)
        window.night_list.addItem(second.path.name)
        window.night_list.setCurrentRow(0)
        del blocker
        pending[0][1]([Galaxy("STALE", 180.0, 20.0, 12.0)])
        pending[1][1]([
            DeepSkyObject("STALE DSO", 180.0, 20.0, "OC", "cluster")
        ])
        assert window.galaxies == []
        assert window.deep_sky_objects == []
        assert window.galaxy_list.count() == 0

        window.find_galaxies()
        assert len(pending) == 4
        window._night_progress(
            (
                "catalogs",
                0,
                [Galaxy("BACKGROUND", 190.0, 20.0, 13.0)],
                [DeepSkyObject("BACKGROUND DSO", 190.0, 20.0, "OC", "cluster")],
            )
        )
        assert window.galaxies == []
        assert [galaxy.name for galaxy in entry.galaxies] == ["BACKGROUND"]
        pending[2][1]([Galaxy("CURRENT", 190.0, 20.0, 12.0)])
        pending[3][1]([
            DeepSkyObject("CURRENT DSO", 190.0, 20.0, "OC", "cluster")
        ])
        assert [galaxy.name for galaxy in window.galaxies] == ["CURRENT"]
        assert [item.name for item in window.deep_sky_objects] == ["CURRENT DSO"]
    finally:
        window._submit = original_submit
        window.close()
        app.processEvents()


def test_night_queue_prepares_multiple_fields_in_background():
    app = QApplication.instance() or QApplication([])
    original_load = app_module.load_fits
    original_galaxies = app_module.query_galaxies
    original_dso = app_module.query_deep_sky_objects
    original_download = app_module.download_reference
    downloads = []
    magnitude_limits = []

    def fake_load(path):
        path = Path(path)
        wcs = WCS(naxis=2)
        wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        wcs.wcs.crval = [180.0, 20.0]
        wcs.wcs.crpix = [50.5, 50.5]
        wcs.wcs.cd = np.array([[-0.01, 0.0], [0.0, 0.01]])
        return FitsImage(np.zeros((100, 100)), fits.Header(), wcs, path=path)

    def fake_galaxies(_ra, _dec, _radius, magnitude_limit, *_args, **_kwargs):
        magnitude_limits.append(magnitude_limit)
        return [Galaxy("TEST GALAXY", 180.0, 20.0, 12.0)]

    def fake_dso(*_args, **_kwargs):
        return [DeepSkyObject("NGC 1", 180.1, 20.0, "OC", "cluster")]

    def fake_download(galaxy, *_args, **_kwargs):
        downloads.append(galaxy.name)
        return fake_load(Path("archive.fits"))

    app_module.load_fits = fake_load
    app_module.query_galaxies = fake_galaxies
    app_module.query_deep_sky_objects = fake_dso
    app_module.download_reference = fake_download
    window = MainWindow()
    try:
        window._add_night_paths([Path("field-1.fit"), Path("field-2.fit")])
        deadline = time.monotonic() + 5.0
        while window.night_processing and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)

        assert not window.night_processing
        assert len(window.night_fields) == 2
        assert all(entry.prepared for entry in window.night_fields)
        assert all(len(entry.galaxies) == 1 for entry in window.night_fields)
        assert all(len(entry.deep_sky_objects) == 1 for entry in window.night_fields)
        assert magnitude_limits == [17.5, 17.5]
        assert downloads == ["TEST GALAXY", "TEST GALAXY"]
        assert "připraveno" in window.night_list.item(0).text()
        assert "připraveno" in window.night_list.item(1).text()

        window.night_list.setCurrentRow(1)
        assert window.current.path == Path("field-2.fit")
        assert window.galaxy_list.count() == 1
        assert len(window.current_panel.dso_markers.points()) == 1
    finally:
        app_module.load_fits = original_load
        app_module.query_galaxies = original_galaxies
        app_module.query_deep_sky_objects = original_dso
        app_module.download_reference = original_download
        window.close()
        app.processEvents()


def test_display_settings_are_saved_and_restored_with_tenth_percent_steps():
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as directory:
        settings_path = str(Path(directory) / "display.ini")
        window = MainWindow()
        window.settings = QSettings(settings_path, QSettings.IniFormat)
        window.candidates = []
        window._refresh_candidate_list()
        candidate_wcs = WCS(naxis=2)
        candidate_wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
        candidate_wcs.wcs.crval = [180.0, 20.0]
        candidate_wcs.wcs.crpix = [50.5, 50.5]
        candidate_wcs.wcs.cd = np.array([[-0.01, 0.0], [0.0, 0.01]])
        candidate_image = FitsImage(
            np.zeros((100, 100)),
            fits.Header(),
            candidate_wcs,
            path=Path(directory) / "night-field.fit",
        )
        other_image = FitsImage(
            np.zeros((100, 100)),
            fits.Header(),
            candidate_wcs,
            path=Path(directory) / "other-field.fit",
        )
        candidate_image.path.touch()
        other_image.path.touch()
        window._display_fits_image(candidate_image, candidate_image.path)
        assert window.mag_limit.value() == 17.5
        window.mag_limit.setValue(15.5)
        window._display_fits_image(candidate_image, candidate_image.path)
        assert window.mag_limit.value() == 15.5
        window.candidate_mode_button.setChecked(True)
        window._candidate_clicked(50.0, 50.0)
        assert len(window.candidates) == 1
        assert window.candidate_list.count() == 1
        assert window.candidate_list_container.isHidden()
        window.candidate_list_button.setChecked(True)
        assert not window.candidate_list_container.isHidden()
        window.resize(1000, 700)
        window.show()
        app.processEvents()
        archive_label = next(
            label
            for label in window.findChildren(app_module.QLabel)
            if label.text() == "Archivní přehlídka"
        )
        assert window.galaxy_list.geometry().bottom() < archive_label.geometry().top()
        window.candidate_list_button.setChecked(False)
        assert window.candidate_list_container.isHidden()
        assert len(window.current_panel.candidate_markers.points()) == 1
        window._display_fits_image(other_image, other_image.path)
        assert window.mag_limit.value() == 17.5
        assert len(window.current_panel.candidate_markers.points()) == 0
        window._display_fits_image(candidate_image, candidate_image.path)
        assert len(window.current_panel.candidate_markers.points()) == 1

        # Selecting a saved point must request an archive centered on that point,
        # even when its FITS field is already open.  Previously the selection only
        # focused the current image and left an unrelated galaxy in the archive.
        requested_targets = []
        original_target_loader = window._load_reference_target
        window._load_reference_target = requested_targets.append
        window._candidate_selected(0)
        window._load_reference_target = original_target_loader
        assert len(requested_targets) == 1
        assert abs(requested_targets[0].ra - window.candidates[0].ra) < 1e-10
        assert abs(requested_targets[0].dec - window.candidates[0].dec) < 1e-10
        assert requested_targets[0].name.startswith("podezřelý bod ")

        # Starting that request must immediately remove the stale archival image.
        window.reference_panel.set_data(np.ones((20, 20)))
        original_submit = window._submit
        window._submit = lambda *args, **kwargs: None
        window._load_reference_target(requested_targets[0])
        window._submit = original_submit
        assert window.reference_panel.raw_data is None
        assert "načítám" in window.reference_panel.title.text()
        assert "podezřelý bod" in window.reference_panel.title.text()
        window.loading_reference_key = None

        window.current_panel.stretch.setCurrentText("power")
        window.current_panel.low.setValue(1.2)
        window.current_panel.high.setValue(99.9)
        window.current_panel.parameter.setValue(2.4)
        window.current_panel.invert.setChecked(False)
        normal_marker_color = window.current_panel.markers.opts["pen"].color()
        window.current_panel.invert.setChecked(True)
        inverted_marker_color = window.current_panel.markers.opts["pen"].color()
        assert normal_marker_color != inverted_marker_color
        assert inverted_marker_color.red() < 200
        assert inverted_marker_color.green() < 80
        inverted_compass_color = window.current_panel.north_line.opts["pen"].color()
        assert inverted_compass_color.red() >= 120
        assert inverted_compass_color.green() == 0
        assert inverted_compass_color.blue() >= 30
        window.current_panel.stretch.setCurrentText("asinh")
        window.current_panel.low.setValue(0.8)
        window.current_panel.high.setValue(99.2)
        window.current_panel.parameter.setValue(18.0)
        window.current_panel.stretch.setCurrentText("power")
        assert window.current_panel.low.value() == 1.2
        assert window.current_panel.high.value() == 99.9
        assert window.current_panel.parameter.value() == 2.4
        window.reference_panel.high.setValue(99.8)
        window.settings.sync()

        restored = MainWindow()
        restored.settings = QSettings(settings_path, QSettings.IniFormat)
        restored._load_candidates()
        restored._refresh_candidate_list()
        controls = (
            restored.current_panel.stretch,
            restored.current_panel.low,
            restored.current_panel.high,
            restored.current_panel.parameter,
            restored.current_panel.invert,
            restored.reference_panel.stretch,
            restored.reference_panel.low,
            restored.reference_panel.high,
            restored.reference_panel.parameter,
            restored.reference_panel.invert,
        )
        blockers = [QSignalBlocker(control) for control in controls]
        restored._restore_display_settings()
        del blockers

        assert restored.current_panel.stretch.currentText() == "power"
        assert restored.current_panel.low.value() == 1.2
        assert restored.current_panel.high.value() == 99.9
        assert restored.current_panel.high.singleStep() == 0.1
        assert restored.current_panel.parameter.value() == 2.4
        assert restored.current_panel.invert.isChecked()
        restored.current_panel.stretch.setCurrentText("asinh")
        assert restored.current_panel.low.value() == 0.8
        assert restored.current_panel.high.value() == 99.2
        assert restored.current_panel.parameter.value() == 18.0
        assert restored.reference_panel.high.value() == 99.8
        assert len(restored.candidates) == 1
        assert restored.candidate_list.count() == 1
        window.close()
        restored.close()
        app.processEvents()


def test_candidate_retention_removes_old_and_missing_sources():
    app = QApplication.instance() or QApplication([])
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        settings = QSettings(str(directory / "retention.ini"), QSettings.IniFormat)
        recent_path = directory / "recent.fit"
        old_path = directory / "old.fit"
        missing_path = directory / "missing.fit"
        recent_path.touch()
        old_path.touch()
        now = datetime.now().astimezone()
        values = [
            {
                "field_path": str(recent_path),
                "ra": 180.0,
                "dec": 20.0,
                "created": (now - timedelta(days=2)).isoformat(timespec="seconds"),
            },
            {
                "field_path": str(old_path),
                "ra": 181.0,
                "dec": 21.0,
                "created": (now - timedelta(days=8)).isoformat(timespec="seconds"),
            },
            {
                "field_path": str(missing_path),
                "ra": 182.0,
                "dec": 22.0,
                "created": now.isoformat(timespec="seconds"),
            },
        ]
        settings.setValue("candidates/json", json.dumps(values))
        settings.sync()

        window = MainWindow()
        window.settings = settings
        window._load_candidates()
        window._refresh_candidate_list()
        assert len(window.candidates) == 1
        assert Path(window.candidates[0].field_path) == recent_path
        assert window.candidate_list.count() == 1
        saved = json.loads(str(settings.value("candidates/json")))
        assert len(saved) == 1
        assert Path(saved[0]["field_path"]) == recent_path
        window.close()
        app.processEvents()


if __name__ == "__main__":
    test_blink_uses_independent_archive_and_current_stretches()
    test_bright_pgc_supplements_hecate_without_duplicates()
    test_wide_field_objects_are_not_transformed_into_small_reference_cutout()
    test_archive_is_centered_when_galaxy_is_near_current_image_edge()
    test_prefetch_runs_for_every_row_and_keeps_displayed_reference()
    test_adaptive_reference_size_and_bounded_cache()
    test_catalog_result_does_not_zoom_to_first_galaxy()
    test_late_manual_catalog_result_cannot_overwrite_a_new_field()
    test_night_queue_prepares_multiple_fields_in_background()
    test_display_settings_are_saved_and_restored_with_tenth_percent_steps()
    test_candidate_retention_removes_old_and_missing_sources()
    print("OK: WCS, candidates, DSO, night queue, archive, and display tests")
