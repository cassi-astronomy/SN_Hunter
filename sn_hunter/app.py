from __future__ import annotations

import sys
import traceback
import math
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.wcs.utils import proj_plane_pixel_scales, skycoord_to_pixel
from PySide6.QtCore import QObject, QSettings, QSignalBlocker, QTimer, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QComboBox,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .imaging import (
    FitsImage,
    field_center_radius,
    field_size_deg,
    load_fits,
    make_comparison_grid,
    reproject_to_image,
)
from .services import (
    SURVEYS,
    SURVEY_PIXEL_SCALE_ARCSEC,
    DeepSkyObject,
    Galaxy,
    download_reference,
    query_deep_sky_objects,
    query_galaxies,
    reference_cache_path,
)
from .widgets import ImagePanel


@dataclass
class NightField:
    path: Path
    image: FitsImage | None = None
    galaxies: list[Galaxy] = field(default_factory=list)
    deep_sky_objects: list[DeepSkyObject] = field(default_factory=list)
    status: str = "čeká"
    processing: bool = False
    prepared: bool = False


@dataclass
class Candidate:
    field_path: str
    ra: float
    dec: float
    created: str


class FutureBridge(QObject):
    completed = Signal(object)
    failed = Signal(str)
    progress = Signal(object)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SN Hunter")
        self.resize(1500, 900)
        self.setAcceptDrops(True)
        self.settings = QSettings("SN Hunter", "SN Hunter")
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="sn-hunter")
        self.bridges: set[FutureBridge] = set()
        self.current: FitsImage | None = None
        self.reference_aligned: np.ndarray | None = None
        self.galaxies: list[Galaxy] = []
        self.deep_sky_objects: list[DeepSkyObject] = []
        self.syncing = False
        self.prefetch_in_progress = False
        self.prefetch_focus_name: str | None = None
        self.loading_reference_key = None
        self.loaded_reference_key = None
        self.night_fields: list[NightField] = []
        self.night_processing = False
        self.candidates: list[Candidate] = []
        self.visible_candidates: list[Candidate] = []

        self.current_panel = ImagePanel("Aktuální snímek")
        self.reference_panel = ImagePanel("Archivní podklad")
        self.current_panel.range_changed.connect(
            lambda ranges: self._sync_range(self.current_panel, self.reference_panel, ranges)
        )
        self.reference_panel.range_changed.connect(
            lambda ranges: self._sync_range(self.reference_panel, self.current_panel, ranges)
        )
        self.current_panel.cursor_moved.connect(
            lambda x, y: self._cursor_moved(self.current_panel, x, y)
        )
        self.reference_panel.cursor_moved.connect(
            lambda x, y: self._cursor_moved(self.reference_panel, x, y)
        )
        self.current_panel.image_clicked.connect(self._candidate_clicked)
        self.current_panel.coordinate_copy_requested.connect(
            lambda x, y: self._copy_coordinates(self.current_panel, x, y)
        )
        self.reference_panel.coordinate_copy_requested.connect(
            lambda x, y: self._copy_coordinates(self.reference_panel, x, y)
        )
        self.galaxy_list = QListWidget()
        self.galaxy_list.setMinimumHeight(180)
        self.galaxy_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.galaxy_list.currentRowChanged.connect(self._galaxy_selected)
        self.galaxy_list.itemSelectionChanged.connect(self._update_download_buttons)
        self.current_panel.marker_clicked.connect(self.galaxy_list.setCurrentRow)
        self.reference_panel.marker_clicked.connect(self.galaxy_list.setCurrentRow)
        self.night_list = QListWidget()
        self.night_list.setMaximumHeight(110)
        self.night_list.currentRowChanged.connect(self._night_field_selected)
        self.candidate_list = QListWidget()
        self.candidate_list.setMaximumHeight(115)
        self.candidate_list.currentRowChanged.connect(self._candidate_selected)
        self.candidate_list_button = QPushButton("Seznam podezřelých bodů (0)")
        self.candidate_list_button.setCheckable(True)
        self.candidate_list_button.toggled.connect(self._set_candidate_list_visible)
        self.mag_limit = QDoubleSpinBox()
        self.mag_limit.setRange(5.0, 22.0)
        self.mag_limit.setValue(16.0)
        self.mag_limit.setDecimals(1)
        self.mag_limit.setToolTip(
            "Automaticky: do 2° = 17,5 mag, 2–5° = 16 mag, "
            "nad 5° = 14,5 mag. "
            "Hodnotu lze pro aktuální pole ručně změnit."
        )
        self.survey = QComboBox()
        self.survey.addItems(SURVEYS.keys())
        self.fov = QDoubleSpinBox()
        self.fov.setRange(1.0, 120.0)
        self.fov.setValue(8.0)
        self.fov.setSuffix("′")

        find_button = QPushButton("Najít galaxie")
        find_button.clicked.connect(self.find_galaxies)
        add_night_button = QPushButton("Přidat snímky noci…")
        add_night_button.clicked.connect(self.add_night_fields)
        self.night_prefetch = QCheckBox("Připravovat archivní podklady na pozadí")
        self.night_prefetch.setChecked(True)
        self.candidate_mode_button = QPushButton("Označovat podezřelé body")
        self.candidate_mode_button.setCheckable(True)
        self.candidate_mode_button.toggled.connect(self._candidate_mode_changed)
        delete_candidate_button = QPushButton("Odstranit vybraný bod")
        delete_candidate_button.clicked.connect(self._delete_candidate)
        self.candidate_list_container = QWidget()
        candidate_list_layout = QVBoxLayout(self.candidate_list_container)
        candidate_list_layout.setContentsMargins(0, 0, 0, 0)
        candidate_list_layout.addWidget(self.candidate_list)
        candidate_list_layout.addWidget(delete_candidate_button)
        self.candidate_list_container.hide()
        load_button = QPushButton("Načíst podklad")
        load_button.clicked.connect(self.load_reference)
        self.center_reference_button = QPushButton("Načíst střed zobrazení")
        self.center_reference_button.setEnabled(False)
        self.center_reference_button.clicked.connect(self.load_reference_at_view_center)
        self.prefetch_button = QPushButton("Stáhnout označené (0)")
        self.prefetch_button.clicked.connect(self.prefetch_selected)
        self.prefetch_all_button = QPushButton("Stáhnout všechny v seznamu (0)")
        self.prefetch_all_button.clicked.connect(self.prefetch_all)
        previous_button = QPushButton("◀ Předchozí")
        previous_button.clicked.connect(lambda: self._step_galaxy(-1))
        next_button = QPushButton("Další ▶")
        next_button.clicked.connect(lambda: self._step_galaxy(1))
        self.blink_button = QPushButton("Blink")
        self.blink_button.setCheckable(True)
        self.blink_button.setEnabled(False)
        self.blink_button.toggled.connect(self._set_blink)
        self.marker_button = QPushButton("Kroužky galaxií")
        self.marker_button.setCheckable(True)
        self.marker_button.setChecked(True)
        self.marker_button.toggled.connect(self.current_panel.set_markers_visible)
        self.marker_button.toggled.connect(self.reference_panel.set_markers_visible)
        self.dso_button = QPushButton("Objekty NGC/IC")
        self.dso_button.setCheckable(True)
        self.dso_button.setChecked(True)
        self.dso_button.toggled.connect(self.current_panel.set_dso_visible)
        self.dso_button.toggled.connect(self.reference_panel.set_dso_visible)
        self.cursor_button = QPushButton("Společný kurzor")
        self.cursor_button.setCheckable(True)
        self.cursor_button.setChecked(True)
        self.cursor_button.toggled.connect(self._cursor_visibility_changed)
        self.reference_panel_button = QPushButton("Skrýt archivní panel")
        self.reference_panel_button.setCheckable(True)
        self.reference_panel_button.setChecked(True)
        self.reference_panel_button.toggled.connect(
            self._set_reference_panel_visible
        )
        self.reference_panel_sizes = [700, 700]

        sidebar_content = QWidget()
        side_layout = QVBoxLayout(sidebar_content)
        side_layout.addWidget(QLabel("<b>Snímky noci</b>"))
        side_layout.addWidget(add_night_button)
        side_layout.addWidget(self.night_prefetch)
        side_layout.addWidget(self.night_list)
        side_layout.addWidget(self.candidate_mode_button)
        side_layout.addWidget(self.candidate_list_button)
        side_layout.addWidget(self.candidate_list_container)
        side_layout.addWidget(QLabel("<b>Galaxie v poli</b>"))
        side_layout.addWidget(QLabel("Limit V magnitudy (HECATE + jasné PGC)"))
        side_layout.addWidget(self.mag_limit)
        side_layout.addWidget(find_button)
        side_layout.addWidget(self.galaxy_list, 1)
        nav = QHBoxLayout()
        nav.addWidget(previous_button)
        nav.addWidget(next_button)
        side_layout.addLayout(nav)
        side_layout.addWidget(QLabel("Archivní přehlídka"))
        side_layout.addWidget(self.survey)
        side_layout.addWidget(QLabel("Velikost výřezu"))
        side_layout.addWidget(self.fov)
        side_layout.addWidget(load_button)
        side_layout.addWidget(self.center_reference_button)
        side_layout.addWidget(self.prefetch_button)
        side_layout.addWidget(self.prefetch_all_button)
        side_layout.addWidget(QLabel("Více řádků: Ctrl nebo Shift"))
        side_layout.addWidget(self.blink_button)
        side_layout.addWidget(self.marker_button)
        side_layout.addWidget(self.dso_button)
        side_layout.addWidget(
            QLabel(
                '<span style="color:#32eb64">○ mlhoviny</span> &nbsp; '
                '<span style="color:#46a0ff">○ hvězdokupy</span> &nbsp; '
                '<span style="color:#dc64ff">○ ostatní</span>'
            )
        )
        side_layout.addWidget(self.cursor_button)
        side_layout.addWidget(self.reference_panel_button)

        # The controls need more vertical space when the candidate list is
        # expanded.  Keep their natural minimum sizes and scroll the whole
        # sidebar instead of letting QLayout squeeze widgets over each other.
        sidebar = QScrollArea()
        sidebar.setWidgetResizable(True)
        sidebar.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sidebar.setFrameShape(QScrollArea.NoFrame)
        sidebar.setMinimumWidth(260)
        sidebar.setWidget(sidebar_content)

        self.images_splitter = QSplitter(Qt.Horizontal)
        self.images_splitter.addWidget(self.current_panel)
        self.images_splitter.addWidget(self.reference_panel)
        self.images_splitter.setSizes(self.reference_panel_sizes)
        root = QSplitter(Qt.Horizontal)
        root.addWidget(sidebar)
        root.addWidget(self.images_splitter)
        root.setSizes([270, 1230])
        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar())

        toolbar = QToolBar("Soubor")
        self.addToolBar(toolbar)
        open_action = QAction("Otevřít FITS…", self)
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self.open_fits)
        toolbar.addAction(open_action)

        self.blink_timer = QTimer(self)
        self.blink_timer.setInterval(650)
        self.blink_timer.timeout.connect(self.reference_panel.toggle_blink_frame)

        self._restore_display_settings()
        self._bind_display_settings()
        self.dso_button.setChecked(
            self.settings.value("layers/ngc_ic_visible", True, type=bool)
        )
        self.dso_button.toggled.connect(
            lambda visible: self.settings.setValue("layers/ngc_ic_visible", visible)
        )
        reference_visible = self.settings.value(
            "layout/reference_panel_visible", True, type=bool
        )
        self.reference_panel_button.setChecked(reference_visible)
        self._set_reference_panel_visible(reference_visible)
        self._load_candidates()
        self._refresh_candidate_list()

    @property
    def cache_dir(self) -> Path:
        return Path.home() / ".cache" / "sn-hunter" / "archive"

    def _bind_display_settings(self):
        for name, panel in (
            ("current", self.current_panel),
            ("reference", self.reference_panel),
        ):
            save = lambda _value=None, n=name, p=panel: self._save_panel_settings(n, p)
            panel.stretch.currentTextChanged.connect(save)
            panel.low.valueChanged.connect(save)
            panel.high.valueChanged.connect(save)
            panel.parameter.valueChanged.connect(save)
            panel.invert.toggled.connect(save)

    def _restore_display_settings(self):
        for name, panel in (
            ("current", self.current_panel),
            ("reference", self.reference_panel),
        ):
            prefix = f"display/{name}"
            stretch = str(
                self.settings.value(
                    f"{prefix}/stretch", panel.stretch.currentText()
                )
            )
            profiles = panel.current_stretch_profiles()
            legacy = {
                "low": self.settings.value(
                    f"{prefix}/low", profiles.get(stretch, profiles["asinh"])["low"],
                    type=float,
                ),
                "high": self.settings.value(
                    f"{prefix}/high", profiles.get(stretch, profiles["asinh"])["high"],
                    type=float,
                ),
                "parameter": self.settings.value(
                    f"{prefix}/parameter",
                    profiles.get(stretch, profiles["asinh"])["parameter"],
                    type=float,
                ),
            }
            for mode, defaults in profiles.items():
                mode_prefix = f"{prefix}/profiles/{mode}"
                profiles[mode] = {
                    "low": self.settings.value(
                        f"{mode_prefix}/low",
                        legacy["low"] if mode == stretch else defaults["low"],
                        type=float,
                    ),
                    "high": self.settings.value(
                        f"{mode_prefix}/high",
                        legacy["high"] if mode == stretch else defaults["high"],
                        type=float,
                    ),
                    "parameter": self.settings.value(
                        f"{mode_prefix}/parameter",
                        legacy["parameter"] if mode == stretch else defaults["parameter"],
                        type=float,
                    ),
                }
            panel.set_stretch_profiles(profiles, stretch)
            panel.invert.setChecked(
                self.settings.value(
                    f"{prefix}/invert", panel.invert.isChecked(), type=bool
                )
            )

    def _save_panel_settings(self, name: str, panel: ImagePanel):
        prefix = f"display/{name}"
        self.settings.setValue(f"{prefix}/stretch", panel.stretch.currentText())
        for mode, values in panel.current_stretch_profiles().items():
            mode_prefix = f"{prefix}/profiles/{mode}"
            self.settings.setValue(f"{mode_prefix}/low", values["low"])
            self.settings.setValue(f"{mode_prefix}/high", values["high"])
            self.settings.setValue(
                f"{mode_prefix}/parameter", values["parameter"]
            )
        self.settings.setValue(f"{prefix}/invert", panel.invert.isChecked())

    def open_fits(self):
        start = self.settings.value("last_directory", str(Path.home()))
        filename, _ = QFileDialog.getOpenFileName(
            self, "Otevřít astronomický snímek", start, "FITS (*.fits *.fit *.fts *.fits.fz);;Všechny soubory (*)"
        )
        if not filename:
            return
        self._load_fits_path(Path(filename))

    def _load_fits_path(self, path: Path):
        filename = str(path)
        self.settings.setValue("last_directory", str(Path(filename).parent))
        try:
            image = load_fits(filename)
        except Exception as exc:
            QMessageBox.critical(self, "Nelze otevřít FITS", str(exc))
            return
        self._display_fits_image(image, path)

    def _display_fits_image(
        self,
        image: FitsImage,
        path: Path,
        galaxies: list[Galaxy] | None = None,
        deep_sky_objects: list[DeepSkyObject] | None = None,
    ):
        previous_path = (
            self.current.path.resolve()
            if self.current is not None and self.current.path is not None
            else None
        )
        new_path = path.resolve()
        field_changed = previous_path != new_path
        self.current = image
        if field_changed:
            blocker = QSignalBlocker(self.mag_limit)
            self.mag_limit.setValue(self._default_magnitude_limit(image))
            del blocker
        self.current_panel.title.setText(f"<b>Aktuální:</b> {path.name}")
        self.current_panel.set_wcs(self.current.wcs)
        self.reference_panel.set_wcs(self.current.wcs)
        self.current_panel.set_data(self.current.data, auto_range=True)
        self.reference_panel.set_data(None)
        self.reference_panel.alternate_data = None
        self.reference_aligned = None
        self.loading_reference_key = None
        self.loaded_reference_key = None
        self.center_reference_button.setEnabled(True)
        self.blink_button.setChecked(False)
        self.blink_button.setEnabled(False)
        self.galaxies.clear()
        self.deep_sky_objects.clear()
        self.galaxy_list.clear()
        self.current_panel.set_markers([])
        self.reference_panel.set_markers([])
        self.current_panel.set_dso_markers([], [], [], [])
        self.reference_panel.set_dso_markers([], [], [], [])
        self.current_panel.set_candidate_markers([], [])
        self.reference_panel.set_candidate_markers([], [])
        if galaxies is not None:
            self._galaxies_ready(galaxies)
        if deep_sky_objects is not None:
            self._deep_sky_objects_ready(deep_sky_objects)
        self._refresh_candidate_markers()
        ra, dec, radius = field_center_radius(self.current)
        field_size = field_size_deg(self.current)
        center = SkyCoord(ra, dec, unit="deg").to_string("hmsdms", precision=1)
        self.statusBar().showMessage(
            f"Načteno {self.current.data.shape[1]}×{self.current.data.shape[0]} px; "
            f"střed {center}; pole {field_size:.2f}°; "
            f"limit galaxií {self.mag_limit.value():.1f} mag"
        )

    @staticmethod
    def _default_magnitude_limit(image: FitsImage) -> float:
        size = field_size_deg(image)
        if size <= 2.0:
            return 17.5
        if size > 5.0:
            return 14.5
        return 16.0

    def add_night_fields(self):
        start = self.settings.value("last_directory", str(Path.home()))
        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "Přidat snímky pořízené během noci",
            start,
            "FITS (*.fits *.fit *.fts *.fits.fz);;Všechny soubory (*)",
        )
        if filenames:
            self._add_night_paths([Path(filename) for filename in filenames])

    def _add_night_paths(self, paths: list[Path]):
        existing = {str(item.path.resolve()).lower() for item in self.night_fields}
        first_new_row = None
        for path in paths:
            resolved = str(path.resolve()).lower()
            if resolved in existing:
                continue
            existing.add(resolved)
            if first_new_row is None:
                first_new_row = len(self.night_fields)
            self.night_fields.append(NightField(path=path))
            self.night_list.addItem("")
            self._update_night_item(len(self.night_fields) - 1)
        if first_new_row is None:
            return
        self.settings.setValue("last_directory", str(paths[0].parent))
        if self.night_list.currentRow() < 0:
            self.night_list.setCurrentRow(first_new_row)
        self._start_night_processing()

    def _night_field_selected(self, row: int):
        if not (0 <= row < len(self.night_fields)):
            return
        entry = self.night_fields[row]
        if entry.image is None:
            try:
                entry.image = load_fits(entry.path)
            except Exception as exc:
                entry.status = "chyba FITS"
                self._update_night_item(row)
                QMessageBox.warning(self, "Nelze otevřít FITS", str(exc))
                return
        self._display_fits_image(
            entry.image,
            entry.path,
            entry.galaxies if entry.galaxies else None,
            entry.deep_sky_objects if entry.deep_sky_objects else None,
        )

    def _update_night_item(self, row: int):
        if not (0 <= row < len(self.night_fields)):
            return
        entry = self.night_fields[row]
        self.night_list.item(row).setText(f"{entry.path.name}  —  {entry.status}")

    def _load_candidates(self):
        self.candidates = []
        try:
            raw = self.settings.value("candidates/json", "[]")
            values = json.loads(str(raw))
            for value in values:
                self.candidates.append(
                    Candidate(
                        field_path=str(value["field_path"]),
                        ra=float(value["ra"]),
                        dec=float(value["dec"]),
                        created=str(value.get("created", "")),
                    )
                )
        except (TypeError, ValueError, KeyError, json.JSONDecodeError):
            self.candidates = []

    def _save_candidates(self):
        values = [
            {
                "field_path": candidate.field_path,
                "ra": candidate.ra,
                "dec": candidate.dec,
                "created": candidate.created,
            }
            for candidate in self.candidates
        ]
        self.settings.setValue(
            "candidates/json", json.dumps(values, ensure_ascii=False)
        )

    def _refresh_candidate_list(self):
        self.visible_candidates = list(self.candidates)
        blocker = QSignalBlocker(self.candidate_list)
        self.candidate_list.clear()
        for candidate in self.visible_candidates:
            coordinate = SkyCoord(candidate.ra, candidate.dec, unit="deg")
            ra = coordinate.ra.to_string(
                unit=u.hourangle, sep=":", precision=1, pad=True
            )
            dec = coordinate.dec.to_string(
                unit=u.deg, sep=":", precision=0, pad=True, alwayssign=True
            )
            self.candidate_list.addItem(
                f"{Path(candidate.field_path).name} — {ra} {dec}"
            )
        del blocker
        count = len(self.visible_candidates)
        if self.candidate_list_button.isChecked():
            self.candidate_list_button.setText(f"Skrýt podezřelé body ({count})")
        else:
            self.candidate_list_button.setText(
                f"Seznam podezřelých bodů ({count})"
            )

    def _set_candidate_list_visible(self, visible: bool):
        self.candidate_list_container.setVisible(visible)
        count = len(self.visible_candidates)
        action = "Skrýt" if visible else "Seznam"
        self.candidate_list_button.setText(
            f"{action} podezřelé body ({count})"
            if visible
            else f"Seznam podezřelých bodů ({count})"
        )

    def _candidate_clicked(self, x: float, y: float):
        if (
            not self.candidate_mode_button.isChecked()
            or self.current is None
            or self.current.path is None
        ):
            return
        height, width = self.current.data.shape
        if not (-0.5 <= x < width - 0.5 and -0.5 <= y < height - 0.5):
            return
        coordinate = self.current.wcs.pixel_to_world(x, y)
        candidate = Candidate(
            field_path=str(self.current.path.resolve()),
            ra=float(coordinate.ra.degree),
            dec=float(coordinate.dec.degree),
            created=datetime.now().astimezone().isoformat(timespec="seconds"),
        )
        self.candidates.append(candidate)
        self._save_candidates()
        self._refresh_candidate_list()
        self._refresh_candidate_markers()
        self.statusBar().showMessage(
            f"Uložen podezřelý bod — RA {candidate.ra:.6f}°, "
            f"Dec {candidate.dec:+.6f}°",
            10000,
        )

    def _candidate_mode_changed(self, enabled: bool):
        self.candidate_mode_button.setText(
            "Klikněte do snímku…" if enabled else "Označovat podezřelé body"
        )

    def _candidate_selected(self, row: int):
        if not (0 <= row < len(self.visible_candidates)):
            return
        candidate = self.visible_candidates[row]
        path = Path(candidate.field_path)
        current_matches = (
            self.current is not None
            and self.current.path is not None
            and self.current.path.resolve() == path.resolve()
        )
        if not current_matches and not path.exists():
            QMessageBox.warning(
                self,
                "Snímek nebyl nalezen",
                f"Původní FITS soubor už není dostupný:\n{path}",
            )
            return
        if not current_matches:
            matching_row = next(
                (
                    index
                    for index, entry in enumerate(self.night_fields)
                    if entry.path.resolve() == path.resolve()
                ),
                None,
            )
            if matching_row is None:
                self._add_night_paths([path])
                matching_row = next(
                    index
                    for index, entry in enumerate(self.night_fields)
                    if entry.path.resolve() == path.resolve()
                )
            self.night_list.setCurrentRow(matching_row)
        coordinate = SkyCoord(candidate.ra, candidate.dec, unit="deg")
        self._focus_coordinate(coordinate)
        ra = coordinate.ra.to_string(
            unit=u.hourangle, sep=":", precision=1, pad=True
        )
        dec = coordinate.dec.to_string(
            unit=u.deg, sep=":", precision=0, pad=True, alwayssign=True
        )
        self._load_reference_target(
            Galaxy(
                name=f"podezřelý bod {ra} {dec}",
                ra=candidate.ra,
                dec=candidate.dec,
            )
        )

    def _delete_candidate(self):
        row = self.candidate_list.currentRow()
        if not (0 <= row < len(self.visible_candidates)):
            return
        candidate = self.visible_candidates[row]
        self.candidates.remove(candidate)
        self._save_candidates()
        self._refresh_candidate_list()
        self._refresh_candidate_markers()

    @staticmethod
    def _objects_inside_image(image: FitsImage, objects: list):
        if not objects:
            return []
        coordinates = SkyCoord(
            [item.ra for item in objects],
            [item.dec for item in objects],
            unit="deg",
        )
        height, width = image.data.shape
        approx_x, approx_y = skycoord_to_pixel(
            coordinates, image.wcs, origin=0, mode="wcs"
        )
        padding_x, padding_y = width * 0.05, height * 0.05
        near = (
            np.isfinite(approx_x)
            & np.isfinite(approx_y)
            & (approx_x >= -padding_x)
            & (approx_x < width + padding_x)
            & (approx_y >= -padding_y)
            & (approx_y < height + padding_y)
        )
        result = []
        if near.any():
            x, y = image.wcs.world_to_pixel(coordinates[near])
            for index, px, py in zip(np.flatnonzero(near), x, y):
                if (
                    np.isfinite(px)
                    and np.isfinite(py)
                    and -0.5 <= px < width - 0.5
                    and -0.5 <= py < height - 0.5
                ):
                    result.append(objects[int(index)])
        return result

    def _start_night_processing(self):
        if self.night_processing:
            return
        pending = [
            (row, entry.path, entry.image)
            for row, entry in enumerate(self.night_fields)
            if not entry.processing and not entry.prepared
        ]
        if not pending:
            return
        for row, _path, _image in pending:
            self.night_fields[row].processing = True
            self.night_fields[row].status = "ve frontě"
            self._update_night_item(row)
        self.night_processing = True
        survey_name = self.survey.currentText()
        requested_fov = self.fov.value()
        prefetch_archives = self.night_prefetch.isChecked()
        cache_dir = self.cache_dir

        def work(report):
            for row, path, existing_image in pending:
                try:
                    report(("status", row, "načítám FITS"))
                    image = existing_image or load_fits(path)
                    report(("image", row, image))
                    ra, dec, radius = field_center_radius(image)
                    magnitude_limit = self._default_magnitude_limit(image)

                    report(("status", row, "načítám katalogy"))
                    catalog_errors = []
                    try:
                        galaxies = query_galaxies(
                            ra, dec, radius, magnitude_limit, cache_dir.parent
                        )
                        galaxies = self._objects_inside_image(image, galaxies)
                    except Exception as exc:
                        galaxies = []
                        catalog_errors.append(f"galaxie: {exc}")
                    try:
                        deep_sky_objects = query_deep_sky_objects(
                            ra, dec, radius, cache_dir.parent
                        )
                        deep_sky_objects = self._objects_inside_image(
                            image, deep_sky_objects
                        )
                    except Exception as exc:
                        deep_sky_objects = []
                        catalog_errors.append(f"NGC/IC: {exc}")
                    report(("catalogs", row, galaxies, deep_sky_objects))

                    download_errors = 0
                    if prefetch_archives and galaxies:
                        _, source_fov, source_pixels = self._reference_geometry(
                            image, galaxies[0], survey_name, requested_fov
                        )
                        total = len(galaxies)
                        for finished, galaxy in enumerate(galaxies, start=1):
                            try:
                                download_reference(
                                    galaxy,
                                    SURVEYS[survey_name],
                                    cache_dir,
                                    source_fov,
                                    pixels=source_pixels,
                                )
                            except Exception:
                                download_errors += 1
                            report(
                                ("download", row, finished, total, download_errors)
                            )
                    if catalog_errors:
                        final_status = "připraveno bez části katalogů"
                    elif download_errors:
                        final_status = f"připraveno ({download_errors} chyb archivu)"
                    else:
                        final_status = "připraveno"
                    report(("ready", row, final_status))
                except Exception as exc:
                    report(("error", row, str(exc)))
            return True

        self._submit(
            work,
            self._night_processing_ready,
            self._night_processing_failed,
            self._night_progress,
        )

    def _night_progress(self, event):
        kind, row, *values = event
        if not (0 <= row < len(self.night_fields)):
            return
        entry = self.night_fields[row]
        if kind == "status":
            entry.status = values[0]
        elif kind == "image":
            if (
                self.night_list.currentRow() == row
                and self.current is not None
                and self.current.path is not None
                and self.current.path.resolve() == entry.path.resolve()
            ):
                entry.image = self.current
            else:
                entry.image = values[0]
        elif kind == "catalogs":
            entry.galaxies, entry.deep_sky_objects = values
            if self.night_list.currentRow() == row and self.current is entry.image:
                self._galaxies_ready(entry.galaxies)
                self._deep_sky_objects_ready(entry.deep_sky_objects)
        elif kind == "download":
            finished, total, errors = values
            entry.status = f"archiv {finished}/{total}"
            if errors:
                entry.status += f" ({errors} chyb)"
        elif kind == "ready":
            entry.status = values[0]
            entry.processing = False
            entry.prepared = True
        elif kind == "error":
            entry.status = f"chyba: {values[0]}"
            entry.processing = False
            entry.prepared = True
        self._update_night_item(row)

    def _night_processing_ready(self, _result):
        self.night_processing = False
        self._start_night_processing()

    def _night_processing_failed(self, message: str):
        self.night_processing = False
        for row, entry in enumerate(self.night_fields):
            if entry.processing:
                entry.processing = False
                entry.status = "zpracování přerušeno"
                self._update_night_item(row)
        self.statusBar().showMessage(f"Noční fronta selhala: {message}", 12000)

    @staticmethod
    def _is_fits_path(path: Path) -> bool:
        name = path.name.lower()
        return path.is_file() and name.endswith(
            (".fits", ".fit", ".fts", ".fits.fz", ".fit.fz", ".fts.fz")
        )

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls() and any(
            self._is_fits_path(Path(url.toLocalFile()))
            for url in event.mimeData().urls()
            if url.isLocalFile()
        ):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        paths = [
            Path(url.toLocalFile())
            for url in event.mimeData().urls()
            if url.isLocalFile() and self._is_fits_path(Path(url.toLocalFile()))
        ]
        if not paths:
            event.ignore()
            return
        self._add_night_paths(paths)
        event.acceptProposedAction()

    def find_galaxies(self):
        if self.current is None:
            QMessageBox.information(self, "Nejdříve snímek", "Nejdříve otevřete FITS snímek s WCS.")
            return
        ra, dec, radius = field_center_radius(self.current)
        self.statusBar().showMessage(
            "Dotazuji VizieR na galaxie HECATE + PGC a objekty NGC/IC…"
        )
        self._submit(
            lambda: query_galaxies(
                ra, dec, radius, self.mag_limit.value(), self.cache_dir.parent
            ),
            self._galaxies_ready,
        )
        self._submit(
            lambda: query_deep_sky_objects(
                ra, dec, radius, self.cache_dir.parent
            ),
            self._deep_sky_objects_ready,
            self._deep_sky_objects_failed,
        )

    def _galaxies_ready(self, galaxies):
        # The TAP query uses a circle enclosing the whole image. Remove objects
        # in the circle's corners that are outside the actual detector area.
        if self.current is not None and galaxies:
            coordinates = SkyCoord(
                [galaxy.ra for galaxy in galaxies],
                [galaxy.dec for galaxy in galaxies],
                unit="deg",
            )
            near = np.asarray(
                self._sky_inside_panel(self.current_panel, coordinates, margin=1.1),
                dtype=bool,
            )
            height, width = self.current.data.shape
            filtered = []
            if near.any():
                x, y = self.current.wcs.world_to_pixel(coordinates[near])
                for index, px, py in zip(np.flatnonzero(near), x, y):
                    if (
                        np.isfinite(px)
                        and np.isfinite(py)
                        and -0.5 <= px < width - 0.5
                        and -0.5 <= py < height - 0.5
                    ):
                        filtered.append(galaxies[int(index)])
            galaxies = filtered
        self.galaxies = galaxies
        night_row = self.night_list.currentRow()
        if 0 <= night_row < len(self.night_fields):
            entry = self.night_fields[night_row]
            if (
                self.current is not None
                and self.current.path is not None
                and self.current.path.resolve() == entry.path.resolve()
            ):
                entry.galaxies = list(galaxies)
        self.galaxy_list.clear()
        self.galaxy_list.addItems([galaxy.label for galaxy in galaxies])
        self._update_download_buttons()
        self._refresh_markers()
        self.statusBar().showMessage(f"Nalezeno {len(galaxies)} galaxií.", 8000)
        if galaxies:
            # Keep the first result ready for the download button, but do not
            # treat catalogue loading as a user navigation request.
            blocker = QSignalBlocker(self.galaxy_list)
            self.galaxy_list.setCurrentRow(0)
            del blocker
            self._update_download_buttons()

    def _deep_sky_objects_ready(self, objects):
        if self.current is not None and objects:
            coordinates = SkyCoord(
                [item.ra for item in objects],
                [item.dec for item in objects],
                unit="deg",
            )
            near = np.asarray(
                self._sky_inside_panel(self.current_panel, coordinates, margin=1.1),
                dtype=bool,
            )
            height, width = self.current.data.shape
            filtered = []
            if near.any():
                x, y = self.current.wcs.world_to_pixel(coordinates[near])
                for index, px, py in zip(np.flatnonzero(near), x, y):
                    if (
                        np.isfinite(px)
                        and np.isfinite(py)
                        and -0.5 <= px < width - 0.5
                        and -0.5 <= py < height - 0.5
                    ):
                        filtered.append(objects[int(index)])
            objects = filtered
        self.deep_sky_objects = objects
        night_row = self.night_list.currentRow()
        if 0 <= night_row < len(self.night_fields):
            entry = self.night_fields[night_row]
            if (
                self.current is not None
                and self.current.path is not None
                and self.current.path.resolve() == entry.path.resolve()
            ):
                entry.deep_sky_objects = list(objects)
        self._refresh_markers()
        self.statusBar().showMessage(
            f"NGC/IC: nalezeno {len(objects)} mlhovin, hvězdokup a dalších DSO.",
            8000,
        )

    def _deep_sky_objects_failed(self, message: str):
        self.deep_sky_objects = []
        self._refresh_markers()
        self.statusBar().showMessage(
            f"NGC/IC objekty se nepodařilo načíst: {message}", 12000
        )

    def _galaxy_selected(self, row: int):
        if self.current is None or not (0 <= row < len(self.galaxies)):
            return
        galaxy = self.galaxies[row]
        coordinate = SkyCoord(galaxy.ra, galaxy.dec, unit="deg")
        if not self._focus_coordinate(coordinate):
            return
        self.statusBar().showMessage(
            f"{galaxy.label} — RA {galaxy.ra:.6f}°, Dec {galaxy.dec:.6f}°"
        )
        self._autoload_cached_reference(galaxy)

    def _focus_coordinate(self, coordinate: SkyCoord) -> bool:
        if self.current is None:
            return False
        x, y = self.current.wcs.world_to_pixel(coordinate)
        if not np.isfinite(x) or not np.isfinite(y):
            return False
        scale = abs(float(np.mean(proj_plane_pixel_scales(self.current.wcs))))
        half_width = max(20.0, (self.fov.value() / 60.0) / scale / 2.0)
        ranges = [[x - half_width, x + half_width], [y - half_width, y + half_width]]
        self._set_ranges(self.current_panel, ranges)
        for panel in (self.current_panel, self.reference_panel):
            if panel.wcs is not None and self._sky_inside_panel(panel, coordinate):
                px, py = panel.wcs.world_to_pixel(coordinate)
                panel.select_marker(float(px), float(py))
            else:
                panel.selected_marker.clear()
        return True

    def load_reference(self):
        row = self.galaxy_list.currentRow()
        if self.current is None or not (0 <= row < len(self.galaxies)):
            QMessageBox.information(self, "Vyberte galaxii", "Nejdříve vyberte galaxii ze seznamu.")
            return
        galaxy = self.galaxies[row]
        self._load_reference_target(galaxy)

    def load_reference_at_view_center(self):
        if self.current is None:
            QMessageBox.information(
                self,
                "Nejdříve snímek",
                "Nejdříve otevřete FITS snímek s WCS.",
            )
            return
        ranges = self.current_panel.view.viewRange()
        x = (ranges[0][0] + ranges[0][1]) / 2.0
        y = (ranges[1][0] + ranges[1][1]) / 2.0
        height, width = self.current.data.shape
        if not (-0.5 <= x < width - 0.5 and -0.5 <= y < height - 0.5):
            QMessageBox.information(
                self,
                "Střed je mimo snímek",
                "Posuňte zobrazení tak, aby jeho střed ležel uvnitř aktuálního snímku.",
            )
            return
        coordinate = self.current.wcs.pixel_to_world(x, y)
        ra = coordinate.ra.to_string(unit=u.hourangle, sep=":", precision=1, pad=True)
        dec = coordinate.dec.to_string(
            unit=u.deg, sep=":", precision=0, pad=True, alwayssign=True
        )
        target = Galaxy(
            name=f"pozice {ra} {dec}",
            ra=float(coordinate.ra.degree),
            dec=float(coordinate.dec.degree),
        )
        self._load_reference_target(target)

    def _load_reference_target(self, galaxy: Galaxy):
        survey_name = self.survey.currentText()
        target_image = self.current
        requested_fov = self.fov.value()
        comparison_grid, source_fov, source_pixels = self._reference_geometry(
            target_image, galaxy, survey_name, requested_fov
        )
        cache_path = reference_cache_path(
            galaxy,
            SURVEYS[survey_name],
            self.cache_dir,
            source_fov,
            source_pixels,
        )
        request_key = (id(target_image), str(cache_path))
        if self.loading_reference_key == request_key:
            return
        self.loading_reference_key = request_key
        self.loaded_reference_key = None
        self.reference_aligned = None
        self.reference_panel.title.setText(
            f"<b>Archiv:</b> načítám — {galaxy.name}"
        )
        self.reference_panel.set_data(None)
        self.reference_panel.alternate_data = None
        self.blink_button.setChecked(False)
        self.blink_button.setEnabled(False)
        if cache_path.exists():
            self.statusBar().showMessage(
                f"Načítám {galaxy.name} z cache a zarovnávám obraz…"
            )
        else:
            self.statusBar().showMessage(f"Stahuji {survey_name} pro {galaxy.name}…")

        def work():
            source = download_reference(
                galaxy,
                SURVEYS[survey_name],
                self.cache_dir,
                source_fov,
                pixels=source_pixels,
            )
            reference_data = reproject_to_image(source, comparison_grid)
            current_data = reproject_to_image(target_image, comparison_grid)
            return target_image, comparison_grid, reference_data, current_data

        self._submit(
            work,
            lambda result: self._reference_ready(
                result[0], result[1], result[2], result[3], galaxy, survey_name,
                request_key,
            ),
            lambda message: self._reference_failed(message, request_key),
        )

    def prefetch_selected(self):
        if self.prefetch_in_progress:
            return
        rows = sorted(
            {index.row() for index in self.galaxy_list.selectionModel().selectedRows()}
        )
        if not rows and self.galaxy_list.currentRow() >= 0:
            rows = [self.galaxy_list.currentRow()]
        if not rows:
            QMessageBox.information(
                self,
                "Vyberte galaxie",
                "Vyberte jednu nebo více galaxií pomocí Ctrl nebo Shift.",
            )
            return
        self._prefetch_rows(rows)

    def prefetch_all(self):
        if self.prefetch_in_progress:
            return
        self._prefetch_rows(list(range(len(self.galaxies))))

    def _prefetch_rows(self, rows: list[int]):
        if self.current is None or not rows:
            QMessageBox.information(
                self, "Žádné galaxie", "V seznamu nejsou žádné galaxie ke stažení."
            )
            return

        galaxies = [self.galaxies[row] for row in rows]
        focus_row = self.galaxy_list.currentRow()
        if focus_row not in rows:
            focus_row = rows[0]
        self.prefetch_focus_name = self.galaxies[focus_row].name
        target_image = self.current
        survey_name = self.survey.currentText()
        requested_fov = self.fov.value()
        _, source_fov, source_pixels = self._reference_geometry(
            target_image, galaxies[0], survey_name, requested_fov
        )
        self.prefetch_in_progress = True
        self.prefetch_button.setEnabled(False)
        self.prefetch_all_button.setEnabled(False)
        self.statusBar().showMessage(
            f"Stahuji {len(galaxies)} výřezů {survey_name} do cache…"
        )

        def work(report):
            failures: list[str] = []
            finished = 0

            def download(galaxy: Galaxy):
                report(("started", galaxy.name, True))
                download_reference(
                    galaxy,
                    SURVEYS[survey_name],
                    self.cache_dir,
                    source_fov,
                    pixels=source_pixels,
                )
                return galaxy.name

            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="sn-prefetch") as pool:
                futures = {pool.submit(download, galaxy): galaxy for galaxy in galaxies}
                for future in as_completed(futures):
                    galaxy = futures[future]
                    try:
                        future.result()
                    except Exception as exc:
                        failures.append(f"{galaxy.name}: {exc}")
                        success = False
                    else:
                        success = True
                    finished += 1
                    report(("finished", finished, len(galaxies), galaxy.name, success))
            return len(galaxies) - len(failures), failures, survey_name

        self._submit(
            work,
            self._prefetch_ready,
            self._prefetch_failed,
            self._prefetch_progress,
        )

    def _prefetch_progress(self, progress):
        if progress[0] == "started":
            _, name, _ = progress
            self.statusBar().showMessage(f"Stahuji archivní výřez: {name}…")
            return
        _, finished, total, name, success = progress
        state = "hotovo" if success else "chyba"
        self.statusBar().showMessage(
            f"Stahování {finished}/{total}: {name} — {state}"
        )
        if success and name == self.prefetch_focus_name:
            row = self.galaxy_list.currentRow()
            if 0 <= row < len(self.galaxies) and self.galaxies[row].name == name:
                self._autoload_cached_reference(self.galaxies[row], force=True)

    def _prefetch_ready(self, result):
        completed, failures, survey_name = result
        self.prefetch_in_progress = False
        self.prefetch_focus_name = None
        self.prefetch_button.setEnabled(True)
        self.prefetch_all_button.setEnabled(True)
        self.statusBar().showMessage(
            f"Do cache uloženo {completed} výřezů {survey_name}.", 12000
        )
        if failures:
            preview = "\n".join(failures[:8])
            if len(failures) > 8:
                preview += f"\n… a dalších {len(failures) - 8}"
            QMessageBox.warning(
                self,
                "Některá stahování selhala",
                f"Úspěšně: {completed}, selhalo: {len(failures)}\n\n{preview}",
            )
        row = self.galaxy_list.currentRow()
        if 0 <= row < len(self.galaxies):
            self._autoload_cached_reference(self.galaxies[row], force=True)

    def _prefetch_failed(self, message: str):
        self.prefetch_in_progress = False
        self.prefetch_focus_name = None
        self.prefetch_button.setEnabled(True)
        self.prefetch_all_button.setEnabled(True)
        self._background_error(message)

    def _update_download_buttons(self):
        selected = len(self.galaxy_list.selectedIndexes())
        total = self.galaxy_list.count()
        self.prefetch_button.setText(f"Stáhnout označené ({selected})")
        self.prefetch_all_button.setText(f"Stáhnout všechny v seznamu ({total})")

    def _reference_ready(
        self,
        target_image: FitsImage,
        comparison_grid: FitsImage,
        reference_data: np.ndarray,
        current_data: np.ndarray,
        galaxy: Galaxy,
        survey_name: str,
        request_key,
    ):
        if self.loading_reference_key != request_key:
            return
        self.loading_reference_key = None
        if self.current is not target_image:
            self.statusBar().showMessage(
                "Stažený podklad patřil k dříve otevřenému snímku a byl ignorován.",
                8000,
            )
            return
        self.loaded_reference_key = request_key
        self.reference_aligned = reference_data
        self.reference_panel.title.setText(f"<b>Archiv:</b> {survey_name} — {galaxy.name}")
        self.reference_panel.set_wcs(comparison_grid.wcs)
        self.reference_panel.set_data(reference_data)
        self.reference_panel.set_blink_data(
            reference_data, current_data, self.current_panel
        )
        self._refresh_markers()
        self.blink_button.setEnabled(True)
        self._focus_coordinate(SkyCoord(galaxy.ra, galaxy.dec, unit="deg"))
        pixel_scale = float(np.mean(proj_plane_pixel_scales(comparison_grid.wcs))) * 3600.0
        self.statusBar().showMessage(
            f"Archiv zachován v jemné mřížce {comparison_grid.data.shape[1]}×"
            f"{comparison_grid.data.shape[0]} px ({pixel_scale:.2f}″/px).",
            10000,
        )

    def _reference_failed(self, message: str, request_key=None):
        if request_key is not None and self.loading_reference_key != request_key:
            return
        self.loading_reference_key = None
        self._background_error(message)

    @staticmethod
    def _reference_geometry(
        target_image: FitsImage,
        galaxy: Galaxy,
        survey_name: str,
        requested_fov: float,
    ):
        comparison_grid = make_comparison_grid(
            target_image,
            galaxy.ra,
            galaxy.dec,
            requested_fov,
            SURVEY_PIXEL_SCALE_ARCSEC[survey_name],
        )
        source_pixels = min(
            3200,
            int(math.ceil(comparison_grid.data.shape[0] * math.sqrt(2.0))),
        )
        return comparison_grid, requested_fov * math.sqrt(2.0), source_pixels

    def _autoload_cached_reference(self, galaxy: Galaxy, force: bool = False):
        if self.current is None or (self.prefetch_in_progress and not force):
            return
        survey_name = self.survey.currentText()
        requested_fov = self.fov.value()
        _, source_fov, source_pixels = self._reference_geometry(
            self.current, galaxy, survey_name, requested_fov
        )
        path = reference_cache_path(
            galaxy,
            SURVEYS[survey_name],
            self.cache_dir,
            source_fov,
            source_pixels,
        )
        request_key = (id(self.current), str(path))
        if (
            path.exists()
            and (force or request_key != self.loaded_reference_key)
            and request_key != self.loading_reference_key
        ):
            QTimer.singleShot(0, self.load_reference)

    def _submit(
        self,
        function,
        callback,
        error_callback=None,
        progress_callback=None,
    ):
        bridge = FutureBridge(self)
        self.bridges.add(bridge)
        bridge.completed.connect(callback)
        bridge.failed.connect(error_callback or self._background_error)
        if progress_callback is not None:
            bridge.progress.connect(progress_callback)
        bridge.completed.connect(lambda _result: self.bridges.discard(bridge))
        bridge.failed.connect(lambda _message: self.bridges.discard(bridge))

        if progress_callback is None:
            future = self.executor.submit(function)
        else:
            future = self.executor.submit(function, bridge.progress.emit)

        def done(completed):
            try:
                result = completed.result()
            except Exception as exc:
                print(traceback.format_exc(), file=sys.stderr)
                bridge.failed.emit(str(exc))
            else:
                bridge.completed.emit(result)

        future.add_done_callback(done)

    def _background_error(self, message: str):
        self.statusBar().showMessage("Operace selhala.", 8000)
        QMessageBox.warning(self, "Operace se nezdařila", message)

    def _step_galaxy(self, delta: int):
        if not self.galaxies:
            return
        self.galaxy_list.setCurrentRow((self.galaxy_list.currentRow() + delta) % len(self.galaxies))

    def _sync_range(self, source: ImagePanel, destination: ImagePanel, ranges):
        if self.syncing:
            return
        if (
            source.wcs is None
            or destination.wcs is None
            or source.raw_data is None
            or destination.raw_data is None
        ):
            return
        self.syncing = True
        try:
            cx = (ranges[0][0] + ranges[0][1]) / 2.0
            cy = (ranges[1][0] + ranges[1][1]) / 2.0
            coordinate = source.wcs.pixel_to_world(cx, cy)
            if not self._sky_inside_panel(destination, coordinate, margin=1.5):
                return
            dx, dy = destination.wcs.world_to_pixel(coordinate)
            source_scale = float(np.mean(proj_plane_pixel_scales(source.wcs)))
            destination_scale = float(np.mean(proj_plane_pixel_scales(destination.wcs)))
            ratio = source_scale / destination_scale
            half_width = abs(ranges[0][1] - ranges[0][0]) * ratio / 2.0
            half_height = abs(ranges[1][1] - ranges[1][0]) * ratio / 2.0
            destination_ranges = [
                [dx - half_width, dx + half_width],
                [dy - half_height, dy + half_height],
            ]
            self._set_ranges(destination, destination_ranges)
        finally:
            self.syncing = False

    @staticmethod
    def _set_ranges(panel: ImagePanel, ranges):
        panel.view.setRange(xRange=ranges[0], yRange=ranges[1], padding=0)

    def _set_blink(self, enabled: bool):
        if enabled:
            self.blink_timer.start()
        else:
            self.blink_timer.stop()
            self.reference_panel.showing_alternate = False
            self.reference_panel.refresh()

    def _cursor_moved(self, source: ImagePanel, x: float, y: float):
        if self.current is None:
            return
        if source.wcs is None or source.raw_data is None:
            return
        height, width = source.raw_data.shape
        inside = -0.5 <= x < width - 0.5 and -0.5 <= y < height - 0.5
        if not inside:
            source.show_cursor(x, y, False)
            return
        try:
            coordinate = source.wcs.pixel_to_world(x, y)
            if not self.cursor_button.isChecked():
                return
            for panel in (self.current_panel, self.reference_panel):
                if panel.wcs is None or panel.raw_data is None:
                    continue
                if not self._sky_inside_panel(panel, coordinate, margin=1.05):
                    panel.show_cursor(0, 0, False)
                    continue
                px, py = panel.wcs.world_to_pixel(coordinate)
                ph, pw = panel.raw_data.shape
                panel.show_cursor(
                    float(px),
                    float(py),
                    -0.5 <= px < pw - 0.5 and -0.5 <= py < ph - 0.5,
                )
            ra = coordinate.ra.to_string(
                unit=u.hourangle, sep=":", precision=2, pad=True
            )
            dec = coordinate.dec.to_string(
                unit=u.deg, sep=":", precision=1, pad=True, alwayssign=True
            )
            self.statusBar().showMessage(
                f"RA {ra}   Dec {dec}    ({coordinate.ra.degree:.6f}°, {coordinate.dec.degree:+.6f}°)"
            )
        except Exception:
            self.statusBar().showMessage(f"Pixel x={x:.1f}, y={y:.1f}")

    def _copy_coordinates(self, source: ImagePanel, x: float, y: float):
        if source.wcs is None or source.raw_data is None:
            return
        height, width = source.raw_data.shape
        if not (-0.5 <= x < width - 0.5 and -0.5 <= y < height - 0.5):
            return
        coordinate = source.wcs.pixel_to_world(x, y)
        text = self._format_coordinate_for_clipboard(coordinate)
        QApplication.clipboard().setText(text)
        self.statusBar().showMessage(f"Zkopírováno: {text}", 8000)

    @staticmethod
    def _format_coordinate_for_clipboard(coordinate: SkyCoord) -> str:
        ra = coordinate.ra.to_string(
            unit=u.hourangle, sep=" ", precision=0, pad=True
        )
        dec = coordinate.dec.to_string(
            unit=u.deg, sep=" ", precision=0, pad=True, alwayssign=True
        )
        return f"{ra} {dec}"

    def _refresh_markers(self):
        for panel in (self.current_panel, self.reference_panel):
            if not self.galaxies or panel.wcs is None or panel.raw_data is None:
                panel.set_markers([])
            else:
                coordinates = SkyCoord(
                    [galaxy.ra for galaxy in self.galaxies],
                    [galaxy.dec for galaxy in self.galaxies],
                    unit="deg",
                )
                points = self._marker_points(panel, coordinates)
                panel.set_markers(points, [galaxy.label for galaxy in self.galaxies])

            if (
                not self.deep_sky_objects
                or panel.wcs is None
                or panel.raw_data is None
            ):
                panel.set_dso_markers([], [], [], [])
            else:
                dso_coordinates = SkyCoord(
                    [item.ra for item in self.deep_sky_objects],
                    [item.dec for item in self.deep_sky_objects],
                    unit="deg",
                )
                panel.set_dso_markers(
                    self._marker_points(panel, dso_coordinates),
                    [item.label for item in self.deep_sky_objects],
                    [item.category for item in self.deep_sky_objects],
                    [item.size_arcmin for item in self.deep_sky_objects],
                )
        self._refresh_candidate_markers()

    def _refresh_candidate_markers(self):
        if self.current is None or self.current.path is None:
            self.current_panel.set_candidate_markers([], [])
            self.reference_panel.set_candidate_markers([], [])
            return
        current_path = str(self.current.path.resolve()).lower()
        candidates = [
            candidate
            for candidate in self.candidates
            if str(Path(candidate.field_path).resolve()).lower() == current_path
        ]
        if not candidates:
            self.current_panel.set_candidate_markers([], [])
            self.reference_panel.set_candidate_markers([], [])
            return
        coordinates = SkyCoord(
            [candidate.ra for candidate in candidates],
            [candidate.dec for candidate in candidates],
            unit="deg",
        )
        labels = [
            f"Podezřelý bod — RA {candidate.ra:.6f}°, "
            f"Dec {candidate.dec:+.6f}°"
            for candidate in candidates
        ]
        for panel in (self.current_panel, self.reference_panel):
            if panel.wcs is None or panel.raw_data is None:
                panel.set_candidate_markers([], [])
            else:
                panel.set_candidate_markers(
                    self._marker_points(panel, coordinates), labels
                )

    def _marker_points(self, panel: ImagePanel, coordinates: SkyCoord):
        inside = np.asarray(
            self._sky_inside_panel(panel, coordinates, margin=1.05), dtype=bool
        )
        points = [(np.nan, np.nan)] * len(coordinates)
        if inside.any():
            x, y = panel.wcs.world_to_pixel(coordinates[inside])
            for index, px, py in zip(np.flatnonzero(inside), x, y):
                points[int(index)] = (float(px), float(py))
        return points

    @staticmethod
    def _sky_inside_panel(
        panel: ImagePanel, coordinates: SkyCoord, margin: float = 1.1
    ):
        if panel.wcs is None or panel.raw_data is None:
            return False
        height, width = panel.raw_data.shape
        # ``mode="wcs"`` deliberately ignores SIP/table distortions.  It gives
        # a cheap first estimate without invoking the iterative all_world2pix
        # solver, which can diverge for catalogue objects far outside a panel.
        x, y = skycoord_to_pixel(coordinates, panel.wcs, origin=0, mode="wcs")
        pad_x = max(0.0, margin - 1.0) * width / 2.0
        pad_y = max(0.0, margin - 1.0) * height / 2.0
        return (
            np.isfinite(x)
            & np.isfinite(y)
            & (x >= -0.5 - pad_x)
            & (x < width - 0.5 + pad_x)
            & (y >= -0.5 - pad_y)
            & (y < height - 0.5 + pad_y)
        )

    def _cursor_visibility_changed(self, visible: bool):
        if not visible:
            self.current_panel.show_cursor(0, 0, False)
            self.reference_panel.show_cursor(0, 0, False)

    def _set_reference_panel_visible(self, visible: bool):
        if visible:
            self.reference_panel.show()
            if hasattr(self, "images_splitter"):
                self.images_splitter.setSizes(self.reference_panel_sizes)
            self.reference_panel_button.setText("Skrýt archivní panel")
        else:
            if hasattr(self, "images_splitter") and not self.reference_panel.isHidden():
                sizes = self.images_splitter.sizes()
                if len(sizes) == 2 and sizes[1] > 0:
                    self.reference_panel_sizes = sizes
            self.reference_panel.hide()
            self.reference_panel_button.setText("Zobrazit archivní panel")
        self.settings.setValue("layout/reference_panel_visible", visible)

    def closeEvent(self, event):
        self._save_candidates()
        self._save_panel_settings("current", self.current_panel)
        self._save_panel_settings("reference", self.reference_panel)
        self.settings.sync()
        self.executor.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)


def main() -> int:
    pg_app = QApplication.instance() or QApplication(sys.argv)
    pg_app.setApplicationName("SN Hunter")
    window = MainWindow()
    window.show()
    return pg_app.exec()


if __name__ == "__main__":
    raise SystemExit(main())

