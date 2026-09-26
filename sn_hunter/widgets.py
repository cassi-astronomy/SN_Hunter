from __future__ import annotations

import math

import numpy as np
import pyqtgraph as pg
from astropy import units as u
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from PySide6.QtCore import QSignalBlocker, Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .imaging import stretch_image


class ImagePanel(QWidget):
    range_changed = Signal(object)
    marker_clicked = Signal(int)
    cursor_moved = Signal(float, float)
    image_clicked = Signal(float, float)
    coordinate_copy_requested = Signal(float, float)

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.raw_data: np.ndarray | None = None
        self.alternate_data: np.ndarray | None = None
        self.alternate_stretch_source: ImagePanel | None = None
        self.showing_alternate = False
        self.wcs: WCS | None = None
        self.marker_labels: list[str] = []
        self.zoom_percent: float | None = None

        self.title = QLabel(f"<b>{title}</b>")
        self.zoom_label = QLabel("Zoom —")
        self.zoom_label.setToolTip(
            "100 % znamená jeden pixel snímku na jeden pixel obrazovky."
        )
        self.zoom_selector = QComboBox()
        self.zoom_selector.setEditable(True)
        self.zoom_selector.addItems(
            [
                "Přizpůsobit",
                "25 %",
                "50 %",
                "75 %",
                "100 %",
                "150 %",
                "200 %",
                "400 %",
                "800 %",
            ]
        )
        self.zoom_selector.setCurrentIndex(0)
        self.zoom_selector.setMaximumWidth(105)
        self.zoom_selector.setToolTip(
            "Vyberte předvolbu nebo napište vlastní procento a stiskněte Enter."
        )
        self.view = pg.PlotWidget()
        self.view.setMenuEnabled(False)
        self.view.setAspectLocked(True)
        self.view.invertY(True)
        self.view.hideAxis("left")
        self.view.hideAxis("bottom")
        self.image_item = pg.ImageItem(axisOrder="row-major")
        self.view.addItem(self.image_item)
        self.markers = pg.ScatterPlotItem(
            pxMode=True,
            symbol="o",
            size=13,
            pen=pg.mkPen(255, 210, 0, 220, width=1.5),
            brush=pg.mkBrush(0, 0, 0, 0),
            hoverable=True,
            hoverPen=pg.mkPen(255, 255, 255, width=2),
            tip=self._marker_tip,
        )
        self.selected_marker = pg.ScatterPlotItem(
            pxMode=True,
            symbol="o",
            size=23,
            pen=pg.mkPen(80, 255, 100, width=2.5),
            brush=pg.mkBrush(0, 0, 0, 0),
        )
        self.dso_markers = pg.ScatterPlotItem(
            pxMode=False,
            symbol="o",
            size=12,
            brush=pg.mkBrush(0, 0, 0, 0),
            hoverable=True,
            hoverPen=pg.mkPen(255, 255, 255, width=2),
            tip=self._dso_tip,
        )
        self.dso_markers.setZValue(5)
        self.markers.setZValue(10)
        self.selected_marker.setZValue(11)
        self.candidate_markers = pg.ScatterPlotItem(
            pxMode=True,
            symbol="x",
            size=19,
            pen=pg.mkPen(255, 70, 40, 255, width=3),
            brush=pg.mkBrush(0, 0, 0, 0),
            hoverable=True,
            hoverPen=pg.mkPen(255, 255, 255, width=4),
            tip=self._dso_tip,
        )
        self.candidate_markers.setZValue(12)
        self.view.addItem(self.dso_markers)
        self.view.addItem(self.markers)
        self.view.addItem(self.selected_marker)
        self.view.addItem(self.candidate_markers)
        self.markers.sigClicked.connect(self._marker_was_clicked)
        self.crosshair_v = pg.InfiniteLine(
            angle=90, movable=False, pen=pg.mkPen(0, 255, 255, 180, width=1)
        )
        self.crosshair_h = pg.InfiniteLine(
            angle=0, movable=False, pen=pg.mkPen(0, 255, 255, 180, width=1)
        )
        self.crosshair_v.setVisible(False)
        self.crosshair_h.setVisible(False)
        self.view.addItem(self.crosshair_v, ignoreBounds=True)
        self.view.addItem(self.crosshair_h, ignoreBounds=True)

        decoration_pen = pg.mkPen(255, 255, 255, 230, width=2.5)
        self.scale_line = pg.PlotCurveItem(pen=decoration_pen)
        self.north_line = pg.PlotCurveItem(pen=decoration_pen)
        self.east_line = pg.PlotCurveItem(pen=decoration_pen)
        self.scale_text = pg.TextItem(color="w", anchor=(0.5, 1.0))
        self.north_text = pg.TextItem("N", color="w", anchor=(0.5, 0.5))
        self.east_text = pg.TextItem("E", color="w", anchor=(0.5, 0.5))
        for item in (
            self.scale_line,
            self.north_line,
            self.east_line,
            self.scale_text,
            self.north_text,
            self.east_text,
        ):
            item.setVisible(False)
            self.view.addItem(item, ignoreBounds=True)

        self.view.scene().sigMouseMoved.connect(self._scene_mouse_moved)
        self.view.scene().sigMouseClicked.connect(self._scene_mouse_clicked)
        self.view.getViewBox().sigRangeChanged.connect(self._emit_range)

        self.stretch = QComboBox()
        self.stretch.addItems(["asinh", "linear", "sqrt", "log", "power"])
        self.low = QDoubleSpinBox()
        self.low.setRange(0.0, 49.9)
        self.low.setValue(0.5)
        self.low.setDecimals(1)
        self.low.setSingleStep(0.1)
        self.low.setSuffix(" %")
        self.high = QDoubleSpinBox()
        self.high.setRange(50.1, 100.0)
        self.high.setValue(99.5)
        self.high.setDecimals(1)
        self.high.setSingleStep(0.1)
        self.high.setSuffix(" %")
        self.parameter = QDoubleSpinBox()
        self.parameter.setRange(0.01, 1000.0)
        self.parameter.setValue(10.0)
        self.parameter.setDecimals(2)
        self.parameter.setSingleStep(0.1)
        self.invert = QPushButton("Negativ")
        self.invert.setCheckable(True)
        self.stretch_profiles = {
            mode: {"low": 0.5, "high": 99.5, "parameter": 10.0}
            for mode in (
                self.stretch.itemText(index) for index in range(self.stretch.count())
            )
        }
        self._active_stretch = self.stretch.currentText()

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Stretch"))
        controls.addWidget(self.stretch)
        controls.addWidget(QLabel("černá"))
        controls.addWidget(self.low)
        controls.addWidget(QLabel("bílá"))
        controls.addWidget(self.high)
        controls.addWidget(QLabel("síla/γ"))
        controls.addWidget(self.parameter)
        controls.addWidget(self.invert)

        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        header.addWidget(self.title)
        header.addStretch(1)
        header.addWidget(self.zoom_label)
        header.addWidget(self.zoom_selector)
        layout.addLayout(header)
        layout.addWidget(self.view, 1)
        layout.addLayout(controls)

        self.stretch.currentTextChanged.connect(self._stretch_changed)
        self.low.valueChanged.connect(self._remember_current_stretch)
        self.high.valueChanged.connect(self._remember_current_stretch)
        self.parameter.valueChanged.connect(self._remember_current_stretch)
        self.low.valueChanged.connect(self.refresh)
        self.high.valueChanged.connect(self.refresh)
        self.parameter.valueChanged.connect(self.refresh)
        self.invert.toggled.connect(self.refresh)
        self.invert.toggled.connect(self._update_overlay_colors)
        self.zoom_selector.activated.connect(self._zoom_preset_selected)
        self.zoom_selector.lineEdit().editingFinished.connect(
            self._zoom_text_entered
        )
        self._update_overlay_colors()

    def _remember_current_stretch(self, _value=None):
        if not hasattr(self, "stretch_profiles"):
            return
        self.stretch_profiles[self._active_stretch] = {
            "low": self.low.value(),
            "high": self.high.value(),
            "parameter": self.parameter.value(),
        }

    def _stretch_changed(self, mode: str):
        self._remember_current_stretch()
        self._active_stretch = mode
        profile = self.stretch_profiles[mode]
        blockers = [
            QSignalBlocker(self.low),
            QSignalBlocker(self.high),
            QSignalBlocker(self.parameter),
        ]
        self.low.setValue(profile["low"])
        self.high.setValue(profile["high"])
        self.parameter.setValue(profile["parameter"])
        del blockers
        self.refresh()

    def set_stretch_profiles(self, profiles: dict, active_mode: str):
        for mode, values in profiles.items():
            if mode in self.stretch_profiles:
                self.stretch_profiles[mode] = {
                    "low": float(values["low"]),
                    "high": float(values["high"]),
                    "parameter": float(values["parameter"]),
                }
        if self.stretch.findText(active_mode) < 0:
            active_mode = "asinh"
        blockers = [
            QSignalBlocker(self.stretch),
            QSignalBlocker(self.low),
            QSignalBlocker(self.high),
            QSignalBlocker(self.parameter),
        ]
        self.stretch.setCurrentText(active_mode)
        self._active_stretch = active_mode
        profile = self.stretch_profiles[active_mode]
        self.low.setValue(profile["low"])
        self.high.setValue(profile["high"])
        self.parameter.setValue(profile["parameter"])
        del blockers
        self.refresh()

    def current_stretch_profiles(self) -> dict:
        self._remember_current_stretch()
        return {
            mode: dict(values) for mode, values in self.stretch_profiles.items()
        }

    def _emit_range(self, _view, ranges):
        self._update_decorations(ranges)
        self._update_zoom_label(ranges)
        self.range_changed.emit(ranges)

    def _update_zoom_label(self, ranges=None):
        if self.raw_data is None:
            self.zoom_percent = None
            self.zoom_label.setText("Zoom —")
            return
        if ranges is None:
            ranges = self.view.viewRange()
        visible_pixels = abs(float(ranges[0][1] - ranges[0][0]))
        screen_pixels = float(self.view.getViewBox().width())
        if visible_pixels <= 0 or screen_pixels <= 0:
            self.zoom_percent = None
            self.zoom_label.setText("Zoom —")
            return
        self.zoom_percent = screen_pixels / visible_pixels * 100.0
        if self.zoom_percent < 10.0:
            value = f"{self.zoom_percent:.1f}"
        else:
            value = f"{self.zoom_percent:.0f}"
        self.zoom_label.setText(f"Zoom {value} %")
        blocker = QSignalBlocker(self.zoom_selector)
        self.zoom_selector.setEditText(f"{value} %")
        del blocker

    def _zoom_preset_selected(self, index: int):
        text = self.zoom_selector.itemText(index)
        if text == "Přizpůsobit":
            if self.raw_data is not None:
                self.view.autoRange()
            return
        self._apply_zoom_text(text)

    def _zoom_text_entered(self):
        self._apply_zoom_text(self.zoom_selector.currentText())

    def _apply_zoom_text(self, text: str):
        cleaned = text.strip().replace("%", "").replace(",", ".")
        try:
            percent = float(cleaned)
        except ValueError:
            self._update_zoom_label()
            return
        self.set_zoom_percent(percent)

    def set_zoom_percent(self, percent: float):
        if self.raw_data is None:
            return
        percent = min(5000.0, max(1.0, float(percent)))
        ranges = self.view.viewRange()
        center_x = (ranges[0][0] + ranges[0][1]) / 2.0
        screen_pixels = float(self.view.getViewBox().width())
        if screen_pixels <= 0:
            return
        visible_pixels = screen_pixels / (percent / 100.0)
        self.view.setXRange(
            center_x - visible_pixels / 2.0,
            center_x + visible_pixels / 2.0,
            padding=0,
        )

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, "zoom_label"):
            self._update_zoom_label()

    def _scene_mouse_moved(self, scene_position):
        if not self.view.sceneBoundingRect().contains(scene_position):
            return
        point = self.view.getViewBox().mapSceneToView(scene_position)
        self.cursor_moved.emit(float(point.x()), float(point.y()))

    def _scene_mouse_clicked(self, event):
        scene_position = event.scenePos()
        if not self.view.sceneBoundingRect().contains(scene_position):
            return
        point = self.view.getViewBox().mapSceneToView(scene_position)
        if event.button() == Qt.RightButton:
            self.coordinate_copy_requested.emit(float(point.x()), float(point.y()))
            event.accept()
        elif event.button() == Qt.LeftButton:
            self.image_clicked.emit(float(point.x()), float(point.y()))

    def set_wcs(self, wcs: WCS | None):
        self.wcs = wcs
        self._update_decorations(self.view.viewRange())

    def show_cursor(self, x: float, y: float, visible: bool = True):
        self.crosshair_v.setPos(x)
        self.crosshair_h.setPos(y)
        self.crosshair_v.setVisible(visible)
        self.crosshair_h.setVisible(visible)

    def _update_decorations(self, ranges):
        visible = self.wcs is not None and self.raw_data is not None
        for item in (
            self.scale_line,
            self.north_line,
            self.east_line,
            self.scale_text,
            self.north_text,
            self.east_text,
        ):
            item.setVisible(visible)
        if not visible:
            return

        xmin, xmax = sorted(ranges[0])
        ymin, ymax = sorted(ranges[1])
        xspan, yspan = xmax - xmin, ymax - ymin
        if xspan <= 0 or yspan <= 0:
            return
        scale_deg = float(np.mean(proj_plane_pixel_scales(self.wcs)))

        target_arcmin = xspan * scale_deg * 60.0 * 0.18
        if target_arcmin >= 60.0:
            target_value = target_arcmin / 60.0
            angular_arcmin = self._nice_length(target_value) * 60.0
        elif target_arcmin >= 1.0:
            angular_arcmin = self._nice_length(target_arcmin)
        else:
            angular_arcmin = self._nice_length(target_arcmin * 60.0) / 60.0
        length_pixels = angular_arcmin / (scale_deg * 60.0)
        sx = xmin + 0.025 * xspan
        sy = ymax - 0.035 * yspan
        self.scale_line.setData([sx, sx + length_pixels], [sy, sy])
        if angular_arcmin >= 60.0:
            label = f"{angular_arcmin / 60.0:g}°"
        elif angular_arcmin >= 1.0:
            label = f"{angular_arcmin:g}′"
        else:
            label = f"{angular_arcmin * 60.0:g}″"
        self.scale_text.setText(label)
        self.scale_text.setPos(sx + length_pixels / 2.0, sy - 0.015 * yspan)

        cx, cy = (xmin + xmax) / 2.0, (ymin + ymax) / 2.0
        try:
            center = self.wcs.pixel_to_world(cx, cy)
            north = center.directional_offset_by(0 * u.deg, max(scale_deg * 20, 0.01) * u.deg)
            east = center.directional_offset_by(
                90 * u.deg, max(scale_deg * 20, 0.01) * u.deg
            )
            nx, ny = self.wcs.world_to_pixel(north)
            ex, ey = self.wcs.world_to_pixel(east)
            ndx, ndy = float(nx - cx), float(ny - cy)
            edx, edy = float(ex - cx), float(ey - cy)
            north_norm = math.hypot(ndx, ndy)
            east_norm = math.hypot(edx, edy)
            if (
                not np.isfinite(north_norm)
                or not np.isfinite(east_norm)
                or north_norm == 0
                or east_norm == 0
            ):
                raise ValueError("Neplatná směrová růžice")
            length = 0.055 * min(xspan, yspan)
            ndx, ndy = ndx / north_norm * length, ndy / north_norm * length
            edx, edy = edx / east_norm * length, edy / east_norm * length
            ox, oy = xmax - 0.075 * xspan, ymin + 0.08 * yspan
            self.north_line.setData([ox, ox + ndx], [oy, oy + ndy])
            self.east_line.setData([ox, ox + edx], [oy, oy + edy])
            self.north_text.setPos(ox + ndx * 1.25, oy + ndy * 1.25)
            self.east_text.setPos(ox + edx * 1.25, oy + edy * 1.25)
        except Exception:
            self.north_line.setVisible(False)
            self.east_line.setVisible(False)
            self.north_text.setVisible(False)
            self.east_text.setVisible(False)

    @staticmethod
    def _nice_length(value: float) -> float:
        exponent = math.floor(math.log10(max(value, 1e-9)))
        unit = 10.0**exponent
        choices = np.array([1.0, 2.0, 5.0, 10.0]) * unit
        return float(choices[np.argmin(np.abs(choices - value))])

    def _marker_was_clicked(self, _item, points, _event):
        if points:
            self.marker_clicked.emit(int(points[0].data()))

    def _marker_tip(self, x: float, y: float, data) -> str:
        index = int(data)
        if 0 <= index < len(self.marker_labels):
            return self.marker_labels[index]
        return f"x={x:.1f}, y={y:.1f}"

    @staticmethod
    def _dso_tip(x: float, y: float, data) -> str:
        if isinstance(data, dict):
            return str(data.get("label", ""))
        return str(data) if data else f"x={x:.1f}, y={y:.1f}"

    def _overlay_palette(self):
        if self.invert.isChecked():
            return {
                "galaxy": (175, 0, 0, 255),
                "selected": (0, 55, 200, 255),
                "nebula": (0, 105, 35, 255),
                "cluster": (0, 65, 185, 255),
                "other": (125, 0, 155, 255),
                "cursor": (145, 0, 145, 230),
                "candidate": (0, 95, 125, 255),
                "decoration": (135, 0, 40, 255),
                "hover": (0, 0, 0, 255),
            }
        return {
            "galaxy": (255, 210, 0, 230),
            "selected": (80, 255, 100, 255),
            "nebula": (50, 235, 100, 230),
            "cluster": (70, 160, 255, 230),
            "other": (220, 100, 255, 230),
            "cursor": (0, 255, 255, 180),
            "candidate": (255, 70, 40, 255),
            "decoration": (255, 255, 255, 230),
            "hover": (255, 255, 255, 255),
        }

    def _update_overlay_colors(self, _checked=None):
        colors = self._overlay_palette()
        self.markers.setPen(pg.mkPen(*colors["galaxy"], width=2))
        self.selected_marker.setPen(pg.mkPen(*colors["selected"], width=2.5))
        self.markers.opts["hoverPen"] = pg.mkPen(*colors["hover"], width=2)
        self.dso_markers.opts["hoverPen"] = pg.mkPen(*colors["hover"], width=2)
        self.candidate_markers.opts["hoverPen"] = pg.mkPen(
            *colors["hover"], width=4
        )
        cursor_pen = pg.mkPen(*colors["cursor"], width=1)
        self.crosshair_v.setPen(cursor_pen)
        self.crosshair_h.setPen(cursor_pen)
        decoration_pen = pg.mkPen(*colors["decoration"], width=2.5)
        self.scale_line.setPen(decoration_pen)
        self.north_line.setPen(decoration_pen)
        self.east_line.setPen(decoration_pen)
        text_color = pg.mkColor(*colors["decoration"])
        self.scale_text.setColor(text_color)
        self.north_text.setColor(text_color)
        self.east_text.setColor(text_color)
        for point in self.dso_markers.points():
            data = point.data()
            category = data.get("category", "other") if isinstance(data, dict) else "other"
            point.setPen(pg.mkPen(*colors.get(category, colors["other"]), width=2))
        candidate_pen = pg.mkPen(*colors["candidate"], width=3)
        for point in self.candidate_markers.points():
            point.setPen(candidate_pen)

    def set_markers(
        self,
        points: list[tuple[float, float]],
        labels: list[str] | None = None,
    ):
        self.marker_labels = labels or []
        height, width = self.raw_data.shape if self.raw_data is not None else (0, 0)
        spots = []
        for index, (x, y) in enumerate(points):
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            if self.raw_data is not None and not (
                -0.5 <= x < width - 0.5 and -0.5 <= y < height - 0.5
            ):
                continue
            spots.append({"pos": (x, y), "data": index})
        self.markers.setData(spots)
        self.selected_marker.clear()

    def select_marker(self, x: float, y: float):
        self.selected_marker.setData([{"pos": (x, y)}])

    def set_markers_visible(self, visible: bool):
        self.markers.setVisible(visible)
        self.selected_marker.setVisible(visible)

    def set_dso_markers(
        self,
        points: list[tuple[float, float]],
        labels: list[str],
        categories: list[str],
        sizes_arcmin: list[float | None],
    ):
        colors = self._overlay_palette()
        height, width = self.raw_data.shape if self.raw_data is not None else (0, 0)
        if self.wcs is not None:
            arcmin_per_pixel = abs(
                float(np.mean(proj_plane_pixel_scales(self.wcs))) * 60.0
            )
        else:
            arcmin_per_pixel = 0.0
        maximum_size = max(12.0, min(height, width) * 0.9)
        spots = []
        for (x, y), label, category, angular_size in zip(
            points, labels, categories, sizes_arcmin
        ):
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            if self.raw_data is not None and not (
                -0.5 <= x < width - 0.5 and -0.5 <= y < height - 0.5
            ):
                continue
            if angular_size is not None and arcmin_per_pixel > 0:
                marker_size = float(angular_size) / arcmin_per_pixel
            else:
                marker_size = 12.0
            marker_size = min(maximum_size, max(10.0, marker_size))
            spots.append(
                {
                    "pos": (x, y),
                    "data": {"label": label, "category": category},
                    "size": marker_size,
                    "pen": pg.mkPen(*colors.get(category, colors["other"]), width=2),
                }
            )
        self.dso_markers.setData(spots)

    def set_dso_visible(self, visible: bool):
        self.dso_markers.setVisible(visible)

    def set_candidate_markers(
        self, points: list[tuple[float, float]], labels: list[str]
    ):
        height, width = self.raw_data.shape if self.raw_data is not None else (0, 0)
        colors = self._overlay_palette()
        spots = []
        for (x, y), label in zip(points, labels):
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            if self.raw_data is not None and not (
                -0.5 <= x < width - 0.5 and -0.5 <= y < height - 0.5
            ):
                continue
            spots.append(
                {
                    "pos": (x, y),
                    "data": label,
                    "pen": pg.mkPen(*colors["candidate"], width=3),
                }
            )
        self.candidate_markers.setData(spots)

    def set_data(self, data: np.ndarray | None, auto_range: bool = False):
        self.raw_data = data
        self.showing_alternate = False
        self.refresh()
        if data is not None and auto_range:
            self.view.autoRange()
        self._update_zoom_label()

    def set_blink_data(
        self,
        primary: np.ndarray,
        alternate: np.ndarray,
        alternate_stretch_source: "ImagePanel | None" = None,
    ):
        self.raw_data = primary
        self.alternate_data = alternate
        self.alternate_stretch_source = alternate_stretch_source

    def toggle_blink_frame(self):
        if self.alternate_data is None:
            return
        self.showing_alternate = not self.showing_alternate
        self.refresh()

    def refresh(self):
        data = self.alternate_data if self.showing_alternate else self.raw_data
        if data is None:
            self.image_item.clear()
            return
        settings = (
            self.alternate_stretch_source
            if self.showing_alternate and self.alternate_stretch_source is not None
            else self
        )
        rendered = stretch_image(
            data,
            settings.stretch.currentText(),
            settings.low.value(),
            settings.high.value(),
            settings.parameter.value(),
            statistics_data=settings.raw_data,
        )
        if settings.invert.isChecked():
            rendered = 1.0 - rendered
        self.image_item.setImage(rendered, autoLevels=False, levels=(0.0, 1.0))

