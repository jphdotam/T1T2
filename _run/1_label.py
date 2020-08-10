import os
import numpy as np
import sys, traceback
import pyqtgraph as pg
from matplotlib import cm
from scipy.stats import iqr
from collections import defaultdict


from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QWidget, QShortcut

from labelling.ui.css import css
from labelling.ui.layout_label import Ui_MainWindow
from utils.cmaps import default_cmap
from utils.labeling import save_pickle, load_pickle, get_hui_report_path, convert_hui_coords_to_peter_coords
from utils.windows import SEQUENCE_WINDOWS, window_numpy
from utils.labeling import get_studies_peter as get_studies

DATADIR_PETER = "E:/Data/T1T2_peter"
DATADIR_HUI = "E:/Data/T1T2_hui"  # False if don't want to check for Hui labels

if QtCore.QT_VERSION >= 0x50501:
    def excepthook(type_, value, traceback_):
        traceback.print_exception(type_, value, traceback_)
        QtCore.qFatal('')
sys.excepthook = excepthook

LABELS = ('endo', 'epi', 'myo')

class MainWindowUI(Ui_MainWindow):
    def __init__(self, mainwindow, data_root_dir=DATADIR_PETER):

        super(MainWindowUI, self).__init__()
        self.data_root_dir = data_root_dir
        self.mainwindow = mainwindow
        self.setupUi(mainwindow)
        self.plotWidget.setAspectLocked(True)
        self.imageItem = None

        self.activelabel_name = LABELS[0]
        self.i_window = 0

        self.numpy_path = None

        self.img_array = None
        self.roi_coords = defaultdict(list)
        self.roi_selectors = {}
        self.shortcuts = {}
        self.labelbuttons = {'endo': self.pushButton_endo,
                             'epi': self.pushButton_epi,
                             'myo': self.pushButton_myo}
        self.labelmode = 'add'

        self.sequences = get_studies(self.data_root_dir, check_hui_label=DATADIR_HUI)
        self.refresh_studyselector()
        self.comboBox_studies.activated.connect(self.load_study)
        self.comboBox_sequences.activated.connect(self.load_sequence)

        self.create_shortcuts()

    def refresh_studyselector(self):
        self.comboBox_studies.clear()
        self.comboBox_studies.addItem("Please select a study")
        n_reported, n_reported_hui = 0, 0
        for sequence_id, sequence_dict in self.sequences.items():
            self.comboBox_studies.addItem(sequence_id)

            if sequence_dict['reported'] == 'peter':
                colour = 'green'
                n_reported += 1
            elif sequence_dict['reported'] == 'hui':
                colour = 'yellow'
                n_reported_hui += 1
            elif sequence_dict['reported'] == 'no':
                colour = 'red'
            else:
                raise ValueError()

            self.comboBox_studies.setItemData(self.comboBox_studies.count()-1, QtGui.QColor(colour), QtCore.Qt.BackgroundRole)
        print(f"{n_reported} of {len(self.sequences)} reported {'(' + str(n_reported_hui) + ' hui reports)' if n_reported_hui else ''}")

    def load_study(self):
        self.i_window = 0
        self.comboBox_sequences.clear()
        try:
            sequence_id = self.comboBox_studies.currentText()
            self.numpy_path = self.sequences[sequence_id]['numpy_path']
            self.report_path = self.sequences[sequence_id]['report_path']

            for i_seq, seq in enumerate(SEQUENCE_WINDOWS.keys()):
                self.comboBox_sequences.addItem(f"{i_seq} - {seq} - {self.numpy_path}")
            self.comboBox_sequences.setCurrentIndex(4)  # Select T2w by default

            hui_reported = self.sequences[sequence_id]['reported']=='hui'
            self.roi_coords = self.load_coords(self.report_path, hui_reported=hui_reported)
            self.load_sequence()

            self.activelabel_name = LABELS[0]
            self.draw_buttons()
        except KeyError as e:  # Selected heading
            print(f"exception: {e}")

    def load_sequence(self):
        i_seq, seq_name, numpy_path = self.comboBox_sequences.currentText().split(' - ', 2)
        print(numpy_path)
        i_seq = int(i_seq)

        img_array = np.load(numpy_path)[:,:,i_seq]
        img_array = img_array.T
        img_array = np.flip(img_array, axis=1)

        try:
            windows_for_class = SEQUENCE_WINDOWS[seq_name]
            window = windows_for_class[self.i_window % len(windows_for_class)]
            window_centre = window['wc']
            window_width = window['ww']
            cmap = default_cmap

        except (KeyError, TypeError):
            # No wc/ww for this, so use median for wc and 2* IQR for WW
            window_centre = np.median(img_array)
            window_width = iqr(img_array) * 2
            cmap = cm.gray

        img_array = window_numpy(img_array,
                                 window_centre=window_centre,
                                 window_width=window_width,
                                 cmap=cmap)

        self.img_array = img_array
        self.draw_image_and_rois()

    def load_coords(self, report_path=None, hui_reported=False):
        if report_path is None:
            report_path = self.report_path
        if os.path.exists(report_path):
            return load_pickle(report_path)
        else:
            if hui_reported:
                hui_report_path = get_hui_report_path(self.numpy_path, DATADIR_HUI)
                if hui_report_path:
                    return convert_hui_coords_to_peter_coords(load_pickle(hui_report_path))
            return defaultdict(list)

    def create_shortcuts(self):
        # Space -> Edit
        shortcut_mode = QShortcut(QKeySequence("Space"), self.pushButton_mode)
        shortcut_mode.activated.connect(lambda: self.action_changemode())
        self.shortcuts['mode'] = shortcut_mode

        # Numbers -> Labels
        for i, label in enumerate(LABELS):
            shortcut_key = f"{i + 1}"
            shortcut = QShortcut(QKeySequence(shortcut_key), self.labelbuttons[label])
            shortcut.activated.connect(lambda labelname=label: self.action_labelbutton(labelname))
            self.shortcuts[label] = shortcut

        # Up/down -> Change study
        shortcut_prevstudy = QShortcut(QKeySequence("Up"), self.pushButton_prevstudy)
        shortcut_prevstudy.activated.connect(lambda: self.action_changestudy(-1))
        shortcut_nextstudy = QShortcut(QKeySequence("Down"), self.pushButton_nextstudy)
        shortcut_nextstudy.activated.connect(lambda: self.action_changestudy(1))

        # Left/right -> Change sequence
        shortcut_prevseq = QShortcut(QKeySequence("Left"), self.pushButton_prevseq)
        shortcut_prevseq.activated.connect(lambda: self.action_changeseq(-1))
        shortcut_nextseq = QShortcut(QKeySequence("Right"), self.pushButton_nextseq)
        shortcut_nextseq.activated.connect(lambda: self.action_changeseq(1))

        # x -> change window
        shortcut_changewindow = QShortcut(QKeySequence("X"), self.pushButton_changeWindow)
        shortcut_changewindow.activated.connect(lambda: self.change_window())

    @pyqtSlot()
    def action_changestudy(self, changeby):
        current_id = self.comboBox_studies.currentIndex()
        new_id = (current_id + changeby) % self.comboBox_studies.count()
        self.comboBox_studies.setCurrentIndex(new_id)
        self.load_study()

    @pyqtSlot()
    def action_changeseq(self, changeby):
        current_id = self.comboBox_sequences.currentIndex()
        new_id = (current_id + changeby) % self.comboBox_sequences.count()
        self.comboBox_sequences.setCurrentIndex(new_id)
        self.load_sequence()

    @pyqtSlot()
    def action_changemode(self):
        if self.labelmode == 'add':
            print(f"Add -> Edit")
            self.labelmode = 'edit'
            self.imageItem.mousePressEvent = None

        elif self.labelmode == 'edit':
            print(f"Edit -> Add")
            self.labelmode = 'add'
            self.imageItem.mousePressEvent = self.add_node_at_mouse
        else:
            raise ValueError()
        self.draw_buttons()

    @pyqtSlot()
    def action_labelbutton(self, labelname):
        self.activelabel_name = labelname
        self.draw_buttons()

    @pyqtSlot()
    def change_window(self):
        print("changing window")
        self.i_window += 1
        self.load_sequence()

    def draw_buttons(self):
        # Edit button
        if self.labelmode == 'add':
            mode_text = 'Add mode'
            mode_style = css['modebutton_add']
        elif self.labelmode == 'edit':
            mode_text = 'Edit mode'
            mode_style = css['modebutton_edit']
        else:
            raise ValueError()
        self.pushButton_mode.setText(mode_text)
        self.pushButton_mode.setStyleSheet(mode_style)
        self.draw_image_and_rois(fix_roi=True)

        # Label buttons
        for name, button in self.labelbuttons.items():
            reported = self.roi_selectors.get(name, None) is not None
            if name == self.activelabel_name:
                style = css['labelbutton_active_green'] if reported else css['labelbutton_active_red']
            else:
                style = css['labelbutton_inactive_green'] if reported else css['labelbutton_inactive_red']
            self.labelbuttons[name].setStyleSheet(style)

    def draw_image_and_rois(self, fix_roi=False):
        # Get current axis range
        x_range = self.plotWidget.getAxis('bottom').range
        y_range = self.plotWidget.getAxis('left').range

        # Remove imageItem
        if self.imageItem:
            self.imageItem.mousePressEvent = None
            self.plotWidget.removeItem(self.imageItem)
            del self.imageItem
        # Remove ROIs from plot and delete
        for roi_name, roi_selector in self.roi_selectors.items():
            self.plotWidget.removeItem(roi_selector)
            self.roi_selectors[roi_name] = None
            del roi_selector
        self.roi_selectors = {}

        # Draw image
        self.imageItem = pg.ImageItem(self.img_array)
        self.plotWidget.addItem(self.imageItem)
        if self.labelmode == 'add':
            self.imageItem.mousePressEvent = self.add_node_at_mouse

        # Draw ROIs
        for roi_name, coords in self.roi_coords.items():
            if roi_name == 'dims':  # Ignore the label containing the image size
                continue
            roi_selector = pg.PolyLineROI(coords, movable=False, closed=True, pen=QtGui.QPen(QtGui.QColor(115, 194, 251)))
            roi_selector.sigRegionChangeFinished.connect(lambda roi: self.update_coords())
            self.roi_selectors[roi_name] = roi_selector
            self.plotWidget.addItem(roi_selector)

        # Restore range
        if fix_roi:
            self.plotWidget.setRange(xRange=x_range, yRange=y_range, padding=0)

    def update_coords(self):
        if self.labelmode == "add":
            # Do not allow adjustments in add mode, recipe for disaster
            return None
        for roi_name, roi in self.roi_selectors.items():
            self.roi_coords[roi_name] = []
            for segment in roi.segments:
                point = segment.listPoints()[0]
                self.roi_coords[roi_name].append([point.img_cl(), point.y()])
        # Finally save
        if self.numpy_path:
            self.save_coords()

    def save_coords(self, report_path=None):
        out_dict = self.roi_coords
        out_dict['dims'] = self.img_array.shape
        if report_path is None:
            report_path = self.report_path
        print(f"Saving as {report_path}")
        save_pickle(report_path, out_dict)

    def add_node_at_mouse(self, event):
        self.labelbuttons[self.activelabel_name].setStyleSheet(css['labelbutton_active_green'])
        x = round(event.pos().img_cl())
        y = round(event.pos().y())
        self.roi_coords[self.activelabel_name].append([x, y])
        print(self.roi_coords[self.activelabel_name])
        self.draw_image_and_rois(fix_roi=True)
        # Finally save
        if self.numpy_path:
            self.save_coords()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = MainWindowUI(MainWindow)
    MainWindow.show()
    print('Showing')
    app.exec_()
