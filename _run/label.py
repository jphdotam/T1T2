import os
import sys, traceback
import pyqtgraph as pg
from glob import glob
from collections import defaultdict

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QWidget, QShortcut

from labelling.ui.css import css
from labelling.ui.layout_label import Ui_MainWindow
from utils.dicoms import save_pickle, load_pickle, dicom_to_img, get_sequences

DICOMDIR = "../data/dicoms/bdbs_new/"

if QtCore.QT_VERSION >= 0x50501:
    def excepthook(type_, value, traceback_):
        traceback.print_exception(type_, value, traceback_)
        QtCore.qFatal('')
sys.excepthook = excepthook

LABELS = ('endo', 'epi', 'myo')


class MainWindowUI(Ui_MainWindow):
    def __init__(self, mainwindow, dicom_root_dir=DICOMDIR):

        super(MainWindowUI, self).__init__()
        self.dicom_root_dir = dicom_root_dir
        self.mainwindow = mainwindow
        self.setupUi(mainwindow)
        self.plotWidget.setAspectLocked(True)
        self.imageItem = None

        self.activelabel_name = LABELS[0]

        self.studypath = None
        self.img_array = None
        self.roi_coords = defaultdict(list)
        self.roi_selectors = {}
        self.shortcuts = {}
        self.labelbuttons = {'endo': self.pushButton_endo,
                             'epi': self.pushButton_epi,
                             'myo': self.pushButton_myo}
        self.labelmode = 'add'

        self.sequences = get_sequences(self.dicom_root_dir)
        self.refresh_studyselector()
        self.comboBox_studies.activated.connect(self.load_study)
        self.comboBox_sequences.activated.connect(self.load_sequence)

        self.create_shortcuts()

    def refresh_studyselector(self):
        self.comboBox_studies.clear()
        self.comboBox_studies.addItem("Please select a study")
        for sequence_id, sequence_dict in self.sequences.items():
            self.comboBox_studies.addItem(sequence_id)
            color = "green" if sequence_dict['reported'] else "red"
            self.comboBox_studies.setItemData(self.comboBox_studies.count()-1, QtGui.QColor(color), QtCore.Qt.BackgroundRole)

    def load_study(self):
        self.comboBox_sequences.clear()
        try:
            sequence_id = self.comboBox_studies.currentText()
            studypath = self.sequences[sequence_id]['path']
            self.studypath = studypath

            dicomnames = [os.path.basename(f) for f in glob(os.path.join(self.studypath, "*.dcm"))]
            for i_dcm, dicomname in enumerate(dicomnames):
                self.comboBox_sequences.addItem(dicomname)
                if 'T2' in dicomname:
                    self.comboBox_sequences.setCurrentIndex(i_dcm)

            self.roi_coords = self.load_coords(self.studypath)
            self.load_sequence()

            self.activelabel_name = LABELS[0]
            self.draw_buttons()
        except KeyError as e:  # Selected heading
            print(e)

    def load_sequence(self):
        dicomname = self.comboBox_sequences.currentText()
        dicompath = os.path.join(self.studypath, dicomname)
        print(f"dicompath is {dicompath}")
        self.img_array = dicom_to_img(dicompath)
        self.draw_image_and_rois()

    def get_pickle_path(self, studypath=None):
        if studypath is None:
            studypath = self.studypath
        return os.path.join(studypath, f"label.pickle")

    def load_coords(self, studypath=None):
        if studypath is None:
            studypath = self.studypath
        pickle_path = self.get_pickle_path(studypath)

        if os.path.exists(pickle_path):
            print(f"Loading {pickle_path}")
            return load_pickle(pickle_path)
        else:
            return defaultdict(list)

    def create_shortcuts(self):
        # Space -> Edit
        shortcut_mode = QShortcut(QKeySequence("Space"), self.pushButton_mode)
        shortcut_mode.activated.connect(lambda: self.action_changemode())
        self.shortcuts['mode'] = shortcut_mode

        # Numbers -> Labels
        for i, label in enumerate(LABELS):
            shortcut_key = f"{i + 1}"
            print(f"sc {shortcut_key}")
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

    def draw_buttons(self):
        # Edit button
        if self.labelmode=='add':
            modetext = 'Add mode'
            modestyle = css['modebutton_add']
        elif self.labelmode=='edit':
            modetext = 'Edit mode'
            modestyle = css['modebutton_edit']
        else:
            raise ValueError()
        self.pushButton_mode.setText(modetext)
        self.pushButton_mode.setStyleSheet(modestyle)
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
            print(roi_name)
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
                self.roi_coords[roi_name].append([point.x(), point.y()])
        # Finally save
        if self.studypath:
            self.save_coords()

    def save_coords(self, studypath=None):
        out_dict = self.roi_coords
        out_dict['dims'] = self.img_array.shape
        if studypath is None:
            studypath = self.studypath
        pickle_path = self.get_pickle_path(studypath)
        print(f"Saving as {pickle_path}")
        save_pickle(pickle_path, out_dict)

    def add_node_at_mouse(self, event):
        self.labelbuttons[self.activelabel_name].setStyleSheet(css['labelbutton_active_green'])
        x = round(event.pos().x())
        y = round(event.pos().y())
        self.roi_coords[self.activelabel_name].append([x, y])
        print(self.roi_coords[self.activelabel_name])
        self.draw_image_and_rois(fix_roi=True)
        # Finally save
        if self.studypath:
            self.save_coords()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = MainWindowUI(MainWindow)
    MainWindow.show()
    print('Showing')
    app.exec_()
