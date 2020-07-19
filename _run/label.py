import os
import numpy as np
import sys, traceback
import pyqtgraph as pg
import matplotlib.pyplot as plt
from scipy.stats import iqr
from pyqtgraph import PlotWidget
from collections import defaultdict, namedtuple

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import QWidget, QShortcut

from labelling.ui.css import css
from labelling.ui.layout_label import Ui_MainWindow
from utils.cmaps import default_cmap
from utils.dicoms import save_pickle, load_pickle, get_studies, window_numpy

DATADIR = "D:/Data/T1T2"
OLD_PATH = "D:\\Dropbox\\Work\\Other projects\\T1T2\\data\\dicoms\\by_date_by_study"

if QtCore.QT_VERSION >= 0x50501:
    def excepthook(type_, value, traceback_):
        traceback.print_exception(type_, value, traceback_)
        QtCore.qFatal('')
sys.excepthook = excepthook

LABELS = ('endo', 'epi', 'myo')

SEQUENCE_WINDOWS = {
    't1map': [{'wc':1300, 'ww':1300}, {'wc':500, 'ww':1000}],
    't2map': [{'wc':60, 'ww':120}]
}

class MainWindowUI(Ui_MainWindow):
    def __init__(self, mainwindow, data_root_dir=DATADIR):

        super(MainWindowUI, self).__init__()
        self.data_root_dir = data_root_dir
        self.mainwindow = mainwindow
        self.setupUi(mainwindow)

        self.roi_coords = defaultdict(list)
        self.roi_selectors = {}
        self.shortcuts = {}
        self.labelbuttons = {'endo': self.pushButton_endo,
                             'epi': self.pushButton_epi,
                             'myo': self.pushButton_myo}
        self.labelmode = 'add'
        self.activelabel_name = LABELS[0]

        # plots will store a list of each of:
        # 1) image_arrays
        # 2) image_items
        # 3) plot_widgets

        self.plots = {}

        self.studies = get_studies(self.data_root_dir)
        self.refresh_studyselector()
        self.comboBox_studies.activated.connect(self.load_study)

        self.create_shortcuts()

    def refresh_studyselector(self):
        self.comboBox_studies.clear()
        self.comboBox_studies.addItem("Please select a study")
        for sequence_id, sequence_dict in self.studies.items():
            self.comboBox_studies.addItem(sequence_id)

            if sequence_dict['reported']:
                colour = 'green'
            else:
                colour = 'red'

            self.comboBox_studies.setItemData(self.comboBox_studies.count()-1, QtGui.QColor(colour), QtCore.Qt.BackgroundRole)

    def reset_plots(self):
        for plot_element_name, plot_elements in self.plots.items():
            for plot_element in plot_elements:
                del plot_element
        while self.horizontalLayout_Plots.count() > 0:
            self.horizontalLayout_Plots.itemAt(0).setParent(None)
        self.plots = {}

    def load_study(self):
        # Reset plot area
        self.reset_plots()

        # Load report
        study_id = self.comboBox_studies.currentText()
        self.report_path = self.studies[study_id]['report_path']
        self.roi_coords = self.load_coords(self.report_path)

        # Refresh buttons
        self.activelabel_name = LABELS[0]
        self.draw_buttons()

        # Create plots
        self.create_plots()

    def create_plots(self):
        img_arrays = []
        image_items = []
        plot_widgets = []

        study_id = self.comboBox_studies.currentText()
        for sequence_type, numpy_path in self.studies[study_id]['sequence_paths'].items():

            # Create plotWidget
            plot_widget = PlotWidget(self.centralwidget)
            plot_widget.setEnabled(True)
            plot_widget.setObjectName(f"plotWidget_{sequence_type}")
            plot_widget.setAspectLocked(True)

            # load image
            img_array = np.load(numpy_path)
            window_centre = SEQUENCE_WINDOWS[sequence_type][0]['wc']
            window_width = SEQUENCE_WINDOWS[sequence_type][0]['ww']
            img_array = window_numpy(img_array,
                                     window_centre=window_centre,
                                     window_width=window_width,
                                     cmap=default_cmap)

            # create imageItem and add to plotWidget
            image_item = pg.ImageItem(img_array)
            if self.labelmode == 'add':
                image_item.mousePressEvent = self.add_node_at_mouse
            plot_widget.addItem(image_item)

            self.horizontalLayout_Plots.addWidget(plot_widget)

            img_arrays.append(img_array)
            plot_widgets.append(plot_widget)
            image_items.append(image_item)

        self.plots['img_arrays'] = img_arrays
        self.plots['plot_widgets'] = plot_widgets
        self.plots['image_items'] = image_items


    # def load_study(self):
    #     # load report and coords
    #     self.report_path = self.studies[study_id]['report_path']
    #     self.roi_coords = self.load_coords(self.report_path)
    #
    #     # Load images into self.image_arrays (list)
    #     self.load_images()
    #     self.plot_images()
    #
    #     # refresh the interface of buttons
    #     self.activelabel_name = LABELS[0]
    #     self.draw_buttons()
    #
    # def load_images(self):
    #     self.img_arrays = []
    #
    #     study_id = self.comboBox_studies.currentText()
    #     for sequence_type, numpy_path in self.studies[study_id]['sequence_paths'].items():
    #
    #         img_array = np.load(numpy_path)
    #         window_centre, window_width = SEQUENCE_WINDOWS[sequence_type][0]['wc'], SEQUENCE_WINDOWS[sequence_type][0]['ww'],
    #         img_array = window_numpy(img_array,
    #                                  window_centre=window_centre,
    #                                  window_width=window_width,
    #                                  cmap=default_cmap)
    #
    #         self.img_arrays.append(img_array)
    #
    # def plot_images(self):
    #     study_id = self.comboBox_studies.currentText()
    #     for sequence_type, numpy_path in self.studies[study_id]['sequence_paths'].items():
    #         plot_widget = PlotWidget(self.centralwidget)
    #         plot_widget.setEnabled(True)
    #         plot_widget.setObjectName(f"plotWidget_{sequence_type}")
    #         plot_widget.setAspectLocked(True)
    #         self.horizontalLayout_Plots.addWidget(plot_widget)

    def load_coords(self, report_path=None):
        if report_path is None:
            report_path = self.report_path
        if os.path.exists(report_path):
            return load_pickle(report_path)
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
            shortcut = QShortcut(QKeySequence(shortcut_key), self.labelbuttons[label])
            shortcut.activated.connect(lambda labelname=label: self.action_labelbutton(labelname))
            self.shortcuts[label] = shortcut

        # Up/down -> Change study
        shortcut_prevstudy = QShortcut(QKeySequence("Up"), self.pushButton_prevstudy)
        shortcut_prevstudy.activated.connect(lambda: self.action_changestudy(-1))
        shortcut_nextstudy = QShortcut(QKeySequence("Down"), self.pushButton_nextstudy)
        shortcut_nextstudy.activated.connect(lambda: self.action_changestudy(1))

    @pyqtSlot()
    def action_changestudy(self, changeby):
        current_id = self.comboBox_studies.currentIndex()
        new_id = (current_id + changeby) % self.comboBox_studies.count()
        self.comboBox_studies.setCurrentIndex(new_id)
        self.load_study()

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
        for img_array, image_item, plot_widget, roi_selectors in zip(self.plots['img_arrays'], self.plots['image_items'], self.plots['plot_widgets'], self.plots['roi_selectors']):
            # Get current axis range
            x_range = plot_widget.getAxis('bottom').range
            y_range = plot_widget.getAxis('left').range

            # Remove imageItem
            image_item.mousePressEvent = None
            plot_widget.removeItem(self.imageItem)
            del image_item

            # Remove ROIs from plot and delete
            for roi_name, roi_selector in roi_selectors:
                plot_widget.removeItem(roi_selector)
                roi_selector[roi_name] = None
                del roi_selector
            self.plots['roi_selectors'] = []

            # Draw image
            image_item = pg.ImageItem(img_array)
            plot_widget.addItem(image_item)
            if self.labelmode == 'add':
                image_item.mousePressEvent = self.add_node_at_mouse

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
                self.roi_coords[roi_name].append([point.x(), point.y()])
        # Finally save
        if self.t1map_path:
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
        x = round(event.pos().x())
        y = round(event.pos().y())
        self.roi_coords[self.activelabel_name].append([x, y])
        self.draw_image_and_rois(fix_roi=True)
        # Finally save
        if self.t1map_path:
            self.save_coords()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = MainWindowUI(MainWindow)
    MainWindow.show()
    print('Showing')
    app.exec_()
