# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\ui_label.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setEnabled(True)
        MainWindow.resize(1680, 1005)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.verticalLayout_Centre = QtWidgets.QVBoxLayout()
        self.verticalLayout_Centre.setObjectName("verticalLayout_Centre")
        self.horizontalLayout_Study = QtWidgets.QHBoxLayout()
        self.horizontalLayout_Study.setObjectName("horizontalLayout_Study")
        self.comboBox_studies = QtWidgets.QComboBox(self.centralwidget)
        self.comboBox_studies.setEnabled(True)
        self.comboBox_studies.setEditable(False)
        self.comboBox_studies.setObjectName("comboBox_studies")
        self.comboBox_studies.addItem("")
        self.horizontalLayout_Study.addWidget(self.comboBox_studies)
        self.pushButton_prevstudy = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_prevstudy.setObjectName("pushButton_prevstudy")
        self.horizontalLayout_Study.addWidget(self.pushButton_prevstudy)
        self.pushButton_nextstudy = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_nextstudy.setObjectName("pushButton_nextstudy")
        self.horizontalLayout_Study.addWidget(self.pushButton_nextstudy)
        self.horizontalLayout_Study.setStretch(0, 8)
        self.horizontalLayout_Study.setStretch(1, 1)
        self.horizontalLayout_Study.setStretch(2, 1)
        self.verticalLayout_Centre.addLayout(self.horizontalLayout_Study)
        self.horizontalLayout_Buttons = QtWidgets.QHBoxLayout()
        self.horizontalLayout_Buttons.setObjectName("horizontalLayout_Buttons")
        self.pushButton_mode = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_mode.setEnabled(True)
        self.pushButton_mode.setStyleSheet("  background-color: #4CAF50; /* Green */\n"
"  border: none;\n"
"  color: white;\n"
"  padding: 15px 32px;\n"
"  text-align: center;\n"
"  text-decoration: none;\n"
"  display: inline-block;\n"
"  font-size: 16px;")
        self.pushButton_mode.setObjectName("pushButton_mode")
        self.horizontalLayout_Buttons.addWidget(self.pushButton_mode)
        self.label_Labels = QtWidgets.QLabel(self.centralwidget)
        self.label_Labels.setStyleSheet("  border: none;\n"
"  color: black;\n"
"  padding: 15px 32px;\n"
"  text-align: right;\n"
"  text-decoration: none;\n"
"  font-size: 16px;")
        self.label_Labels.setObjectName("label_Labels")
        self.horizontalLayout_Buttons.addWidget(self.label_Labels)
        self.pushButton_endo = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_endo.setEnabled(True)
        self.pushButton_endo.setStyleSheet("  background-color: #FF0000; /* Green */\n"
"  border:5px solid #000000;\n"
"  color: white;\n"
"  padding: 15px 32px;\n"
"  text-align: center;\n"
"  text-decoration: none;\n"
"  display: inline-block;\n"
"  font-size: 16px;")
        self.pushButton_endo.setObjectName("pushButton_endo")
        self.horizontalLayout_Buttons.addWidget(self.pushButton_endo)
        self.pushButton_epi = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_epi.setEnabled(True)
        self.pushButton_epi.setStyleSheet("  background-color: #FF0000; /* Green */\n"
"  border: none;\n"
"  color: white;\n"
"  padding: 15px 32px;\n"
"  text-align: center;\n"
"  text-decoration: none;\n"
"  display: inline-block;\n"
"  font-size: 16px;")
        self.pushButton_epi.setObjectName("pushButton_epi")
        self.horizontalLayout_Buttons.addWidget(self.pushButton_epi)
        self.pushButton_myo = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_myo.setEnabled(True)
        self.pushButton_myo.setStyleSheet("  background-color: #FF0000; /* Green */\n"
"  border: none;\n"
"  color: white;\n"
"  padding: 15px 32px;\n"
"  text-align: center;\n"
"  text-decoration: none;\n"
"  display: inline-block;\n"
"  font-size: 16px;")
        self.pushButton_myo.setObjectName("pushButton_myo")
        self.horizontalLayout_Buttons.addWidget(self.pushButton_myo)
        self.horizontalLayout_Buttons.setStretch(0, 1)
        self.horizontalLayout_Buttons.setStretch(2, 3)
        self.horizontalLayout_Buttons.setStretch(3, 3)
        self.horizontalLayout_Buttons.setStretch(4, 3)
        self.verticalLayout_Centre.addLayout(self.horizontalLayout_Buttons)
        self.horizontalLayout_Plots = QtWidgets.QHBoxLayout()
        self.horizontalLayout_Plots.setObjectName("horizontalLayout_Plots")
        self.verticalLayout_Centre.addLayout(self.horizontalLayout_Plots)
        self.verticalLayout_Centre.setStretch(0, 1)
        self.verticalLayout_Centre.setStretch(1, 1)
        self.verticalLayout_Centre.setStretch(2, 40)
        self.gridLayout.addLayout(self.verticalLayout_Centre, 0, 1, 1, 1)
        self.verticalLayout_Left = QtWidgets.QVBoxLayout()
        self.verticalLayout_Left.setObjectName("verticalLayout_Left")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.label_StudyData = QtWidgets.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setPointSize(24)
        self.label_StudyData.setFont(font)
        self.label_StudyData.setObjectName("label_StudyData")
        self.verticalLayout.addWidget(self.label_StudyData)
        self.label_PatientID = QtWidgets.QLabel(self.centralwidget)
        self.label_PatientID.setObjectName("label_PatientID")
        self.verticalLayout.addWidget(self.label_PatientID)
        self.label_StudyDate = QtWidgets.QLabel(self.centralwidget)
        self.label_StudyDate.setObjectName("label_StudyDate")
        self.verticalLayout.addWidget(self.label_StudyDate)
        self.label_ExportDate = QtWidgets.QLabel(self.centralwidget)
        self.label_ExportDate.setObjectName("label_ExportDate")
        self.verticalLayout.addWidget(self.label_ExportDate)
        self.verticalLayout_Left.addLayout(self.verticalLayout)
        self.verticalLayout_Info = QtWidgets.QVBoxLayout()
        self.verticalLayout_Info.setObjectName("verticalLayout_Info")
        self.textBrowser = QtWidgets.QTextBrowser(self.centralwidget)
        self.textBrowser.setObjectName("textBrowser")
        self.verticalLayout_Info.addWidget(self.textBrowser)
        self.verticalLayout_Left.addLayout(self.verticalLayout_Info)
        self.gridLayout.addLayout(self.verticalLayout_Left, 0, 0, 1, 1)
        self.gridLayout.setColumnStretch(0, 1)
        self.gridLayout.setColumnStretch(1, 6)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.toolBar = QtWidgets.QToolBar(MainWindow)
        self.toolBar.setEnabled(True)
        self.toolBar.setObjectName("toolBar")
        MainWindow.addToolBar(QtCore.Qt.TopToolBarArea, self.toolBar)
        self.actionLoad_study = QtWidgets.QAction(MainWindow)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap("../../BARD/ui/icons/Patient.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionLoad_study.setIcon(icon)
        self.actionLoad_study.setObjectName("actionLoad_study")
        self.actionAbout = QtWidgets.QAction(MainWindow)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap("../../BARD/ui/icons/About.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.actionAbout.setIcon(icon1)
        self.actionAbout.setObjectName("actionAbout")
        self.actionExit = QtWidgets.QAction(MainWindow)
        self.actionExit.setObjectName("actionExit")
        self.toolBar.addAction(self.actionLoad_study)
        self.toolBar.addAction(self.actionAbout)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Image Annotator"))
        self.comboBox_studies.setCurrentText(_translate("MainWindow", "Please select a file"))
        self.comboBox_studies.setItemText(0, _translate("MainWindow", "Please select a file"))
        self.pushButton_prevstudy.setText(_translate("MainWindow", "Prev"))
        self.pushButton_nextstudy.setText(_translate("MainWindow", "Next"))
        self.pushButton_mode.setText(_translate("MainWindow", "Add mode"))
        self.label_Labels.setText(_translate("MainWindow", "Labels:"))
        self.pushButton_endo.setText(_translate("MainWindow", "1 Endocardium"))
        self.pushButton_epi.setText(_translate("MainWindow", "2 Epicardium"))
        self.pushButton_myo.setText(_translate("MainWindow", "3 Myocardium"))
        self.label_StudyData.setText(_translate("MainWindow", "Study Data"))
        self.label_PatientID.setText(_translate("MainWindow", "Patient ID:"))
        self.label_StudyDate.setText(_translate("MainWindow", "Study Date:"))
        self.label_ExportDate.setText(_translate("MainWindow", "Export Date:"))
        self.textBrowser.setHtml(_translate("MainWindow", "<!DOCTYPE HTML PUBLIC \"-//W3C//DTD HTML 4.0//EN\" \"http://www.w3.org/TR/REC-html40/strict.dtd\">\n"
"<html><head><meta name=\"qrichtext\" content=\"1\" /><style type=\"text/css\">\n"
"p, li { white-space: pre-wrap; }\n"
"</style></head><body style=\" font-family:\'MS Shell Dlg 2\'; font-size:8.25pt; font-weight:400; font-style:normal;\">\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:16pt;\">Information</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:16pt;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">This system will look for studies set by the </span><span style=\" font-family:\'Courier New\'; font-size:12pt; color:#000000; background-color:#ffffff;\">DICOMDIR </span><span style=\" font-size:12pt;\">variable defined at the top of the </span><span style=\" font-family:\'Courier New\'; font-size:12pt; color:#000000; background-color:#ffffff;\">label</span><span style=\" font-family:\'Courier New\'; font-size:12pt; color:#808030;\">.</span><span style=\" font-family:\'Courier New\'; font-size:12pt; color:#000000;\">py </span><span style=\" font-size:12pt;\">file.</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:12pt;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">One loaded, studies can be selected from the top dropbox box, and sequences from the box below that.</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:12pt;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">The selected sequence will always default to T2.</span></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:12pt;\"><br /></p>\n"
"<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt;\">Using the keyboard combinations is strongly recommended, as follows:</span></p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" font-size:8pt;\" style=\" margin-top:12px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-size:12pt; font-weight:600;\">Up</span><span style=\" font-size:12pt;\"> - Previous study</span></li></ul>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:1; text-indent:0px; font-size:12pt;\"><br /></p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" font-size:12pt;\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">Down</span> - Next study</li></ul>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:1; text-indent:0px; font-size:12pt;\"><br /></p>\n"
"<ul style=\"margin-top: 0px; margin-bottom: 0px; margin-left: 0px; margin-right: 0px; -qt-list-indent: 1;\"><li style=\" font-size:12pt;\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">Left</span> - Previous sequence in study study</li>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:1; text-indent:0px; font-size:12pt;\"><br /></p>\n"
"<li style=\" font-size:12pt;\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">Right</span> - Next sequence in study</li>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:1; text-indent:0px; font-size:12pt;\"><br /></p>\n"
"<li style=\" font-size:12pt;\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">1</span> - Label endocardium</li>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:1; text-indent:0px; font-size:12pt;\"><br /></p>\n"
"<li style=\" font-size:12pt;\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">2</span> - Label epicardium</li>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:1; text-indent:0px; font-size:12pt;\"><br /></p>\n"
"<li style=\" font-size:12pt;\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">3</span> - Label other myocardium (e.g. for pap muscles)</li>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:1; text-indent:0px; font-size:12pt;\"><br /></p>\n"
"<li style=\" font-size:12pt;\" style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\"><span style=\" font-weight:600;\">Spacebar</span> - Alternate between \'edit\' and \'add\' mode</li></ul>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:12pt;\"><br /></p>\n"
"<p style=\"-qt-paragraph-type:empty; margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px; font-size:12pt;\"><br /></p></body></html>"))
        self.toolBar.setWindowTitle(_translate("MainWindow", "toolBar"))
        self.actionLoad_study.setText(_translate("MainWindow", "Load patient"))
        self.actionLoad_study.setShortcut(_translate("MainWindow", "Ctrl+N"))
        self.actionAbout.setText(_translate("MainWindow", "About"))
        self.actionAbout.setShortcut(_translate("MainWindow", "Ctrl+I"))
        self.actionExit.setText(_translate("MainWindow", "Exit"))
        self.actionExit.setShortcut(_translate("MainWindow", "Ctrl+Q"))

