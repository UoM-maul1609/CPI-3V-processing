# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow
from PyQt5.QtWidgets import QApplication, QDialog

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Selector(object):
    def setupUi(self, Selector):
        Selector.setObjectName("Selector")
        Selector.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(Selector)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalSlider = QtWidgets.QSlider(self.centralwidget)
        self.horizontalSlider.setGeometry(QtCore.QRect(230, 300, 160, 22))
        self.horizontalSlider.setOrientation(QtCore.Qt.Horizontal)
        self.horizontalSlider.setObjectName("horizontalSlider")
        Selector.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(Selector)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        self.menuHi = QtWidgets.QMenu(self.menubar)
        self.menuHi.setObjectName("menuHi")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        Selector.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(Selector)
        self.statusbar.setObjectName("statusbar")
        Selector.setStatusBar(self.statusbar)
        self.actionYes = QtWidgets.QAction(Selector)
        self.actionYes.setCheckable(True)
        self.actionYes.setObjectName("actionYes")
        self.actionNo = QtWidgets.QAction(Selector)
        self.actionNo.setObjectName("actionNo")
        self.actionOpen = QtWidgets.QAction(Selector)
        self.actionOpen.setObjectName("actionOpen")
        self.menuHi.addAction(self.actionYes)
        self.menuHi.addAction(self.actionNo)
        self.menuFile.addAction(self.actionOpen)
        self.menubar.addAction(self.menuHi.menuAction())
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(Selector)
        QtCore.QMetaObject.connectSlotsByName(Selector)

    def retranslateUi(self, Selector):
        _translate = QtCore.QCoreApplication.translate
        Selector.setWindowTitle(_translate("Selector", "MainWindow"))
        self.menuHi.setTitle(_translate("Selector", "Hi"))
        self.menuFile.setTitle(_translate("Selector", "File"))
        self.actionYes.setText(_translate("Selector", "Yes"))
        self.actionNo.setText(_translate("Selector", "No"))
        self.actionOpen.setText(_translate("Selector", "Open"))


if __name__ == "__main__":
    import sys
    

    app = QApplication(sys.argv)
    window = QMainWindow()
    ui = Ui_Selector()
    ui.setupUi(window)

    window.show()


sys.exit(app.exec_())