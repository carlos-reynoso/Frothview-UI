import sys
import cv2
from PyQt5.QtWidgets import QApplication, QWidget, QGraphicsView, QGraphicsScene, QVBoxLayout, QPushButton, QSlider, QSizePolicy
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt, QMutex
from PyQt5.QtGui import QPixmap, QImage, QIcon
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QMainWindow
from helper_functions import VideoPlayer, OptionsMenu,process_frame
from PyQt5.QtWidgets import QMainWindow, QApplication, QDockWidget, QTextEdit
from PyQt5.QtWidgets import QApplication, QMainWindow, QDesktopWidget


class MainApplication(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Main Application")
        self.video_player = VideoPlayer(parent=self,process_frame=process_frame)
        self.setCentralWidget(self.video_player)

        # Create and add the dock widgets
        self.create_docks()

        # Set the window to maximized state
        self.showMaximized()

        # Get the screen size
        screen = QDesktopWidget().screenGeometry()
        self.setMinimumSize(screen.width(), screen.height())

    def create_docks(self):
        # Dock for Options on the Left
        options_dock = QDockWidget("Options", self)
        options_dock_widget = QTextEdit()  # Placeholder widget
        options_dock.setWidget(options_dock_widget)
        self.addDockWidget(Qt.LeftDockWidgetArea, options_dock)
        options_dock_widget = OptionsMenu(self)
        options_dock.setWidget(options_dock_widget)


        # Dock for Time Series Plot below the Video
        timeseries_dock = QDockWidget("Time Series Plot", self)
        timeseries_dock_widget = QTextEdit()  # Placeholder widget
        timeseries_dock.setWidget(timeseries_dock_widget)
        self.addDockWidget(Qt.BottomDockWidgetArea, timeseries_dock)

        # Dock for Plots on the Right
        plots_dock = QDockWidget("Plots", self)
        plots_dock_widget = QTextEdit()  # Placeholder widget
        plots_dock.setWidget(plots_dock_widget)
        self.addDockWidget(Qt.RightDockWidgetArea, plots_dock)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_app = MainApplication()
    main_app.show()
    sys.exit(app.exec_())
