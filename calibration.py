import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QGraphicsView, QGraphicsScene, QGraphicsEllipseItem, QGraphicsItem
from PyQt5.QtCore import Qt, QRectF, QLineF
from PyQt5.QtGui import QPen, QBrush
from PyQt5.QtWidgets import QGraphicsItem
from PyQt5.QtGui import QPainter
import math 
import cv2
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QGraphicsPixmapItem
from PyQt5.QtWidgets import QDialog, QPushButton, QVBoxLayout, QApplication, QMainWindow
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout
from PyQt5.QtWidgets import QLineEdit, QVBoxLayout, QWidget
#import qpointf
from PyQt5.QtCore import QPointF



class HandleItem(QGraphicsEllipseItem):
    def __init__(self, line_item, is_start_handle):
        super().__init__(-5, -5, 10, 10)
        self.setBrush(QBrush(Qt.red))
        self.setFlag(QGraphicsItem.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)

        #set transparency
        self.setOpacity(0.5)

        #set border red
        self.setPen(QPen(Qt.red, 2))

        self.line_item = line_item
        self.is_start_handle = is_start_handle

        if is_start_handle:
            self.setPos(line_item.line.p1())
        else:
            self.setPos(line_item.line.p2())

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionChange:
            if self.is_start_handle:
                self.line_item.line.setP1(value)
                
            else:
                self.line_item.line.setP2(value)
            self.line_item.update()

            self.line_item.update_perpendicular_lines()

        return super().itemChange(change, value)
    
                
  

    def mousePressEvent(self, event):
        self.setCursor(Qt.ClosedHandCursor)
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self.setCursor(Qt.OpenHandCursor)
        super().mouseReleaseEvent(event)

    def mouseMoveEvent(self, event):
        # The position update is handled by the itemChange function
        super().mouseMoveEvent(event)

class DraggableLineItem(QGraphicsItem):
    def __init__(self, line):
        super().__init__()
        self.line = line
        self.setFlag(QGraphicsItem.ItemIsSelectable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges)

        # Create handles for draggable endpoints
        self.start_handle = HandleItem(self, True)
        self.end_handle = HandleItem(self, False)

        # Add handles to the scene as children of this line item
        self.start_handle.setParentItem(self)
        self.end_handle.setParentItem(self)

        self.perp_line1 = None
        self.perp_line2 = None
        self.update_perpendicular_lines()

    def boundingRect(self):
        return QRectF(self.line.p1(), self.line.p2())
        
    def paint(self, painter, option, widget=None):
        painter.setPen(QPen(Qt.black, 2))
        painter.drawLine(self.line)

    def update_perpendicular_lines(self):
        line_len = 20  # Length of the perpendicular lines
        angle = math.atan2(self.line.dy(), self.line.dx()) + math.pi / 2

        # Calculate half-length offsets for the start and end points
        offset = QPointF(math.cos(angle) * line_len / 2, math.sin(angle) * line_len / 2)

        # Start Perpendicular Line
        p1_start = self.line.p1() - offset
        p1_end = self.line.p1() + offset
        self.perp_line1 = QLineF(p1_start, p1_end)

        # End Perpendicular Line
        p2_start = self.line.p2() - offset
        p2_end = self.line.p2() + offset
        self.perp_line2 = QLineF(p2_start, p2_end)

    def paint(self, painter, option, widget=None):
        painter.setPen(QPen(Qt.black, 2))
        painter.drawLine(self.line)
        painter.drawLine(self.perp_line1)
        painter.drawLine(self.perp_line2)


#import qimage
from PyQt5.QtGui import QImage, QPixmap
#import qgraphicsitem
from PyQt5.QtWidgets import QGraphicsPixmapItem

class DraggableLineWidget(QGraphicsView):
    def __init__(self, parent=None, image=None):
        super().__init__(parent)
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        self.image = image

        self.start_point = None
        self.end_point = None
        self.current_line = None
        self.is_drawing = False  # Flag to track if we are currently drawing a line
        self.last_line = None
        self.setRenderHint(QPainter.Antialiasing)
        self.setMouseTracking(True)

        if self.image is not None:
            self.set_image(self.image)



    def set_image(self, image):
        # Convert the NumPy array (image) to QImage
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert the image to RGBimage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width, channel = image.shape
        bytesPerLine = 3 * width
        q_img = QImage(image.data, width, height, bytesPerLine, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)

        # Add the QPixmap to the scene
        self.scene.addPixmap(pixmap)
        self.setSceneRect(QRectF(pixmap.rect()))

    def wheelEvent(self, event):
        factor = 1.01  # Define the scaling factor (zoom speed)
        if event.angleDelta().y() > 0:
            # Zoom in
            self.scale(factor, factor)
        else:
            # Zoom out
            self.scale(1 / factor, 1 / factor)

    def mousePressEvent(self, event):
        print("Mouse Pressed")
        clicked_item = self.itemAt(event.pos())
        #check if clicked item is a pixmap


        if clicked_item and not self.is_drawing and not isinstance(clicked_item, QGraphicsPixmapItem):
        
            print("Clicked on an existing item:", clicked_item)
            self.scene.clearSelection()
            clicked_item.setSelected(True)
            self.current_line = None
        elif not self.is_drawing and event.button() == Qt.LeftButton:
            print("Starting a new line")
            self.start_point = self.mapToScene(event.pos())
            self.current_line = DraggableLineItem(QLineF(self.start_point, self.start_point))
            self.scene.addItem(self.current_line)
            self.is_drawing = True
        elif self.is_drawing and event.button() == Qt.LeftButton:
            print("Finalizing the line")
            #store most recent line
            self.last_line = self.current_line

            self.is_drawing = False
            self.current_line = None

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self.is_drawing and self.current_line:
            print("Drawing the line")
            self.end_point = self.mapToScene(event.pos())
            self.current_line.line.setP2(self.end_point)
            self.current_line.update()
            #upddate handles
            self.current_line.end_handle.setPos(self.end_point)
        else:
            super().mouseMoveEvent(event)

        #update scene
        self.scene.update()



from PyQt5.QtWidgets import QDialog, QPushButton, QVBoxLayout, QApplication, QMainWindow
from PyQt5.QtCore import Qt
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QHBoxLayout

class CalibrationDialog(QDialog):
    lineLengthSignal = pyqtSignal(float)  # Define a signal to emit the line length

    def __init__(self, parent=None, current_frame=None):
        super().__init__(parent)
        self.setWindowTitle("Calibration")

        #set size
        self.setMinimumSize(800, 600)

        # Initialize the Draggable Line Widget with the current frame
        self.draggable_line_widget = DraggableLineWidget(self, image=current_frame)

        # Layout for buttons
        self.button_layout = QHBoxLayout()
        self.button_layout.addStretch(1)  # This will push the buttons to the right


        # Accept Button
        self.accept_button = QPushButton("Accept")
        self.accept_button.clicked.connect(self.accept)
        self.button_layout.addWidget(self.accept_button)


        # Apply Button
        self.apply_button = QPushButton("Apply")
        self.apply_button.clicked.connect(self.apply_calibration)
        self.button_layout.addWidget(self.apply_button)


        # Cancel Button
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        self.button_layout.addWidget(self.cancel_button)



        # Main Layout
        self.main_layout = QVBoxLayout()
        self.main_layout.addWidget(self.draggable_line_widget)
        # Add the button layout at the bottom of the main layout
        self.main_layout.addLayout(self.button_layout)
        self.setLayout(self.main_layout)


    def accept(self):
        print("Calibration accepted")
        self.emit_line_length()
        super().accept()

    def apply_calibration(self):
        print("Calibration applied")
        self.emit_line_length()

    def emit_line_length(self):
        if self.draggable_line_widget.last_line:
            line = self.draggable_line_widget.last_line.line
            # Calculate Euclidean distance
            p1 = line.p1()
            p2 = line.p2()
            length = math.sqrt((p2.x() - p1.x()) ** 2 + (p2.y() - p1.y()) ** 2)
            print("Line length (Euclidean):", length)
            self.lineLengthSignal.emit(round(length, 3))

from PyQt5.QtWidgets import QLineEdit, QVBoxLayout, QWidget
class TestMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Draggable Line Widget Test")
        self.setGeometry(100, 100, 800, 600)

        # Add a button to open the calibration dialog
        self.calibration_button = QPushButton("Open Calibration Dialog")



     # Add an edit line to display the line length
        self.line_length_edit = QLineEdit(self)
        self.calibration_button.clicked.connect(self.open_calibration_dialog)

        # Use a vertical layout to include the button and the edit line
        layout = QVBoxLayout()
        layout.addWidget(self.calibration_button)
        layout.addWidget(self.line_length_edit)
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

    def open_calibration_dialog(self):
        self.calibration_dialog = CalibrationDialog(self)
        self.calibration_dialog.lineLengthSignal.connect(self.update_line_length)
        self.calibration_dialog.exec_()

    def update_line_length(self, length):
        self.line_length_edit.setText(str(round(length, 3)))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestMainWindow()
    window.show()
    sys.exit(app.exec_())