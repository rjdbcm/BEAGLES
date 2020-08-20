from PyQt5.QtGui import QPainter, QPixmap, QBrush, QColor
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import QWidget

from libs.shape import Shape


class CanvasWidget(QWidget):
    CREATE, EDIT = list(range(2))
    CURSOR_DEFAULT = Qt.ArrowCursor
    CURSOR_POINT = Qt.PointingHandCursor
    CURSOR_DRAW = Qt.CrossCursor
    CURSOR_MOVE = Qt.ClosedHandCursor
    CURSOR_GRAB = Qt.OpenHandCursor

    def __init__(self, *args, **kwargs):
        super(CanvasWidget, self).__init__(*args, **kwargs)
        self.mode = self.EDIT
        self.pixmap = QPixmap()
        self._painter = QPainter()

    @property
    def drawing(self):
        return self.mode == self.CREATE

    @property
    def editing(self):
        return self.mode == self.EDIT

    @editing.setter
    def editing(self, value=True):
        self.mode = self.EDIT if value else self.CREATE
        if not value:  # Create
            self.unHighlight()
            self.deSelectShape()
        self.prevPoint = QPointF()
        self.repaint()

    # These two, along with a call to adjustSize are required for the
    # scroll area.
    def sizeHint(self):
        return self.minimumSizeHint()

    def minimumSizeHint(self):
        if self.pixmap:
            return self.scale * self.pixmap.size()
        return super(CanvasWidget, self).minimumSizeHint()

    def enterEvent(self, ev):
        self.overrideCursor(self._cursor)

    def leaveEvent(self, ev):
        self.restoreCursor()

    def focusOutEvent(self, ev):
        self.restoreCursor()

    def isVisible(self, shape):
        return self.visible.get(shape, True)

    def mouseMoveEvent(self, ev):
        """Update line with last point and current coordinates."""
        pos = self.transformPos(ev.pos())

        # Update coordinates in status bar if image is opened
        window = self.parent().window()
        if window.filePath is not None:
            self.parent().window().labelCoordinates.setText(
                'X: %d; Y: %d' % (pos.x(), pos.y()))

        # Polygon drawing.
        if self.drawing:
            self.overrideCursor(self.CURSOR_DRAW)
            if self.current:
                # Display annotation width and height while drawing
                currentWidth = abs(self.current[0].x() - pos.x())
                currentHeight = abs(self.current[0].y() - pos.y())
                self.parent().window().labelCoordinates.setText(
                    'Width: %d, Height: %d / X: %d; Y: %d' % (
                        currentWidth, currentHeight, pos.x(), pos.y()))
                color = self.drawingLineColor
                if self.outOfPixmap(pos):
                    # Don't allow the user to draw outside the pixmap.
                    # Project the point to the pixmap's edges.
                    pos = self.intersectionPoint(self.current[-1], pos)
                elif len(self.current) > 1 and self.closeEnough(pos,
                                                                self.current[
                                                                    0]):
                    # Attract line to starting point and colorise to alert the
                    # user:
                    pos = self.current[0]
                    color = self.current.line_color
                    self.overrideCursor(self.CURSOR_POINT)
                    self.current.highlightVertex(0, Shape.NEAR_VERTEX)

                if self.drawSquare:
                    initPos = self.current[0]
                    minX = initPos.x()
                    minY = initPos.y()
                    min_size = min(abs(pos.x() - minX), abs(pos.y() - minY))
                    directionX = -1 if pos.x() - minX < 0 else 1
                    directionY = -1 if pos.y() - minY < 0 else 1
                    self.line[1] = QPointF(minX + directionX * min_size,
                                           minY + directionY * min_size)
                else:
                    self.line[1] = pos

                self.line.line_color = color
                self.prevPoint = QPointF()
                self.current.highlightClear()
            else:
                self.prevPoint = pos
            self.repaint()
            return

        # Polygon copy moving.
        if Qt.RightButton & ev.buttons():
            if self.selectedShapeCopy and self.prevPoint:
                self.overrideCursor(self.CURSOR_MOVE)
                self.boundedMoveShape(self.selectedShapeCopy, pos)
                self.repaint()
            elif self.selectedShape:
                self.selectedShapeCopy = self.selectedShape.copy()
                self.repaint()
            return

        # Polygon/Vertex moving.
        if Qt.LeftButton & ev.buttons():
            if self.selectedVertex():
                self.boundedMoveVertex(pos)
                self.shapeMoved.emit()
                self.repaint()
            elif self.selectedShape and self.prevPoint:
                self.overrideCursor(self.CURSOR_MOVE)
                self.boundedMoveShape(self.selectedShape, pos)
                self.shapeMoved.emit()
                self.repaint()
            return

        # Just hovering over the canvas, 2 posibilities:
        # - Highlight shapes
        # - Highlight vertex
        # Update shape/vertex fill and tooltip value accordingly.
        self.setToolTip("Image")
        for shape in reversed([s for s in self.shapes if self.isVisible(s)]):
            # Look for a nearby vertex to highlight. If that fails,
            # check if we happen to be inside a shape.
            index = shape.nearestVertex(pos, self.epsilon)
            if index is not None:
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.hVertex, self.hShape = index, shape
                shape.highlightVertex(index, shape.MOVE_VERTEX)
                self.overrideCursor(self.CURSOR_POINT)
                self.setToolTip("Click & drag to move point")
                self.setStatusTip(self.toolTip())
                self.update()
                break
            elif shape.containsPoint(pos):
                if self.selectedVertex():
                    self.hShape.highlightClear()
                self.hVertex, self.hShape = None, shape
                self.setToolTip(
                    "Click & drag to move shape '%s'" % shape.label)
                self.setStatusTip(self.toolTip())
                self.overrideCursor(self.CURSOR_GRAB)
                self.update()
                break
        else:  # Nothing found, clear highlights, reset state.
            if self.hShape:
                self.hShape.highlightClear()
                self.update()
            self.hVertex, self.hShape = None, None
            self.overrideCursor(self.CURSOR_DEFAULT)

    def mousePressEvent(self, ev):
        pos = self.transformPos(ev.pos())

        if ev.button() == Qt.LeftButton:
            if self.drawing:
                self.handleDrawing(pos)
            else:
                self.selectShapePoint(pos)
                self.prevPoint = pos
                self.repaint()
        elif ev.button() == Qt.RightButton and self.editing:
            self.selectShapePoint(pos)
            self.prevPoint = pos
            self.repaint()

    def mouseReleaseEvent(self, ev):
        if ev.button() == Qt.RightButton:
            menu = self.menus[bool(self.selectedShapeCopy)]
            self.restoreCursor()
            if not menu.exec_(self.mapToGlobal(ev.pos())) \
                    and self.selectedShapeCopy:
                # Cancel the move by deleting the shadow copy.
                self.selectedShapeCopy = None
                self.repaint()
        elif ev.button() == Qt.LeftButton and self.selectedShape:
            if self.selectedVertex():
                self.overrideCursor(self.CURSOR_POINT)
            else:
                self.overrideCursor(self.CURSOR_GRAB)
        elif ev.button() == Qt.LeftButton:
            pos = self.transformPos(ev.pos())
            if self.drawing:
                self.handleDrawing(pos)

    def mouseDoubleClickEvent(self, ev):
        # We need at least 4 points here, since the mousePress handler
        # adds an extra one before this handler is called.
        if self.canCloseShape() and len(self.current) > 3:
            self.current.popPoint()
            self.finalise()

    def wheelEvent(self, ev):
        delta = ev.angleDelta()
        h_delta = delta.x()
        v_delta = delta.y()

        mods = ev.modifiers()
        if Qt.ControlModifier == int(mods) and v_delta:
            self.zoomRequest.emit(v_delta)
        else:
            v_delta and self.scrollRequest.emit(v_delta, Qt.Vertical)
            h_delta and self.scrollRequest.emit(h_delta, Qt.Horizontal)
        ev.accept()

    def keyPressEvent(self, ev):
        key = ev.key()
        if key == Qt.Key_Escape and self.current:
            print('ESC press')
            self.current = None
            self.drawingPolygon.emit(False)
            self.update()
        elif key == Qt.Key_Return and self.canCloseShape():
            self.finalise()
        elif key == Qt.Key_Left and self.selectedShape:
            self.moveOnePixel('Left')
        elif key == Qt.Key_Right and self.selectedShape:
            self.moveOnePixel('Right')
        elif key == Qt.Key_Up and self.selectedShape:
            self.moveOnePixel('Up')
        elif key == Qt.Key_Down and self.selectedShape:
            self.moveOnePixel('Down')

    def paintEvent(self, event):
        if not self.pixmap:
            return super(CanvasWidget, self).paintEvent(event)

        p = self._painter
        p.begin(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.HighQualityAntialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        p.scale(self.scale, self.scale)
        p.translate(self.offsetToCenter())

        p.drawPixmap(0, 0, self.pixmap)
        Shape.scale = self.scale
        for shape in self.shapes:
            if (shape.selected or not self._hideBackground) and self.isVisible(shape):
                shape.fill = shape.selected or shape == self.hShape
                shape.paint(p)
        if self.current:
            self.current.paint(p)
            self.line.paint(p)
        if self.selectedShapeCopy:
            self.selectedShapeCopy.paint(p)

        # Paint rect
        if self.current is not None and len(self.line) == 2:
            leftTop = self.line[0]
            rightBottom = self.line[1]
            rectWidth = rightBottom.x() - leftTop.x()
            rectHeight = rightBottom.y() - leftTop.y()
            p.setPen(self.drawingRectColor)
            brush = QBrush(Qt.BDiagPattern)
            p.setBrush(brush)
            p.drawRect(leftTop.x(), leftTop.y(), rectWidth, rectHeight)

        if self.drawing and not self.prevPoint.isNull() and not self.outOfPixmap(self.prevPoint):
            p.setPen(QColor(0, 0, 0))
            p.drawLine(self.prevPoint.x(), 0, self.prevPoint.x(), self.pixmap.height())
            p.drawLine(0, self.prevPoint.y(), self.pixmap.width(), self.prevPoint.y())

        self.setAutoFillBackground(True)
        if self.verified:
            pal = self.palette()
            pal.setColor(self.backgroundRole(), QColor(184, 239, 38, 128))
            self.setPalette(pal)
        else:
            pal = self.palette()
            pal.setColor(self.backgroundRole(), QColor(25, 35, 45, 255))
            self.setPalette(pal)

        p.end()



