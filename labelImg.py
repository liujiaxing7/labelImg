#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import codecs
import distutils.spawn
import os.path
import platform
import re
import sys
import subprocess

from functools import partial
from collections import defaultdict

from PyQt5 import QtGui

CURRENT_DIR = os.path.dirname(__file__)

try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    # needed for py3+qt4
    # Ref:
    # http://pyqt.sourceforge.net/Docs/PyQt4/incompatible_apis.html
    # http://stackoverflow.com/questions/21217399/pyqt4-qtcore-qvariant-object-instead-of-a-string
    if sys.version_info.major >= 3:
        import sip
        sip.setapi('QVariant', 2)
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

from libs.combobox import ComboBox
from libs.resources import *
from libs.constants import *
from libs.utils import *
from libs.settings import Settings
from libs.shape import Shape, DEFAULT_LINE_COLOR, DEFAULT_FILL_COLOR, box2Shape
from libs.stringBundle import StringBundle
from libs.canvas import Canvas
from libs.zoomWidget import ZoomWidget
from libs.labelDialog import LabelDialog
from libs.classDialog import ClassDialog
from libs.colorDialog import ColorDialog
from libs.labelFile import LabelFile, LabelFileError, LabelFileFormat
from libs.toolBar import ToolBar
from libs.pascal_voc_io import PascalVocReader
from libs.pascal_voc_io import XML_EXT
from libs.yolo_io import YoloReader
from libs.yolo_io import TXT_EXT
from libs.create_ml_io import CreateMLReader
from libs.create_ml_io import JSON_EXT
from libs.ustr import ustr
from libs.hashableQListWidgetItem import HashableQListWidgetItem
try:
    from libs.detector.ssd.onnxmodel import ONNXModel
    is_onnxok=True
except:
    is_onnxok=False
from libs.detector.ssd.postprocess.ssd import PostProcessor_SSD
import numpy as np
import cv2
from libs.utils.file import MkdirSimple
from libs.utils.utils import *

from libs.detector.ssd.model import SSD
from libs.detector.centernet.model import CenterNet
from libs.detector.yolov5.model import YOLOv5
from libs.detector.yolov3.model import YOLOv3
from libs.detector.mobileNetv2.model import MobileNetV2
from libs.detector.yolov3.postprocess.postprocess import load_class_names, weighted_nms


onnxModelIndex = 0
MODEL_PARAMS = {0: "_SSD", 1: "_CENTER_NET", 2: "_YOLOv5", 3: "_YOLOv5s", 4: "_YOLOv5_i18R", 5: "_YOLOv3", 6: "_MobileNetV2"}  # TODO models later should be added here
MODEL_PATH = {"_SSD": "config/cleaner/ssd.onnx",
              "_CENTER_NET": "config/human/centernet.onnx",
              "_YOLOv5": "config/human/yolov5.onnx",
              "_YOLOv5s": "config/human/yolov5s.onnx",
              "_YOLOv5_i18R": "config/i18R/yolov5X.onnx",
              "_YOLOv3": "config/i18R/yolov3.onnx",
              "_MobileNetV2": "config/rubby/mobilenetV2.onnx"}
MAX_IOU_FOR_DELETE = 0.6
ADD_RECTBOX_BY_SERIES_NUM = 10
IOU_NMS = 0.5
IMAGE_LIST_FILE =  "all.txt"
# IMG_SIZE_DICT = {'IMAGE_SIZE'+MODEL_PARAMS[0]: 320,
#                  'IMAGE_SIZE'+MODEL_PARAMS[1]: 320,
#                  'IMAGE_SIZE'+MODEL_PARAMS[2]: 640,}

__appname__ = 'labelImg'

def format_shape_qt(s):
    return dict(label=s.label,
                line_color=s.line_color.getRgb(),
                fill_color=s.fill_color.getRgb(),
                points=[(p.x(), p.y()) for p in s.points],
                difficult = s.difficult,
                distance = s.distance,
                score = s.score)


class WindowMixin(object):

    def menu(self, title, actions=None):
        menu = self.menuBar().addMenu(title)
        if actions:
            addActions(menu, actions)
        return menu

    def toolbar(self, title, actions=None):
        toolbar = ToolBar(title)
        toolbar.setObjectName(u'%sToolBar' % title)
        # toolbar.setOrientation(Qt.Vertical)
        toolbar.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        if actions:
            addActions(toolbar, actions)
        self.addToolBar(Qt.LeftToolBarArea, toolbar)
        return toolbar


class MainWindow(QMainWindow, WindowMixin):
    FIT_WINDOW, FIT_WIDTH, MANUAL_ZOOM = list(range(3))

    def __init__(self, defaultFilename=None, defaultPrefdefClassFile=None, defaultSaveDir=None):
        super(MainWindow, self).__init__()
        self.setWindowTitle(__appname__)

        self.fullyAutoMode = False
        self.EqualizeHist=False
        self.denoise = False
        self.timer4autolabel = QTimer(self)

        self.class_name_file_4_detect = os.path.join(CURRENT_DIR, "config/human/classes.names")
        self.model_file_4_detect = os.path.join(CURRENT_DIR, MODEL_PATH[MODEL_PARAMS[onnxModelIndex]])
        self._initDetect()

        # Load setting in the main thread
        self.settings = Settings()
        self.settings.load()
        settings = self.settings

        # Load string bundle for i18n
        self.stringBundle = StringBundle.getBundle()
        getStr = lambda strId: self.stringBundle.getString(strId)

        # Save as Pascal voc xml
        self.defaultSaveDir = defaultSaveDir
        self.labelFileFormat = settings.get(SETTING_LABEL_FILE_FORMAT, LabelFileFormat.PASCAL_VOC)

        # For loading all image under a directory
        self.mImgList = []
        self.dirname = None
        self.labelHist = []
        self.classes={}
        self.classes_list=[]
        self.lastOpenDir = None

        # Whether we need to save or not.
        self.dirty = False

        self._noSelectionSlot = False
        self._beginner = True
        self.screencastViewer = self.getAvailableScreencastViewer()
        self.screencast = "https://youtu.be/p0nR2YsCY_U"

        # Load predefined classes to the list
        self.loadPredefinedClasses(defaultPrefdefClassFile)

        # Main widgets and related state.
        self.labelDialog = LabelDialog(parent=self, listItem=self.labelHist)

        self.itemsToShapes = {}
        self.shapesToItems = {}
        self.prevLabelText = ''

        self.autoDelete = 0
        self.autoAdd = 0

        listLayout = QVBoxLayout()
        listLayout.setContentsMargins(0, 0, 0, 0)

        # Create a widget for using default label
        self.useDefaultLabelCheckbox = QCheckBox(getStr('useDefaultLabel'))
        self.useDefaultLabelCheckbox.setChecked(False)
        self.defaultLabelTextLine = QLineEdit()
        useDefaultLabelQHBoxLayout = QHBoxLayout()
        useDefaultLabelQHBoxLayout.addWidget(self.useDefaultLabelCheckbox)
        useDefaultLabelQHBoxLayout.addWidget(self.defaultLabelTextLine)
        useDefaultLabelContainer = QWidget()
        useDefaultLabelContainer.setLayout(useDefaultLabelQHBoxLayout)

        # Create a widget for edit and diffc button
        self.diffcButton = QCheckBox(getStr('useDifficult'))
        self.diffcButton.setChecked(False)
        self.diffcButton.stateChanged.connect(self.btnstate)
        self.editButton = QToolButton()
        self.editButton.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)

        # Create distance label
        distanceLayout = QHBoxLayout()
        self.labelDistance = QLabel("distance: ")
        self.labelDistanceUnit = QLabel("(cm)")
        self.textDistance = QLineEdit()
        self.textDistance.textChanged[str].connect(self.distanceChange)
        distanceLayout.addWidget(self.labelDistance)
        distanceLayout.addWidget(self.textDistance)
        distanceLayout.addWidget(self.labelDistanceUnit)
        distanceLayout.addStretch(2)
        distanceContainer = QWidget()
        distanceContainer.setLayout(distanceLayout)

        # Add some of widgets to listLayout
        listLayout.addWidget(self.editButton)
        listLayout.addWidget(self.diffcButton)
        listLayout.addWidget(distanceContainer)
        listLayout.addWidget(useDefaultLabelContainer)

        # Create and add combobox for showing unique labels in group
        self.comboBox = ComboBox(self)
        listLayout.addWidget(self.comboBox)

        # Create and add a widget for showing current label items
        self.labelList = QListWidget()
        labelListContainer = QWidget()
        labelListContainer.setLayout(listLayout)
        self.labelList.itemActivated.connect(self.labelSelectionChanged)
        self.labelList.itemSelectionChanged.connect(self.labelSelectionChanged)
        self.labelList.itemDoubleClicked.connect(self.editLabel)
        # Connect to itemChanged to detect checkbox changes.
        self.labelList.itemChanged.connect(self.labelItemChanged)
        listLayout.addWidget(self.labelList)



        self.dock = QDockWidget(getStr('boxLabelText'), self)
        self.dock.setObjectName(getStr('labels'))
        self.dock.setWidget(labelListContainer)

        self.fileListWidget = QListWidget()
        self.fileListWidget.itemDoubleClicked.connect(self.fileitemDoubleClicked)
        filelistLayout = QVBoxLayout()
        filelistLayout.setContentsMargins(0, 0, 0, 0)
        filelistLayout.addWidget(self.fileListWidget)
        fileListContainer = QWidget()
        fileListContainer.setLayout(filelistLayout)
        self.filedock = QDockWidget(getStr('fileList'), self)
        self.filedock.setObjectName(getStr('files'))
        self.filedock.setWidget(fileListContainer)

        self.zoomWidget = ZoomWidget()
        self.colorDialog = ColorDialog(parent=self)

        self.canvas = Canvas(parent=self)
        self.canvas.zoomRequest.connect(self.zoomRequest)
        self.canvas.setDrawingShapeToSquare(settings.get(SETTING_DRAW_SQUARE, False))

        scroll = QScrollArea()
        scroll.setWidget(self.canvas)
        scroll.setWidgetResizable(True)
        self.scrollBars = {
            Qt.Vertical: scroll.verticalScrollBar(),
            Qt.Horizontal: scroll.horizontalScrollBar()
        }
        self.scrollArea = scroll
        self.canvas.scrollRequest.connect(self.scrollRequest)

        self.canvas.newShape.connect(self.newShape)
        self.canvas.shapeMoved.connect(self.setDirty)
        self.canvas.selectionChanged.connect(self.shapeSelectionChanged)
        self.canvas.drawingPolygon.connect(self.toggleDrawingSensitive)
        self.canvas.deleteSerials.connect(self.deleteSeries)

        self.setCentralWidget(scroll)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock)
        self.addDockWidget(Qt.RightDockWidgetArea, self.filedock)
        self.filedock.setFeatures(QDockWidget.DockWidgetFloatable)

        self.dockFeatures = QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetFloatable
        self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)

        # Actions
        action = partial(newAction, self)
        quit = action(getStr('quit'), self.close,
                      'Ctrl+Q', 'quit', getStr('quitApp'))

        global autoLabel
        autoLabel = action(getStr('autoLabel'), self.autoLabel,
                      's', 'autoLabel', getStr('autoLabelDetail'))

        open = action(getStr('openFile'), self.openFile,
                      'Ctrl+O', 'open', getStr('openFileDetail'))

        opendir = action(getStr('openDir'), self.openDirDialog,
                         'Ctrl+u', 'open', getStr('openDir'))

        copyPrevBounding = action(getStr('copyPrevBounding'), self.copyPreviousBoundingBoxes,
                         'Ctrl+v', 'paste', getStr('copyPrevBounding'))

        changeSavedir = action(getStr('changeSaveDir'), self.changeSavedirDialog,
                               'Ctrl+r', 'open', getStr('changeSavedAnnotationDir'))

        openAnnotation = action(getStr('openAnnotation'), self.openAnnotationDialog,
                                'Ctrl+Shift+O', 'open', getStr('openAnnotationDetail'))

        openNextImg = action(getStr('nextImg'), self.openNextImg,
                             'd', 'next', getStr('nextImgDetail'))

        openPrevImg = action(getStr('prevImg'), self.openPrevImg,
                             'a', 'prev', getStr('prevImgDetail'))

        verify = action(getStr('verifyImg'), self.verifyImg,
                        '', 'verify', getStr('verifyImgDetail'))

        save = action(getStr('save'), self.saveFile,
                      'Ctrl+S', 'save', getStr('saveDetail'), enabled=False)

        def getFormatMeta(format):
            """
            returns a tuple containing (title, icon_name) of the selected format
            """
            if format == LabelFileFormat.PASCAL_VOC:
                return ('&PascalVOC', 'format_voc')
            elif format == LabelFileFormat.YOLO:
                return ('&YOLO', 'format_yolo')
            elif format == LabelFileFormat.CREATE_ML:
                return ('&CreateML', 'format_createml')

        save_format = action(getFormatMeta(self.labelFileFormat)[0],
                             self.change_format, 'Ctrl+',
                             getFormatMeta(self.labelFileFormat)[1],
                             getStr('changeSaveFormat'), enabled=True)

        saveAs = action(getStr('saveAs'), self.saveFileAs,
                        'Ctrl+Shift+S', 'save-as', getStr('saveAsDetail'), enabled=False)

        close = action(getStr('closeCur'), self.closeFile, 'Ctrl+W', 'close', getStr('closeCurDetail'))

        deleteImg = action(getStr('deleteImg'), self.deleteImg, 'Ctrl+Shift+D', 'close', getStr('deleteImgDetail'))

        resetAll = action(getStr('resetAll'), self.resetAll, None, 'resetall', getStr('resetAllDetail'))

        color1 = action(getStr('boxLineColor'), self.chooseColor1,
                        'Ctrl+L', 'color_line', getStr('boxLineColorDetail'))

        createMode = action(getStr('crtBox'), self.setCreateMode,
                            'w', 'new', getStr('crtBoxDetail'), enabled=False)
        editMode = action('&Edit\nRectBox', self.setEditMode,
                          'Ctrl+J', 'edit', u'Move and edit Boxs', enabled=False)

        create = action(getStr('crtBox'), self.createShape,
                        'w', 'new', getStr('crtBoxDetail'), enabled=False)
        delete = action(getStr('delBox'), self.deleteSelectedShape,
                        'space', 'delete', getStr('delBoxDetail'), enabled=False)
        deleteSeries = action('Delete Rectbox by IoU', self.deleteSeries,
                        'Ctrl+Delete', 'deleteSeries', 'This func is configued through variable MAX_IOU_FOR_DELETE', enabled=False)
        addSeries = action('Add Rectboxs by Series', self.addSelectedShape,
                        'Shift+D', 'copy', 'This func is configued through variable ADD_RECTBOX_BY_SERIES_NUM', enabled=False)
        copy = action(getStr('dupBox'), self.copySelectedShape,
                      'Ctrl+D', 'copy', getStr('dupBoxDetail'),
                      enabled=False)

        advancedMode = action(getStr('advancedMode'), self.toggleAdvancedMode,
                              'Ctrl+Shift+A', 'expert', getStr('advancedModeDetail'),
                              checkable=True)

        hideAll = action('&Hide\nRectBox', partial(self.togglePolygons, False),
                         'Ctrl+H', 'hide', getStr('hideAllBoxDetail'),
                         enabled=False)
        showAll = action('&Show\nRectBox', partial(self.togglePolygons, True),
                         'Ctrl+A', 'hide', getStr('showAllBoxDetail'),
                         enabled=False)

        help = action(getStr('tutorial'), self.showTutorialDialog, None, 'help', getStr('tutorialDetail'))
        showInfo = action(getStr('info'), self.showInfoDialog, None, 'help', getStr('info'))

        zoom = QWidgetAction(self)
        zoom.setDefaultWidget(self.zoomWidget)
        self.zoomWidget.setWhatsThis(
            u"Zoom in or out of the image. Also accessible with"
            " %s and %s from the canvas." % (fmtShortcut("Ctrl+[-+]"),
                                             fmtShortcut("Ctrl+Wheel")))
        self.zoomWidget.setEnabled(False)

        zoomIn = action(getStr('zoomin'), partial(self.addZoom, 10),
                        'Ctrl++', 'zoom-in', getStr('zoominDetail'), enabled=False)
        zoomOut = action(getStr('zoomout'), partial(self.addZoom, -10),
                         'Ctrl+-', 'zoom-out', getStr('zoomoutDetail'), enabled=False)
        zoomOrg = action(getStr('originalsize'), partial(self.setZoom, 100),
                         'Ctrl+=', 'zoom', getStr('originalsizeDetail'), enabled=False)
        fitWindow = action(getStr('fitWin'), self.setFitWindow,
                           'Ctrl+F', 'fit-window', getStr('fitWinDetail'),
                           checkable=True, enabled=False)
        fitWidth = action(getStr('fitWidth'), self.setFitWidth,
                          'Ctrl+Shift+F', 'fit-width', getStr('fitWidthDetail'),
                          checkable=True, enabled=False)
        # Group zoom controls into a list for easier toggling.
        zoomActions = (self.zoomWidget, zoomIn, zoomOut,
                       zoomOrg, fitWindow, fitWidth)
        self.zoomMode = self.MANUAL_ZOOM
        self.scalers = {
            self.FIT_WINDOW: self.scaleFitWindow,
            self.FIT_WIDTH: self.scaleFitWidth,
            # Set to one to scale to 100% when loading files.
            self.MANUAL_ZOOM: lambda: 1,
        }

        edit = action(getStr('editLabel'), self.editLabel,
                      'Ctrl+E', 'edit', getStr('editLabelDetail'),
                      enabled=False)
        edit1 = action(getStr('editLabel'), self.editLabel1,
                      '1', 'edit', getStr('editLabelDetail'),
                      enabled=False)
        edit2 = action(getStr('editLabel'), self.editLabel2,
                       '2', 'edit', getStr('editLabelDetail'),
                       enabled=False)
        edit3 = action(getStr('editLabel'), self.editLabel3,
                       '3', 'edit', getStr('editLabelDetail'),
                       enabled=False)
        edit4 = action(getStr('editLabel'), self.editLabel4,
                       '4', 'edit', getStr('editLabelDetail'),
                       enabled=False)
        edit5 = action(getStr('editLabel'), self.editLabel5,
                       '5', 'edit', getStr('editLabelDetail'),
                       enabled=False)
        edit6 = action(getStr('editLabel'), self.editLabel6,
                       '6', 'edit', getStr('editLabelDetail'),
                       enabled=False)
        edit7 = action(getStr('editLabel'), self.editLabel7,
                       '7', 'edit', getStr('editLabelDetail'),
                       enabled=False)
        edit8 = action(getStr('editLabel'), self.editLabel8,
                       '8', 'edit', getStr('editLabelDetail'),
                       enabled=False)
        self.editButton.setDefaultAction(edit)
        self.editButton.setDefaultAction(edit1)
        self.editButton.setDefaultAction(edit2)
        self.editButton.setDefaultAction(edit3)
        self.editButton.setDefaultAction(edit4)
        self.editButton.setDefaultAction(edit5)
        self.editButton.setDefaultAction(edit6)
        self.editButton.setDefaultAction(edit7)
        self.editButton.setDefaultAction(edit8)

        shapeLineColor = action(getStr('shapeLineColor'), self.chshapeLineColor,
                                icon='color_line', tip=getStr('shapeLineColorDetail'),
                                enabled=False)
        shapeFillColor = action(getStr('shapeFillColor'), self.chshapeFillColor,
                                icon='color', tip=getStr('shapeFillColorDetail'),
                                enabled=False)

        labels = self.dock.toggleViewAction()
        labels.setText(getStr('showHide'))
        labels.setShortcut('Ctrl+Shift+L')

        # Label list context menu.
        labelMenu = QMenu()
        addActions(labelMenu, (edit, delete))
        self.labelList.setContextMenuPolicy(Qt.CustomContextMenu)
        self.labelList.customContextMenuRequested.connect(
            self.popLabelListMenu)

        # Draw squares/rectangles
        self.drawSquaresOption = QAction('Draw Squares', self)
        self.drawSquaresOption.setShortcut('Ctrl+Shift+R')
        self.drawSquaresOption.setCheckable(True)
        self.drawSquaresOption.setChecked(settings.get(SETTING_DRAW_SQUARE, False))
        self.drawSquaresOption.triggered.connect(self.toogleDrawSquare)

        # Store actions for further handling.
        self.actions = struct(save=save, save_format=save_format, saveAs=saveAs, open=open, close=close, resetAll = resetAll, deleteImg = deleteImg,
                              lineColor=color1, create=create, delete=delete, deleteSeries=deleteSeries, addSeries=addSeries, edit=edit,edit1=edit1,
                              edit2=edit2, edit3=edit3, edit4=edit4, edit5=edit5, edit6=edit6, edit7=edit7, edit8=edit8, copy=copy, createMode=createMode, editMode=editMode, advancedMode=advancedMode,
                              shapeLineColor=shapeLineColor, shapeFillColor=shapeFillColor,
                              zoom=zoom, zoomIn=zoomIn, zoomOut=zoomOut, zoomOrg=zoomOrg,
                              fitWindow=fitWindow, fitWidth=fitWidth,
                              zoomActions=zoomActions,
                              fileMenuActions=(
                                  open, opendir, save, saveAs, close, resetAll, quit),
                              beginner=(), advanced=(),
                              editMenu=(edit, copy, delete,
                                        None, color1, self.drawSquaresOption),
                              beginnerContext=(deleteSeries, addSeries, create, edit, copy, delete),
                              advancedContext=(createMode, editMode, edit, copy,
                                               delete, shapeLineColor, shapeFillColor),
                              onLoadActive=(
                                  close, create, createMode, editMode),
                              onShapesPresent=(saveAs, hideAll, showAll))

        self.menus = struct(
            file=self.menu('&File'),
            edit=self.menu('&Edit'),
            view=self.menu('&View'),
            help=self.menu('&Help'),
            models=self.menu('&Models'),
            recentFiles=QMenu('Open &Recent'),
            labelList=labelMenu)

        # Auto saving : Enable auto saving if pressing next
        self.autoSaving = QAction(getStr('autoSaveMode'), self)
        self.autoSaving.setCheckable(True)
        self.autoSaving.setChecked(settings.get(SETTING_AUTO_SAVE, False))
        # Sync single class mode from PR#106
        self.singleClassMode = QAction(getStr('singleClsMode'), self)
        self.singleClassMode.setShortcut("Ctrl+Shift+S")
        self.singleClassMode.setCheckable(True)
        self.singleClassMode.setChecked(settings.get(SETTING_SINGLE_CLASS, False))
        self.lastLabel = None
        # Add option to enable/disable labels being displayed at the top of bounding boxes
        self.displayLabelOption = QAction(getStr('displayLabel'), self)
        self.displayLabelOption.setShortcut("Ctrl+Shift+L")
        self.displayLabelOption.setCheckable(True)
        self.displayLabelOption.setChecked(settings.get(SETTING_PAINT_LABEL, False))
        self.displayLabelOption.triggered.connect(self.togglePaintLabelsOption)

        # Add option to enable/disable score being displayed at the top of bounding boxes
        self.displayScoreOption = QAction(getStr('displayScore'), self)
        self.displayScoreOption.setShortcut("Ctrl+Shift+P")
        self.displayScoreOption.setCheckable(True)
        self.displayScoreOption.setChecked(settings.get(SETTING_PAINT_SCORE, False))
        self.displayScoreOption.triggered.connect(self.togglePaintScoresOption)

        # Full auto label controller
        self.FullyAutoLabelOption = QAction("Fully Auto Label Mode", self)
        self.FullyAutoLabelOption.setCheckable(True)
        self.FullyAutoLabelOption.setChecked(False)
        self.FullyAutoLabelOption.triggered.connect(self.toggleFullyAutoLabel)


        # denoise
        self.denoiseOption = QAction("denoise by Gauss", self)
        self.denoiseOption.setCheckable(True)
        self.denoiseOption.setChecked(False)
        self.denoiseOption.triggered.connect(self.toggleDenoise)

        # Histogram Equalization
        self.EqualizeHistOption = QAction("Histogram Equalization", self)
        self.EqualizeHistOption.setCheckable(True)
        self.EqualizeHistOption.setChecked(False)
        self.EqualizeHistOption.triggered.connect(self.setEqualizeHistStatus)

        # Models
        # self.SSD = QAction('SSD', self)
        # self.SSD.setCheckable(True)
        # self.SSD.setChecked()

        self.SSD = None
        self.model0 = QAction("SSD", self)
        self.model0.setCheckable(True)
        self.model0.setChecked(False)
        self.model0.triggered.connect(self.changeStatusModel0)

        self.centerNet=None
        self.model1 = QAction("CenterNet", self)
        self.model1.setCheckable(True)
        self.model1.setChecked(True)
        self.model1.triggered.connect(self.changeStatusModel1)

        self.YOLOv5=None
        self.model2 = QAction("YOLOv5_coco", self)
        self.model2.setCheckable(True)
        self.model2.setChecked(False)
        self.model2.triggered.connect(self.changeStatusModel2)

        self.YOLOv5s=None
        self.model3 = QAction("YOLOv5s", self)
        self.model3.setCheckable(True)
        self.model3.setChecked(False)
        self.model3.triggered.connect(self.changeStatusModel3)

        self.YOLOv5_i18R=None
        self.model4 = QAction("YOLOv5_i18R", self)
        self.model4.setCheckable(True)
        self.model4.setChecked(False)
        self.model4.triggered.connect(self.changeStatusModel4)

        self.YOLOv3 = None
        self.model5 = QAction("YOLOv3", self)
        self.model5.setCheckable(True)
        self.model5.setChecked(False)
        self.model5.triggered.connect(self.changeStatusModel5)
        
        self.MobileNetV2 = None
        self.model6 = QAction("MobileNetV2", self)
        self.model6.setCheckable(True)
        self.model6.setChecked(False)
        self.model6.triggered.connect(self.changeStatusModel6)

        addActions(self.menus.models, (self.model0, self.model1, self.model2, self.model3, self.model4, self.model5, self.model6))

        addActions(self.menus.file,
                   (open, opendir, copyPrevBounding, changeSavedir, openAnnotation, self.menus.recentFiles, save, save_format, saveAs, close, resetAll, deleteImg, quit))
        addActions(self.menus.help, (help, showInfo))
        addActions(self.menus.view, (
            self.autoSaving,
            self.singleClassMode,
            self.displayLabelOption,
            self.displayScoreOption,
            self.FullyAutoLabelOption,
            self.EqualizeHistOption,
            self.denoiseOption,
            labels, advancedMode, None,
            hideAll, showAll, None,
            zoomIn, zoomOut, zoomOrg, None,
            fitWindow, fitWidth))

        self.menus.file.aboutToShow.connect(self.updateFileMenu)

        # Custom context menu for the canvas widget:
        addActions(self.canvas.menus[0], self.actions.beginnerContext)
        addActions(self.canvas.menus[1], (
            action('&Copy here', self.copyShape),
            action('&Move here', self.moveShape)))

        self.tools = self.toolbar('Tools')
        self.actions.beginner = (
            autoLabel, open, opendir, changeSavedir, openNextImg, openPrevImg, verify, save, save_format, None, create, copy, delete, None,
            zoomIn, zoom, zoomOut, fitWindow, fitWidth)

        self.actions.advanced = (
            autoLabel, open, opendir, changeSavedir, openNextImg, openPrevImg, save, save_format, None,
            createMode, editMode, None,
            hideAll, showAll)

        self.statusBar().showMessage('%s started.' % __appname__)
        self.statusBar().show()

        # Application state.
        self.image = QImage()
        self.filePath = ustr(defaultFilename)
        self.lastOpenDir= None
        self.recentFiles = []
        self.maxRecent = 7
        self.lineColor = None
        self.fillColor = None
        self.zoom_level = 100
        self.fit_window = False
        # Add Chris
        self.difficult = False

        ## Fix the compatible issue for qt4 and qt5. Convert the QStringList to python list
        if settings.get(SETTING_RECENT_FILES):
            if have_qstring():
                recentFileQStringList = settings.get(SETTING_RECENT_FILES)
                self.recentFiles = [ustr(i) for i in recentFileQStringList]
            else:
                self.recentFiles = recentFileQStringList = settings.get(SETTING_RECENT_FILES)

        size = settings.get(SETTING_WIN_SIZE, QSize(600, 500))
        position = QPoint(0, 0)
        saved_position = settings.get(SETTING_WIN_POSE, position)
        # Fix the multiple monitors issue
        for i in range(QApplication.desktop().screenCount()):
            if QApplication.desktop().availableGeometry(i).contains(saved_position):
                position = saved_position
                break
        self.resize(size)
        self.move(position)
        saveDir = ustr(settings.get(SETTING_SAVE_DIR, None))
        self.lastOpenDir = ustr(settings.get(SETTING_LAST_OPEN_DIR, None))
        if self.defaultSaveDir is None and saveDir is not None and os.path.exists(saveDir):
            self.defaultSaveDir = saveDir
            self.statusBar().showMessage('%s started. Annotation will be saved to %s' %
                                         (__appname__, self.defaultSaveDir))
            self.statusBar().show()

        self.restoreState(settings.get(SETTING_WIN_STATE, QByteArray()))
        Shape.line_color = self.lineColor = QColor(settings.get(SETTING_LINE_COLOR, DEFAULT_LINE_COLOR))
        Shape.fill_color = self.fillColor = QColor(settings.get(SETTING_FILL_COLOR, DEFAULT_FILL_COLOR))
        self.canvas.setDrawingColor(self.lineColor)
        # Add chris
        Shape.difficult = self.difficult

        def xbool(x):
            if isinstance(x, QVariant):
                return x.toBool()
            return bool(x)

        if xbool(settings.get(SETTING_ADVANCE_MODE, False)):
            self.actions.advancedMode.setChecked(True)
            self.toggleAdvancedMode()

        # Populate the File menu dynamically.
        self.updateFileMenu()

        # Since loading the file may take some time, make sure it runs in the background.
        if self.filePath and os.path.isdir(self.filePath):
            self.queueEvent(partial(self.importDirImages, self.filePath or ""))
        elif self.filePath:
            self.queueEvent(partial(self.loadFile, self.filePath or ""))

        # Callbacks:
        self.zoomWidget.valueChanged.connect(self.paintCanvas)

        self.populateModeActions()

        # Display cursor coordinates at the right of status bar
        self.labelCoordinates = QLabel('')
        self.statusBar().addPermanentWidget(self.labelCoordinates)

        # Open Dir if deafult file
        if self.filePath and os.path.isdir(self.filePath):
            self.openDirDialog(dirpath=self.filePath, silent=True)

        # Models to be used to inference are controlled in this dict
        self.theseModels = {0: False, 1: True, 2: False, 3: False, 4: False, 5: False, 6: False}    # by default, CenterNet is used for inference


    def _loadClassNames4Detect(self):
        if os.path.isfile(self.class_name_file_4_detect):
            self.class_names_4_detect = [name.strip('\n')for name in open(self.class_name_file_4_detect).readlines()]
        else:
            raise IOError("no such file {}".format(self.class_name_file_4_detect))

    def _initModel(self):
        if os.path.isfile(self.model_file_4_detect):
            # self.net = ONNXModel(self.model_file_4_detect)
            pass
        else:
            raise IOError("no such file {}".format(self.model_file_4_detect))

    def _initDetect(self):
        self._loadClassNames4Detect()
        self._initModel()

        return self

    def keyReleaseEvent(self, event):
        if event.key() == Qt.Key_Tab:
            self.canvas.keyPressEvent(event)
            return

        if event.key() == Qt.Key_Control:
            self.canvas.setDrawingShapeToSquare(False)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Control:
            # Draw rectangle if Ctrl is pressed
            self.canvas.setDrawingShapeToSquare(True)

    ## Support Functions ##
    def set_format(self, save_format):
        if save_format == FORMAT_PASCALVOC:
            self.actions.save_format.setText(FORMAT_PASCALVOC)
            self.actions.save_format.setIcon(newIcon("format_voc"))
            self.labelFileFormat = LabelFileFormat.PASCAL_VOC
            LabelFile.suffix = XML_EXT

        elif save_format == FORMAT_YOLO:
            self.actions.save_format.setText(FORMAT_YOLO)
            self.actions.save_format.setIcon(newIcon("format_yolo"))
            self.labelFileFormat = LabelFileFormat.YOLO
            LabelFile.suffix = TXT_EXT

        elif save_format == FORMAT_CREATEML:
            self.actions.save_format.setText(FORMAT_CREATEML)
            self.actions.save_format.setIcon(newIcon("format_createml"))
            self.labelFileFormat = LabelFileFormat.CREATE_ML
            LabelFile.suffix = JSON_EXT

    def change_format(self):
        if self.labelFileFormat == LabelFileFormat.PASCAL_VOC:
            self.set_format(FORMAT_YOLO)
        elif self.labelFileFormat == LabelFileFormat.YOLO:
            self.set_format(FORMAT_CREATEML)
        elif self.labelFileFormat == LabelFileFormat.CREATE_ML:
            self.set_format(FORMAT_PASCALVOC)
        else:
            raise ValueError('Unknown label file format.')
        self.setDirty()

    def noShapes(self):
        return not self.itemsToShapes

    def toggleAdvancedMode(self, value=True):
        self._beginner = not value
        self.canvas.setEditing(True)
        self.populateModeActions()
        self.editButton.setVisible(not value)
        if value:
            self.actions.createMode.setEnabled(True)
            self.actions.editMode.setEnabled(False)
            self.dock.setFeatures(self.dock.features() | self.dockFeatures)
        else:
            self.dock.setFeatures(self.dock.features() ^ self.dockFeatures)

    def populateModeActions(self):
        if self.beginner():
            tool, menu = self.actions.beginner, self.actions.beginnerContext
        else:
            tool, menu = self.actions.advanced, self.actions.advancedContext
        self.tools.clear()
        addActions(self.tools, tool)
        self.canvas.menus[0].clear()
        addActions(self.canvas.menus[0], menu)
        self.menus.edit.clear()
        actions = (self.actions.create,) if self.beginner()\
            else (self.actions.createMode, self.actions.editMode)
        addActions(self.menus.edit, actions + self.actions.editMenu)

    def setBeginner(self):
        self.tools.clear()
        addActions(self.tools, self.actions.beginner)

    def setAdvanced(self):
        self.tools.clear()
        addActions(self.tools, self.actions.advanced)

    def setDirty(self):
        self.dirty = True
        self.actions.save.setEnabled(True)

    def setClean(self):
        self.dirty = False
        self.actions.save.setEnabled(False)
        self.actions.create.setEnabled(True)

    def toggleActions(self, value=True):
        """Enable/Disable widgets which depend on an opened image."""
        for z in self.actions.zoomActions:
            z.setEnabled(value)
        for action in self.actions.onLoadActive:
            action.setEnabled(value)

    def queueEvent(self, function):
        QTimer.singleShot(0, function)

    def status(self, message, delay=5000):
        self.statusBar().showMessage(message, delay)

    def resetState(self):
        self.itemsToShapes.clear()
        self.shapesToItems.clear()
        self.labelList.clear()
        self.filePath = None
        self.imageData = None
        self.labelFile = None
        self.canvas.resetState()
        self.labelCoordinates.clear()
        # self.comboBox.cb.clear()

    def currentItem(self):
        items = self.labelList.selectedItems()
        if items:
            return items[0]
        return None

    def addRecentFile(self, filePath):
        if filePath in self.recentFiles:
            self.recentFiles.remove(filePath)
        elif len(self.recentFiles) >= self.maxRecent:
            self.recentFiles.pop()
        self.recentFiles.insert(0, filePath)

    def beginner(self):
        return self._beginner

    def advanced(self):
        return not self.beginner()

    def getAvailableScreencastViewer(self):
        osName = platform.system()

        if osName == 'Windows':
            return ['C:\\Program Files\\Internet Explorer\\iexplore.exe']
        elif osName == 'Linux':
            return ['xdg-open']
        elif osName == 'Darwin':
            return ['open']

    ## Callbacks ##
    def showTutorialDialog(self):
        subprocess.Popen(self.screencastViewer + [self.screencast])

    def changeStatusModel0(self):
        # SSD
        global onnxModelIndex
        onnxModelIndex = 0
        if self.model0.isChecked():
            self.theseModels[onnxModelIndex] = True
        else:
            self.theseModels[onnxModelIndex] = False

    def changeStatusModel1(self):
        # CenterNet
        global onnxModelIndex
        onnxModelIndex = 1
        if self.model1.isChecked():
            self.theseModels[onnxModelIndex] = True
        else:
            self.theseModels[onnxModelIndex] = False

    def changeStatusModel2(self):
        # YOLOv5
        global onnxModelIndex
        onnxModelIndex = 2
        if self.model2.isChecked():
            self.theseModels[onnxModelIndex] = True
        else:
            self.theseModels[onnxModelIndex] = False

    def changeStatusModel3(self):
        # YOLOv5s
        global onnxModelIndex
        onnxModelIndex = 3
        if self.model3.isChecked():
            self.theseModels[onnxModelIndex] = True
        else:
            self.theseModels[onnxModelIndex] = False

    def changeStatusModel4(self):
        # YOLOv5_i18R
        global onnxModelIndex
        onnxModelIndex = 4
        if self.model4.isChecked():
            self.theseModels[onnxModelIndex] = True
        else:
            self.theseModels[onnxModelIndex] = False

    def changeStatusModel5(self):
        # YOLOv3
        global onnxModelIndex
        onnxModelIndex = 5
        if self.model5.isChecked():
            self.theseModels[onnxModelIndex] = True
        else:
            self.theseModels[onnxModelIndex] = False
            
    def changeStatusModel6(self):
        # YOLOv3
        global onnxModelIndex
        onnxModelIndex = 6
        if self.model6.isChecked():
            self.theseModels[onnxModelIndex] = True
        else:
            self.theseModels[onnxModelIndex] = False

    def showInfoDialog(self):
        from libs.__init__ import __version__
        msg = u'Name:{0} \nApp Version:{1} \n{2} '.format(__appname__, __version__, sys.version_info)
        QMessageBox.information(self, u'Information', msg)

    def createShape(self):
        assert self.beginner()
        self.canvas.setEditing(False)
        self.actions.create.setEnabled(False)

    def toggleDrawingSensitive(self, drawing=True):
        """In the middle of drawing, toggling between modes should be disabled."""
        self.actions.editMode.setEnabled(not drawing)
        if not drawing and self.beginner():
            # Cancel creation.
            print('Cancel creation.')
            self.canvas.setEditing(True)
            self.canvas.restoreCursor()
            self.actions.create.setEnabled(True)

    def toggleDrawMode(self, edit=True):
        self.canvas.setEditing(edit)
        self.actions.createMode.setEnabled(edit)
        self.actions.editMode.setEnabled(not edit)

    def setCreateMode(self):
        assert self.advanced()
        self.toggleDrawMode(False)

    def setEditMode(self):
        assert self.advanced()
        self.toggleDrawMode(True)
        self.labelSelectionChanged()

    def updateFileMenu(self):
        currFilePath = self.filePath

        def exists(filename):
            return os.path.exists(filename)
        menu = self.menus.recentFiles
        menu.clear()
        files = [f for f in self.recentFiles if f !=
                 currFilePath and exists(f)]
        for i, f in enumerate(files):
            icon = newIcon('labels')
            action = QAction(
                icon, '&%d %s' % (i + 1, QFileInfo(f).fileName()), self)
            action.triggered.connect(partial(self.loadRecent, f))
            menu.addAction(action)

    def popLabelListMenu(self, point):
        self.menus.labelList.exec_(self.labelList.mapToGlobal(point))

    def editLabel(self):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:
            return
        text = self.labelDialog.popUp(item.text())
        if text is not None:
            item.setText(text)
            item.setBackground(generateColorByText(text))
            self.setDirty()
            self.updateComboBox()

    def editLabel1(self):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:
            return
        if len(self.labelHist) >= 1:
            text = self.labelHist[0]
        if text is not None:
            item.setText(text)
            item.setBackground(generateColorByText(text))
            self.setDirty()
            self.updateComboBox()

    def editLabel2(self):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:
            return
        if len(self.labelHist) >= 1:
            text = self.labelHist[1]
        if text is not None:
            item.setText(text)
            item.setBackground(generateColorByText(text))
            self.setDirty()
            self.updateComboBox()

    def editLabel3(self):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:
            return
        if len(self.labelHist) >= 1:
            text = self.labelHist[2]
        if text is not None:
            item.setText(text)
            item.setBackground(generateColorByText(text))
            self.setDirty()
            self.updateComboBox()

    def editLabel4(self):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:
            return
        if len(self.labelHist) >= 1:
            text = self.labelHist[3]
        if text is not None:
            item.setText(text)
            item.setBackground(generateColorByText(text))
            self.setDirty()
            self.updateComboBox()

    def editLabel5(self):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:
            return
        if len(self.labelHist) >= 1:
            text = self.labelHist[4]
        if text is not None:
            item.setText(text)
            item.setBackground(generateColorByText(text))
            self.setDirty()
            self.updateComboBox()

    def editLabel6(self):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:
            return
        if len(self.labelHist) >= 1:
            text = self.labelHist[5]
        if text is not None:
            item.setText(text)
            item.setBackground(generateColorByText(text))
            self.setDirty()
            self.updateComboBox()

    def editLabel7(self):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:
            return
        if len(self.labelHist) >= 1:
            text = self.labelHist[6]
        if text is not None:
            item.setText(text)
            item.setBackground(generateColorByText(text))
            self.setDirty()
            self.updateComboBox()

    def editLabel8(self):
        if not self.canvas.editing():
            return
        item = self.currentItem()
        if not item:
            return
        if len(self.labelHist) >= 1:
            text = self.labelHist[7]
        if text is not None:
            item.setText(text)
            item.setBackground(generateColorByText(text))
            self.setDirty()
            self.updateComboBox()

    # Tzutalin 20160906 : Add file list and dock to move faster
    def fileitemDoubleClicked(self, item=None):
        currIndex = self.mImgList.index(ustr(item.text()))
        if currIndex < len(self.mImgList):
            filename = self.mImgList[currIndex]
            if filename:
                self.loadFile(filename)

    def distanceChange(self):
        distance = self.textDistance.text()
        if not distance.isnumeric():
            return

        if not self.canvas.editing():
            return

        item = self.currentItem()
        if not item: # If not selected Item, take the first one
            item = self.labelList.item(self.labelList.count()-1)

        try:
            shape = self.itemsToShapes[item]
        except:
            pass
        # Checked and Update
        try:
            if distance != shape.distance:
                shape.distance = distance
                self.setDirty()
            else:  # User probably changed item visibility
                self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)
        except:
            pass

    # Add chris
    def btnstate(self, item= None):
        """ Function to handle difficult examples
        Update on each object """
        if not self.canvas.editing():
            return

        item = self.currentItem()
        if not item: # If not selected Item, take the first one
            item = self.labelList.item(self.labelList.count()-1)

        difficult = self.diffcButton.isChecked()

        try:
            shape = self.itemsToShapes[item]
        except:
            pass
        # Checked and Update
        try:
            if difficult != shape.difficult:
                shape.difficult = difficult
                self.setDirty()
            else:  # User probably changed item visibility
                self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)
        except:
            pass

    # React to canvas signals.
    def shapeSelectionChanged(self, selected=False):
        if self._noSelectionSlot:
            self._noSelectionSlot = False
        else:
            shape = self.canvas.selectedShape
            if shape:
                self.shapesToItems[shape].setSelected(True)
            else:
                self.labelList.clearSelection()
        self.actions.delete.setEnabled(selected)
        self.actions.deleteSeries.setEnabled(selected)
        self.actions.addSeries.setEnabled(selected)
        self.actions.copy.setEnabled(selected)
        self.actions.edit.setEnabled(selected)
        self.actions.edit1.setEnabled(selected)
        self.actions.edit2.setEnabled(selected)
        self.actions.edit3.setEnabled(selected)
        self.actions.edit4.setEnabled(selected)
        self.actions.edit5.setEnabled(selected)
        self.actions.edit6.setEnabled(selected)
        self.actions.edit7.setEnabled(selected)
        self.actions.edit8.setEnabled(selected)
        self.actions.shapeLineColor.setEnabled(selected)
        self.actions.shapeFillColor.setEnabled(selected)

    def addLabel(self, shape):
        shape.paintLabel = self.displayLabelOption.isChecked()
        shape.paintScore = self.displayScoreOption.isChecked()
        item = HashableQListWidgetItem(shape.label)
        item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
        item.setCheckState(Qt.Checked)
        item.setBackground(generateColorByText(shape.label))
        self.itemsToShapes[item] = shape
        self.shapesToItems[shape] = item
        self.labelList.addItem(item)
        for action in self.actions.onShapesPresent:
            action.setEnabled(True)
        self.updateComboBox()

    def remLabel(self, shape):
        if shape is None:
            # print('rm empty label')
            return
        item = self.shapesToItems[shape]
        self.labelList.takeItem(self.labelList.row(item))
        del self.shapesToItems[shape]
        del self.itemsToShapes[item]
        self.updateComboBox()

    def loadLabels(self, shapes):
        s = []
        for label, points, line_color, fill_color, difficult, distance, score in shapes:
            for i, (x, y) in enumerate(points):
                # Ensure the labels are within the bounds of the image. If not, fix them.
                x, y, snapped = self.canvas.snapPointToCanvas(x, y)
                points[i] = (x, y)
                if snapped:
                    self.setDirty()

            shape = box2Shape(label, points, line_color, fill_color, difficult, distance, score)
            s.append(shape)
            self.addLabel(shape)

        self.updateComboBox()

        if len(self.canvas.shapes) > 0:
            s.extend(self.canvas.shapes)
        self.canvas.loadShapes(s)

    def updateComboBox(self):
        # Get the unique labels and add them to the Combobox.
        combobox_add=[]
        itemsTextList = [str(self.labelList.item(i).text()) for i in range(self.labelList.count())]
        combobox_list=[str(self.comboBox.cb.itemText(i)) for i in range(self.comboBox.cb.count())]

        uniqueTextList = list(set(itemsTextList))
        # Add a null row for showing all the labels
        uniqueTextList.append("")
        uniqueTextList.sort()

        for i in uniqueTextList:
            if not (i in combobox_list):
                combobox_add.append(i)

        self.comboBox.update_items(combobox_add)

    def saveLabels(self, annotationFilePath, shapes:map):
        annotationFilePath = ustr(annotationFilePath)
        if self.labelFile is None:
            self.labelFile = LabelFile()
            self.labelFile.verified = self.canvas.verified

        # Can add differrent annotation formats here
        try:
            if self.labelFileFormat == LabelFileFormat.PASCAL_VOC:
                if annotationFilePath[-4:].lower() != ".xml":
                    annotationFilePath += XML_EXT
                self.labelFile.savePascalVocFormat(annotationFilePath, shapes, self.filePath, self.imageData,
                                                   self.lineColor.getRgb(), self.fillColor.getRgb())
            elif self.labelFileFormat == LabelFileFormat.YOLO:
                if annotationFilePath[-4:].lower() != ".txt":
                    annotationFilePath += TXT_EXT
                self.labelFile.saveYoloFormat(annotationFilePath, shapes, self.filePath, self.imageData, self.labelHist,
                                              self.lineColor.getRgb(), self.fillColor.getRgb())
            elif self.labelFileFormat == LabelFileFormat.CREATE_ML:
                if annotationFilePath[-5:].lower() != ".json":
                    annotationFilePath += JSON_EXT
                self.labelFile.saveCreateMLFormat(annotationFilePath, shapes, self.filePath, self.imageData,
                                                  self.labelHist, self.lineColor.getRgb(), self.fillColor.getRgb())
            else:
                self.labelFile.save(annotationFilePath, shapes, self.filePath, self.imageData,
                                    self.lineColor.getRgb(), self.fillColor.getRgb())
            # print('Image:{0} -> Annotation:{1}'.format(self.filePath, annotationFilePath))
            return True
        except LabelFileError as e:
            self.errorMessage(u'Error saving label data', u'<b>%s</b>' % e)
            return False

    def copySelectedShape(self):
        self.addLabel(self.canvas.copySelectedShape())
        # fix copy and delete
        self.shapeSelectionChanged(True)

    def addSelectedShape(self):
        self.tmpShape = self.canvas.selectedShape.copy()
        self.autoAdd = 1
        currIndex = self.mImgList.index(self.filePath)
        rest = len(self.mImgList) - currIndex - 1
        for i in range(min(ADD_RECTBOX_BY_SERIES_NUM, rest)):
            self.openNextImg()
        self.autoAdd = 0

    def addOneShape(self):
        if self.shapeNMS(self.canvas.shapes, self.tmpShape):
            self.addLabel(self.canvas.copyOneShape(self.tmpShape))
            self.setDirty()

    def shapeNMS(self, shape_list, shapeToBeSupressed):
        '''

        return: true for keep, and false for drop
        '''
        keep = True
        xs1, ys1, xs2, ys2 = shapeToBeSupressed.points[0].x(), shapeToBeSupressed.points[0].y(), \
                         shapeToBeSupressed.points[2].x(), shapeToBeSupressed.points[2].y()
        areaS = max(0, (xs2 - xs1) * (ys2 - ys1))
        for i in shape_list:
            xo1, yo1, xo2, yo2 = i.points[0].x(), i.points[0].y(), i.points[2].x(), i.points[2].y()

            xx1 = max(xs1, xo1)
            yy1 = max(ys1, yo1)
            xx2 = min(xs2, xo2)
            yy2 = min(ys2, yo2)
            w = max(0, xx2 - xx1)
            h = max(0, yy2 - yy1)
            intersection = w * h

            areaO = max(0, (xo2 - xo1) * (yo2 - yo1))
            if (areaO + areaS - intersection) > 0 :
                iou = intersection / (areaO + areaS - intersection)

            if iou > IOU_NMS:
                keep = False

        return keep

    def comboSelectionChanged(self, index):
        text = self.comboBox.cb.itemText(index)
        for i in range(self.labelList.count()):
            if text == "":
                self.labelList.item(i).setCheckState(2)
            elif text != self.labelList.item(i).text():
                self.labelList.item(i).setCheckState(0)
            else:
                self.labelList.item(i).setCheckState(2)

    def labelSelectionChanged(self):
        item = self.currentItem()
        if item and self.canvas.editing():
            self._noSelectionSlot = True
            self.canvas.selectShape(self.itemsToShapes[item])
            shape = self.itemsToShapes[item]
            # Add Chris
            self.diffcButton.setChecked(shape.difficult)
            self.textDistance.setText(str(shape.distance))

    def labelItemChanged(self, item):
        shape = self.itemsToShapes[item]
        label = item.text()
        if label != shape.label:
            shape.label = item.text()
            shape.line_color = generateColorByText(shape.label)
            self.setDirty()
        else:  # User probably changed item visibility
            self.canvas.setShapeVisible(shape, item.checkState() == Qt.Checked)

    # Callback functions:
    def newShape(self):
        """Pop-up and give focus to the label editor.

        position MUST be in global coordinates.
        """
        # TODO : get label
        if not self.useDefaultLabelCheckbox.isChecked() or not self.defaultLabelTextLine.text():
            if len(self.labelHist) > 0:
                self.labelDialog = LabelDialog(
                    parent=self, listItem=self.labelHist)

            # Sync single class mode from PR#106
            if self.singleClassMode.isChecked() and self.lastLabel:
                text = self.lastLabel
            else:
                text = self.labelDialog.popUp(text=self.prevLabelText)
                self.lastLabel = text
        else:
            text = self.defaultLabelTextLine.text()

        # Add Chris
        self.diffcButton.setChecked(False)
        self.textDistance.setText("")
        if text is not None:
            self.prevLabelText = text
            generate_color = generateColorByText(text)
            shape = self.canvas.setLastLabel(text, generate_color, generate_color)
            self.addLabel(shape)
            if self.beginner():  # Switch to edit mode.
                self.canvas.setEditing(True)
                self.actions.create.setEnabled(True)
            else:
                self.actions.editMode.setEnabled(True)
            self.setDirty()

            if text not in self.labelHist:
                self.labelHist.append(text)
        else:
            # self.canvas.undoLastLine()
            self.canvas.resetAllLines()

    def scrollRequest(self, delta, orientation):
        units = - delta / (8 * 15)
        bar = self.scrollBars[orientation]
        bar.setValue(bar.value() + bar.singleStep() * units)

    def setZoom(self, value):
        self.actions.fitWidth.setChecked(False)
        self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.MANUAL_ZOOM
        self.zoomWidget.setValue(value)

    def addZoom(self, increment=10):
        self.setZoom(self.zoomWidget.value() + increment)

    def zoomRequest(self, delta):
        # get the current scrollbar positions
        # calculate the percentages ~ coordinates
        h_bar = self.scrollBars[Qt.Horizontal]
        v_bar = self.scrollBars[Qt.Vertical]

        # get the current maximum, to know the difference after zooming
        h_bar_max = h_bar.maximum()
        v_bar_max = v_bar.maximum()

        # get the cursor position and canvas size
        # calculate the desired movement from 0 to 1
        # where 0 = move left
        #       1 = move right
        # up and down analogous
        cursor = QCursor()
        pos = cursor.pos()
        relative_pos = QWidget.mapFromGlobal(self, pos)

        cursor_x = relative_pos.x()
        cursor_y = relative_pos.y()

        w = self.scrollArea.width()
        h = self.scrollArea.height()

        # the scaling from 0 to 1 has some padding
        # you don't have to hit the very leftmost pixel for a maximum-left movement
        margin = 0.1
        move_x = (cursor_x - margin * w) / (w - 2 * margin * w)
        move_y = (cursor_y - margin * h) / (h - 2 * margin * h)

        # clamp the values from 0 to 1
        move_x = min(max(move_x, 0), 1)
        move_y = min(max(move_y, 0), 1)

        # zoom in
        units = delta / (8 * 15)
        scale = 10
        self.addZoom(scale * units)

        # get the difference in scrollbar values
        # this is how far we can move
        d_h_bar_max = h_bar.maximum() - h_bar_max
        d_v_bar_max = v_bar.maximum() - v_bar_max

        # get the new scrollbar values
        new_h_bar_value = h_bar.value() + move_x * d_h_bar_max
        new_v_bar_value = v_bar.value() + move_y * d_v_bar_max

        h_bar.setValue(new_h_bar_value)
        v_bar.setValue(new_v_bar_value)

    def setFitWindow(self, value=True):
        if value:
            self.actions.fitWidth.setChecked(False)
        self.zoomMode = self.FIT_WINDOW if value else self.MANUAL_ZOOM
        self.adjustScale()

    def setFitWidth(self, value=True):
        if value:
            self.actions.fitWindow.setChecked(False)
        self.zoomMode = self.FIT_WIDTH if value else self.MANUAL_ZOOM
        self.adjustScale()

    def togglePolygons(self, value):
        for item, shape in self.itemsToShapes.items():
            item.setCheckState(Qt.Checked if value else Qt.Unchecked)

    def loadFile(self, filePath=None):
        """Load the specified file, or the last opened file if None."""
        self.resetState()
        self.canvas.setEnabled(False)
        if filePath is None:
            filePath = self.settings.get(SETTING_FILENAME)

        # Make sure that filePath is a regular python string, rather than QString
        filePath = ustr(filePath)

        # Fix bug: An  index error after select a directory when open a new file.
        unicodeFilePath = ustr(filePath)
        unicodeFilePath = os.path.abspath(unicodeFilePath)
        # Tzutalin 20160906 : Add file list and dock to move faster
        # Highlight the file item
        if unicodeFilePath and self.fileListWidget.count() > 0:
            if unicodeFilePath in self.mImgList:
                index = self.mImgList.index(unicodeFilePath)
                fileWidgetItem = self.fileListWidget.item(index)
                fileWidgetItem.setSelected(True)
            else:
                self.fileListWidget.clear()
                self.mImgList.clear()

        if unicodeFilePath and os.path.exists(unicodeFilePath):
            if LabelFile.isLabelFile(unicodeFilePath):
                try:
                    self.labelFile = LabelFile(unicodeFilePath)
                except LabelFileError as e:
                    self.errorMessage(u'Error opening file',
                                      (u"<p><b>%s</b></p>"
                                       u"<p>Make sure <i>%s</i> is a valid label file.")
                                      % (e, unicodeFilePath))
                    self.status("Error reading %s" % unicodeFilePath)
                    return False
                self.imageData = self.labelFile.imageData
                self.lineColor = QColor(*self.labelFile.lineColor)
                self.fillColor = QColor(*self.labelFile.fillColor)
                self.canvas.verified = self.labelFile.verified
            else:
                # Load image:
                # read data first and store for saving into label file.
                self.imageData = read(unicodeFilePath, None)
                self.labelFile = None
                self.canvas.verified = False

            if isinstance(self.imageData, QImage):
                image = self.imageData
            else:
                image = QImage.fromData(self.imageData)
            if image.isNull():
                self.errorMessage(u'Error opening file',
                                  u"<p>Make sure <i>%s</i> is a valid image file." % unicodeFilePath)
                self.status("Error reading %s" % unicodeFilePath)
                return False
            self.status("Loaded %s" % os.path.basename(unicodeFilePath))
            if self.EqualizeHist:
                image=self.setEqualizeHistNew(image)

            if self.denoise:
                image = self.Denoise(image)

            self.image = image
            self.filePath = unicodeFilePath
            self.canvas.loadPixmap(QPixmap.fromImage(image))
            if self.labelFile:
                self.loadLabels(self.labelFile.shapes)
            self.setClean()
            self.canvas.setEnabled(True)
            self.adjustScale(initial=True)
            self.paintCanvas()
            self.addRecentFile(self.filePath)
            self.toggleActions(True)

            self.showBoundingBoxFromAnnotationFile(filePath)

            self.setWindowTitle(__appname__ + ' ' + filePath)

            # Default : select last item if there is at least one item
            if self.labelList.count():
               self.labelList.setCurrentItem(self.labelList.item(self.labelList.count()-1))
               self.labelList.item(self.labelList.count()-1).setSelected(True)

            self.canvas.setFocus(True)
            return True
        return False

    def showBoundingBoxFromAnnotationFile(self, filePath):
        if self.defaultSaveDir is not None:

            subfilepath = self.filePath[self.dirname_len+1:]
            basename = os.path.splitext(subfilepath)[0]

            xmlPath = os.path.join(self.defaultSaveDir, basename + XML_EXT)
            txtPath = os.path.join(self.defaultSaveDir, basename + TXT_EXT)

            # TODO : for json file output
            basename = os.path.basename(os.path.splitext(filePath)[0])
            filedir = filePath.split(basename)[0].split(os.path.sep)[-2:-1][0]
            jsonPath = os.path.join(self.defaultSaveDir, filedir + JSON_EXT)

            """Annotation file priority:
            PascalXML > YOLO
            """
            if os.path.isfile(xmlPath):
                self.loadPascalXMLByFilename(xmlPath)
            elif os.path.isfile(txtPath):
                self.loadYOLOTXTByFilename(txtPath)
            elif os.path.isfile(jsonPath):
                self.loadCreateMLJSONByFilename(jsonPath, filePath)

        else:
            xmlPath = os.path.splitext(filePath)[0] + XML_EXT
            txtPath = os.path.splitext(filePath)[0] + TXT_EXT
            if os.path.isfile(xmlPath):
                self.loadPascalXMLByFilename(xmlPath)
            elif os.path.isfile(txtPath):
                self.loadYOLOTXTByFilename(txtPath)

    def resizeEvent(self, event):
        if self.canvas and not self.image.isNull()\
           and self.zoomMode != self.MANUAL_ZOOM:
            self.adjustScale()
        super(MainWindow, self).resizeEvent(event)

    def paintCanvas(self):
        assert not self.image.isNull(), "cannot paint null image"
        self.canvas.scale = 0.01 * self.zoomWidget.value()
        self.canvas.labelFontSize = int(0.02 * max(self.image.width(), self.image.height()))
        self.canvas.adjustSize()
        self.canvas.update()

    def adjustScale(self, initial=False):
        value = self.scalers[self.FIT_WINDOW if initial else self.zoomMode]()
        self.zoomWidget.setValue(int(100 * value))

    def scaleFitWindow(self):
        """Figure out the size of the pixmap in order to fit the main widget."""
        e = 2.0  # So that no scrollbars are generated.
        w1 = self.centralWidget().width() - e
        h1 = self.centralWidget().height() - e
        a1 = w1 / h1
        # Calculate a new scale value based on the pixmap's aspect ratio.
        w2 = self.canvas.pixmap.width() - 0.0
        h2 = self.canvas.pixmap.height() - 0.0
        a2 = w2 / h2
        return w1 / w2 if a2 >= a1 else h1 / h2

    def scaleFitWidth(self):
        # The epsilon does not seem to work too well here.
        w = self.centralWidget().width() - 2.0
        return w / self.canvas.pixmap.width()

    def closeEvent(self, event):
        if not self.mayContinue():
            event.ignore()
        settings = self.settings
        # If it loads images from dir, don't load it at the begining
        if self.dirname is None:
            settings[SETTING_FILENAME] = self.filePath if self.filePath else ''
        else:
            settings[SETTING_FILENAME] = ''

        settings[SETTING_WIN_SIZE] = self.size()
        settings[SETTING_WIN_POSE] = self.pos()
        settings[SETTING_WIN_STATE] = self.saveState()
        settings[SETTING_LINE_COLOR] = self.lineColor
        settings[SETTING_FILL_COLOR] = self.fillColor
        settings[SETTING_RECENT_FILES] = self.recentFiles
        settings[SETTING_ADVANCE_MODE] = not self._beginner
        if self.defaultSaveDir and os.path.exists(self.defaultSaveDir):
            settings[SETTING_SAVE_DIR] = ustr(self.defaultSaveDir)
        else:
            settings[SETTING_SAVE_DIR] = ''

        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            settings[SETTING_LAST_OPEN_DIR] = self.lastOpenDir
        else:
            settings[SETTING_LAST_OPEN_DIR] = ''

        settings[SETTING_AUTO_SAVE] = self.autoSaving.isChecked()
        settings[SETTING_SINGLE_CLASS] = self.singleClassMode.isChecked()
        settings[SETTING_PAINT_LABEL] = self.displayLabelOption.isChecked()
        settings[SETTING_DRAW_SQUARE] = self.drawSquaresOption.isChecked()
        settings[SETTING_LABEL_FILE_FORMAT] = self.labelFileFormat
        settings.save()

    def loadRecent(self, filename):
        if self.mayContinue():
            self.loadFile(filename)

    def scanAllImages(self, folderPath):
        extensions = ['.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        images = []

        for root, dirs, files in os.walk(folderPath):
            for file in files:
                if file.lower().endswith(tuple(extensions)):
                    relativePath = os.path.join(root, file)
                    path = ustr(os.path.abspath(relativePath))
                    images.append(path)
        natural_sort(images, key=lambda x: x.lower())
        return images

    def changeSavedirDialog(self, _value=False):
        if self.defaultSaveDir is not None:
            path = ustr(self.defaultSaveDir)
        else:
            path = '.'

        dirpath = ustr(QFileDialog.getExistingDirectory(self,
                                                       '%s - Save annotations to the directory' % __appname__, path,  QFileDialog.ShowDirsOnly
                                                       | QFileDialog.DontResolveSymlinks))

        if dirpath is not None and len(dirpath) > 1:
            self.defaultSaveDir = dirpath

        self.statusBar().showMessage('%s . Annotation will be saved to %s' %
                                     ('Change saved folder', self.defaultSaveDir))
        self.statusBar().show()

    def openAnnotationDialog(self, _value=False):
        if self.filePath is None:
            self.statusBar().showMessage('Please select image first')
            self.statusBar().show()
            return

        path = os.path.dirname(ustr(self.filePath))\
            if self.filePath else '.'
        if self.labelFileFormat == LabelFileFormat.PASCAL_VOC:
            filters = "Open Annotation XML file (%s)" % ' '.join(['*.xml'])
            filename = ustr(QFileDialog.getOpenFileName(self,'%s - Choose a xml file' % __appname__, path, filters))
            if filename:
                if isinstance(filename, (tuple, list)):
                    filename = filename[0]
            self.loadPascalXMLByFilename(filename)

    def openDirDialog(self, _value=False, dirpath=None, silent=False):
        if not self.mayContinue():
            return

        defaultOpenDirPath = dirpath if dirpath else '.'
        if self.lastOpenDir and os.path.exists(self.lastOpenDir):
            defaultOpenDirPath = self.lastOpenDir
        else:
            defaultOpenDirPath = os.path.dirname(self.filePath) if self.filePath else '.'
        if silent!=True :
            targetDirPath = ustr(QFileDialog.getExistingDirectory(self,
                                                         '%s - Open Directory' % __appname__, defaultOpenDirPath,
                                                         QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks))
        else:
            targetDirPath = ustr(defaultOpenDirPath)
        self.lastOpenDir = targetDirPath
        self.importDirImages(targetDirPath)

    def importDirImages(self, dirpath):
        if not self.mayContinue() or not dirpath:
            return

        self.lastOpenDir = dirpath
        self.dirname = dirpath
        self.dirname_len = len(self.dirname)
        self.filePath = None
        self.fileListWidget.clear()

        image_list_file = os.path.join(dirpath, IMAGE_LIST_FILE)
        if (os.path.exists(image_list_file)):
            self.mImgList = [f.strip().strip('\n') for f in open(image_list_file).readlines()]
        else:
            self.mImgList = self.scanAllImages(dirpath)
        self.openNextImg()
        for imgPath in self.mImgList:
            item = QListWidgetItem(imgPath)
            self.fileListWidget.addItem(item)

        print("image num: ", len(self.mImgList))

        return self

    def verifyImg(self, _value=False):
        # Proceding next image without dialog if having any label
        if self.filePath is not None:
            try:
                self.labelFile.toggleVerify()
            except AttributeError:
                # If the labelling file does not exist yet, create if and
                # re-save it with the verified attribute.
                self.saveFile()
                if self.labelFile != None:
                    self.labelFile.toggleVerify()
                else:
                    return

            self.canvas.verified = self.labelFile.verified
            self.paintCanvas()
            self.saveFile()

    def openPrevImg(self, _value=False):
        # Proceding prev image without dialog if having any label
        if self.autoSaving.isChecked():
            if self.defaultSaveDir is not None:
                if self.dirty is True:
                    self.saveFile()
            else:
                self.changeSavedirDialog()
                return

        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        if self.filePath is None:
            return

        currIndex = self.mImgList.index(self.filePath)
        if currIndex - 1 >= 0:
            filename = self.mImgList[currIndex - 1]
            if filename:
                self.loadFile(filename)

    def openNextImg(self, _value=False):
        # Proceding prev image without dialog if having any label
        self.textDistance.setText("")
        if self.autoSaving.isChecked():
            if self.defaultSaveDir is not None:
                if self.dirty is True:
                    self.saveFile()
            else:
                self.changeSavedirDialog()
                return

        if not self.mayContinue():
            return

        if len(self.mImgList) <= 0:
            return

        filename = None
        if self.filePath is None:
            filename = self.mImgList[0]
        else:
            currIndex = self.mImgList.index(self.filePath)
            if currIndex + 1 < len(self.mImgList):
                filename = self.mImgList[currIndex + 1]

        if filename:
            self.loadFile(filename)

        if self.autoAdd == 1:
            self.addOneShape()

        if self.autoDelete == 1:
            if self.canvas.shapes != []:
                tmp_judge = []
                for shape_other in self.canvas.shapes:
                    tmp_judge.append(self.computeIoU(self.tmpShape, shape_other, Iou_max=MAX_IOU_FOR_DELETE))
                if 0 not in tmp_judge:
                    self.autoDelete = 0
            else:
                self.autoDelete = 0

        return filename

    def openFile(self, _value=False):
        if not self.mayContinue():
            return
        path = os.path.dirname(ustr(self.filePath)) if self.filePath else '.'
        formats = ['*.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
        filters = "Image & Label files (%s)" % ' '.join(formats + ['*%s' % LabelFile.suffix])
        filename = QFileDialog.getOpenFileName(self, '%s - Choose Image or Label file' % __appname__, path, filters)
        if filename:
            if isinstance(filename, (tuple, list)):
                filename = filename[0]
            self.loadFile(filename)

    def _loadImage4Detect(self):
        image = self.image
        size = image.size()
        s = image.bits().asstring(size.width() * size.height() * image.depth() // 8)  # format 0xffRRGGBB
        image = np.fromstring(s, dtype=np.uint8).reshape((size.height(), size.width(), image.depth() // 8))
        gray = image[:, :, 0]
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

        return gray

    def load_classes(self):
        self.classes = {}
        self.classes_list=[]
        for i in range(len(self.theseModels)):
            if self.theseModels[i]:
                classes = load_class_names(os.path.join(CURRENT_DIR,os.path.split(MODEL_PATH[MODEL_PARAMS[i]])[0]+'/classes.names'))
                self.classes[MODEL_PARAMS[i]] = [i for i in classes]

        for m in self.classes.values():
            for n in m:
                if n not in self.classes_list:
                    self.classes_list.append(n)

    def autoLabel(self):
        self.load_classes()
        if not self.fullyAutoMode:
            # if not in the fullyAutoMode
            if is_onnxok:
                if self.theseModels[0] and self.SSD is None:
                    self.SSD = SSD(os.path.join(CURRENT_DIR, MODEL_PATH[MODEL_PARAMS[0]]),
                                   self.classes[MODEL_PARAMS[0]])
                elif self.theseModels[0]:self.SSD.class_sel = self.classes[MODEL_PARAMS[0]]

                if self.theseModels[1] and self.centerNet is None:
                    self.centerNet = CenterNet(os.path.join(CURRENT_DIR, MODEL_PATH[MODEL_PARAMS[1]]),
                                               self.classes[MODEL_PARAMS[1]])
                elif self.theseModels[1]:self.centerNet.class_sel = self.classes[MODEL_PARAMS[1]]

                if self.theseModels[2] and self.YOLOv5 is None:
                    self.YOLOv5 = YOLOv5(os.path.join(CURRENT_DIR, MODEL_PATH[MODEL_PARAMS[2]]),
                                         self.classes[MODEL_PARAMS[2]])
                elif self.theseModels[2]:self.YOLOv5.class_sel = self.classes[MODEL_PARAMS[2]]

                if self.theseModels[3] and self.YOLOv5s is None:
                    self.YOLOv5s = YOLOv5(os.path.join(CURRENT_DIR, MODEL_PATH[MODEL_PARAMS[3]]),
                                          self.classes[MODEL_PARAMS[3]])
                elif self.theseModels[3]:self.YOLOv5s.class_sel= self.classes[MODEL_PARAMS[3]]

                if self.theseModels[4] and self.YOLOv5_i18R is None:
                    self.YOLOv5_i18R = YOLOv5(os.path.join(CURRENT_DIR, MODEL_PATH[MODEL_PARAMS[4]]),
                                          self.classes[MODEL_PARAMS[4]])
                elif self.theseModels[4]:self.YOLOv5_i18R.class_sel= self.classes[MODEL_PARAMS[4]]

                if self.theseModels[5] and self.YOLOv3 is None:
                    self.YOLOv3 = YOLOv3(os.path.join(CURRENT_DIR, MODEL_PATH[MODEL_PARAMS[5]]),
                                         self.classes[MODEL_PARAMS[5]])
                elif self.theseModels[5]:self.YOLOv3.class_sel = self.classes[MODEL_PARAMS[5]]
                
                if self.theseModels[6] and self.YOLOv3 is None:
                    self.MobileNetV2 = MobileNetV2(os.path.join(CURRENT_DIR, MODEL_PATH[MODEL_PARAMS[6]]),
                                         self.classes[MODEL_PARAMS[6]])
                elif self.theseModels[6]:self.YOLOv3.class_sel = self.classes[MODEL_PARAMS[6]]


                self.auto()
            else:
                return
        else:
            # in the fullyAutoMode
            if self.timer4autolabel.isActive():
                self.timer4autolabel.stop()
                autoLabel.setText("Fully autoLabel")

            elif is_onnxok:
                class_sel = ClassDialog(parent=self, classDicts=self.classes).popUp()
                if class_sel is not None:
                    if self.theseModels[0] and self.SSD is None:
                        self.SSD = SSD(os.path.join(CURRENT_DIR, MODEL_PATH[MODEL_PARAMS[0]]),
                                       class_sel[MODEL_PARAMS[0]])
                    elif self.theseModels[0]:self.SSD.class_sel = class_sel[MODEL_PARAMS[0]]

                    if self.theseModels[1] and self.centerNet is None:
                        self.centerNet = CenterNet(os.path.join(CURRENT_DIR, MODEL_PATH[MODEL_PARAMS[1]]),
                                                   class_sel[MODEL_PARAMS[1]])
                    elif self.theseModels[1]:self.centerNet.class_sel = class_sel[MODEL_PARAMS[1]]

                    if self.theseModels[2] and self.YOLOv5 is None:
                        self.YOLOv5 = YOLOv5(os.path.join(CURRENT_DIR, MODEL_PATH[MODEL_PARAMS[2]]),
                                             class_sel[MODEL_PARAMS[2]])
                    elif self.theseModels[2]: self.YOLOv5.class_sel = class_sel[MODEL_PARAMS[2]]

                    if self.theseModels[3] and self.YOLOv5s is None:
                        self.YOLOv5s = YOLOv5(os.path.join(CURRENT_DIR, MODEL_PATH[MODEL_PARAMS[3]]),
                                              class_sel[MODEL_PARAMS[3]])
                    elif self.theseModels[3]:self.YOLOv5s.class_sel = class_sel[MODEL_PARAMS[3]]

                    if self.theseModels[4] and self.YOLOv5_i18R is None:
                        self.YOLOv5_i18R = YOLOv5(os.path.join(CURRENT_DIR, MODEL_PATH[MODEL_PARAMS[4]]),
                                              class_sel[MODEL_PARAMS[4]])
                    elif self.theseModels[4]:self.YOLOv5_i18R.class_sel = class_sel[MODEL_PARAMS[4]]

                    if self.theseModels[5] and self.YOLOv3 is None:
                        self.YOLOv3 = YOLOv3(os.path.join(CURRENT_DIR, MODEL_PATH[MODEL_PARAMS[5]]),
                                             class_sel[MODEL_PARAMS[5]])
                    elif self.theseModels[5]:self.YOLOv3.class_sel = class_sel[MODEL_PARAMS[5]]

                    if self.theseModels[6] and self.MobileNetV2 is None:
                        self.MobileNetV2 = MobileNetV2(os.path.join(CURRENT_DIR, MODEL_PATH[MODEL_PARAMS[6]]),
                                             class_sel[MODEL_PARAMS[6]])
                    elif self.theseModels[6]:self.MobileNetV2.class_sel = class_sel[MODEL_PARAMS[6]]

                    self.timer4autolabel.start(20)
                    self.timer4autolabel.timeout.connect(self.autoThreadFunc)
                    autoLabel.setText("stop autoLabel")

            else:
                return

    def autoThreadFunc(self):

        if self.fullyAutoMode:
            try:
                next_id = self.mImgList.index(self.filePath) + 1
                tlen = len(self.mImgList)
                print("processing {} , total {} |{}{}|\r".format(next_id, tlen, "*"*int(float(next_id/tlen)*50),
                                                                                        " "*int(float(1-next_id/tlen)*50)))
                if next_id <= tlen:
                    self.auto()
                    self.openNextImg()
                if next_id == tlen:
                    self.timer4autolabel.stop()
                    autoLabel.setText("Fully autoLabel")
            except:
                print(self.filePath)

        else:
            self.timer4autolabel.stop()

    def auto(self):
        shapes = []
        results_box=[]
        for i in range(len(MODEL_PARAMS)):
            if self.theseModels[i]:
                result,box = eval("self.autoLabel" + MODEL_PARAMS[i] + "()")
                for j in box:
                    j[5]=self.classes_list.index(j[5])
                    results_box.append(j)

        results_box=np.array(results_box)

        box_nms=weighted_nms(results_box)
        for box in box_nms:
            x, y, x2, y2, score = int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(box[4])
            item = (self.classes_list[int(box[5])], [(x, y), (x2, y), (x2, y2), (x, y2)], None, None, False, 0, round(score, 2))
            new_shape = box2Shape(*item)
            if (self.shapeNMS(self.canvas.shapes, new_shape)):
                shapes.append(item)

        self.loadLabels(shapes)
        self.setDirty()

    def autoLabel_SSD(self):
        return self.SSD.forward(self._loadImage4Detect(), self.class_names_4_detect)

    def autoLabel_CENTER_NET(self):
        return self.centerNet.forward(self._loadImage4Detect())

    def autoLabel_YOLOv5(self):
        return self.YOLOv5.forward(self._loadImage4Detect())
        # return self.YOLOv5.forward(cv2.imread(self.filePath))

    def autoLabel_YOLOv5s(self):
        return self.YOLOv5s.forward(self._loadImage4Detect())
        # return self.YOLOv5.forward(cv2.imread(self.filePath))

    def autoLabel_YOLOv5_i18R(self):
        return self.YOLOv5_i18R.forward(self._loadImage4Detect())
        # return self.YOLOv5.forward(cv2.imread(self.filePath))

    def autoLabel_YOLOv3(self):
        return self.YOLOv3.forward(self._loadImage4Detect())
        # return self.YOLOv3.forward(cv2.imread(self.filePath))
    
    def autoLabel_MobileNetV2(self):
        return self.MobileNetV2.forward(self._loadImage4Detect())

    def saveFile(self, _value=False):
        if self.defaultSaveDir is not None and len(ustr(self.defaultSaveDir)):
            if self.filePath:
                imgFileName = self.filePath[self.dirname_len+1:]
                savedFileName = os.path.splitext(imgFileName)[0]
                savedPath = os.path.join(ustr(self.defaultSaveDir), savedFileName)
                if not os.path.exists(os.path.dirname(savedPath)):
                    os.makedirs(os.path.dirname(savedPath))
                self._saveFile(savedPath)
        else:
            imgFileDir = os.path.dirname(self.filePath)
            imgFileName = os.path.basename(self.filePath)
            savedFileName = os.path.splitext(imgFileName)[0]
            savedPath = os.path.join(imgFileDir, savedFileName)
            self._saveFile(savedPath if self.labelFile
                           else self.saveFileDialog(removeExt=False))

    def saveFileAs(self, _value=False):
        assert not self.image.isNull(), "cannot save empty image"
        self._saveFile(self.saveFileDialog())

    def saveFileDialog(self, removeExt=True):
        caption = '%s - Choose File' % __appname__
        filters = 'File (*%s)' % LabelFile.suffix
        openDialogPath = self.currentPath()
        dlg = QFileDialog(self, caption, openDialogPath, filters)
        dlg.setDefaultSuffix(LabelFile.suffix[1:])
        dlg.setAcceptMode(QFileDialog.AcceptSave)
        filenameWithoutExtension = os.path.splitext(self.filePath)[0]
        dlg.selectFile(filenameWithoutExtension)
        dlg.setOption(QFileDialog.DontUseNativeDialog, False)
        if dlg.exec_():
            fullFilePath = ustr(dlg.selectedFiles()[0])
            if removeExt:
                return os.path.splitext(fullFilePath)[0] # Return file path without the extension.
            else:
                return fullFilePath
        return ''

    def _saveFile(self, annotationFilePath):
        shapes = [format_shape_qt(shape) for shape in self.canvas.shapes]
        if annotationFilePath and self.saveLabels(annotationFilePath, shapes):
            self.setClean()
            self.statusBar().showMessage('Saved to  %s' % annotationFilePath)
            self.statusBar().show()

    def closeFile(self, _value=False):
        if not self.mayContinue():
            return
        self.resetState()
        self.setClean()
        self.toggleActions(False)
        self.canvas.setEnabled(False)
        self.actions.saveAs.setEnabled(False)

    def deleteImg(self):
        deletePath = self.filePath
        if deletePath is not None:
            self.openNextImg()
            if os.path.exists(deletePath):
                os.remove(deletePath)
            self.importDirImages(self.lastOpenDir)

    def resetAll(self):
        self.settings.reset()
        self.close()
        proc = QProcess()
        proc.startDetached(os.path.abspath(__file__))

    def mayContinue(self):
        if not self.dirty:
            return True
        else:
            discardChanges = self.discardChangesDialog()
            if discardChanges == QMessageBox.No:
                return True
            elif discardChanges == QMessageBox.Yes:
                self.saveFile()
                return True
            else:
                return False

    def discardChangesDialog(self):
        yes, no, cancel = QMessageBox.Yes, QMessageBox.No, QMessageBox.Cancel
        msg = u'You have unsaved changes, would you like to save them and proceed?\nClick "No" to undo all changes.'
        return QMessageBox.warning(self, u'Attention', msg, yes | no | cancel)

    def errorMessage(self, title, message):
        return QMessageBox.critical(self, title,
                                    '<p><b>%s</b></p>%s' % (title, message))

    def currentPath(self):
        return os.path.dirname(self.filePath) if self.filePath else '.'

    def chooseColor1(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.lineColor = color
            Shape.line_color = color
            self.canvas.setDrawingColor(color)
            self.canvas.update()
            self.setDirty()

    def deleteSelectedShape(self):
        self.remLabel(self.canvas.deleteSelected())
        self.setDirty()
        if self.noShapes():
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)

    def deleteShape(self, shape):
        self.remLabel(shape)
        self.setDirty()
        self.canvas.deleteOneShape(shape)
        if self.noShapes():
            for action in self.actions.onShapesPresent:
                action.setEnabled(False)

    def computeIoU(self, shape_1, shape_2, Iou_max=0.9):
        shape1 = shape_1.points
        shape2 = shape_2.points

        firs = [shape1[0].x(), shape1[0].y(), shape1[2].x(), shape1[2].y()]
        secs = [shape2[0].x(), shape2[0].y(), shape2[2].x(), shape2[2].y()]
        bbox_intersect = [max(secs[0], firs[0]), max(secs[1], firs[1]),
                          min(secs[2], firs[2]), min(secs[3], firs[3])]
        intersect_width = bbox_intersect[2] - bbox_intersect[0] + 1
        intersect_height = bbox_intersect[3] - bbox_intersect[1] + 1
        if intersect_height > 0 and intersect_width > 0:
            union_area = ((firs[2] - firs[0] + 1) * (firs[3] - firs[1] + 1)) \
                         + ((secs[2] - secs[0] + 1) * (secs[3] - secs[1] + 1)) \
                         - (intersect_width * intersect_height)
            Iou = intersect_height * intersect_width / union_area
            if Iou > Iou_max:
                self.deleteShape(shape_2)
                return 0
            else:
                return 1
        else:
            return 1

    def deleteSeries(self):

        self.tmpShape = self.canvas.deleteSelected()
        if self.tmpShape is None:
            return
        self.autoDelete = 1
        self.deleteShape(self.tmpShape)
        self.canvas.deleteOneShape(self.tmpShape)


        while self.autoDelete == 1:
            self.openNextImg()

    def chshapeLineColor(self):
        color = self.colorDialog.getColor(self.lineColor, u'Choose line color',
                                          default=DEFAULT_LINE_COLOR)
        if color:
            self.canvas.selectedShape.line_color = color
            self.canvas.update()
            self.setDirty()

    def chshapeFillColor(self):
        color = self.colorDialog.getColor(self.fillColor, u'Choose fill color',
                                          default=DEFAULT_FILL_COLOR)
        if color:
            self.canvas.selectedShape.fill_color = color
            self.canvas.update()
            self.setDirty()

    def copyShape(self):
        self.canvas.endMove(copy=True)
        self.addLabel(self.canvas.selectedShape)
        self.setDirty()

    def moveShape(self):
        self.canvas.endMove(copy=False)
        self.setDirty()

    def loadPredefinedClasses(self, predefClassesFile):
        if os.path.exists(predefClassesFile) is True:
            with codecs.open(predefClassesFile, 'r', 'utf8') as f:
                for line in f:
                    line = line.strip()
                    if self.labelHist is None:
                        self.labelHist = [line]
                    else:
                        self.labelHist.append(line)

    def loadPascalXMLByFilename(self, xmlPath):
        if self.filePath is None:
            return
        if os.path.isfile(xmlPath) is False:
            return

        self.set_format(FORMAT_PASCALVOC)

        tVocParseReader = PascalVocReader(xmlPath)
        shapes = tVocParseReader.getShapes()
        self.loadLabels(shapes)
        self.canvas.verified = tVocParseReader.verified

    def loadYOLOTXTByFilename(self, txtPath):
        if self.filePath is None:
            return
        if os.path.isfile(txtPath) is False:
            return

        self.set_format(FORMAT_YOLO)
        tYoloParseReader = YoloReader(txtPath, self.image)
        shapes = tYoloParseReader.getShapes()
        print (shapes)
        self.loadLabels(shapes)
        self.canvas.verified = tYoloParseReader.verified

    def loadCreateMLJSONByFilename(self, jsonPath, filePath):
        if self.filePath is None:
            return
        if os.path.isfile(jsonPath) is False:
            return

        self.set_format(FORMAT_CREATEML)

        crmlParseReader = CreateMLReader(jsonPath, filePath)
        shapes = crmlParseReader.get_shapes()
        self.loadLabels(shapes)
        self.canvas.verified = crmlParseReader.verified

    def copyPreviousBoundingBoxes(self):
        currIndex = self.mImgList.index(self.filePath)
        if currIndex - 1 >= 0:
            prevFilePath = self.mImgList[currIndex - 1]
            self.showBoundingBoxFromAnnotationFile(prevFilePath)
            self.saveFile()

    def togglePaintLabelsOption(self):
        for shape in self.canvas.shapes:
            shape.paintLabel = self.displayLabelOption.isChecked()

    def togglePaintScoresOption(self):
        for shape in self.canvas.shapes:
            shape.paintScore = self.displayScoreOption.isChecked()

    def toggleFullyAutoLabel(self):
        if self.FullyAutoLabelOption.isChecked():
            # if this button is checked
            autoLabel.setText("Fully autoLabel")
            self.fullyAutoMode = True
        else:
            # if this button is NOT checked
            autoLabel.setText("autoLabel")
            self.fullyAutoMode = False

    def reloadImage(self):
        currIndex = self.mImgList.index(self.filePath)
        filename = self.mImgList[currIndex]
        self.loadFile(filename)

    def toggleDenoise(self):
        self.denoise = self.denoiseOption.isChecked()
        self.reloadImage()

    def setEqualizeHistStatus(self):
        if self.EqualizeHistOption.isChecked():
            # if this button is checked
            self.EqualizeHist = True
        else:
            # if this button is NOT checked
            self.EqualizeHist = False

        self.reloadImage()

    def preProcess(self, img):
        # img = cv2.medianBlur(img, 3)
        # img = cv2.bilateralFilter(img,5,75,75)
        img = cv2.fastNlMeansDenoising(img, None, 5, 8, 25)

        img =cv2.normalize(img,dst=None,alpha=350,beta=10,norm_type=cv2.NORM_MINMAX)

        clahe = cv2.createCLAHE(3,(8,8))
        img = clahe.apply(img)

        return img

    def setEqualizeHist(self,image):
        size = image.size()
        s = image.bits().asstring(size.width() * size.height() * image.depth() // 8)
        img_arr = np.fromstring(s, dtype=np.uint8).reshape((size.height(), size.width(), image.depth() // 8))

        img = self.toRGB(img_arr)
        R, G, B = cv2.split(img)  # get single 8-bits channel
        EB = self.preProcess(B)
        EG = self.preProcess(G)
        ER = self.preProcess(R)
        img_arr = cv2.merge((ER, EG, EB))
        image = QtGui.QImage(img_arr[:], img_arr.shape[1], img_arr.shape[0],
                             img_arr.shape[1] * img_arr.shape[2],
                             QtGui.QImage.Format_RGB888)

        return image

    def compute(self,img, min_percentile, max_percentile):

        max_percentile_pixel = np.percentile(img, max_percentile)
        min_percentile_pixel = np.percentile(img, min_percentile)

        return max_percentile_pixel, min_percentile_pixel

    def toRGB(self, image):
        if image.shape[2] >= 3:
            if image.shape[2] == 3:
                B, G, R = cv2.split(image)  # get single 8-bits channel
                image = cv2.merge((R, G, B))
            elif image.shape[2] == 4:
                B, G, R, t = cv2.split(image)  # get single 8-bits channel
                image = cv2.merge((R, G, B))
        elif image.shape[2] == 1:
            image = cv2.merge((image, image, image))

        return image

    def setEqualizeHistNew(self,image):
        size = image.size()
        s = image.bits().asstring(size.width() * size.height() * image.depth() // 8)
        img_arr = np.fromstring(s, dtype=np.uint8).reshape((size.height(), size.width(), image.depth() // 8))

        img = self.toRGB(img_arr)  # get single 8-bits channel

        hsv_image=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
        if hsv_image[:, :, 2].mean()>130:
            return image

        max_percentile_pixel, min_percentile_pixel = self.compute(img, 1, 94)

        img[img >= max_percentile_pixel] = max_percentile_pixel
        img[img <= min_percentile_pixel] = min_percentile_pixel

        out = np.zeros(img.shape, img.dtype)
        cv2.normalize(img, out, 255 * 0.1, 255 * 0.9, cv2.NORM_MINMAX)

        image = QtGui.QImage(out[:], out.shape[1], out.shape[0],
                             out.shape[1] * out.shape[2],
                             QtGui.QImage.Format_RGB888)

        return image


    def Denoise(self, image):
        size = image.size()
        s = image.bits().asstring(size.width() * size.height() * image.depth() // 8)
        img_arr = np.fromstring(s, dtype=np.uint8).reshape((size.height(), size.width(), image.depth() // 8))

        img = self.toRGB(img_arr)  # get single 8-bits channel

        out = cv2.GaussianBlur(img, (3, 3), 0)

        image = QtGui.QImage(out[:], out.shape[1], out.shape[0],
                             out.shape[1] * out.shape[2],
                             QtGui.QImage.Format_RGB888)

        return image

    def toogleDrawSquare(self):
        self.canvas.setDrawingShapeToSquare(self.drawSquaresOption.isChecked())


def inverted(color):
    return QColor(*[255 - v for v in color.getRgb()])


def read(filename, default=None):
    try:
        reader = QImageReader(filename)
        reader.setAutoTransform(True)
        return reader.read()
    except:
        return default


def get_main_app(argv=[]):
    """
    Standard boilerplate Qt application code.
    Do everything but app.exec_() -- so that we can test the application in one thread
    """
    app = QApplication(argv)
    app.setApplicationName(__appname__)
    app.setWindowIcon(newIcon("app"))
    # Tzutalin 201705+: Accept extra agruments to change predefined class file
    argparser = argparse.ArgumentParser()
    argparser.add_argument("image_dir", nargs="?")
    argparser.add_argument("predefined_classes_file",
                           default=os.path.join(os.path.dirname(__file__), "data", "predefined_classes.txt"),
                           nargs="?")
    argparser.add_argument("save_dir", nargs="?")
    args = argparser.parse_args(argv[1:])
    # Usage : labelImg.py image predefClassFile saveDir
    win = MainWindow(args.image_dir,
                     args.predefined_classes_file,
                     args.save_dir)
    win.show()
    return app, win


def main():
    '''construct main app and run it'''
    app, _win = get_main_app(sys.argv)
    return app.exec_()

if __name__ == '__main__':
    sys.exit(main())
