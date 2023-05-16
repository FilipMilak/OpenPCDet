import os
from pathlib import Path
import re
import sys
import torch

import numpy as np
import pyqtgraph.opengl as gl
import pandas as pd
from scipy import stats

from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *

import warnings

import demo
from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from tools.visual_utils.open3d_vis_utils import get_box

# In order to ignore some weird FutureWarning
warnings.filterwarnings("ignore")

# CONSOLE PARAMETERS

DATE = '2022-12-06_11-56-16'
LIDAR_DATA_PATH = "/external/10g/charlesnas/datasets/ImmendingenSprayTestsSpraymakerv2_new/"


#UPCOMING PARAMETERS

CFG_FILES = '/lhome/fimilak/Documents/OpenPCDet/tools/cfgs/dense_models/'#pointpillar_newaugs.yaml'
CKPT_FILES = '/lhome/fimilak/Documents/OpenPCDet/output/dense_models/'
CKPT_FILE_EXT = '/full/ckpt/checkpoint_epoch_80.pth'

# Change for working on work stations
POINTCLOUD_PATH = LIDAR_DATA_PATH + f'{DATE}/lidar_hdl64_s3/velodyne_pointclouds/strongest_echo/'

# Each path leads to the folder, in which the stereo photos are located
IMAGE_LEFT_PATH = LIDAR_DATA_PATH + f"{DATE}/cam_stereo/left/image_rect"
IMAGE_RIGHT_PATH = LIDAR_DATA_PATH + f"{DATE}/cam_stereo/left/image_rect"

# How many pictures will be loaded into the application
AMOUNT_OF_SCENES = 2

# Where to begin the scene
START_SCENE = 0

# Image scale divisor
WIDTH_DIVISOR = 3
HEIGHT_DIVISOR = 3

# Constant to define in which column what is located
X_AXIS   = 0
Y_AXIS   = 1
Z_AXIS   = 2
COLOR    = 3
DISTANCE = 4

# Define the minium of boxes
MINIMUM_AXIS = -10000

# Define the steps size per unit
SCALE_FACTOR = 10

# Define the three colors for the boxes
NONE_SELECTED_COLOR = (1,1,1,1) # white
SELECTED_COLOR = (1,0.7,1,1)    # light magenta
EDITING_COLOR = (1,0,1,1)       # magenta

DEFAULT_BOX_VALUE = 1

# Translate the int to keys
W_KEY, S_KEY, D_KEY, A_KEY, Q_KEY, E_KEY = 87, 83, 65, 68, 81, 69

# Translate the number keypad
ONE_KEY, TWO_KEY, THREE_KEY, FOUR_KEY, FIVE_KEY, SIX_KEY = 49, 50, 51, 52, 53, 54

# Input keys to position the box
POSITIVE_X = D_KEY
NEGATIVE_X = A_KEY

POSITIVE_Y = W_KEY
NEGATIVE_Y = S_KEY

POSITIVE_Z = E_KEY
NEGATIVE_Z = Q_KEY

POSITIVE_X_LEN = ONE_KEY
NEGATIVE_X_LEN = TWO_KEY

POSITIVE_Y_LEN = THREE_KEY
NEGATIVE_Y_LEN = FOUR_KEY

POSITIVE_Z_LEN = FIVE_KEY
NEGATIVE_Z_LEN = SIX_KEY

# Labels are saved here
SHOW_EXTRACT_POINTS_BUTTON_TEXT = "Show extractable"
SHOW_EXTRACT_POINTS_BUTTON_INVERSE_TEXT = "Show points"
EXTRACT_ALL_BUTTON_TEXT = "Extract all"
EXTRACT_SELECTED_BUTTON_TEXT = "Extract selected"
CREATE_A_BOX_BUTTON_TEXT = "Create a box"
CREATE_A_BOX_BUTTON_TEXT_INVERSE = "Stop editing"
DELETE_SELECTED_BUTTON_TEXT = "Delete selected"
DELETE_ALL_BUTTON_TEXT = "Delete all"

FILE_NAME_INPUT_PLACEHOLDER_TEXT = "file name"

# Function to convert integer into RGB
def inteToRGB(pc_inte):
    minimum, maximum = np.min(pc_inte), np.max(pc_inte)
    ratio = 2 * (pc_inte-minimum) / (maximum - minimum)
    b = (np.maximum((1 - ratio), 0))
    r = (np.maximum((ratio - 1), 0))
    g = 1 - b - r
    return np.stack([r, g, b, np.ones_like(r)]).transpose()

# Function to get the int of the box view
def getIntOfString(text : str):
    
    return int(re.findall(r'\b\d+\b',text)[0])
    
# class to handle coordinates
class Coordinate():
    
    # Constructor
    def __init__(self, x : float = 0, y : float = 0, z : float = 0):
        
        self.x : float = x
        
        self.y : float = y
        
        self.z : float = z
        
    # Return all points as a numpy array for compatibility
    def render(self) : return np.array([self.x, self.y, self.z]) 
    
#TODO: Wird nicht genutzt, könnte gelöscht werden
class Lines():
    
    def __init__(self, cod1 : Coordinate(), cod2 : Coordinate()) -> None:
        
        self.cod1 : Coordinate = cod1
        
        self.cod2 : Coordinate = cod2
        
    def render(self) : return 

# Class box
class Box():
    
    # Constructor
    def __init__(self, x : float, y : float, z : float, x_len : float, y_len : float, z_len : float, points : pd.DataFrame, color : list = EDITING_COLOR) -> None:
        
        # Define the current color of the box, including its edges and the points within
        self.color = color
        
        # Create the box with the given parameters
        self.assignVariables(x, y, z, x_len, y_len, z_len)
        
        # Update and display the current 
        self.updateAndDisplay(points)
        
    # Set the variable color
    def setColor(self, color):
        
        self.color = color

    # Create each point of the box   
    def createCoordinates(self):
        
        self.coordinates : list[Coordinate] = []
        
        self._x_y_z = Coordinate(self.x  , self.y  , self.z )
    
        self.xx_y_z = Coordinate(self.xx , self.y  , self.z )
    
        self.xxyy_z = Coordinate(self.xx , self.yy , self.z )
    
        self.xxyyzz = Coordinate(self.xx , self.yy , self.zz)
    
        self._xyy_z = Coordinate(self.x  , self.yy , self.z )
    
        self._xyyzz = Coordinate(self.x  , self.yy , self.zz)
    
        self.xx_yzz = Coordinate(self.xx , self.y  , self.zz)

        self._x_yzz = Coordinate(self.x  , self.y  , self.zz)
    
    # Create a line by passing two coordinates
    def createLine(self, cod1 : Coordinate, cod2 : Coordinate, width : int = 3, antialias : bool = True) -> gl.GLLinePlotItem : 
        
        return gl.GLLinePlotItem(pos=np.array([cod1.render(),cod2.render()]), mode='lines', color=self.color, width=width, antialias=antialias)
        
    # Create and define the lines of the box and its points within
    def createLinesPoints(self, points : pd.DataFrame) -> None : 
        
        self.line_list = [
        
        self.createLine(self._x_y_z, self.xx_y_z),
        
        self.createLine(self._x_y_z, self._xyy_z),
        
        self.createLine(self._x_y_z, self._x_yzz),
        
        self.createLine(self.xx_y_z, self.xxyy_z),
        
        self.createLine(self.xx_y_z, self.xx_yzz),
        
        self.createLine(self.xxyy_z, self.xxyyzz),
        
        self.createLine(self._xyy_z, self._xyyzz),
        
        self.createLine(self._xyy_z, self.xxyy_z),
        
        self.createLine(self._xyyzz, self.xxyyzz),
        
        self.createLine(self.xx_yzz, self.xxyyzz),
        
        self.createLine(self.xx_yzz, self._x_yzz),
        
        self.createLine(self._xyyzz, self._x_yzz)]
        
        # Select all points of the current scene, which are located within the box
        self.points_within = points[
            (points[X_AXIS] >= self.x) & (points[X_AXIS] <= self.xx) &
            (points[Y_AXIS] >= self.y) & (points[Y_AXIS] <= self.yy) &
            (points[Z_AXIS] >= self.z) & (points[Z_AXIS] <= self.zz)
        ]
        
        self.points = None
        
        # if the there are points within the box, create a scatter plot
        if not self.points_within.empty :
            
            self.points = gl.GLScatterPlotItem(pos=self.points_within[[X_AXIS, Y_AXIS, Z_AXIS]], size = 4, color = self.color)
        
    # Assign the sizes to the box and its axis points
    def assignVariables(self, x : float = 0, y : float = 0, z : float = 0, x_len : float = 0, y_len : float = 0, z_len : float = 0) -> None :

        self.x_len : float = x_len
        
        self.y_len : float = y_len
        
        self.z_len : float = z_len
        
        self.x     : float = x
        
        self.y     : float = y
        
        self.z     : float = z
        
        self.xx    : float = x + x_len
        
        self.yy    : float = y + y_len
        
        self.zz    : float = z + z_len
    
    # get the current box sizes for setting the box settings
    def getCurrentBoxSettings(self) -> list[float]:
        
        # return the the input parameters
        return [
            
            self.x,
            
            self.y,
            
            self.z,
            
            self.x_len,
            
            self.y_len,
            
            self.z_len
        ]
    
    # update the current box by creating the box`s points and its lines`
    def updateAndDisplay(self, points : pd.DataFrame) -> None:
        
        self.createCoordinates()
        
        self.createLinesPoints(points=points)
    
    # get all lines of the box
    def getLines(self) -> list[gl.GLLinePlotItem]:
        
        return self.line_list
    
    # get all points which are located within the box
    def getPoints(self) -> gl.GLScatterPlotItem :#| None:
        
        return self.points #if (self.color != NONE_SELECTED_COLOR) else None

# Class to define the prediction boxes
class BoundingBox():

    def __init__(self, bounding_box: list[gl.GLLinePlotItem], score_3d: gl.GLScatterPlotItem, class_3d: gl.GLScatterPlotItem, confidence: float):

        self.bounding_box = bounding_box
        self.score_3d = score_3d
        self.class_3d = class_3d
        self.confidence = confidence


# Class of the application inherits from basic QWidget
class Window(QWidget):
    
    # Constructor
    def __init__(self):

        # Execute the super constructor
        super().__init__()

        # Define layout of the window title
        self.setWindowTitle("Lidar Viewer")

        # contains all bounding of all scenes of all models
        self.models_bounding_boxes : dict[str, list[list[BoundingBox]]] = {}

        # contains the current selected model which display its predicted bounding boxes+
        self.current_model: str = None

        # represents the thold
        self.thold: int = 0
        
        # Read all important files, including point clods and model
        self.readAllPointClouds()
        self.readImageFiles()
        
        
        # Where does the scene end?
        self.max_scence_index = len(self.points)-1

        # Initialize the scene
        self.scene_index = START_SCENE
        
        # Initialize the filter with full tolerance
        self.distance = self.max_distance
        self.higher_z_axis = self.max_z_axis
        self.lower_z_axis = self.min_z_axis
        
        # Initialize variables which handles the editing system
        self.current_points = None
        self.current_box = None
        self.editing_mode = False
        # Defines the names of the boxes and is increased for each created
        self.inc_index = 0
        
        

        # Contains all current displayed boxes
        self.box : list[Box] = []
        
        #initialize the 3D coordinate system
        self.view = gl.GLViewWidget()

        ######## All Layouts ########

        # this is the whole layout
        main_layout = QVBoxLayout()

        #this layout defines the body without the forward and backward button
        central_layout = QHBoxLayout()

        #it contains the right image of the current scene
        self.image_right_label = QLabel("right stereo image")
        self.image_right = QLabel()

        #it contains the left image of the current scene
        self.image_left_label = QLabel("left stereo image")
        self.image_left = QLabel()

        #It contains images of the stereo cameras
        picture_layout = QVBoxLayout()

        #This layout contains the forward and backward button
        button_layout = QHBoxLayout()

        #This layout contains the bounding box prediction settings like model and threshold
        bbox_layout = QHBoxLayout()

        #This layout contains the two sliders, which configure the displayed height and distance of point clouds
        point_cloud_layout = QVBoxLayout()
        
        #This layout contains the control panel, which can create a box
        box_layout = QVBoxLayout()
        box_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        #Create a slider which selects what scene is displayed chronically ordered
        self.scene_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.scene_slider_label = QLabel(f"current scene: {self.scene_index}")
        self.createSlider(self.scene_slider, self.max_scence_index, 0, self.sceneSliderChanged)

        self.box_thold_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.box_thold_slider_label = QLabel(f"bounding box threshold: {self.thold}")
        self.createSlider(self.box_thold_slider, 100, 0, self.boundingBoxTholdChanged, self.thold)
        
        box_thold = QHBoxLayout()
        box_thold.addWidget(self.box_thold_slider_label)
        box_thold.addWidget(self.box_thold_slider)

        self.models_selection = QComboBox()
        self.models_selection.currentTextChanged.connect(self.modelChanged)

        # search for all models in folder
        self.searchAvailModels()

        bbox_layout.addLayout(box_thold)
        bbox_layout.addWidget(self.models_selection)
        
        #Create a slider which filters points with a larger distance than the slider's value
        self.distance_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.distance_slider_label = QLabel("distance view")
        self.distance_slider_label.adjustSize()
        self.createSlider(self.distance_slider, self.max_distance, 0, self.distanceSliderChanged, start_value=self.max_distance)
        
        #Create a slider which filter points above the slider's value
        self.higher_z_axis_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.higher_z_axis_slider_label = QLabel("higher z-axis view")
        self.higher_z_axis_slider_label.adjustSize()
        self.createSlider(self.higher_z_axis_slider, self.max_z_axis, self.min_z_axis, self.higherZAxisSliderChanged, start_value=self.max_z_axis)

        #Create a slider which filter points above the slider's value
        self.lower_z_axis_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.lower_z_axis_slider_label = QLabel("lower z-axis view")
        self.lower_z_axis_slider_label.adjustSize()
        self.createSlider(self.lower_z_axis_slider, self.max_z_axis, self.min_z_axis, self.lowerZAxisSliderChanged, start_value=self.min_z_axis)
        
        #Create a backward button to decrease the scene index
        self.backward_button = QPushButton("Backward")
        self.backward_button.clicked.connect(self.backward)        
        #If there are no scenes before, deactivate the button
        if self.scene_index == 0 : self.backward_button.setEnabled(False)        
        #add the button to the layout
        button_layout.addWidget(self.backward_button, 1)
        
        #Create a forward button to increase the scene index
        self.forward_button = QPushButton("Forward")
        self.forward_button.clicked.connect(self.forward) 
        #If there are no scenes after, deactivate the button 
        if self.scene_index == self.max_scence_index: self.forward_button.setEnabled(False)
        #Add the button to the layout
        button_layout.addWidget(self.forward_button, 1)

        #Add both images and their labels to the layout
        picture_layout.addWidget(self.image_left_label,1)
        picture_layout.addWidget(self.image_left,10000)
        picture_layout.addWidget(self.image_right_label,1)
        picture_layout.addWidget(self.image_right,10000)

        #Add the distance and z-axis slider and their labels to the point cloud layout
        point_cloud_layout.addWidget(self.distance_slider_label,1)
        point_cloud_layout.addWidget(self.distance_slider,1)
        point_cloud_layout.addWidget(self.lower_z_axis_slider_label,1)
        point_cloud_layout.addWidget(self.lower_z_axis_slider,1)
        point_cloud_layout.addWidget(self.higher_z_axis_slider_label,1)
        point_cloud_layout.addWidget(self.higher_z_axis_slider,1)
        point_cloud_layout.addWidget(self.view,100000)
        
        # Box view, which displays all current boxes in a list
        self.box_view = QListWidget()
        # By that you are able to select multiple items in the list
        self.box_view.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        # Dont allow Drag&Drop
        self.box_view.setAcceptDrops(False)
        # If a selection has changed then execute a function
        self.box_view.itemSelectionChanged.connect(self.itemSelectionChanged)
        # If an item was double clicked, execute a function
        self.box_view.itemDoubleClicked.connect(self.itemDoubleClicked)
        
        # Create a button which en- or disables the editing mode, by creating a new box
        self.create_box_button = QPushButton(CREATE_A_BOX_BUTTON_TEXT)
        # If this button was clicked execute create box function
        self.create_box_button.clicked.connect(self.createBox)
        
        # Create a button which delete only selected boxes
        self.delete_selected_button = QPushButton(DELETE_SELECTED_BUTTON_TEXT)
        # Executre the deleted function if clicked
        self.delete_selected_button.clicked.connect(self.deleteSelected)
        # Disable the button when the window is created, since there is no selection yet
        self.delete_selected_button.setEnabled(False)
        
        # Create a button which delete all boxes at once
        self.delete_all_button = QPushButton(DELETE_ALL_BUTTON_TEXT)
        # Execute the delete all function if button is clicked
        self.delete_all_button.clicked.connect(self.deleteAll)
        
        # Create a button which extracts all boxes and saves them as bin files
        self.extract_selected_button = QPushButton(EXTRACT_SELECTED_BUTTON_TEXT)
        # Execute the function if button is clicked
        self.extract_selected_button.clicked.connect(self.extractSelected)
        # Disable the button since right after initialize this window nothing is selected
        self.extract_selected_button.setEnabled(False)
        
        # Create a button which extracts all buttons
        self.extract_all_button = QPushButton(EXTRACT_ALL_BUTTON_TEXT)
        # If button is clicked execute the function
        self.extract_all_button.clicked.connect(self.extractAll)
        
        # Create an input field in which the file name must be written
        self.extract_file_name_input = QLineEdit()
        # Set the placeholder for the input
        self.extract_file_name_input.setPlaceholderText(FILE_NAME_INPUT_PLACEHOLDER_TEXT)
        # Regex to validate if file name is alphanumerical
        reg_ex = QRegularExpression("^\w+$")
        # Create the regular expression validator
        input_validator = QRegularExpressionValidator(reg_ex, self.extract_file_name_input)
        # Set the validator for the input field
        self.extract_file_name_input.setValidator(input_validator)
        
        # Create a box which removes or adds the scatter points again
        self.show_extract_points_button = QPushButton(SHOW_EXTRACT_POINTS_BUTTON_TEXT)
        # If button is clicked executer the function
        self.show_extract_points_button.clicked.connect(self.showExtractedPoints)
        
        # Disable all buttons which need to have at least one box in the box view
        self.enableAllBoxButton(False)
        
        # Create a layout which contains the box configurations
        box_column = QVBoxLayout()
        # Set alignment to remove the gaps between the buttons
        box_column.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Both lists contain the input field for the box and its labels
        self.box_settings : list[QDoubleSpinBox] = []
        self.box_settings_label : list[QLabel] = []
        
        # For each dimension create an input box and a label
        for i, label in enumerate(["X-coordinate", "Y-coordinate", "Z-coordinate", "X-length", "Y-length", "Z-length"]):
            
            # Create a box which contains the input field and its label
            layout = QHBoxLayout()
            # Remove the space between label and input
            layout.setAlignment(Qt.AlignmentFlag.AlignTop)
            
            # Append a spin box to configure the position and size
            self.box_settings.append(QDoubleSpinBox())
            self.box_settings_label.append(QLabel(label))
            # If it is about a position, there is a negative 
            if i < 3 : self.box_settings[-1].setMinimum(MINIMUM_AXIS)
            
            # Set the value of the box configurations
            self.box_settings[-1].setValue(DEFAULT_BOX_VALUE)
            
            # Hide the settings if application has just started
            self.box_settings[-1].setVisible(False)
            self.box_settings_label[-1].setVisible(False)
            
            # If value of box settings has changed then update the box's position and size
            self.box_settings[-1].valueChanged.connect(lambda: self.updateBox())
            
            # Add the labels and the configuration input to the layout 
            layout.addWidget(self.box_settings_label[-1],1)
            layout.addWidget(self.box_settings[-1],1)
            box_layout.addLayout(layout)
        
        # Assemble all elements which configures the box
        box_column.addWidget(self.create_box_button, 1)
        box_column.addLayout(box_layout,1)
        box_column.addWidget(self.box_view,10000)
        box_column.addWidget(self.extract_selected_button, 1)
        box_column.addWidget(self.extract_all_button, 1)
        box_column.addWidget(self.extract_file_name_input, 1)
        box_column.addWidget(self.show_extract_points_button, 1)
        box_column.addWidget(self.delete_selected_button, 1)
        box_column.addWidget(self.delete_all_button, 1)

        #Add the right and left column to the central layout, whereas point cloud layout is two times bigger than the images
        central_layout.addLayout(picture_layout,3)
        central_layout.addLayout(point_cloud_layout,5)
        central_layout.addLayout(box_column, 1)

        #Merge all layouts and scene control 
        main_layout.addLayout(button_layout,2)
        main_layout.addWidget(self.scene_slider_label,1)
        main_layout.addWidget(self.scene_slider,1)
        main_layout.addLayout(bbox_layout)
        main_layout.addLayout(central_layout,1000000)        
        
        # Set the layout on the application's window
        self.setLayout(main_layout)
        
        #Create a new axis item since it was deleted and add it to the coordinate system
        cord = gl.GLAxisItem()
        cord.setSize(3,3,3)
        self.view.addItem(cord)
        
        #Display the pictures and point clouds
        self.updatePoints()

        # Update all bounding boxes
        self.addBoundingBoxes()

        #Load both images and scale the images relatively to the window size down
        self.image_left.setPixmap(QPixmap(os.path.join(IMAGE_LEFT_PATH, self.images_left[self.scene_index])).scaled(int(self.width()/2.1),int(self.height()/3), aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio))
        self.image_right.setPixmap(QPixmap(os.path.join(IMAGE_RIGHT_PATH, self.images_right[self.scene_index])).scaled(int(self.width()/2.1),int(self.height()/3), aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio))


    # Function of slider event to change threshold
    def boundingBoxTholdChanged(self):

        old_thold: float = self.thold

        # get value and divide by 100
        self.thold: float = self.box_thold_slider.value()/100

        #set new texts
        self.box_thold_slider_label.setText(f"bounding box threshold: {'%4.2f' % self.thold}")

        if self.thold > old_thold:

            self.removeBoundingBox(start_thold=old_thold, end_thold=self.thold)

        elif self.thold < old_thold:

            self.addBoundingBoxes(start_thold=self.thold, end_thold=old_thold)

    # Reset the editing mode     
    def resetEditingMode(self):
        
        # Setting flag variable to false
        self.editing_mode = False
        
        # Set the text of the create button box to default
        self.create_box_button.setText(CREATE_A_BOX_BUTTON_TEXT)

        # reset all settable 
        self.enableBoxSettings(False)
    
    # Hide the current Points except for those which are located within boxes
    def showExtractedPoints(self):
        
        # If the button was not clicked yet, remove the scatter points
        if self.show_extract_points_button.text() == SHOW_EXTRACT_POINTS_BUTTON_TEXT:
            
            # Set the text of the button
            self.show_extract_points_button.setText(SHOW_EXTRACT_POINTS_BUTTON_INVERSE_TEXT)           
            
            # Remove the current points from the view
            self.view.removeItem(self.current_points)
            
            # for box in self.box.values():
                
            #     print(box.points)
                
            #     if box.points is not None and box.points not in self.view.items:
                    
            #         self.view.addItem(box.points)
        # If the button was already clicked, change the function and reset
        else:
            # Reset the text of the button to default
            self.show_extract_points_button.setText(SHOW_EXTRACT_POINTS_BUTTON_TEXT)
            
            # Add the scatter points to the view again
            self.view.addItem(self.current_points)
        
            # for key, box in self.box.items():
                
            #     indexes = [getIntOfString(item.text()) for item in self.box_view.selectedItems()]

            #     if box.points is not None and not (self.editing_mode and key == self.current_box) and key not in indexes:
                    
            #         self.view.removeItem(box.points)
    
    # Function which removes and deletes all boxes
    def deleteAll(self):
        
        # Disable editing mode
        self.resetEditingMode()
        
        # Clear the box dictionary
        self.box_view.clear()
        
        self.current_box = None
        
        # Remove the box from the 3D plot and the list
        [self.removeBox(idx) for idx in reversed(range(len(self.box)))] 
    
    # Enable all box buttons
    def enableAllBoxButton(self, enable : bool) -> None : 
        
        self.delete_all_button.setEnabled(enable)
        self.extract_all_button.setEnabled(enable)
        self.extract_file_name_input.setEnabled(enable)
        self.show_extract_points_button.setEnabled(enable)
    
    # Delete selected buttons
    def deleteSelected(self):
        
        for q_idx in self.box_view.selectedIndexes():
            
            idx = q_idx.row()
            
            print(f"row_index {idx}")
            
            # If the current edited box is selected to be deleted, then disable editing mode
            if self.current_box == idx: self.resetEditingMode()
            
            # Extract the item in a special index
            self.box_view.takeItem(idx)
            
            # Remove box with index
            self.removeBox(idx)
            
            # If the box to be deleted is the same as the box in the current_box variable then reset the variable
            if self.current_box == idx: self.current_box = None
            
            # If the box to be deleted is before the current_box in the list then decrement its value
            elif self.current_box > idx: self.current_box -= 1
        
        # If there is no box left
        if not self.box:
            
            self.current_box = None
            
        self.delete_selected_button.setEnabled(False)
        self.extract_selected_button.setEnabled(False)

    # Extract all boxes and save them
    def extractAll(self):

        self.save_extracted(self.extract(self.box))

    # Extract only selected boxes and save them
    def extractSelected(self):
        
        self.save_extracted(self.extract([self.box[idx.row()] for idx in self.box_view.selectedIndexes()]))

    # Save the extracted Box into a .bin file
    def save_extracted(self, extracted_points : np.ndarray):

        file_name = self.extract_file_name_input.text()
        
        if file_name:
            extracted_points.tofile(f"{file_name}.bin")
    
    # Get all points of given boxes and return their points
    def extract(self, boxes : list[Box]) -> np.ndarray :
        
        extracted_points = []
        
        for box in boxes:
            
            if box.points_within is not None:
                
                extracted_points.append(box.points_within)
        # Concatenate each points of each boxes and drop the duplicates and distance colum,
        # since boxes can overlap and distance is not important anymore
        return pd.concat(extracted_points, ignore_index=True).drop_duplicates().drop(DISTANCE).values
    
    # If a key on keyboard is presse
    def keyPressEvent(self, event):
        
        #FIXME: When 3D-View Widget is clicked the keyevent can not be caught anymore
        
        # If the editing mode is on
        if(self.editing_mode):
            
            # Change the values of the box according to what is selected
            if event.key() == POSITIVE_X: # A
                self.box_settings[0].setValue(self.box_settings[0].value() + .1)
            if event.key() == NEGATIVE_X: # D
                self.box_settings[0].setValue(self.box_settings[0].value() - .1)
                
            if event.key() == POSITIVE_Y: # W
                self.box_settings[1].setValue(self.box_settings[1].value() + .1)
            if event.key() == NEGATIVE_Y: # S
                self.box_settings[1].setValue(self.box_settings[1].value() - .1)
                
            if event.key() == POSITIVE_Z: # Q
                self.box_settings[2].setValue(self.box_settings[2].value() + .1)
            if event.key() == NEGATIVE_Z: # E
                self.box_settings[2].setValue(self.box_settings[2].value() - .1)
                
            if event.key() == POSITIVE_X_LEN: # Q
                self.box_settings[3].setValue(self.box_settings[3].value() + .1)
            if event.key() == NEGATIVE_X_LEN: # E
                self.box_settings[3].setValue(self.box_settings[3].value() - .1)
                
            if event.key() == POSITIVE_Y_LEN: # Q
                self.box_settings[4].setValue(self.box_settings[4].value() + .1)
            if event.key() == NEGATIVE_Y_LEN: # E
                self.box_settings[4].setValue(self.box_settings[4].value() - .1)
                
            if event.key() == POSITIVE_Z_LEN: # Q
                self.box_settings[5].setValue(self.box_settings[5].value() + .1)
            if event.key() == NEGATIVE_Z_LEN: # E
                self.box_settings[5].setValue(self.box_settings[5].value() - .1)
            
        event.accept()
        
    # Disable the editing mode
    def disableEditingMode(self, resetEditingMode = True):
    
        # Set the color of the current color to none selected if it is noneselected
        # Otherwise set color to selected color
        if self.current_box in [idx.row() for idx in self.box_view.selectedIndexes()]:
            self.box[self.current_box].setColor(SELECTED_COLOR)
        else:
            self.box[self.current_box].setColor(NONE_SELECTED_COLOR)
        
        # Update the boxes and points
        self.updateBox()
        
        # Get the current item of the list
        curr_item : QListWidgetItem = self.box_view.item(self.current_box)
        
        self.changeEditingLabel(curr_item, False)
        
        # Reset the editing mode if wanted
        if resetEditingMode:
            self.resetEditingMode()
    
    # Enable or disable the box
    def createBox(self):
        
        # If the editing mode is on then disable the editing mode
        if self.editing_mode:
            
            self.disableEditingMode()
            
        # Else enable the editing mode
        else: 
            
            self.enableEditingMode()

    # function to enable and create or edit a current box
    def enableEditingMode(self, idx : int = None):
        
        # If there is a current box edited or has been edited
        if(self.current_box is not None):
        
            # Get the box which was edited before
            # TODO: CHECK FUNCTIONALITY
            oldEditedItem : QListWidgetItem = self.box_view.item(self.current_box)
            
            # If there is an old item then extract ">" and "<" from its label
            if oldEditedItem is not None and ">" in oldEditedItem.text():
                
                self.changeEditingLabel(oldEditedItem, False)
        
        # set the editing mode to true
        self.editing_mode = True
            
        # change the text of the button to indicate the change of function
        self.create_box_button.setText(CREATE_A_BOX_BUTTON_TEXT_INVERSE)
        
        #print(self.box)
        
        # if there is no existing box with that index, create a new box
        if idx is None:
            
            self.current_box = len(self.box)
            
            # set text with two brackets in order to indicate that this box is currently edited
            list_widget = QListWidgetItem(">Box " + str(self.inc_index) + "<")
            list_widget.setTextAlignment(Qt.AlignmentFlag.AlignHCenter)
            self.box_view.addItem(list_widget)
            #Reset all box setting values 
            [setting.setValue(DEFAULT_BOX_VALUE) for setting in self.box_settings]
            
            # create a new box with the current values
            self.box.append(Box(*self.getCurrentBoxSettings(), points = self.points[self.scene_index]))
            
            # if there is a box now then enable the editing inputs
            if self.box : self.enableAllBoxButton(True)
        
        else:
            
            self.current_box = idx
            
            # If there is already a box matchting the index, set the box setting inputs to its dimension
            [setting.setValue(value) for setting, value in zip(self.box_settings, self.box[self.current_box].getCurrentBoxSettings())]
            
            # Add the prefix and suffix
            self.changeEditingLabel(self.box_view.item(self.current_box), True)
        
        self.inc_index += 1
        
        # Set the color of the current box 
        self.box[self.current_box].setColor(EDITING_COLOR)
        
        # Enable the inputs
        self.enableBoxSettings(True)

        # Update the current box position and dimension
        self.updateBox()

    # Change the label by adding or extracting the preffix and suffix
    def changeEditingLabel(self, item : QListWidgetItem, isEdited : bool):
        
        if isEdited:
            # Add prefix and suffix
            item.setText(">" + item.text() + "<")
        
        else:
            # Extract the ">" and "<"
            item.setText(item.text()[1:-1])

    # Enable/disable the visibility of the settings and labels of the box configuration
    def enableBoxSettings(self, enable : bool):
        
        for setting, label in zip(self.box_settings, self.box_settings_label):
            setting.setVisible(enable)
            label.setVisible(enable)      
        
    #Function to create a slider
    def createSlider(self, slider, max, min, func, start_value = None):
        slider.setGeometry(50,50, 200, 50)
        slider.setMinimum(min)
        slider.setMaximum(max)
        slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        slider.setTickInterval(1)
        slider.valueChanged.connect(func)
        
        #If there is a start value then set it as that
        if start_value != None:
            slider.setValue(start_value)
    
    # If the selection of items in the list has changed
    def itemSelectionChanged(self):
        
        changedBoxes = []
        
        # Get all indexes which are selected
        selectedIndexes = [idx.row() for idx in self.box_view.selectedIndexes()]
        
        # Get all indexes which are not selected
        unselectedIndexes = [idx for idx in range(self.box_view.count()) if idx not in selectedIndexes]
        
        #print(f"unselectedIndexes {unselectedIndexes}")
        
        # For each unselected item change its color if necessary
        for idx in unselectedIndexes:
            
            item = self.box[idx]
            
            # if there is no editing mode, every box can be changed, but if not
            # check if the item is the currently edited item. 
            if not self.editing_mode or idx != self.current_box and item.color != NONE_SELECTED_COLOR:
                
                changedBoxes.append(idx)
                
                item.setColor(NONE_SELECTED_COLOR)
        
        #For each selected item
        for idx in selectedIndexes:

            # Get the current item            
            item = self.box[idx]
        
            # if the item is not edited currently set the color
            if not self.editing_mode or idx != self.current_box:
            
                item.setColor(SELECTED_COLOR)
                
                changedBoxes.append(idx)
                
        #print(changedBoxes)

        # Update the box and its points
        self.updateBox(changedBoxes, toMove=False)
        
        # If there are selected Items then enable the function to manage them
        if self.box_view.selectedItems():
            
            self.delete_selected_button.setEnabled(True)
            self.extract_selected_button.setEnabled(True)
            
        else:
            
            self.delete_selected_button.setEnabled(False)
            self.extract_selected_button.setEnabled(False)
        
    # If an item was double clicked
    def itemDoubleClicked(self, listItem : QListWidgetItem):
        
        # If there is already is a box been edited, change only the items
        if self.editing_mode:
            
            self.disableEditingMode(resetEditingMode = False)
        
        # Change the item by passing the index 
        self.enableEditingMode(self.box_view.indexFromItem(listItem).row())
    
    #Function to change scene index and update the current
    def sceneSliderChanged(self):

        # if scene was changed by forward button, don't change twice
        if self.scene_index != self.scene_slider.value():

            #Update the current scene index with the slider's value
            self.changeScene(self.scene_slider.value())
 
    #Function to change the current maximum distance view
    def distanceSliderChanged(self):
        
        #Update the current tolerated distance with slider's value
        self.distance = self.distance_slider.value()
        
        #Update the scene
        self.updatePoints()
        
    #Function to change the z-axis maximum 
    def higherZAxisSliderChanged(self):
        
        #Update maximum z-axis value
        self.higher_z_axis = self.higher_z_axis_slider.value()
        
        #Update the scene
        self.updatePoints()

    #Function to change the z-axis maximum 
    def lowerZAxisSliderChanged(self):
        
        #Update maximum z-axis value
        self.lower_z_axis = self.lower_z_axis_slider.value()
        
        #Update the scene
        self.updatePoints()

    def changeScene(self, new_value: int):

        #If there is a scene after the current one
        if(new_value <= self.max_scence_index and new_value >= 0): 

            self.removeBoundingBox(scene_idx=self.scene_index)

            #Increase the current scene by one
            self.scene_index = new_value

            #set new texts
            self.scene_slider_label.setText(f"current scene: {self.scene_index}")

            #Sync current scene with the slider
            self.scene_slider.setValue(self.scene_index)
            
            #Update the current scene
            self.updatePoints()

            # if buttons are not loaded yet, skip the if condition
            if hasattr(self, 'forward_button'):
            
                #After using this button and the forward one is disabled, enable it again
                if( self.scene_index < self.max_scence_index and not self.forward_button.isEnabled()):
                    
                    #Enable the forward button
                    self.forward_button.setEnabled(True)
                    
                #After changing the scene enable the backward button
                elif(self.scene_index > 0 and not self.backward_button.isEnabled()):
                    
                    #Enable the backward button
                    self.backward_button.setEnabled(True)   
                    
                #If there is no scene before the current one
                elif(self.scene_index == 0):
                    
                    #Disable the backward button
                    self.backward_button.setEnabled(False)
                
                #If the maximum is reached then disable the forward button
                elif(self.scene_index == self.max_scence_index):
                    
                    #Disable the forward button
                    self.forward_button.setEnabled(False) 

                #Load both images and scale the images relatively to the window size down
                self.image_left.setPixmap(QPixmap(os.path.join(IMAGE_LEFT_PATH, self.images_left[self.scene_index])).scaled(int(self.width()/2.1),int(self.height()/3), aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio))
                self.image_right.setPixmap(QPixmap(os.path.join(IMAGE_RIGHT_PATH, self.images_right[self.scene_index])).scaled(int(self.width()/2.1),int(self.height()/3), aspectRatioMode=Qt.AspectRatioMode.KeepAspectRatio))
        
                # Update all boxes with their points
                self.updateBox(changedBoxes = list(range(len(self.box))))

                # Update all bounding boxes
                self.addBoundingBoxes()
           
    #Increase the scene index
    def forward(self):

        self.changeScene(self.scene_index + 1)    
    
    #Function for backward button in order to decrease the scene index   
    def backward(self):
    
        self.changeScene(self.scene_index - 1)
        
    #Function to load and show both images and the point clouds
    def updatePoints(self):
            
        #Clear all elements in the coordinate system
        
        if self.current_points is not None:
            self.view.removeItem(self.current_points)
        
        #Select the point cloud of the current scene
        points = self.points[self.scene_index].copy()

        #Select all points which are located beneath the maximum z-axis and closer to the origin than the maximum distance
        points = points[points[DISTANCE] <= self.distance]
        points = points[points[Z_AXIS] <= self.higher_z_axis/SCALE_FACTOR]
        points = points[self.lower_z_axis/SCALE_FACTOR <= points[Z_AXIS]]
        
        #Extract the x,y,z coordinates
        pc_xyz = points[[X_AXIS, Y_AXIS, Z_AXIS]]
        
        #Extract the color value and convert to RGB
        pc_colors = inteToRGB(points[[COLOR]])
        
        #Create a scatter plot item with the coordinates and colors
        self.current_points = gl.GLScatterPlotItem(pos=pc_xyz, size=2, color=pc_colors)
        
        #Add the scatter plot to the three dimensional coordinate system object
        self.view.addItem(self.current_points)

    # remove a box from the 3D-Plot and from the list of boxes
    def removeBox(self, idx : int, deleteObject : bool = True) -> None:
        
        item : Box = self.box[idx]
        
        # Get all lines of a box
        [self.view.removeItem(line) for line in item.getLines()] 
        
        # If there are points within the box
        if item.getPoints() is not None:
            
            # Remove from the 3D-Plot
            self.view.removeItem(item.getPoints())
        
        # If the box needs to be deleted as well
        if deleteObject : 
            self.box.remove(item)
        
        # If there no boxes anymore disable the buttons which controls the boxes
        if deleteObject and not self.box : self.enableAllBoxButton(False)
    
    # remove bounding box
    def removeBoundingBox(self, scene_idx: int = None, start_thold: float = None, end_thold: float = None) -> None:

        print(f"remove boxes of scene {scene_idx} with current scene {self.scene_index} of model {self.current_model}")

        if scene_idx is not None:
            for bbox in self.models_bounding_boxes[self.current_model][scene_idx]:

                if bbox.confidence >= self.thold:
                
                    [self.view.removeItem(line) for line in bbox.bounding_box]
                    self.view.removeItem(bbox.score_3d)
                    self.view.removeItem(bbox.class_3d)

        elif start_thold is not None:

            for bbox in self.models_bounding_boxes[self.current_model][self.scene_index]:

                if bbox.confidence > start_thold and bbox.confidence < end_thold:

                    try:
                        [self.view.removeItem(line) for line in bbox.bounding_box]
                        self.view.removeItem(bbox.score_3d)
                        self.view.removeItem(bbox.class_3d)
                    except ValueError as e:
                        print(e, f"\nConfidene: {bbox.confidence}")

    # function to update the shape, color and position of all changed boxes
    def updateBox(self, changedBoxes : list[int] = [], toMove : bool = True) -> None:
        
        # if there is a box in the list
        if self.box:
            
            # the current box will only be added if:
            # 1. The editing mode is on
            # 2. The current box is not deleted yet
            # 3. The current box is not already located in the changed boxes
            changedBoxes = (changedBoxes + [self.current_box]) if self.editing_mode and self.current_box < len(self.box) and self.current_box not in changedBoxes else changedBoxes
            
            #print(f"changedboxes : {changedBoxes}")
            
            # iterate through all indexes of boxes to be updated
            for idx in changedBoxes:
            
                item : Box = self.box[idx]
            
                # try to remove the box from the view if there is one
                try:
                    self.removeBox(idx, False)
                # if there is no box then pass
                except:
                    pass
                
                if toMove and idx == self.current_box:
                    # move the box to the current settings
                    item.assignVariables(*self.getCurrentBoxSettings())
                    
                item.updateAndDisplay(self.points[self.scene_index])

                # add the box to the view
                [self.view.addItem(line) for line in item.getLines()]
                
                # Get the points which are located within the item
                points_to_draw = item.getPoints()
                
                if(points_to_draw is not None):
                    # add the points to the view
                    self.view.addItem(points_to_draw)
    
    def addBoundingBoxes(self, start_thold: float = None, end_thold: float = 1) -> None:

        if start_thold is None: start_thold = self.thold

        for bbox in self.models_bounding_boxes[self.current_model][self.scene_index]:

            if bbox.confidence >= start_thold and bbox.confidence <= end_thold:

                [self.view.addItem(line) for line in bbox.bounding_box]
                self.view.addItem(bbox.score_3d)
                self.view.addItem(bbox.class_3d)

    # function to get the current box settings
    def getCurrentBoxSettings(self) -> list[float]:
        
        # return the current value of the widgets
        return [box.value() for box in self.box_settings]
    
    # Function to read all point clouds available
    def readAllPointClouds(self):
        
        # Initialize the maximum 
        self.max_z_axis = 0
        self.min_z_axis = 0
        
        self.max_distance = 0
        
        # List, in which the point clouds are loaded
        self.points : list[pd.DataFrame] = []

        # Search for every bin file in Folder and sort them 
        # since they are not ordered alphabetically
        self.point_cloud_files = sorted([os.path.join(POINTCLOUD_PATH, file) for file in os.listdir(POINTCLOUD_PATH) if ".bin" in file])[:AMOUNT_OF_SCENES]
        
        # Load only a specific amount of files to not reduce perfomance during application start
        for file in self.point_cloud_files:
            
            print(file)            
            
            # Load each point cloud as a pandas dataframe to increase usability
            # Since thhose bin-files contain a column next to the x,y,z and color columns which ist not necessary, so it will be dropped
            points = pd.DataFrame(np.fromfile(file, dtype=np.float32).reshape((-1, 5))[:, :4])
            
            # Remove the outliers from the z-axis
            points = points[(np.abs(stats.zscore(points[Z_AXIS])) < 3)]
            
            # Calculate the distance to the LIDAR-system
            points[DISTANCE] = points.agg(np.linalg.norm,axis="columns").astype(int)
            
            # Get the minimum and maximum z-axis value to determine the scale
            max_z_axis = int((max(points[Z_AXIS]))*SCALE_FACTOR)
            min_z_axis = int((min(points[Z_AXIS]))*SCALE_FACTOR)
            
            # Get the max distance relatively to the origin
            max_distance = int(points[DISTANCE].max())
            
            # If this point cloud has got a bigger extremum it will be defined as the global maximum oder minimum
            if(self.max_z_axis < max_z_axis):
                self.max_z_axis = max_z_axis
                
            if(self.min_z_axis > min_z_axis):
                self.min_z_axis = min_z_axis  
            
            if(self.max_distance < max_distance):
                self.max_distance = max_distance

            # Add the point cloud to the object
            self.points.append(points)

    # Function to read all files from the image files
    def readImageFiles(self):

        
        # Save all files, which contains the extension .png
        self.images_right = sorted([file for file in os.listdir(IMAGE_LEFT_PATH) if ".png" in file])
        self.images_left = sorted([file for file in os.listdir(IMAGE_RIGHT_PATH) if ".png" in file])

    # Function to load model and produce bounding boxes on pointcloud
    def evalBoxesByModel(self, model: str):

        cfg_file = self.config_files[model]
        ckpt_file = self.checkpoint_files[model]

        logger = common_utils.create_logger()

        cfg_from_yaml_file(cfg_file, cfg)

        self.demo_dataset = demo.DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            root_path=None, ext='.bin', logger=logger, data_file_list=self.point_cloud_files
        )

        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.demo_dataset)
        self.model.load_params_from_file(filename=ckpt_file, logger=logger, to_cpu=True)
        self.model.cuda()
        self.model.eval()

        self.models_bounding_boxes[model] = []

        with torch.no_grad():
            for idx, data_dict in enumerate(self.demo_dataset):
                logger.info(f'Visualized sample index: \t{idx + 1}')
                data_dict = self.demo_dataset.collate_batch([data_dict])
                load_data_to_gpu(data_dict)

                pred_dicts, _ = self.model.forward(data_dict)

                pred_dicts  =   pred_dicts[0]
                ref_boxes   =   pred_dicts['pred_boxes']
                ref_scores  =   pred_dicts['pred_scores']
                ref_labels  =   pred_dicts['pred_labels']

                if ref_boxes is not None and not isinstance(ref_boxes, np.ndarray):
                    ref_boxes = ref_boxes.cpu().numpy()
                #if gt_boxes is not None and not isinstance(gt_boxes, np.ndarray):
                #    gt_boxes = gt_boxes.cpu().numpy()
                if ref_scores is not None and not isinstance(ref_scores, np.ndarray):
                    ref_scores = ref_scores.cpu().numpy()
                if ref_labels is not None and not isinstance(ref_labels, np.ndarray):
                    ref_labels = ref_labels.cpu().numpy()

                # if thold != 0:
                #     i = 0

                #     while True:

                #         if ref_scores[i] < thold:
                            
                #             ref_boxes = np.delete(ref_boxes,i, 0)
                #             ref_scores = np.delete(ref_scores,i)
                #             ref_labels = np.delete(ref_labels,i)
                            
                #             i -= 1

                #         i += 1

                #         if i == len(ref_boxes):
                #             break
                
                #if gt_boxes is not None:
                #    vis = draw_box(vis, gt_boxes, (0, 0, 1))

                if ref_boxes is not None:
                    self.models_bounding_boxes[model].append([BoundingBox(*boundingBox) for boundingBox in get_box(ref_boxes, (0, 1., 0, 1.), ref_labels, ref_scores)])

    # Check for all available models and provide them to the user
    def searchAvailModels(self):

        self.config_files = {}
        self.checkpoint_files = {}

        for file_dir in os.listdir(CKPT_FILES):
            if self.current_model is None: self.current_model = file_dir
            self.config_files[file_dir] = CFG_FILES + file_dir + ".yaml"
            self.checkpoint_files[file_dir] = CKPT_FILES + file_dir + CKPT_FILE_EXT
            try:
                self.evalBoxesByModel(file_dir)
                self.models_selection.addItem(file_dir)
            except KeyError as e:
                print(e)
            
    # model has changed
    def modelChanged(self, value: str):

        print(f"change model to {value}")

        # remove all boundingboxes of the prior model
        self.removeBoundingBox(self.scene_index)

        # update the model
        self.current_model = value

        # add new bounding boxes
        self.addBoundingBoxes()

#Main application which executes when this file is directly runned
if __name__ == "__main__":
    
    #Create an application object
    app = QApplication(sys.argv)
    
    #Create a window
    window = Window()
    
    #And show the window then
    window.show()
    
    #If the window closes then stop the process
    sys.exit(app.exec())