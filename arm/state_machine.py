"""!
The state machine that implements the logic.
"""
from PyQt4.QtCore import (QThread, Qt, pyqtSignal, pyqtSlot, QTimer)
import time
import numpy as np
from color_detection import BlockDetector
from rxarm import RXArm
import rospy
import cv2
from apriltag_ros.msg import AprilTagDetectionArray
from kinematics import IK_geometric, IK_numerical
from listeners import AprilTagListener
from copy import deepcopy
import matplotlib.pyplot as plt
import time

class StateMachine():
    """!
    @brief      This class describes a state machine.

                TODO: Add states and state functions to this class to implement all of the required logic for the armlab
    """

    def __init__(self, rxarm, camera):
        """!
        @brief      Constructs a new instance.

        @param      rxarm   The rxarm
        @param      planner  The planner
        @param      camera   The camera
        """
        self.rxarm = rxarm
        self.camera = camera
        # self.VideoThread = VideoThread(self.camera)
        # self.VideoThread.updateFrame.connect(self.setImage)
        # self.VideoThread.start()
        self.status_message = "State: Idle"
        self.current_state = "idle"
        self.next_state = "idle"
        self.april_tag_listener = AprilTagListener()
        self.april_tag_object_points = np.array([[-250, -25, 0], #LL
                    [250, -25, 0], #LR
                    [250, 275, 0], #UR
                    [-250, 275, 0], #UL
                    [0, 425, 0], #UM
                    [0, 175, 0]], dtype=np.float32) #MM
        self.D = np.array([0.1536298245191574, -0.4935448169708252, -0.0008808146812953055, 0.0008218809380196035, 0.4401721954345703])
        self.waypoints = [
            [-np.pi/2,       -0.5,      -0.3,            0.0,       0.0],
            [0.75*-np.pi/2,   0.5,      0.3,      0.0,       np.pi/2],
            [0.5*-np.pi/2,   -0.5,     -0.3,     np.pi / 2,     0.0],
            [0.25*-np.pi/2,   0.5,     0.3,     0.0,       np.pi/2],
            [0.0,             0.0,      0.0,         0.0,     0.0],
            [0.25*np.pi/2,   -0.5,      -0.3,      0.0,       np.pi/2],
            [0.5*np.pi/2,     0.5,     0.3,     np.pi / 2,     0.0],
            [0.75*np.pi/2,   -0.5,     -0.3,     0.0,       np.pi/2],
            [np.pi/2,         0.5,     0.3,      0.0,     0.0],
            [0.0,             0.0,     0.0,      0.0,     0.0]]
        self.recorded_position = []
        self.IK = IK_geometric
        self.initial_high = np.array([0, -33.75 * np.pi/180, 12.8 * np.pi/180, -120 * np.pi/180, 0])


    def calcMoveAccelTime(self, current_pos, goal):
        displacement = np.abs(goal - current_pos)
        max_displacement_ind = np.argmax(displacement)
        max_displacement = displacement[max_displacement_ind]
        max_velocity = 2 * np.pi * 10/60
        t = (5 * max_displacement)/(3 * max_velocity)
        return t, 2*t/5

    def sign(self, number):
        if number >= 0:
            return 1
        else:
            return -1

    def angle_mod(self, theta):
        if theta >= np.pi:
            theta -= np.pi

        if theta > 0:
            while theta > np.pi/4:
                theta -= np.pi/2
        else:
            while theta < -np.pi/4:
                theta += np.pi/2
        return theta

    def moveArm(self, move_time, accel_time, location):
        self.rxarm.set_moving_time(move_time)
        self.rxarm.set_accel_time(accel_time)
        location_new = deepcopy(location)
        location_new[1] -= 2.3 * np.pi/180
        # location_new[2] += 1.7 * np.pi/180
        location_new[0] += (3.1 + 0 * -self.sign(location_new[0])) * np.pi/180
        # if location[0] > 0:
        #     location_new[0] += 1 * np.pi/180
        # location_new[0] += (3.4 + location_new[0]) * np.pi/180
        self.rxarm.set_positions(location_new)
        rospy.sleep(move_time)
        self.rxarm.set_moving_time(move_time)
        self.rxarm.set_accel_time(accel_time)

    def navToGoal(self, goal):
        current_pos = self.rxarm.get_positions()
        move_time, accel_time = self.calcMoveAccelTime(current_pos, goal)
        self.moveArm(move_time, accel_time, goal)

    def motion_plan(self, pickup, block_orientation, dropoff, dropoff_fixed_orientation=None, dropoff_base_orientation=False):
        pickup_original_angles = IK_numerical(np.array([pickup[0], pickup[1], pickup[2]]), self.rxarm)
        pickup_original_angles[-1] = self.angle_mod(block_orientation + pickup_original_angles[0])
        pickup_high = np.array([pickup[0], pickup[1], 319])
        pickup_high_angles = IK_numerical(pickup_high, self.rxarm)
        pickup_high_angles[-1] = self.angle_mod(block_orientation + pickup_high_angles[0])

        dropoff_original_angles = IK_numerical(np.array([dropoff[0], dropoff[1], dropoff[2]]), self.rxarm)
        dropoff_high = np.array([dropoff[0], dropoff[1], 319])
        dropoff_high_angles = IK_numerical(dropoff_high, self.rxarm)
        dropoff_block_orientation = self.angle_mod(block_orientation + dropoff_high_angles[0])
        if dropoff_base_orientation:
            dropoff_block_orientation = 0
        elif dropoff_fixed_orientation is not None:
            dropoff_block_orientation = (dropoff_fixed_orientation + dropoff_high_angles[0] + (3.1 + 0 * -self.sign(dropoff_high_angles[0])) * np.pi/180) % np.pi
        dropoff_orientation = dropoff_block_orientation
        dropoff_original_angles[-1] = dropoff_orientation
        dropoff_high_angles[-1] = dropoff_orientation

        # current_angle = self.rxarm.get_positions()
        # if self.sign(current_angle[0]) != self.sign(pickup_high_angles[0]):
        #     self.navToGoal(self.initial_high)
        print('going to pickup:', pickup_high_angles)
        self.navToGoal(pickup_high_angles)
        print('at pickup:', pickup_high_angles - self.rxarm.get_positions())
        # print('target - actual', ))
        print('descending:', pickup_original_angles)
        self.navToGoal(pickup_original_angles)
        print('descended:', pickup_original_angles - self.rxarm.get_positions())
        self.rxarm.close_gripper()
        print('ascending:', pickup_high_angles)
        self.navToGoal(pickup_high_angles)
        print('ascended:', pickup_high_angles - self.rxarm.get_positions())
        pickup_initial_high = self.initial_high.copy()
        pickup_initial_high[0] = pickup_high_angles[0]
        print('reaching back to initial:', pickup_initial_high)
        self.navToGoal(pickup_initial_high)
        print('back to initial with base modification:', pickup_initial_high - self.rxarm.get_positions())
        # current_angle = self.rxarm.get_positions()
        # if self.sign(current_angle[0]) != self.sign(dropoff_high_angles[0]):
        #     self.navToGoal(self.initial_high)
        print('moving to dropoff high:', dropoff_high_angles)
        self.navToGoal(dropoff_high_angles)
        print('moved to dropoff high:', dropoff_high_angles - self.rxarm.get_positions())
        print('moving to dropoff initial:', dropoff_original_angles)
        self.navToGoal(dropoff_original_angles)
        print('moved to dropoff initial:', dropoff_original_angles - self.rxarm.get_positions())
        self.rxarm.open_gripper()
        print('moving to dropoff high:', dropoff_high_angles)
        self.navToGoal(dropoff_high_angles)
        print('moved to dropoff high:', dropoff_high_angles - self.rxarm.get_positions())
        dropoff_initial_high = self.initial_high.copy()
        dropoff_initial_high[0] = dropoff_high_angles[0]
        print('reaching back to initial:', dropoff_initial_high)
        self.navToGoal(dropoff_initial_high)
        print('reached back to initial:', dropoff_initial_high - self.rxarm.get_positions())

    def set_next_state(self, state, data=None):
        """!
        @brief      Sets the next state.

            This is in a different thread than run so we do nothing here and let run handle it on the next iteration.

        @param      state  a string representing the next state.
        """
        self.next_state = state
        self.data = data

    def run(self):
        """!
        @brief      Run the logic for the next state

                    This is run in its own thread.

                    TODO: Add states and funcitons as needed.
        """
        if self.next_state == "initialize_rxarm":
            self.initialize_rxarm()

        if self.next_state == "idle":
            self.idle()

        if self.next_state == "estop":
            self.estop()

        if self.next_state == "execute":
            self.execute()

        if self.next_state == "calibrate":
            self.calibrate()

        if self.next_state == "detect":
            self.detect()

        if self.next_state == "manual":
            self.manual()

        if self.next_state == "teach":
            self.teach()

        if self.next_state == "repeat":
            self.repeat()

        if self.next_state == 'record_waypoint':
            self.record_waypoint()

        if self.next_state == 'record_open_gripper':
            self.record_open_gripper()

        if self.next_state == 'record_close_gripper':
            self.record_close_gripper()

        if self.next_state == 'task_1':
            self.task_1()

        if self.next_state == 'task_2':
            self.task_2()

        if self.next_state == 'task_3':
            self.task_3()

        if self.next_state == 'task_4':
            self.task_4()

        if self.next_state == 'pick':
            self.pick()

        if self.next_state == 'place':
            self.place()

    """Functions run for each state"""

    def pick(self):
        self.camera.block_Detector.detect_blocks(self.camera.VideoFrame, self.camera.DepthFrameRaw)

        self.status_message = "Picking block"
        self.current_state = 'pick'
        self.next_state = 'idle'
        pose = self.data
        block_centroid, block_centroid_dist = self.camera.block_Detector.nearest_centroid_world_coordinate(pose[:3])
        print('pose_original')
        pose_original_angles = IK_numerical(np.array([block_centroid[0], block_centroid[1], block_centroid[2]]))
        pose_high = np.array([block_centroid[0], block_centroid[1], 319])
        print('pose_high')
        pose_high_angles = IK_numerical(pose_high)
        print('before processing:', pose_high_angles)

        block_orientation = self.camera.block_Detector.orientations_dict[(block_centroid[0], block_centroid[1], block_centroid[2])]
        pose_original_angles[-1] = self.angle_mod(block_orientation + pose_original_angles[0])
        pose_high_angles[-1] = self.angle_mod(block_orientation + pose_high_angles[0])
        print('after processing:', pose_high_angles)

        print('going to pickup')

        # current_angle = self.rxarm.get_positions()
        # if self.sign(current_angle[0]) != self.sign(pose_high_angles[0]):
        #     self.navToGoal(self.initial_high)
        self.navToGoal(pose_high_angles)
        print('at pickup')
        print('motor angles:', self.rxarm.get_positions())
        print('target - actual', pose_high_angles - self.rxarm.get_positions())
        print(pose_high_angles)
        print(block_orientation)
        print(block_orientation + pose_high_angles[0])
        self.navToGoal(pose_original_angles)

        self.rxarm.close_gripper()

        self.navToGoal(pose_high_angles)
        pose_initial_high = self.initial_high.copy()
        pose_initial_high[0] = pose_high_angles[0]
        self.navToGoal(pose_initial_high)

    def place(self):
        self.camera.block_Detector.detect_blocks(self.camera.VideoFrame, self.camera.DepthFrameRaw)

        self.status_message = "place block"
        self.current_state = 'place'
        self.next_state = 'idle'
        pose = self.data
        pose[2] += 20
        pose_original_angles = IK_numerical(np.array([pose[0], pose[1], pose[2]]))
        pose_high = np.array([pose[0], pose[1], 319])
        pose_high_angles = IK_numerical(pose_high)
        pose_high_angles[-1] = pose_high_angles[0]
        pose_original_angles[-1] = pose_original_angles[0]

        current_angle = self.rxarm.get_positions()
        if self.sign(current_angle[0]) != self.sign(pose_high_angles[0]):
            self.navToGoal(self.initial_high)

        self.navToGoal(pose_high_angles)
        self.navToGoal(pose_original_angles)

        self.rxarm.open_gripper()

        self.navToGoal(pose_high_angles)
        pose_initial_high = self.initial_high.copy()
        pose_initial_high[0] = pose_high_angles[0]
        self.navToGoal(pose_initial_high)

    def manual(self):
        """!
        @brief      Manually control the rxarm
        """
        self.status_message = "State: Manual - Use sliders to control arm"
        self.current_state = "manual"

    def idle(self):
        """!
        @brief      Do nothing
        """
        self.status_message = "State: Idle - Waiting for input"
        self.current_state = "idle"

    def estop(self):
        """!
        @brief      Emergency stop disable torque.
        """
        self.status_message = "EMERGENCY STOP - Check rxarm and restart program"
        self.current_state = "estop"
        self.rxarm.disable_torque()

    def execute(self):
        """!
        @brief      Go through all waypoints
        TODO: Implement this function to execute a waypoint plan
              Make sure you respect estop signal
        """
        self.status_message = "State: Execute - Executing motion plan"

        for wp in self.waypoints:
            print("WAYPOINT:", wp)
            self.rxarm.set_positions(wp)
            rospy.sleep(5)

        self.next_state = "idle"

    def calibrate(self):
        """!
        @brief      Gets the user input to perform the calibration
        """
        self.current_state = "calibrate"
        self.next_state = "idle"

        """TODO Perform camera calibration routine here"""
        image_points = []
        image_points_with_z = []
        correct_apriltag_z = [.983, .970, .970, .979, .972, .974]
        for i, detection in enumerate(self.april_tag_listener.apriltag_data.detections):
            image_points.append((detection.pose.pose.pose.position.x, detection.pose.pose.pose.position.y))
            image_points_with_z.append((detection.pose.pose.pose.position.x, detection.pose.pose.pose.position.y, correct_apriltag_z[i]))



        success, rotation_vector, translation_vector = cv2.solvePnP(self.april_tag_object_points, np.array(image_points), np.eye(3, dtype=np.float32), self.D, flags=0)
        rotation_matrix = None
        rotation_matrix = cv2.Rodrigues(rotation_vector, rotation_matrix)[0]

        print('Rotation:')
        print(rotation_matrix)

        print('Translation:')
        print(translation_vector)

        self.camera.extrinsic_matrix[:3, :3] = rotation_matrix
        self.camera.extrinsic_matrix[:3, -1] = translation_vector.flatten()
        self.camera.extrinsic_matrix[2, 3] = 982 #Using manually measured z
        print(self.camera.extrinsic_matrix)

        homogenous_camera_coords = np.concatenate((np.array(image_points_with_z)*1000, np.ones((6, 1), dtype=np.float32)), axis=1)
        print('homogenous camera')
        print(homogenous_camera_coords)
        print()
        world_coords_projection = np.matmul(np.linalg.inv(self.camera.extrinsic_matrix), homogenous_camera_coords.T)
        print('World coordinate projection')
        print(world_coords_projection)
        print()

        reprojection_error_mm = np.linalg.norm(world_coords_projection[:3, :].T-self.april_tag_object_points, axis=1).mean()

        print("Reprojection Error: ", reprojection_error_mm)

        self.status_message = "Calibration - Completed Calibration"
        self.camera.block_Detector.intrinsic = self.camera.intrinsic_matrix
        self.camera.block_Detector.extrinsic = self.camera.extrinsic_matrix
        # while True:
        #     try:
        self.camera.block_Detector.detect_blocks(self.camera.VideoFrame, self.camera.DepthFrameRaw)
            #     break
            # except:
            #     pass

    """ TODO """
    def detect(self):
        """!
        @brief      Detect the blocks
        """


        rospy.sleep(1)

    def task_1_2(self, dropoff_locations):
        not_all_clear = True
        while not_all_clear:
            not_all_clear = False
            self.camera.block_Detector.detect_blocks(self.camera.VideoFrame, self.camera.DepthFrameRaw)
            for centroid, block_orientation in self.camera.block_Detector.orientations_dict.items():
                if centroid[1] > 0:
                    dropoff_location = dropoff_locations.pop(0)
                    self.motion_plan(centroid, block_orientation, dropoff_location, dropoff_fixed_orientation=np.pi/2)
                    not_all_clear = True
                    break


    def task_2_helper(self, dropoff_locations, size):
        print('in task 2 helper')

        not_all_clear = True
        while not_all_clear:
            not_all_clear = False
            self.camera.block_Detector.detect_blocks(self.camera.VideoFrame, self.camera.DepthFrameRaw)

            size_filtered_centroids = [(centroid, block_orientation) for centroid, block_orientation in self.camera.block_Detector.orientations_dict.items() if self.camera.block_Detector.block_sizes_dict[centroid] == size]
            for centroid, block_orientation in size_filtered_centroids:
                if centroid[1] > 0:
                    dropoff_location = dropoff_locations.pop(0)
                    self.motion_plan(centroid, block_orientation, dropoff_location, dropoff_fixed_orientation=np.pi/2)
                    not_all_clear = True
                    break

    def task_1(self):
        """!
        @brief      Perform task 1
        """
        self.current_state = "task_1"
        self.status_message = "Performing task 1"
        self.next_state = "idle"
        dropoff_locations_small = [np.array([-400, -75, 20]), np.array([-325, -75, 20]), np.array([-250, -75, 20]), np.array([-200, -75, 20]),
                                   np.array([-400, -125, 20]), np.array([-325, -125, 20]), np.array([-250, -125, 20]), np.array([-200, -125, 20]),
                                   np.array([-400, -25, 20])]
        dropoff_locations_big = [np.array([400, -75, 20]), np.array([325, -75, 20]), np.array([250, -75, 20]), np.array([200, -75, 20]),
                                   np.array([400, -125, 20]), np.array([325, -125, 20]), np.array([250, -125, 20]), np.array([200, -125, 20]),
                                   np.array([400, -25, 20])]
        try:
            self.task_2_helper(dropoff_locations_small, 'Small')
            self.task_2_helper(dropoff_locations_big, 'Large')
        except:
            return

    def task_2(self):
        """!
        @brief      Perform task 2
        """
        print('task 2')
        self.current_state = "task_2"
        self.status_message = "Performing task 2"
        self.next_state = "idle"
        # dropoff_locations_big = [np.array([-250, -75, 20]), np.array([-250, -75, 60]), np.array([-250, -75, 100])]
        # dropoff_locations_small = [np.array([325, -75, 15]), np.array([325, -75, 40]), np.array([325, -75, 65])]

        dropoff_locations_small = [np.array([-300, -75, 15]), np.array([-300, -75, 55]), np.array([-300, -75, 95]),
                                   np.array([-250, -75, 15]), np.array([-250, -75, 55]), np.array([-250, -75, 95]),
                                   np.array([-400, -75, 15]), np.array([-400, -75, 55]), np.array([-400, -75, 95]),

                                   ]
        dropoff_locations_big = [np.array([325, -75, 20]), np.array([325, -75, 60]), np.array([325, -75, 100]),
                                   np.array([250, -75, 20]), np.array([250, -75, 60]), np.array([250, -75, 100]),
                                   np.array([400, -75, 20]), np.array([400, -75, 60]), np.array([400, -75, 100]),
                                   ]
        # try:
        self.task_2_helper(dropoff_locations_small, 'Small')
        self.task_2_helper(dropoff_locations_big, 'Large')
        # except:
        #     return

    def break_stack(self, dropoff_locations, size='Large'):
        # empty_dropoff_loc = [np.array([-350, 0, 15]), np.array([-350, -75, 15]), np.array([-350, 75, 15]),
        #                      np.array([350, 0, 15]), np.array([350, -75, 15]), np.array([350, 75, 15])]

        if size == 'Large':
            z_offset = 20
        else:
            z_offset = 12.5

        for centroid in self.camera.block_Detector.orientations_dict:
            if centroid[-1] >= 50:
                in_line = True
                while in_line:
                    drop_x = np.random.uniform(-300, 300)
                    drop_y = np.random.uniform(175, 300)
                    random_drop_loc = np.array([drop_x, drop_y, z_offset])
                    in_line = False
                    for drop_loc in dropoff_locations:
                        if abs(drop_loc[0] - drop_x) < 60 and abs(drop_loc[1] - drop_y) < 60:
                            in_line = True
                            break
                    for center_ in self.camera.block_Detector.colors_dict:
                        if in_line:
                            break
                        else:
                            if abs(center_[0] - drop_x) < 60 and abs(center_[1] - drop_y) < 60:
                                in_line = True
                                break

                block_orientation = self.camera.block_Detector.orientations_dict[centroid]
                self.motion_plan(centroid, block_orientation, random_drop_loc)


    def task_3(self):
        """!
        @brief      Perform task 3
        """
        self.current_state = "task_3"
        self.status_message = "Performing task 3"
        self.next_state = "idle"
        self.camera.block_Detector.detect_blocks(self.camera.VideoFrame, self.camera.DepthFrameRaw)
        offset = 50
        dropoff_locations = []
        dropoff_angles = []
        while True:
            start_location_x = np.random.uniform(-300, 300)
            start_location_y = np.random.uniform(175, 300)
            nearest_centroid, nearest_centroid_distance = self.camera.block_Detector.nearest_centroid_world_coordinate_no_z(np.array([start_location_x, start_location_y, 20]))
            if nearest_centroid_distance < offset + 5:
                continue
            for direction, dropoff_angle in [(np.array([1, 0, 0]), np.pi/2), (np.array([-1, 0, 0]), np.pi/2)]:
                dropoff_locations_temp = []
                max_min_x = start_location_x + 5 * offset * direction[0] + 25 * direction[0]
                max_min_y = start_location_y + 5 * offset * direction[1] + 25 * direction[1]
                if max_min_x >= 425 or max_min_x <= -425 or max_min_y >= 400 or max_min_y <= -100:
                    continue
                dropoff_locations_temp.append(np.array([start_location_x, start_location_y, 20]))
                dropoff_angles.append(dropoff_angle)
                while len(dropoff_locations_temp) < 6:
                    this_iter_block = np.array([start_location_x + direction[0] * offset * len(dropoff_locations_temp), start_location_y + direction[1] * offset * len(dropoff_locations_temp), 20])
                    nearest_centroid, nearest_centroid_distance = self.camera.block_Detector.nearest_centroid_world_coordinate_no_z(this_iter_block)
                    if nearest_centroid_distance < offset + 10:
                        break
                    dropoff_locations_temp.append(this_iter_block)
                    dropoff_angles.append(dropoff_angle)
                if len(dropoff_locations_temp) == 6:
                    break
            if len(dropoff_locations_temp) == 6:
                break
        dropoff_locations = dropoff_locations_temp
        print("Available Dropoff Locations : ", dropoff_locations)

        for color in ['red', 'orange', 'yellow','green', 'blue', 'purple']:
            found_color = False
            while not found_color:
                self.camera.block_Detector.detect_blocks(self.camera.VideoFrame, self.camera.DepthFrameRaw)

                print('beginning of color loop:', color)
                for centroid, color_centroid in self.camera.block_Detector.colors_dict.items():
                    if color_centroid == color:
                        found_color = True
                        break
                print('found_color:', found_color)
                print()
                if found_color:
                    block_orientation = self.camera.block_Detector.orientations_dict[centroid]
                    self.motion_plan(centroid, block_orientation, dropoff_locations.pop(0),  dropoff_angles.pop(0))
                else:
                    self.break_stack(dropoff_locations)


    def task_4(self):
        """!
        @brief      Perform task 4
        """
        self.current_state = "task_4"
        self.status_message = "Performing task 4"
        self.next_state = "idle"
        self.camera.block_Detector.detect_blocks(self.camera.VideoFrame, self.camera.DepthFrameRaw)
        dropoff_locations = []
        offset = 100
        while True:
            start_location_x = np.random.uniform(-200, 200)
            start_location_y = np.random.uniform(150, 250)
            nearest_centroid, nearest_centroid_distance = self.camera.block_Detector.nearest_centroid_world_coordinate_no_z(np.array([start_location_x, start_location_y, 20]))
            if nearest_centroid_distance < offset:
                continue
            else:
                break
        dropoff_location = np.array([start_location_x, start_location_y, 20])
        dropoff_locations.append(dropoff_location)
        print("Available Dropoff Locations : ", dropoff_locations)

        height_multiplier = 0
        for color in ['red', 'orange', 'yellow','green', 'blue', 'purple']:
            found_color = False
            dropoff_location = np.array([start_location_x, start_location_y, 20 + 40 * height_multiplier])
            while not found_color:
                self.camera.block_Detector.detect_blocks(self.camera.VideoFrame, self.camera.DepthFrameRaw)
                for centroid, color_centroid in self.camera.block_Detector.colors_dict.items():
                    if color_centroid == color:
                        found_color = True
                        break
                if found_color:
                    block_orientation = self.camera.block_Detector.orientations_dict[centroid]
                    self.motion_plan(centroid, block_orientation, dropoff_location, dropoff_base_orientation=True)
                else:
                    self.break_stack(dropoff_locations)

            height_multiplier += 1



    def initialize_rxarm(self):
        """!
        @brief      Initializes the rxarm.
        """
        self.current_state = "initialize_rxarm"
        self.status_message = "RXArm Initialized!"
        if not self.rxarm.initialize():
            print('Failed to initialize the rxarm')
            self.status_message = "State: Failed to initialize the rxarm!"
            rospy.sleep(5)
        self.navToGoal(self.initial_high)
        self.next_state = "idle"

    def teach(self):
        self.current_state = "teach"
        self.recorded_position = []
        self.rxarm.disable_torque()
        self.next_state = "idle"
        self.status_message = "State: Teach - Manually move the robot to a desired position and click Record Waypoint button"

    def repeat(self):
        self.current_state = "repeat"
        self.status_message = "State: Repeat - execute recorded motion plan"
        self.rxarm.enable_torque()

        joint_angles = []
        times = []
        joint_angles.append(self.rxarm.get_positions())
        times.append(time.time())
        i = 0
        for wp in self.recorded_position:
            print("WAYPOINT:", wp)

            if wp == 'Open Gripper':
                self.rxarm.open_gripper()
            elif wp == 'Close Gripper':
                self.rxarm.close_gripper()
            else:
                current_pos = self.rxarm.get_positions()
                displacement = np.abs(wp - current_pos)
                max_displacement_ind = np.argmax(displacement)
                max_displacement = displacement[max_displacement_ind]
                max_velocity = 2 * np.pi * 5/60
                t = (5 * max_displacement)/(3 * max_velocity)
                self.rxarm.set_moving_time(t)
                self.rxarm.set_accel_time(2*t/5)
                self.rxarm.set_positions(wp)
                self.rxarm.set_moving_time(2)
                self.rxarm.set_accel_time(.3)
            rospy.sleep(t)
            print('Current position: ', self.rxarm.get_positions(), '\n')
            joint_angles.append(self.rxarm.get_positions())
            times.append(time.time())
            i += 1

        print("joint angles over time: ", joint_angles)
        joint_angles = np.stack(joint_angles, axis=0)
        fig, ax = plt.subplots()
        for i in range(joint_angles.shape[1]):
            ax.plot(times, joint_angles[:, i])
        ax.legend(['Base', 'Shoulder', 'Elbow', 'Wrist Angle', 'Wrist Rotation'])
        plt.savefig('report materials/teach_repeat.png')
        self.next_state = "idle"

    def record_waypoint(self):
        """
        Record waypoints and play them back
        """
        self.current_state = "record_waypoint"
        self.next_state = "idle"
        self.recorded_position.append(self.rxarm.get_positions())
        recorded_position = self.recorded_position[-1]
        self.status_message = 'State: Record - recorded waypoint: '+str(recorded_position)

    def record_open_gripper(self):
        """
        Record waypoints and play them back
        """
        self.current_state = "record_open_gripper"
        self.next_state = "idle"
        self.recorded_position.append('Open Gripper')
        recorded_position = self.recorded_position[-1]
        self.status_message = 'State: Record - recorded waypoint: '+str(recorded_position)

    def record_close_gripper(self):
        """
        Record waypoints and play them back
        """
        self.current_state = "record_close_gripper"
        self.next_state = "idle"
        self.recorded_position.append('Close Gripper')
        recorded_position = self.recorded_position[-1]
        self.status_message = 'State: Record - recorded waypoint: '+str(recorded_position)

    def IK_test(self):
        """
        Record waypoints and play them back
        """
        self.current_state = "IK"
        self.next_state = "idle"
        self.IK([-402.575, 0, 303.91], self.rxarm)


class StateMachineThread(QThread):
    """!
    @brief      Runs the state machine
    """
    updateStatusMessage = pyqtSignal(str)

    def __init__(self, state_machine, parent=None):
        """!
        @brief      Constructs a new instance.

        @param      state_machine  The state machine
        @param      parent         The parent
        """
        QThread.__init__(self, parent=parent)
        self.sm=state_machine

    def run(self):
        """!
        @brief      Update the state machine at a set rate
        """
        while True:
            self.sm.run()
            self.updateStatusMessage.emit(self.sm.status_message)
            rospy.sleep(0.05)
