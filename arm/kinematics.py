"""!
Implements Forward and Inverse kinematics with DH parametrs and product of exponentials

TODO: Here is where you will write all of your kinematics functions
There are some functions to start with, you may need to implement a few more
"""
from lib2to3.pygram import Symbols

from operator import inv
from re import I
from sys import api_version
from turtle import position
from webbrowser import get
import numpy as np
from sympy import *
from scipy.optimize import fsolve
# expm is a matrix exponential function
from scipy.linalg import expm



dh_params = np.array([[   0, np.pi/2, 103.91, -np.pi/2],
                      [   0,       0,      0,  np.pi/2],
                      [ 200,       0,      0,        0],
                      [  50,       0,      0,  np.pi/2],
                      [ 200,       0,      0,        0],
                      [   0, np.pi/2,      0,  np.pi/2],
                      [   0,       0,     65,        0],
                      [   0,       0, 109.15,        0]])


# theta1 = theta2 = theta3 = theta4 = theta5 = np.radians(0.7)

# joint_angles = np.array([theta1, theta2, theta3, theta4, theta5])

def clamp(angle):
    """!
    @brief      Clamp angles between (-pi, pi]

    @param      angle  The angle

    @return     Clamped angle
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle <= -np.pi:
        angle += 2 * np.pi
    return angle

def get_transform_from_dh(a, alpha, d, theta):
    """!
    @brief      Gets the transformation matrix from dh parameters.

    TODO: Find the T matrix from a row of a DH table

    @param      a      a meters
    @param      alpha  alpha radians
    @param      d      d meters
    @param      theta  theta radians

    @return     The 4x4 transform matrix.
    """

    transformation = np.array([[np.cos(theta), -np.sin(theta)*np.cos(alpha),  np.sin(theta)*np.sin(alpha), a*np.cos(theta)],
                               [np.sin(theta),  np.cos(theta)*np.cos(alpha), -np.cos(theta)*np.sin(alpha), a*np.sin(theta)],
                               [            0,                np.sin(alpha),                np.cos(alpha),               d],
                               [            0,                            0,                            0,               1]])
    return transformation

def FK_dh(dh_params, joint_angles, link):
    """!
    @brief      Get the 4x4 transformation matrix from link to world

                TODO: implement this function

                Calculate forward kinematics for rexarm using DH convention

                return a transformation matrix representing the pose of the desired link

                note: phi is the euler angle about the y-axis in the base frame

    @param      dh_params     The dh parameters as a 2D list each row represents a link and has the format [a, alpha, d,
                              theta]
    @param      joint_angles  The joint angles of the links
    @param      link          The link to transform from

    @return     a transformation matrix representing the pose of the desired link
    """
    count = 0
    transformation = np.identity(4)

    # add each joint angles to certain D-H table row
    dh_params[0][3] += joint_angles[0]
    dh_params[1][3] += joint_angles[1]
    dh_params[4][3] += joint_angles[2]
    dh_params[5][3] += joint_angles[3]
    dh_params[7][3] += joint_angles[4]

    if link == 1:
        count = 0

    elif link == 2:
        count = 1

    elif link == 3:
        count = 4

    elif link == 4:
        count = 5

    elif link == 5:
        count = 7

    # get transformation matrix from link
    for i in range(count + 1):
        transformation = np.dot(transformation, get_transform_from_dh(dh_params[i][0], dh_params[i][1], dh_params[i][2], dh_params[i][3]))

    return transformation


def get_euler_angles_from_T(T):
    """!
    @brief      Gets the euler angles from a transformation matrix.

                TODO: Implement this function return the Euler angles from a T matrix

    @param      T     transformation matrix

    @return     The euler angles from T.
    """
    pass


def get_pose_from_T(T):
    """!
    @brief      Gets the pose from T.

                TODO: implement this function return the joint pose from a T matrix of the form (x,y,z,phi) where phi is
                rotation about base frame y-axis

    @param      T     transformation matrix

    @return     The pose from T.
    """
    pass


def FK_pox(joint_angles, m_mat, s_lst):
    """!
    @brief      Get a 4-tuple (x, y, z, phi) representing the pose of the desired link

                TODO: implement this function, Calculate forward kinematics for rexarm using product of exponential
                formulation return a 4-tuple (x, y, z, phi) representing the pose of the desired link note: phi is the euler
                angle about y in the base frame

    @param      joint_angles  The joint angles
                m_mat         The M matrix
                s_lst         List of screw vectors

    @return     a 4-tuple (x, y, z, phi) representing the pose of the desired link
    """
    pass


def to_s_matrix(w, v):
    """!
    @brief      Convert to s matrix.

    TODO: implement this function
    Find the [s] matrix for the POX method e^([s]*theta)

    @param      w     { parameter_description }
    @param      v     { parameter_description }

    @return     { description_of_the_return_value }
    """
    pass

def get_R0_4(theta1, theta2, theta3):

    return np.array([[-np.cos(theta3)*np.cos(theta1-np.pi/2)*np.sin(theta2+np.pi/2)-np.cos(theta1-np.pi/2)*np.cos(theta2+np.pi/2)*np.sin(theta3), np.cos(theta1-np.pi/2)*np.sin(theta3)*np.sin(theta2+np.pi/2)-np.cos(theta3)*np.cos(theta1-np.pi/2)*np.cos(theta2+np.pi/2),  np.sin(theta1-np.pi/2)],
                     [-np.cos(theta3)*np.sin(theta1-np.pi/2)*np.sin(theta2+np.pi/2)-np.cos(theta2+np.pi/2)*np.sin(theta3)*np.sin(theta1-np.pi/2), np.sin(theta1-np.pi/2)*np.sin(theta3)*np.sin(theta2+np.pi/2)-np.cos(theta3)*np.cos(theta2+np.pi/2)*np.sin(theta1-np.pi/2), -np.cos(theta1-np.pi/2)],
                     [                        np.cos(theta3)*np.cos(theta2+np.pi/2)-np.sin(theta3)*np.sin(theta2+np.pi/2),                        -np.cos(theta3)*np.sin(theta2+np.pi/2)-np.cos(theta2+np.pi/2)*np.sin(theta3),                                                                    0]])

def get_possible_joint_angle(theta1, theta2, theta3, R0_7):
    inverse_R0_4 = np.linalg.inv(get_R0_4(theta1, theta2, theta3))
    R4_7 = np.dot(inverse_R0_4, R0_7)
    """
    BY Forward Kinematics
    R4_7 = [cos(theta5)*cos(theta4 + pi/2), -cos(theta4 + pi/2)*sin(theta5),  sin(theta4 + pi/2)]
           [cos(theta5)*sin(theta4 + pi/2), -sin(theta5)*sin(theta4 + pi/2), -cos(theta4 + pi/2)]
           [                   sin(theta5),                     cos(theta5),                   0]

    """
    IK_theta5 = np.arctan2(R4_7[2][0],  R4_7[2][1])
    IK_theta4 = np.arctan2(R4_7[0][2], -R4_7[1][2]) - np.pi/2
    return np.array([theta1, theta2, theta3, IK_theta4])


def IK_transformation(a, alpha, d, theta):
    transformation = Matrix([[cos(theta), -sin(theta)*cos(alpha),  sin(theta)*sin(alpha), a*cos(theta)],
                             [sin(theta),  cos(theta)*cos(alpha), -cos(theta)*sin(alpha), a*sin(theta)],
                             [         0,             sin(alpha),             cos(alpha),            d],
                             [         0,                      0,                      0,            1]])

    return transformation


def predict_end_position():
    theta1, theta2, theta3, theta4, theta5 = symbols('theta1, theta2, theta3, theta4, theta5')
    # transformation = eye(4)
    # dh_params_IK = Matrix([[   0,    pi/2, 103.91, -pi/2 + theta1],
    #                        [   0,       0,      0,  pi/2 + theta2],
    #                        [ 200,       0,      0,              0],
    #                        [  50,       0,      0,           pi/2],
    #                        [ 200,       0,      0,        -theta3],
    #                        [   0,    pi/2,      0, -pi/2 - theta4],
    #                        [   0,       0,     65,              0],
    #                        [   0,       0, 109.15,         theta5]])

    #Figure out what 0 degree config looks like, base it off of that.
    # dh_params_IK = Matrix([[      0,   pi/2,  103.91,               theta1 - pi/2],
    #                        [ 205.73,      pi,       0,           theta2 + pi/2 + arctan(.25)],
    #                        [    200,       0,       0, theta3 - (pi/2 - arctan(.25))],
    #                        [      0,    pi/2,       0,                      theta4 + pi/2],
    #                        [      0,      0,  152.575,                      theta5 - pi/2]])
    # for i in range(dh_params_IK.shape[0]):
    #     transformation = transformation * IK_transformation(dh_params_IK[i, 0], dh_params_IK[i, 1], dh_params_IK[i, 2], dh_params_IK[i, 3])
    # p7 = transformation * Matrix([0, 0, 0, 1])
    # position_end = Matrix([p7[0], p7[1], p7[2]])
    position_end = Matrix([-(199.587416554899*sin(theta2) + 49.8968541387248*cos(theta2) + 200.0*cos(theta2 - theta3) + 174.15*cos(-theta2 + theta3 + theta4))*sin(theta1),
(199.587416554899*sin(theta2) + 49.8968541387248*cos(theta2) + 200.0*cos(theta2 - theta3) + 174.15*cos(-theta2 + theta3 + theta4))*cos(theta1),
-49.8968541387248*sin(theta2) - 200.0*sin(theta2 - theta3) + 174.15*sin(-theta2 + theta3 + theta4) + 199.587416554899*cos(theta2) + 103.91])

    # t1 = -61.4/180 * np.pi
    # t2 = -58.8/180 * np.pi + np.pi/2
    # t3 = -43.9/180 * np.pi
    # t4 = -17/180 * np.pi
    # t5 =0
    # print('br')
    # t1 = pi/2
    # t2 = 0
    # t3 = 0
    # t4 = 0
    # subs = position_end.subs({theta1: t1, theta2: t2, theta3: t3, theta4: t4})
    # print(np.array(simplify(subs)))

    # t1 = -58.71/180*np.pi
    # t2 = -55/180*np.pi
    # t3 = -45.7/180*np.pi
    # t4 = -9.76/180*np.pi
    # t5 = 0
    # subs = position_end.subs({theta1: t1, theta2: t2, theta3: t3, theta4: t4, theta5: t5})
    # print(np.array(simplify(subs)))
    # x = (-12.1267812518166*sqrt(17)*sin(t2) - 200.0*sin(t2 - t3) + 48.5071250072666*sqrt(17)*cos(t2) - 152.575*cos(-t2 + t3 + t4))*cos(t1)
    # y = (-12.1267812518166*sqrt(17)*sin(t2) - 200.0*sin(t2 - t3) + 48.5071250072666*sqrt(17)*cos(t2) - 152.575*cos(-t2 + t3 + t4))*sin(t1)
    # z = -48.5071250072666*sqrt(17)*sin(t2) - 152.575*sin(-t2 + t3 + t4) - 12.1267812518166*sqrt(17)*cos(t2) - 200.0*cos(t2 - t3) + 103.91
    # print(x, y, z)
    return position_end

def jacobian():
    theta1, theta2, theta3, theta4, theta5= symbols('theta1, theta2, theta3, theta4, theta5')
    theta = Matrix([theta1, theta2, theta3, theta4, theta5])
    end = predict_end_position()
    jacobian_matrix = end.jacobian(theta)
    return jacobian_matrix


def get_joint_angle(rxarm):
    '''
    get the each present joint angle from theta1 to theta5
    return 1 by 5 matrix
    '''
    position = rxarm.get_positions()
    joint_angle = Matrix([position[0], position[1], position[2], position[3], position[4]])
    #joint_angle = Matrix([0, 0, 0, 0, 0])
    return joint_angle


def get_error(joint_angle, pose):
    x_d = pose[0]
    y_d = pose[1]
    z_d = pose[2]

    theta1, theta2, theta3, theta4, theta5 = symbols('theta1, theta2, theta3, theta4, theta5')
    position_now = predict_end_position()
    subs_position_now = position_now.subs({theta1: joint_angle[0],
                                           theta2: joint_angle[1],
                                           theta3: joint_angle[2],
                                           theta4: joint_angle[3],
                                           theta5: joint_angle[4]}).evalf()
    err = Matrix([[x_d - subs_position_now[0]],
                  [y_d - subs_position_now[1]],
                  [z_d - subs_position_now[2]],
                  ])
    return err

def IK_delta_theta(error, joint_angle):
    '''
    joint_angle = [theta1, theata2, theta3, theta4] joint angle from now
    pose = [x, y, z, phi] desired position

    '''
    theta1, theta2, theta3, theta4, theta5 = symbols('theta1, theta2, theta3, theta4, theta5')
    jacobian_matrix = jacobian()

    subs_jacobian = jacobian_matrix.subs({theta1: joint_angle[0],
                                          theta2: joint_angle[1],
                                          theta3: joint_angle[2],
                                          theta4: joint_angle[3],
                                          theta5: joint_angle[4]}).evalf()
    pseudo_subs_j = subs_jacobian.pinv()

    delta_theta = pseudo_subs_j * error
    return delta_theta


# def set_joint_angle(new_angle, rxarm):
#     '''
#     set each joint to new angle
#     '''
#     angle = np.array(new_angle).astype(np.float64)
#     current_pos = get_joint_angle(rxarm)
#     displacement = np.abs(angle - current_pos)
#     max_displacement_ind = np.argmax(displacement)
#     max_displacement = displacement[max_displacement_ind]
#     max_velocity = 2 * np.pi * 5/60
#     t = (5 * max_displacement)/(3 * max_velocity)
#     rxarm.set_moving_time(t)
#     rxarm.set_accel_time(2*t/5)
#     rxarm.set_positions(angle)


def IK_numerical(pose):
    # joint_angle = get_joint_angle(rxarm)
    print('goal pose', pose)
    # print('initial joint_angle', joint_angle)
    joint_angle = Matrix([1, 0.5, 0.5, 0.5, 0])

    error = get_error(joint_angle, pose)
    while abs(error[0]) > 1 or abs(error[1]) > 1 or abs(error[2]) > 1:
        step = 0.25
        delta_angle = step * IK_delta_theta(error, joint_angle)
        joint_angle += delta_angle

        while(joint_angle[0] > np.pi or joint_angle[0] < -np.pi):
            if(joint_angle[0] > np.pi):
                joint_angle[0] -= 2 * np.pi
            else:
                joint_angle[0] += 2 * np.pi

        while(joint_angle[1] > np.pi or joint_angle[1] < -np.pi):
            if(joint_angle[1] > np.pi):
                joint_angle[1] -= 2 * np.pi
            else:
                joint_angle[1] += 2 * np.pi

        while(joint_angle[2] > np.pi / 2  or joint_angle[2] < -np.pi / 2):
            if(joint_angle[2] > np.pi / 2):
                joint_angle[2] -=  np.pi
            elif(joint_angle[2] < np.pi / 2):
                joint_angle[2] += np.pi

        while(joint_angle[3] > np.pi or joint_angle[3] < -np.pi):
            if(joint_angle[3] > np.pi):
                joint_angle[3] -= 2 * np.pi
            else:
                joint_angle[3] += 2 * np.pi
        # set_joint_angle(joint_angle + delta_angle, rxarm)

        # joint_angle = get_joint_angle(rxarm)


        error = get_error(joint_angle, pose)
        print("delta_angle: ", delta_angle)
        print("joint_angle: ", joint_angle)
        print("error: ", error)
        print('--------------------------------------')


    return np.array(joint_angle).astype(np.float64).flatten()

def eq2_1(A, x):
    return 2*np.arctan((np.sqrt(-A**2 - 400*A*np.sin(x) - 40000*np.sin(x)**2 + 42500) - 50)/(A + 200*np.sin(x) + 200))

def eq2_2(A, x):
    return -2*np.arctan((np.sqrt(-A**2 - 400*A*np.sin(x) - 40000*np.sin(x)**2 + 42500) + 50)/(A + 200*np.sin(x) + 200))

def IK_geometric(pose):
    """!
    @brief      Get all possible joint configs that produce the pose.

                TODO: Convert a desired end-effector pose as np.array x,y,z,phi to joint angles

    @param      dh_params  The dh parameters
    @param      pose       The desired pose as np.array x,y,z,phi

    @return     All four possible joint configurations in a numpy array 4x4 where each row is one possible joint
                configuration
    """

    all_possible_joint_angle = np.empty((0, 5))

    # total length of arm
    theta1, theta2, theta3, theta4, theta5 = symbols('theta1, theta2, theta3, theta4, theta5')


    x, y, z, phi = pose[0], pose[1], pose[2], pose[3]
    A = z - 174.15 * np.sin(-phi) - 103.91
    theta1_1 = np.arctan2(y, x) - np.pi / 2
    B = y / np.cos(theta1_1) - 174.15 * np.cos(-phi)
    C = -x / np.sin(theta1_1) - 174.15*np.cos(-phi)
    # print("b first: ", B)
    theta2_1_theta3_1 = -2*np.arctan((400*A - np.sqrt(-A**4 - 2*A**2*B**2 + 165000*A**2 - B**4 + 165000*B**2 - 6250000))/(A**2 + B**2 + 400*B - 2500))
    theta2_2_theta3_2 = -2*np.arctan((400*A + np.sqrt(-A**4 - 2*A**2*B**2 + 165000*A**2 - B**4 + 165000*B**2 - 6250000))/(A**2 + B**2 + 400*B - 2500))
    theta2_1_1 = eq2_1(A, theta2_1_theta3_1)
    theta2_1_2 = eq2_2(A, theta2_1_theta3_1)
    theta2_2_1 = eq2_1(A, theta2_2_theta3_2)
    theta2_2_2 = eq2_2(A, theta2_2_theta3_2)
    theta3_1_1 = -theta2_1_theta3_1 + theta2_1_1
    theta3_1_2 = -theta2_1_theta3_1 + theta2_1_2
    theta3_2_1 = -theta2_2_theta3_2 + theta2_2_1
    theta3_2_2 = -theta2_2_theta3_2 + theta2_2_2
    theta4_1_1 = -phi + theta2_1_1 - theta3_1_1
    theta4_1_2 = -phi + theta2_1_2 - theta3_1_2
    theta4_2_1 = -phi + theta2_2_1 - theta3_2_1
    theta4_2_2 = -phi + theta2_2_2 - theta3_2_2

    theta1_2 = np.arctan2(y, x) + np.pi / 2
    B = y / np.cos(theta1_2) - 174.15 * np.cos(-phi)
    # print("b second: ", B)
    C = -x / np.sin(theta1_2) - 174.15*np.cos(-phi)
    theta2_3_theta3_3 = -2*np.arctan((400*A - np.sqrt(-A**4 - 2*A**2*B**2 + 165000*A**2 - B**4 + 165000*B**2 - 6250000))/(A**2 + B**2 + 400*B - 2500))    # print(theta1)
    theta2_4_theta3_4 = -2*np.arctan((400*A + np.sqrt(-A**4 - 2*A**2*B**2 + 165000*A**2 - B**4 + 165000*B**2 - 6250000))/(A**2 + B**2 + 400*B - 2500))


    theta2_3_1 = eq2_1(A, theta2_3_theta3_3)
    theta2_3_2 = eq2_2(A, theta2_3_theta3_3)
    theta2_4_1 = eq2_1(A, theta2_4_theta3_4)
    theta2_4_2 = eq2_2(A, theta2_4_theta3_4)
    theta3_3_1 = -theta2_3_theta3_3 + theta2_3_1
    theta3_3_2 = -theta2_3_theta3_3 + theta2_3_2
    theta3_4_1 = -theta2_4_theta3_4 + theta2_4_1
    theta3_4_2 = -theta2_4_theta3_4 + theta2_4_2
    theta4_3_1 = -phi + theta2_3_1 - theta3_3_1
    theta4_3_2 = -phi + theta2_3_2 - theta3_3_2
    theta4_4_1 = -phi + theta2_4_1 - theta3_4_1
    theta4_4_2 = -phi + theta2_4_2 - theta3_4_2

    # theta2_3 = -2*atan((sqrt(-A**2 - 400*A*sin(theta2_3_theta3_3) - 40000*sin(theta2_3_theta3_3)**2 + 42500) + 50)/(A + 200*sin(theta2_3_theta3_3) + 200))
    # theta2_4 = 2*atan((sqrt(-A**2 - 400*A*sin(theta2_4_theta3_4) - 40000*sin(theta2_4_theta3_4)**2 + 42500) - 50)/(A + 200*sin(theta2_4_theta3_4) + 200))
    # theta3_3 = -theta2_3_theta3_3 + theta2_3
    # theta3_4 = -theta2_4_theta3_4 + theta2_4

    # theta4_3 = -phi + theta2_3 - theta3_3
    # theta4_4 = -phi + theta2_4 - theta3_4
    # print("theta2_1_theta3_1: ", theta2_1_theta3_1.evalf())
    # print("theta2_2_theta3_2: ", theta2_2_theta3_2.evalf())
    # print("theta2_3_theta3_3: ", theta2_3_theta3_3.evalf())
    # print("theta2_4_theta3_4: ", theta2_4_theta3_4.evalf())
    # print("-------------------------------")
    # print("theta1_1: ", theta1_1.evalf())
    # print("theta1_2: ", theta1_2.evalf())
    # print("theta2_1: ", theta2_1.evalf())
    # print("theta2_2: ", theta2_2.evalf())
    # print("theta3_1: ", theta3_1.evalf())
    # print("theta3_2: ", theta3_2.evalf())
    # print("theta4_1: ", theta4_1.evalf())
    # print("theta4_2: ", theta4_2.evalf())
    # print("-------------------------------")
    # print("theta1_2: ", theta1_2.evalf())
    # print("theta1_2: ", theta1_2.evalf())
    # print("theta2_3: ", theta2_3.evalf())
    # print("theta2_4: ", theta2_4.evalf())
    # print("theta3_3: ", theta3_3.evalf())
    # print("theta3_4: ", theta3_4.evalf())
    # print("theta4_3: ", theta4_3.evalf())
    # print("theta4_4: ", theta4_4.evalf())
    solution = np.array([[theta1_1, theta2_1_1, theta3_1_1, theta4_1_1, 0],
                         [theta1_1, theta2_1_2, theta3_1_2, theta4_1_2, 0],
                         [theta1_1, theta2_2_1, theta3_2_1, theta4_2_1, 0],
                         [theta1_1, theta2_2_2, theta3_2_2, theta4_2_2, 0],
                         [theta1_2, theta2_3_1, theta3_3_1, theta4_3_1, 0],
                         [theta1_2, theta2_3_2, theta3_3_2, theta4_3_2, 0],
                         [theta1_2, theta2_4_1, theta3_4_1, theta4_4_1, 0],
                         [theta1_2, theta2_4_2, theta3_4_2, theta4_4_2, 0]])
    position_now = predict_end_position()
    for sol in solution:
        subs_position_now = position_now.subs({theta1: sol[0],
                                               theta2: sol[1],
                                               theta3: sol[2],
                                               theta4: sol[3],
                                               theta5: 0}).evalf()
        # print(subs_position_now)
        # print("0:", abs(subs_position_now[2] - z) < 2)
        print(x, y, z)

        try:
            if abs(subs_position_now[0] - x) < 2 and abs(subs_position_now[1] - y) < 2 and abs(subs_position_now[2] - z) < 2:
                # print(sol)
                all_possible_joint_angle = np.append(all_possible_joint_angle, [sol], axis = 0)
        except:
            pass
    print(all_possible_joint_angle)
    # if cos(theta1) != 0:
    #     B = y / cos(theta1) - 152.575 * cos(-phi)
    # elif sin(theta1) != 0:
    #     B = - x / sin(theta1) - 152.575 * cos(-phi)
    # theta3_1 = -2*arctan((400*A - sqrt(-A**4 - 2*A**2*B**2 + 165000*A**2 - B**4 + 165000*B**2 - 6250000))/(A**2 + B**2 + 400*B - 2500))
    # theta3_2 = -2*arctan((400*A + sqrt(-A**4 - 2*A**2*B**2 + 165000*A**2 - B**4 + 165000*B**2 - 6250000))/(A**2 + B**2 + 400*B - 2500))
    # theta2 = 2*arctan((sqrt(-A**2 - 400*A*sin(theta3_1) - 40000*sin(theta3_1)**2 + 42500) - 50)/(A + 200*sin(theta3_1) + 200))
    # theta4 = -phi + theta2 - theta3_1
    # joint_angle = np.array([theta1, theta2, theta3_1, theta4, 0])

    # R0_7 = np.array([[1,  0,  0],
    #                  [0, -1,  0],
    #                  [0,  0, -1]])

    # # o0_7 = np.array([[pose_x], [pose_y], [pose_z]])
    # # o0_C = o0_7 - link_length_7 * np.dot(R0_7, np.array([[0],[0],[1]]))
    # # x_c, y_c, z_c = o0_C[0][0], o0_C[1][0], o0_C[2][0]
    # x_c, y_c, z_c = pose_x, pose_y, pose_z
    # print(link_length_7)
    # print(x_c, y_c, z_c)
    # # all possible angles are two theta1 combine two set of theta2 and theta3
    # # positive theta1
    # angle2_1, angle3_1 = symbols('angle2_1, angle3_1')
    # eq1_1 = 200 * cos(angle2_1) + 50 * sin(angle2_1) + 200 * cos(angle3_1) - (x_c ** 2 + y_c ** 2) ** 0.5
    # eq2_1 = 200 * sin(angle2_1) - 50 * cos(angle2_1) + 200 * sin(angle3_1) + 103.91 - z_c
    # solution = solve((eq1_1, eq2_1), (angle2_1, angle3_1))
    # print(solution)
    # # solution_1 = np.array(solution).astype(np.float64)
    # # IK_theta1_1 = np.arctan2(y_c, x_c)
    # # IK_theta2_1 = solution_1[0][0]
    # # IK_theta2_2 = solution_1[1][0]
    # # IK_theta3_1 = solution_1[0][1]
    # # IK_theta3_2 = solution_1[1][1]

    # # negeative theta1
    # IK_theta1_2 = np.arctan2(y_c, x_c) + np.pi
    # angle2_2, angle3_2 = symbols('angle2_2, angle3_2')
    # eq1_2 = 200 * cos(angle2_2) - 50 * sin(angle2_2) + 200 * cos(angle3_2) - (x_c ** 2 + y_c ** 2) ** 0.5
    # eq2_2 = 200 * sin(angle2_2) + 50 * cos(angle2_2) + 200 * sin(angle3_2) + 103.91 - z_c
    # solution = solve((eq1_2, eq2_2), (angle2_2, angle3_2))
    # print((solution))
    # solution_2 = np.array(solution).astype(np.float64)
    # IK_theta1_2 = np.arctan2(y_c, x_c) + np.pi
    # IK_theta2_3 = np.pi - solution_2[0][0]
    # IK_theta2_4 = np.pi - solution_2[1][0]
    # IK_theta3_3 = np.pi - solution_2[0][1]
    # IK_theta3_4 = np.pi - solution_2[1][1]


    # all_possible_joint_angle =  np.append(all_possible_joint_angle, [get_possible_joint_angle(IK_theta1_1, IK_theta2_1, IK_theta3_1, R0_7)], axis = 0)
    # all_possible_joint_angle =  np.append(all_possible_joint_angle, [get_possible_joint_angle(IK_theta1_1, IK_theta2_2, IK_theta3_2, R0_7)], axis = 0)
    # all_possible_joint_angle =  np.append(all_possible_joint_angle, [get_possible_joint_angle(IK_theta1_2, IK_theta2_3, IK_theta3_3, R0_7)], axis = 0)
    # all_possible_joint_angle =  np.append(all_possible_joint_angle, [get_possible_joint_angle(IK_theta1_2, IK_theta2_4, IK_theta3_4, R0_7)], axis = 0)

    return all_possible_joint_angle



def solve_eq():
    A, B, theta3 = symbols('A, B, theta3')
    eq = 50 ** 2 - A ** 2 - B ** 2 - 400 * sin(theta3) * A + 400 * cos(theta3) * B
    solution = solve((eq), (theta3))
    print("solution 1: ", simplify(solution[0]))
    print("solution 2: ", simplify(solution[1]))

def solve_eq2():
    A, theta2, s3 = symbols('A, theta2, s3')
    eq = -50 * sin(theta2) - 200 * s3 + 200 * cos(theta2) - A
    solution = solve((eq), (theta2))
    print("solution 1: ", simplify(solution[0]))
    print("solution 2: ", simplify(solution[1]))

# solve_eq2()
# IK_geometric([0, 424.15, 303.91, 0])

# IK_numerical([200, -312, 25])
# print(simplify(IK_geometric([200, 75, 300, 0])))

# A = z - 152.575 * sin(-phi) - 103.91
# B = y / cos(theta1) - 152.575 * cos(-phi)
# C = - x / sin(theta1) - 152.575 * cos(-phi)
# theta1 = arctan2(y, x) - pi / 2
# first solution for theta3  = -2*arctan((400*A - sqrt(-A**4 - 2*A**2*B**2 + 165000*A**2 - B**4 + 165000*B**2 - 6250000))/(A**2 + B**2 + 400*B - 2500))
# second solution for theta3 = -2*arctan((400*A + sqrt(-A**4 - 2*A**2*B**2 + 165000*A**2 - B**4 + 165000*B**2 - 6250000))/(A**2 + B**2 + 400*B - 2500))
# theta2 = 2*arctan((sqrt(-A**2 - 400*A*s3 - 40000*s3**2 + 42500) - 50)/(A + 200*s3 + 200))
# theta4 = -phi + theta2 - theta3
# predict_end_position()
