#!/usr/bin/env python
#
# Copyright (c) 2017 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
# Copyright (c) 2019 Intel Corporation
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
Welcome to CARLA ROS manual control.

Use ARROWS or WASD keys for control.

    W            : throttle
    S            : brake
    AD           : steer
    Q            : toggle reverse
    Space        : hand-brake
    P            : toggle autopilot
    M            : toggle manual transmission
    ,/.          : gear up/down
    C            : toggle manual control
    V            : start ddpg when 'c' key ready
    B            : respawn vehicle

    F1           : toggle HUD
    H/?          : toggle help
    ESC          : quit
"""

from __future__ import print_function

import datetime
import math
import numpy
import rospy
import tf


from std_msgs.msg import Bool
from std_msgs.msg import Float32
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import NavSatFix
from sensor_msgs.msg import Image
from carla_msgs.msg import CarlaCollisionEvent
from carla_msgs.msg import CarlaLaneInvasionEvent
from carla_msgs.msg import CarlaEgoVehicleControl
from carla_msgs.msg import CarlaEgoVehicleStatus
from carla_msgs.msg import CarlaEgoVehicleInfo
from carla_msgs.msg import CarlaStatus
from geometry_msgs.msg import PoseWithCovarianceStamped #s
from geometry_msgs.msg import Twist #s

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_b
    from pygame.locals import K_v
    from pygame.locals import K_c

except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================


class World(object):
    """
    Handle the rendering
    """

    def __init__(self, role_name, hud):
        self._control = CarlaEgoVehicleControl()
        self._steer_cache = 0.0
        self._surface = None
        self.hud = hud
        self.role_name = role_name
        self.image_subscriber = rospy.Subscriber(
            "/carla/{}/camera/rgb/view/image_color".format(self.role_name),
            Image, self.on_view_image)
        self.collision_subscriber = rospy.Subscriber(
            "/carla/{}/collision".format(self.role_name), CarlaCollisionEvent, self.on_collision)
        self.lane_invasion_subscriber = rospy.Subscriber(
            "/carla/{}/lane_invasion".format(self.role_name),
            CarlaLaneInvasionEvent, self.on_lane_invasion)
        
        self.input_topic_subscriber = rospy.Subscriber( # ddpg
            "/TFF/ddpg_camera_input",
            Float32MultiArray, self.on_ddpg)

        self.input_vel_subscriber = rospy.Subscriber( # ddpg
            "/TFF/cmd_vel", Twist, self.on_ddpg_vel)


        self.vehicle_control_publisher = rospy.Publisher(
            "/carla/{}/vehicle_control_cmd_manual".format(self.role_name),
            CarlaEgoVehicleControl, queue_size=1)

        self.input_x = [0 for i in range(5)]
        self.input_y = [0 for i in range(5)]
        self.input_dist = [0 for i in range(5)]
        self.initialpose_publisher = rospy.Publisher(
            "/carla/{}/initialpose".format(self.role_name),
            PoseWithCovarianceStamped, queue_size=1) #s, initialpose publisher
        self.twist_cmd_publisher = rospy.Publisher(
            "/carla/{}/twist_cmd".format(self.role_name),
            Twist, queue_size=1) #s, twist_cmd publisher
        #self.vehicle_status_publisher = rospy.Publisher(
        #    "/carla/{}/vehicle_status".format(self.role_name),
        #    CarlaEgoVehicleStatus, queue_size=1) #s, vehicle_status.velocity publisher
        self.collision_flag_publisher = rospy.Publisher(
            "/carla/{}/collsion_flag".format(self.role_name),
            Bool, queue_size=1) #s, collision_flag publisher for vehicle_status.velocity == 0
        self.collision_flag = False

        self.lane_invasion_flag_publisher = rospy.Publisher(
            "/carla/{}/lane_invasion_flag".format(self.role_name),
            Bool, queue_size=1) #s, lane_invasion_flag publisher for reward
        self.lane_invasion_flag = True

    def on_ddpg(self, fma123):
        """
        Callback on ddpg event
        """
        #self.x = Float32MultiArray()
        if fma123.data[3] == 1:
            self.input_x[0] = fma123.data[0]
            self.input_y[0] = fma123.data[1]
            self.input_dist[0] = fma123.data[2]
        elif fma123.data[3] == 2:
            self.input_x[1] = fma123.data[0]
            self.input_y[1] = fma123.data[1]
            self.input_dist[1] = fma123.data[2]
        elif fma123.data[3] == 3:
            self.input_x[2] = fma123.data[0]
            self.input_y[2] = fma123.data[1]
            self.input_dist[2] = fma123.data[2]
        elif fma123.data[3] == 4:
            self.input_x[3] = fma123.data[0]
            self.input_y[3] = fma123.data[1]
            self.input_dist[3] = fma123.data[2]
        elif fma123.data[3] == 5:
            self.input_x[4] = fma123.data[0]
            self.input_y[4] = fma123.data[1]
            self.input_dist[4] = fma123.data[2]
        #print(self.input_x)
        #print(self.input_y)
        #print(self.input_dist) # ddpg
        if self.collision_flag == True:
            self.initial_pose = PoseWithCovarianceStamped()
            self.initial_pose.pose.pose.position.x = 225.252029
            self.initial_pose.pose.pose.position.y = 369.109375
            self.initial_pose.pose.pose.position.z = -2
            self.twist_cmd = Twist()
            self.twist_cmd.linear.x = 0
            self.twist_cmd.linear.y = 0
            self.twist_cmd.linear.z = 0
            self.twist_cmd.angular.x = 0
            self.twist_cmd.angular.y = 0
            self.twist_cmd.angular.z = 0
            pygame.time.delay(100)
            self.initialpose_publisher.publish(self.initial_pose)
            self.twist_cmd_publisher.publish(self.twist_cmd)
            self.collision_flag_publisher.publish(self.collision_flag)
            self.collision_flag = False
            
        #if self.input_dist[0] < 30 or self.input_dist[4] < 30 :
        #    self.initialpose_publisher.publish(self.initial_pose)
        #print(self.collision_flag)

    def on_ddpg_vel(self, vel_data):
        """
        Callback on ddpg event
        """
        self.twist_cmd = Twist()
        #print(self.twist_cmd)


        steer_increment = 5e-1 * vel_data.angular.z
        self._steer_cache += steer_increment

        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self.twist_cmd.linear.x = vel_data.linear.x
        self.twist_cmd.angular.z = self._control.steer
        self.twist_cmd_publisher.publish(self.twist_cmd)
            

    def on_collision(self, data): #s
        """
        Callback on collision event
        """
        intensity = math.sqrt(data.normal_impulse.x**2 +
                              data.normal_impulse.y**2 + data.normal_impulse.z**2)
        self.hud.notification('Collision with {} (impulse {})'.format(
            data.other_actor_id, intensity))
        #if intensity > 100:
        #   print("Too big collision, adjusting header...")
        #   self.initial_pose = PoseWithCovarianceStamped()
        #   self.initial_pose.pose.pose.position.x = 225.252029
        #   self.initial_pose.pose.pose.position.y = 369.109375
        #   self.initial_pose.pose.pose.position.z = 4
        #   self.initialpose_publisher.publish(self.initial_pose)
        print("collision")
        #self.initial_pose = PoseWithCovarianceStamped()
        #self.initial_pose.pose.pose.position.x = 225.252029
        #self.initial_pose.pose.pose.position.y = 369.109375
        #self.initial_pose.pose.pose.position.z = -1
        #pygame.time.delay(500)
        #self.initialpose_publisher.publish(self.initial_pose)
        self.collision_flag = True
        #self.collision_flag_publisher.publish(self.collision_flag)
        #for i in range(100):
        #    self.vehicle_status = CarlaEgoVehicleStatus()
        #    self.vehicle_status.velocity = 0
        #    self.vehicle_status_publisher.publish(self.vehicle_status)

    def on_lane_invasion(self, data):
        """
        Callback on lane invasion event
        """
        text = []
        for marking in data.crossed_lane_markings:
            if marking is CarlaLaneInvasionEvent.LANE_MARKING_OTHER:
                text.append("Other")
            elif marking is CarlaLaneInvasionEvent.LANE_MARKING_BROKEN:
                text.append("Broken")
            elif marking is CarlaLaneInvasionEvent.LANE_MARKING_SOLID:
                text.append("Solid")
            else:
                text.append("Unknown ")
        self.hud.notification('Crossed line %s' % ' and '.join(text))
        self.lane_invasion_flag_publisher.publish(self.lane_invasion_flag)
        #self.vehicle_status = CarlaEgoVehicleStatus()
        #self.vehicle_status.velocity = 0
        #self.vehicle_status_publisher.publish(self.vehicle_status)

    def on_view_image(self, image):
        """
        Callback when receiving a camera image
        """
        array = numpy.frombuffer(image.data, dtype=numpy.dtype("uint8"))
        array = numpy.reshape(array, (image.height, image.width, 4))
        array = array[:, :, :3]
        array = array[:, :, ::-1]
        self._surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

    def render(self, display):
        """
        render the current image
        """
        if self._surface is not None:
            display.blit(self._surface, (0, 0))
        self.hud.render(display)

    def destroy(self):
        """
        destroy all objects
        """
        self.image_subscriber.unregister()
        self.collision_subscriber.unregister()
        self.lane_invasion_subscriber.unregister()


# ==============================================================================
# -- KeyboardControl -----------------------------------------------------------
# ==============================================================================

class KeyboardControl(object):
    """
    Handle input events
    """

    def __init__(self, role_name, hud):
        self.role_name = role_name
        self.hud = hud

        self._autopilot_enabled = False
        self._control = CarlaEgoVehicleControl()
        self._steer_cache = 0.0
        self.input_x = [0 for i in range(5)]
        self.input_y = [0 for i in range(5)]
        self.input_dist = [0 for i in range(5)]

        self.vehicle_control_manual_override_publisher = rospy.Publisher(
            "/carla/{}/vehicle_control_manual_override".format(self.role_name),
            Bool, queue_size=1, latch=True)
        self.vehicle_control_ddpg_override_publisher = rospy.Publisher(
            "/carla/{}/vehicle_control_ddpg_override".format(self.role_name),
            Bool, queue_size=1, latch=True) #s, ddpg_on publisher
        self.vehicle_control_manual_override = False
        self.vehicle_control_ddpg_override = False
        self.auto_pilot_enable_publisher = rospy.Publisher(
            "/carla/{}/enable_autopilot".format(self.role_name), Bool, queue_size=1)
        self.vehicle_control_publisher = rospy.Publisher(
            "/carla/{}/vehicle_control_cmd_manual".format(self.role_name),
            CarlaEgoVehicleControl, queue_size=1)
        self.initialpose_publisher = rospy.Publisher(
            "/carla/{}/initialpose".format(self.role_name),
            PoseWithCovarianceStamped, queue_size=5) #s, initialpose publisher 
        self.twist_cmd_publisher = rospy.Publisher(
            "/carla/{}/twist_cmd".format(self.role_name),
            Twist, queue_size=1) #s, twist_cmd publisher
        self.input_topic_subscriber = rospy.Subscriber( # ddpg
            "/TFF/ddpg_camera_input",
            Float32MultiArray, self.on_ddpg)

        self.carla_status_subscriber = rospy.Subscriber(
            "/carla/status", CarlaStatus, self._on_new_carla_frame)

        self.set_autopilot(self._autopilot_enabled)

        self.set_vehicle_control_manual_override(
            self.vehicle_control_manual_override)  # disable manual override
        self.set_vehicle_control_ddpg_override(
            self.vehicle_control_ddpg_override)  # disable ddpg override

    def __del__(self):
        self.auto_pilot_enable_publisher.unregister()
        self.vehicle_control_publisher.unregister()
        self.vehicle_control_manual_override_publisher.unregister()

    def set_vehicle_control_manual_override(self, enable):
        """
        Set the manual control override
        """
        self.hud.notification('Set vehicle control manual override to: {}'.format(enable))
        self.vehicle_control_manual_override_publisher.publish((Bool(data=enable)))

    def set_vehicle_control_ddpg_override(self, enable): #s
        """
        Set the manual control override
        """
        self.hud.notification('Set vehicle control ddpg override to: {}'.format(enable))
        self.vehicle_control_ddpg_override_publisher.publish((Bool(data=enable)))

    def set_autopilot(self, enable):
        """
        enable/disable the autopilot
        """
        self.auto_pilot_enable_publisher.publish(Bool(data=enable))

    def on_ddpg(self, fma123): #s
        """
        Callback on ddpg event
        """
        #self.x = Float32MultiArray()
        if fma123.data[3] == 1:
            self.input_x[0] = fma123.data[0]
            self.input_y[0] = fma123.data[1]
            self.input_dist[0] = fma123.data[2]
        elif fma123.data[3] == 2:
            self.input_x[1] = fma123.data[0]
            self.input_y[1] = fma123.data[1]
            self.input_dist[1] = fma123.data[2]
        elif fma123.data[3] == 3:
            self.input_x[2] = fma123.data[0]
            self.input_y[2] = fma123.data[1]
            self.input_dist[2] = fma123.data[2]
        elif fma123.data[3] == 4:
            self.input_x[3] = fma123.data[0]
            self.input_y[3] = fma123.data[1]
            self.input_dist[3] = fma123.data[2]
        elif fma123.data[3] == 5:
            self.input_x[4] = fma123.data[0]
            self.input_y[4] = fma123.data[1]
            self.input_dist[4] = fma123.data[2]

    # pylint: disable=too-many-branches
    def parse_events(self, clock):
        """
        parse an input event
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_F1:
                    self.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and
                                          pygame.key.get_mods() & KMOD_SHIFT):
                    self.hud.help.toggle()
                elif event.key == K_b: #s
                    print("respawn vehicle")
                    self.initial_pose = PoseWithCovarianceStamped()
                    self.initial_pose.pose.pose.position.x = 225.252029
                    self.initial_pose.pose.pose.position.y = 369.109375
                    self.initial_pose.pose.pose.position.z = -2
                    self.initial_pose.pose.pose.orientation.x = 0
                    self.initial_pose.pose.pose.orientation.y = 0
                    self.initial_pose.pose.pose.orientation.z = 0
                    self.initial_pose.pose.pose.orientation.w = 0
                    self.twist_cmd = Twist()
                    self.twist_cmd.linear.x = 0
                    self.twist_cmd.linear.y = 0
                    self.twist_cmd.linear.z = 0
                    self.twist_cmd.angular.x = 0
                    self.twist_cmd.angular.y = 0
                    self.twist_cmd.angular.z = 0
                    self.initialpose_publisher.publish(self.initial_pose)
                    self.twist_cmd_publisher.publish(self.twist_cmd)
                elif event.key == K_c:
                    self.vehicle_control_manual_override = not self.vehicle_control_manual_override
                    self.set_vehicle_control_manual_override(self.vehicle_control_manual_override)
                elif event.key == K_v:
                    self.vehicle_control_ddpg_override = not self.vehicle_control_ddpg_override
                    self.set_vehicle_control_ddpg_override(self.vehicle_control_ddpg_override)
                if event.key == K_q:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.key == K_m:
                    self._control.manual_gear_shift = not self._control.manual_gear_shift
                    self.hud.notification('%s Transmission' % (
                        'Manual' if self._control.manual_gear_shift else 'Automatic'))
                elif self._control.manual_gear_shift and event.key == K_COMMA:
                    self._control.gear = max(-1, self._control.gear - 1)
                elif self._control.manual_gear_shift and event.key == K_PERIOD:
                    self._control.gear = self._control.gear + 1
                elif event.key == K_p:
                    self._autopilot_enabled = not self._autopilot_enabled
                    self.set_autopilot(self._autopilot_enabled)
                    self.hud.notification('Autopilot %s' % (
                        'On' if self._autopilot_enabled else 'Off'))                     
        if not self._autopilot_enabled and self.vehicle_control_manual_override:
            self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
            self._control.reverse = self._control.gear < 0

        return False

    def _on_new_carla_frame(self, _):
        """
        callback on new frame

        As CARLA only processes one vehicle control command per tick,
        send the current from within here (once per frame)
        """
        if not self._autopilot_enabled and self.vehicle_control_manual_override:
            try:
                self.vehicle_control_publisher.publish(self._control)
            except rospy.ROSException as error:
                rospy.logwarn("Could not send vehicle control: {}".format(error))

    def _parse_vehicle_keys(self, keys, milliseconds):
        """
        parse key events
        """
        #print(self.vehicle_control_ddpg_override)
        #self._control.throttle = 0.5 if keys[K_UP] or keys[K_w] or self.vehicle_control_ddpg_override else 0.0
        self._control.throttle = 0.5 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        #if ddpg_true:
        #    self._steer_cache = self.ddpg_data * steer_increment
        #if self.input_dist[0] > self.input_dist[4]: # simple rule-based driving
        #    self._control.steer = -0.7
        #else:
        #    self._control.steer = 0.7
        #print("-------steer_cache--------")       
        #print(self._steer_cache)
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]


    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================


class HUD(object):
    """
    Handle the info display
    """

    def __init__(self, role_name, width, height):
        self.role_name = role_name
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        fonts = [x for x in pygame.font.get_fonts() if 'mono' in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self._show_info = True
        self._info_text = []
        self.vehicle_status = CarlaEgoVehicleStatus()
        self.tf_listener = tf.TransformListener()
        self.vehicle_status_subscriber = rospy.Subscriber(
            "/carla/{}/vehicle_status".format(self.role_name),
            CarlaEgoVehicleStatus, self.vehicle_status_updated)
        self.vehicle_info = CarlaEgoVehicleInfo()
        self.vehicle_info_subscriber = rospy.Subscriber(
            "/carla/{}/vehicle_info".format(self.role_name),
            CarlaEgoVehicleInfo, self.vehicle_info_updated)
        self.latitude = 0
        self.longitude = 0
        self.manual_control = False
        self.gnss_subscriber = rospy.Subscriber(
            "/carla/{}/gnss/gnss1/fix".format(self.role_name), NavSatFix, self.gnss_updated)
        self.manual_control_subscriber = rospy.Subscriber(
            "/carla/{}/vehicle_control_manual_override".format(self.role_name),
            Bool, self.manual_control_override_updated)

        self.carla_status = CarlaStatus()
        self.status_subscriber = rospy.Subscriber(
            "/carla/status", CarlaStatus, self.carla_status_updated)

        self.vehicle_velocity_publisher = rospy.Publisher(
            "/TFF/vehicle_velocity",
            Float32, queue_size=1) #s, vehicle velocity publisher

    def __del__(self):
        self.gnss_subscriber.unregister()
        self.vehicle_status_subscriber.unregister()
        self.vehicle_info_subscriber.unregister()

    def tick(self, clock):
        """
        tick method
        """
        self._notifications.tick(clock)

    def carla_status_updated(self, data):
        """
        Callback on carla status
        """
        self.carla_status = data
        self.vehicle_velocity_publisher.publish(self.vehicle_status.velocity)
        self.update_info_text()

    def manual_control_override_updated(self, data):
        """
        Callback on vehicle status updates
        """
        self.manual_control = data.data
        self.update_info_text()

    def vehicle_status_updated(self, vehicle_status):
        """
        Callback on vehicle status updates
        """
        self.vehicle_status = vehicle_status
        self.update_info_text()

    def vehicle_info_updated(self, vehicle_info):
        """
        Callback on vehicle info updates
        """
        self.vehicle_info = vehicle_info
        self.update_info_text()

    def gnss_updated(self, data):
        """
        Callback on gnss position updates
        """
        self.latitude = data.latitude
        self.longitude = data.longitude
        self.update_info_text()

    def update_info_text(self):
        """
        update the displayed info text
        """
        if not self._show_info:
            return
        try:
            (position, quaternion) = self.tf_listener.lookupTransform(
                '/map', self.role_name, rospy.Time())
            _, _, yaw = tf.transformations.euler_from_quaternion(quaternion)
            yaw = -math.degrees(yaw)
            x = position[0]
            y = -position[1]
            z = position[2]
        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
            x = 0
            y = 0
            z = 0
            yaw = 0
        heading = 'N' if abs(yaw) < 89.5 else ''
        heading += 'S' if abs(yaw) > 90.5 else ''
        heading += 'E' if 179.5 > yaw > 0.5 else ''
        heading += 'W' if -0.5 > yaw > -179.5 else ''
        fps = 0
        if self.carla_status.fixed_delta_seconds:
            fps = 1 / self.carla_status.fixed_delta_seconds
        self._info_text = [
            'Frame: % 22s' % self.carla_status.frame,
            'Simulation time: % 12s' % datetime.timedelta(
                seconds=int(rospy.get_rostime().to_sec())),
            'FPS: % 24.1f' % fps,
            '',
            'Vehicle: % 20s' % ' '.join(self.vehicle_info.type.title().split('.')[1:]),
            'Speed:   % 15.0f km/h' % (3.6 * self.vehicle_status.velocity),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (x, y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (self.latitude, self.longitude)),
            'Height:  % 18.0f m' % z,
            '']
        self._info_text += [
            ('Throttle:', self.vehicle_status.control.throttle, 0.0, 1.0),
            ('Steer:', self.vehicle_status.control.steer, -1.0, 1.0),
            ('Brake:', self.vehicle_status.control.brake, 0.0, 1.0),
            ('Reverse:', self.vehicle_status.control.reverse),
            ('Hand brake:', self.vehicle_status.control.hand_brake),
            ('Manual:', self.vehicle_status.control.manual_gear_shift),
            'Gear:        %s' % {-1: 'R', 0: 'N'}.get(self.vehicle_status.control.gear,
                                                      self.vehicle_status.control.gear),
            '']
        self._info_text += [('Manual ctrl:', self.manual_control)]
        if self.carla_status.synchronous_mode:
            self._info_text += [('Sync mode running:', self.carla_status.synchronous_mode_running)]
        self._info_text += ['', '', 'Press <H> for help']

    def toggle_info(self):
        """
        show/hide the info text
        """
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        """
        display a notification for x seconds
        """
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        """
        display an error
        """
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        """
        render the display
        """
        if self._show_info:
            info_surface = pygame.Surface(
                (220, self.dim[1]))  # pylint: disable=too-many-function-args
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30)
                                  for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset + 50, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect(
                                (bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            f = 0.0
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)

# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================


class FadingText(object):
    """
    Support Class for info display, fade out text
    """

    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)  # pylint: disable=too-many-function-args

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        """
        set the text
        """
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)  # pylint: disable=too-many-function-args
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, clock):
        """
        tick for fading
        """
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        """
        render the fading
        """
        display.blit(self.surface, self.pos)

# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================


class HelpText(object):
    """
    Show the help text
    """

    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)  # pylint: disable=too-many-function-args
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        """
        Show/hide the help
        """
        self._render = not self._render

    def render(self, display):
        """
        render the help
        """
        if self._render:
            display.blit(self.surface, self.pos)

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================


def main():
    """
    main function
    """
    rospy.init_node('carla_manual_control', anonymous=True)

    role_name = rospy.get_param("~role_name", "ego_vehicle")

    # resolution should be similar to spawned camera with role-name 'view'
    resolution = {"width": 800, "height": 600}

    pygame.init()
    pygame.font.init()
    pygame.display.set_caption("CARLA ROS manual control")
    world = None
    try:
        display = pygame.display.set_mode(
            (resolution['width'], resolution['height']),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        hud = HUD(role_name, resolution['width'], resolution['height'])
        world = World(role_name, hud)
        controller = KeyboardControl(role_name, hud)

        clock = pygame.time.Clock()

        while not rospy.core.is_shutdown():
            clock.tick_busy_loop(60)
            if controller.parse_events(clock):
                return
            hud.tick(clock)
            world.render(display)
            pygame.display.flip()

    finally:
        if world is not None:
            world.destroy()
        pygame.quit()


if __name__ == '__main__':

    main()
