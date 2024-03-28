# Code based on Carla examples, which are authored by 
# Computer Vision Center (CVC) at the Universitat Autonoma de Barcelona (UAB).

import carla
import random
from pathlib import Path
import numpy as np
import pygame
from util.carla_util import *
from util.geometry_util import dist_point_linestring
import argparse
import cv2
import glob
import os
import sys
import time
from datetime import datetime
from matplotlib import cm
import open3d as o3d


VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
LABEL_COLORS = np.array([
    (255, 255, 255), # None
    (70, 70, 70),    # Building
    (100, 40, 40),   # Fences
    (55, 90, 80),    # Other
    (220, 20, 60),   # Pedestrian
    (153, 153, 153), # Pole
    (157, 234, 50),  # RoadLines
    (128, 64, 128),  # Road
    (244, 35, 232),  # Sidewalk
    (107, 142, 35),  # Vegetation
    (0, 0, 142),     # Vehicle
    (102, 102, 156), # Wall
    (220, 220, 0),   # TrafficSign
    (70, 130, 180),  # Sky
    (81, 0, 81),     # Ground
    (150, 100, 100), # Bridge
    (230, 150, 140), # RailTrack
    (180, 165, 180), # GuardRail
    (250, 170, 30),  # TrafficLight
    (110, 190, 160), # Static
    (170, 120, 50),  # Dynamic
    (45, 60, 150),   # Water
    (145, 170, 100), # Terrain
]) / 255.0 # normalize each channel [0-1] since is what Open3D uses


def lidar_callback(point_cloud, point_list):
    """Prepares a point cloud with intensity
    colors ready to be consumed by Open3D"""
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))

    # Isolate the intensity and compute a color for it
    intensity = data[:, -1]
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

    # Isolate the 3D data
    points = data[:, :-1]

    # We're negating the y to correclty visualize a world that matches
    # what we see in Unreal since Open3D uses a right-handed coordinate system
    points[:, :1] = -points[:, :1]

    # # An example of converting points from sensor to vehicle space if we had
    # # a carla.Transform variable named "tran":
    # points = np.append(points, np.ones((points.shape[0], 1)), axis=1)
    # points = np.dot(tran.get_matrix(), points.T).T
    # points = points[:, :-1]

    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)


def semantic_lidar_callback(point_cloud, point_list):
    """Prepares a point cloud with semantic segmentation
    colors ready to be consumed by Open3D"""
    data = np.frombuffer(point_cloud.raw_data, dtype=np.dtype([
        ('x', np.float32), ('y', np.float32), ('z', np.float32),
        ('CosAngle', np.float32), ('ObjIdx', np.uint32), ('ObjTag', np.uint32)]))

    # We're negating the y to correclty visualize a world that matches
    # what we see in Unreal since Open3D uses a right-handed coordinate system
    points = np.array([data['x'], -data['y'], data['z']]).T

    # # An example of adding some noise to our data if needed:
    # points += np.random.uniform(-0.05, 0.05, size=points.shape)

    # Colorize the pointcloud based on the CityScapes color palette
    labels = np.array(data['ObjTag'])
    int_color = LABEL_COLORS[labels]

    # # In case you want to make the color intensity depending
    # # of the incident ray angle, you can use:
    # int_color *= np.array(data['CosAngle'])[:, None]

    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)


def generate_lidar_bp(arg, world, blueprint_library, delta):
    """Generates a CARLA blueprint based on the script parameters"""
    if arg.semantic:
        lidar_bp = world.get_blueprint_library().find('sensor.lidar.ray_cast_semantic')
    else:
        lidar_bp = blueprint_library.find('sensor.lidar.ray_cast')
        if arg.no_noise:
            lidar_bp.set_attribute('dropoff_general_rate', '0.0')
            lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
            lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
        else:
            lidar_bp.set_attribute('noise_stddev', '0.2')

    lidar_bp.set_attribute('upper_fov', str(arg.upper_fov))
    lidar_bp.set_attribute('lower_fov', str(arg.lower_fov))
    lidar_bp.set_attribute('channels', str(arg.channels))
    lidar_bp.set_attribute('range', str(arg.range))
    lidar_bp.set_attribute('rotation_frequency', str(1.0 / delta))
    lidar_bp.set_attribute('points_per_second', str(arg.points_per_second))
    return lidar_bp


def add_open3d_axis(vis):
    """Add a small 3D axis on Open3D Visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)


def get_trajectory_from_lane_detector(lane_detector, image):
    # get lane boundaries using the lane detector
    image_arr = carla_img_to_array(image)

    poly_left, poly_right, img_left, img_right = lane_detector(image_arr)
    # https://stackoverflow.com/questions/50966204/convert-images-from-1-1-to-0-255
    img = img_left + img_right
    img = cv2.normalize(img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    img = img.astype(np.uint8)
    img = cv2.resize(img, (600,400))
    
    # trajectory to follow is the mean of left and right lane boundary
    # note that we multiply with -0.5 instead of 0.5 in the formula for y below
    # according to our lane detector x is forward and y is left, but
    # according to Carla x is forward and y is right.
    x = np.arange(-2,60,1.0)
    y = -0.5*(poly_left(x)+poly_right(x))
    # x,y is now in coordinates centered at camera, but camera is 0.5 in front of vehicle center
    # hence correct x coordinates
    x += 0.5
    trajectory = np.stack((x,y)).T
    return trajectory, img

def get_trajectory_from_map(CARLA_map, vehicle):
    # get 80 waypoints each 1m apart. If multiple successors choose the one with lower waypoint.id
    waypoint = CARLA_map.get_waypoint(vehicle.get_transform().location)
    list_waypoint = [waypoint]
    for _ in range(20):
        next_wps = waypoint.next(1.0)
        if len(next_wps) > 0:
            waypoint = sorted(next_wps, key=lambda x: x.id)[0]
        list_waypoint.append(waypoint)

    # transform waypoints to vehicle ref frame
    trajectory = np.array(
        [np.array([*carla_vec_to_np_array(x.transform.location), 1.]) for x in list_waypoint]
    ).T
    trafo_matrix_world_to_vehicle = np.array(vehicle.get_transform().get_inverse_matrix())

    trajectory = trafo_matrix_world_to_vehicle @ trajectory
    trajectory = trajectory.T
    trajectory = trajectory[:,:2]
    return trajectory

def send_control(vehicle, throttle, steer, brake,
                 hand_brake=False, reverse=False):
    throttle = np.clip(throttle, 0.0, 1.0)
    steer = np.clip(steer, -1.0, 1.0)
    brake = np.clip(brake, 0.0, 1.0)
    control = carla.VehicleControl(throttle, steer, brake, hand_brake, reverse)
    vehicle.apply_control(control)



def main(arg, fps_sim, mapid, weather_idx, showmap, model_type):
    # Imports
    from lane_detection.openvino_lane_detector import OpenVINOLaneDetector
    from lane_detection.lane_detector import LaneDetector
    from lane_detection.camera_geometry import CameraGeometry
    from control.pure_pursuit import PurePursuitPlusPID

    actor_list = []
    pygame.init()

    display, font, clock, world = create_carla_world(pygame, mapid)

    weather_presets = find_weather_presets()
    world.set_weather(weather_presets[weather_idx][0])

    controller = PurePursuitPlusPID()
    cross_track_list = []
    fps_list = []

    client = carla.Client(arg.host, arg.port)
    client.set_timeout(10.0)
    world = client.get_world()

    try:
        original_settings = world.get_settings()
        settings = world.get_settings()
        traffic_manager = client.get_trafficmanager(8000)
        traffic_manager.set_synchronous_mode(True)

        delta = 0.05

        settings.fixed_delta_seconds = delta
        settings.synchronous_mode = True
        settings.no_rendering_mode = arg.no_rendering
        world.apply_settings(settings)

        CARLA_map = world.get_map()

        # create a vehicle
        blueprint_library = world.get_blueprint_library()
        veh_bp = random.choice(blueprint_library.filter('vehicle.audi.tt'))
        veh_bp.set_attribute('color','64,81,181')
        spawn_point = random.choice(CARLA_map.get_spawn_points())

        vehicle = world.spawn_actor(veh_bp, spawn_point)
        actor_list.append(vehicle)

        # Show map
        if showmap:
            plot_map(CARLA_map, mapid, vehicle)

        startPoint = spawn_point
        startPoint = carla_vec_to_np_array(startPoint.location)

        #Lidar
        lidar_bp = generate_lidar_bp(arg, world, blueprint_library, delta)

        user_offset = carla.Location(arg.x, arg.y, arg.z)
        lidar_transform = carla.Transform(carla.Location(x=-0.5, z=1.8) + user_offset)

        lidar = world.spawn_actor(lidar_bp, lidar_transform, attach_to=vehicle)

        point_list = o3d.geometry.PointCloud()
        if arg.semantic:
            lidar.listen(lambda data: semantic_lidar_callback(data, point_list))
        else:
            lidar.listen(lambda data: lidar_callback(data, point_list))

        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name='Carla Lidar',
            width=960,
            height=540,
            left=480,
            top=270)
        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1
        vis.get_render_option().show_coordinate_frame = True

        if arg.show_axis:
            add_open3d_axis(vis)

        frame = 0
        dt0 = datetime.now()

        # visualization cam (no functionality)
        camera_rgb = world.spawn_actor(
            blueprint_library.find('sensor.camera.rgb'),
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            attach_to=vehicle)
        actor_list.append(camera_rgb)
        sensors = [camera_rgb]
        
        # Lane Detector Model
        # ---------------------------------
        cg = CameraGeometry()
        
        # TODO: Change model here
        if model_type == "openvino":
            lane_detector = OpenVINOLaneDetector()
        else:
            lane_detector = LaneDetector(model_path=Path("lane_detection/Deeplabv3+(MobilenetV2).pth").absolute())

        # Windshield cam
        cam_windshield_transform = carla.Transform(carla.Location(x=0.5, z=cg.height), carla.Rotation(pitch=-1*cg.pitch_deg))
        bp = blueprint_library.find('sensor.camera.rgb')
        fov = cg.field_of_view_deg
        bp.set_attribute('image_size_x', str(cg.image_width))
        bp.set_attribute('image_size_y', str(cg.image_height))
        bp.set_attribute('fov', str(fov))
        camera_windshield = world.spawn_actor(
            bp,
            cam_windshield_transform,
            attach_to=vehicle)
        actor_list.append(camera_windshield)
        sensors.append(camera_windshield)
        # ---------------------------------

        flag = True
        max_error = 0
        FPS = fps_sim

        # Create a synchronous mode context.
        with CarlaSyncMode(world, *sensors, fps=FPS) as sync_mode:
            while True:
                if should_quit():
                    return
                clock.tick()          
                
                # Advance the simulation and wait for the data. 
                tick_response = sync_mode.tick(timeout=2.0)

                snapshot, image_rgb, image_windshield = tick_response
                try:
                    trajectory, img = get_trajectory_from_lane_detector(lane_detector, image_windshield)
                except:
                    trajectory = get_trajectory_from_map(CARLA_map, vehicle)

                max_curvature = get_curvature(np.array(trajectory))
                if max_curvature > 0.005 and flag == False:
                    move_speed = np.abs(25 - 100*max_curvature)
                else:
                    move_speed = 25

                speed = np.linalg.norm( carla_vec_to_np_array(vehicle.get_velocity()))
                throttle, steer = controller.get_control(trajectory, speed, desired_speed=move_speed, dt=1./FPS)
                send_control(vehicle, throttle, steer, 0)

                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                dist = dist_point_linestring(np.array([0,0]), trajectory)

                cross_track_error = int(dist)
                max_error = max(max_error, cross_track_error)
                if cross_track_error > 0:
                    cross_track_list.append(cross_track_error)
                waypoint = CARLA_map.get_waypoint(vehicle.get_transform().location)
                vehicle_loc = carla_vec_to_np_array(waypoint.transform.location)

                if np.linalg.norm(vehicle_loc-startPoint) > 20:
                    flag = False

                if np.linalg.norm(vehicle_loc-startPoint) < 20 and flag == False:
                    print('done.')
                    break
                
                if speed < 1 and flag == False:
                    print("----------------------------------------\nSTOP, car accident !!!")
                    break

                fontText = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 0.75
                fontColor = (255,255,255)
                lineType = 2
                laneMessage = "No Lane Detected"
                steerMessage = ""
                
                if dist < 0.75:
                    laneMessage = "Lane Tracking: Good"
                else:
                    laneMessage = "Lane Tracking: Bad"

                cv2.putText(img, laneMessage,
                        (350,50),
                        fontText,
                        fontScale,
                        fontColor,
                        lineType)             

                if steer > 0:
                    steerMessage = "Right"
                else:
                    steerMessage = "Left"

                cv2.putText(img, "Steering: {}".format(steerMessage),
                        (400,90),
                        fontText,
                        fontScale,
                        fontColor,
                        lineType)

                steerMessage = ""
                laneMessage = "No Lane Detected"

                cv2.putText(img, "X: {:.2f}, Y: {:.2f}".format((vehicle_loc[0]), vehicle_loc[1], vehicle_loc[2]),
                            (20,50),
                            fontText,
                            0.5,
                            fontColor,
                            lineType)

                cv2.imshow('Lane detect', img)
                cv2.waitKey(1)

                fps_list.append(clock.get_fps())

                # Draw the display pygame.
                draw_image(display, image_rgb)
                display.blit(
                    font.render('     FPS (real) % 5d ' % clock.get_fps(), True, (255, 255, 255)),
                    (8, 10))
                display.blit(
                    font.render('     FPS (simulated): % 5d ' % fps, True, (255, 255, 255)),
                    (8, 28))
                display.blit(
                    font.render('     speed: {:.2f} km/h'.format(speed*3.6), True, (255, 255, 255)),
                    (8, 46))
                display.blit(
                    font.render('     cross track error: {:03d} m'.format(cross_track_error*100), True, (255, 255, 255)),
                    (8, 64))
                display.blit(
                    font.render('     max cross track error: {:03d} m'.format(max_error), True, (255, 255, 255)),
                    (8, 82))

                pygame.display.flip()
                if frame == 2:
                    vis.add_geometry(point_list)
                vis.update_geometry(point_list)

                vis.poll_events()
                vis.update_renderer()
                # # This can fix Open3D jittering issues:
                time.sleep(0.005)
                world.tick()

                process_time = datetime.now() - dt0
                sys.stdout.write('\r' + 'FPS: ' + str(1.0 / process_time.total_seconds()))
                sys.stdout.flush()
                dt0 = datetime.now()
                frame += 1


    finally:
        world.apply_settings(original_settings)
        traffic_manager.set_synchronous_mode(False)
        vehicle.destroy()
        lidar.destroy()
        vis.destroy_window()
        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()
        print('mean cross track error: ',np.mean(np.array(cross_track_list)))
        print('mean fps: ',np.mean(np.array(fps_list)))
        pygame.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs Carla simulation with your control algorithm.')
    parser.add_argument("--mapid", default = "4", help="Choose map from 1 to 5")
    parser.add_argument("--fps", type=int, default=20, help="Setting FPS")
    parser.add_argument("--weather", type=int, default=6, help="Check function find_weather in carla_util.py for mor information")
    parser.add_argument("--showmap", type=bool, default=False, help="Display Map")
    parser.add_argument("--model", default="openvino", help="Choose between OpenVINO model and PyTorch model")
    parser.add_argument(
        '--host',
        metavar='H',
        default='localhost',
        help='IP of the host CARLA Simulator (default: localhost)')
    parser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port of CARLA Simulator (default: 2000)')
    parser.add_argument(
        '--no-rendering',
        action='store_true',
        help='use the no-rendering mode which will provide some extra'
        ' performance but you will lose the articulated objects in the'
        ' lidar, such as pedestrians')
    parser.add_argument(
        '--semantic',
        action='store_true',
        help='use the semantic lidar instead, which provides ground truth'
        ' information')
    parser.add_argument(
        '--no-noise',
        action='store_true',
        help='remove the drop off and noise from the normal (non-semantic) lidar')
    parser.add_argument(
        '--no-autopilot',
        action='store_false',
        help='disables the autopilot so the vehicle will remain stopped')
    parser.add_argument(
        '--show-axis',
        action='store_true',
        help='show the cartesian coordinates axis')
    parser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='model3',
        help='actor filter (default: "vehicle.*")')
    parser.add_argument(
        '--upper-fov',
        default=15.0,
        type=float,
        help='lidar\'s upper field of view in degrees (default: 15.0)')
    parser.add_argument(
        '--lower-fov',
        default=-25.0,
        type=float,
        help='lidar\'s lower field of view in degrees (default: -25.0)')
    parser.add_argument(
        '--channels',
        default=64.0,
        type=float,
        help='lidar\'s channel count (default: 64)')
    parser.add_argument(
        '--range',
        default=100.0,
        type=float,
        help='lidar\'s maximum range in meters (default: 100.0)')
    parser.add_argument(
        '--points-per-second',
        default=500000,
        type=int,
        help='lidar\'s points per second (default: 500000)')
    parser.add_argument(
        '-x',
        default=0.0,
        type=float,
        help='offset in the sensor position in the X-axis in meters (default: 0.0)')
    parser.add_argument(
        '-y',
        default=0.0,
        type=float,
        help='offset in the sensor position in the Y-axis in meters (default: 0.0)')
    parser.add_argument(
        '-z',
        default=0.0,
        type=float,
        help='offset in the sensor position in the Z-axis in meters (default: 0.0)')
    args = parser.parse_args()

    try:
        main(args, fps_sim = args.fps, mapid = args.mapid, weather_idx=args.weather, showmap=args.showmap, model_type=args.model)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
