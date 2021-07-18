# Code based on Carla examples, which are authored by 
# Computer Vision Center (CVC) at the Universitat Autonoma de Barcelona (UAB).

# How to run: 
# cd into the parent directory of the 'code' directory and run
# python -m code.tests.control.carla_sim


import carla
import random
from pathlib import Path
import numpy as np
import pygame
from util.carla_util import carla_vec_to_np_array, carla_img_to_array, CarlaSyncMode, find_weather_presets, draw_image, get_font, should_quit
from util.geometry_util import dist_point_linestring
import argparse
import cv2


def plot_map(m, vehicle):
    import matplotlib.pyplot as plt

    wp_list = m.generate_waypoints(2.0)
    loc_list = np.array(
        [carla_vec_to_np_array(wp.transform.location) for wp in wp_list]
    )
    plt.scatter(loc_list[:, 0], loc_list[:, 1])

    wp = m.get_waypoint(vehicle.get_transform().location)
    vehicle_loc = carla_vec_to_np_array(wp.transform.location)
    plt.scatter([vehicle_loc[0]], [vehicle_loc[1]])
    plt.show()

def get_curvature(polyline):
    dx_dt = np.gradient(polyline[:, 0])
    dy_dt = np.gradient(polyline[:, 1])
    d2x_dt2 = np.gradient(dx_dt)
    d2y_dt2 = np.gradient(dy_dt)
    curvature = (
        np.abs(d2x_dt2 * dy_dt - dx_dt * d2y_dt2)
        / (dx_dt * dx_dt + dy_dt * dy_dt) ** 1.5
    )
    # print(curvature)
    return np.max(curvature)

def get_trajectory_from_lane_detector(ld, image):
    # get lane boundaries using the lane detector
    image_arr = carla_img_to_array(image)

    poly_left, poly_right, img_left, img_right = ld(image_arr)
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
    traj = np.stack((x,y)).T
    return traj, img

def get_trajectory_from_map(m, vehicle):
    # get 80 waypoints each 1m apart. If multiple successors choose the one with lower waypoint.id
    wp = m.get_waypoint(vehicle.get_transform().location)
    wps = [wp]
    for _ in range(20):
        next_wps = wp.next(1.0)
        if len(next_wps) > 0:
            wp = sorted(next_wps, key=lambda x: x.id)[0]
        wps.append(wp)

    # transform waypoints to vehicle ref frame
    traj = np.array(
        [np.array([*carla_vec_to_np_array(x.transform.location), 1.]) for x in wps]
    ).T
    trafo_matrix_world_to_vehicle = np.array(vehicle.get_transform().get_inverse_matrix())

    traj = trafo_matrix_world_to_vehicle @ traj
    traj = traj.T
    traj = traj[:,:2]
    return traj

def send_control(vehicle, throttle, steer, brake,
                 hand_brake=False, reverse=False):
    throttle = np.clip(throttle, 0.0, 1.0)
    steer = np.clip(steer, -1.0, 1.0)
    brake = np.clip(brake, 0.0, 1.0)
    control = carla.VehicleControl(throttle, steer, brake, hand_brake, reverse)
    vehicle.apply_control(control)



def main(fps_sim, mapid):
    # Imports
    from lane_detection.lane_detector import LaneDetector
    from lane_detection.camera_geometry import CameraGeometry
    from control.pure_pursuit import PurePursuitPlusPID

    actor_list = []
    pygame.init()

    display = pygame.display.set_mode(
        (800, 600),
        pygame.HWSURFACE | pygame.DOUBLEBUF)
    font = get_font()
    clock = pygame.time.Clock()

    client = carla.Client('localhost', 2000)
    client.set_timeout(40.0)

    client.load_world('Town0' + mapid)
    world = client.get_world()
    weather_presets = find_weather_presets()
    world.set_weather(weather_presets[3][0])

    controller = PurePursuitPlusPID()

    try:
        m = world.get_map()

        if mapid == '4':
            spawn_id = 90
        elif mapid == '5':
            spawn_id = 49
        elif mapid == '2':
            spawn_id = 18
        elif mapid == '3':
            spawn_id = 70
        else:
            spawn_id = 40

        blueprint_library = world.get_blueprint_library()

        veh_bp = random.choice(blueprint_library.filter('vehicle.audi.tt'))
        veh_bp.set_attribute('color','64,81,181')
        vehicle = world.spawn_actor(
            veh_bp,
            m.get_spawn_points()[spawn_id])
        actor_list.append(vehicle)

        # Show map here
        # plot_map(m, vehicle)

        startPoint = m.get_spawn_points()[spawn_id]
        startPoint = carla_vec_to_np_array(startPoint.location)

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
        
        # Change model here
        ld = LaneDetector(model_path=Path("lane_detection/best_model_multi_dice_loss-5.pth").absolute())

        #windshield cam
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
        FPS = int(fps_sim)
        cross_track_list = []
        fps_list = []

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
                    traj, img = get_trajectory_from_lane_detector(ld, image_windshield)
                except:
                    traj = get_trajectory_from_map(m, vehicle)

                max_curvature = get_curvature(np.array(traj))
                if max_curvature > 0.005 and flag == False:
                    move_speed = np.abs(25 - 100*max_curvature)
                else:
                    move_speed = 25

                speed = np.linalg.norm( carla_vec_to_np_array(vehicle.get_velocity()))
                throttle, steer = controller.get_control(traj, speed, desired_speed=move_speed, dt=1./FPS)
                send_control(vehicle, throttle, steer, 0)

                fps = round(1.0 / snapshot.timestamp.delta_seconds)

                dist = dist_point_linestring(np.array([0,0]), traj)

                cross_track_error = int(dist)
                max_error = max(max_error, cross_track_error)
                if cross_track_error > 0:
                    cross_track_list.append(cross_track_error)
                wp = m.get_waypoint(vehicle.get_transform().location)
                vehicle_loc = carla_vec_to_np_array(wp.transform.location)

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


    finally:
        print('destroying actors.')
        for actor in actor_list:
            actor.destroy()
        print('mean cross track error: ',np.mean(np.array(cross_track_list)))
        print('mean fps: ',np.mean(np.array(fps_list)))
        pygame.quit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Runs Carla simulation with your control algorithm.')
    parser.add_argument("--mapid", default = "4", help="Choose map from 1 to 5")
    parser.add_argument("--fps", default="20", help="Setting FPS")
    args = parser.parse_args()

    try:
        main(fps_sim = args.fps, mapid = args.mapid)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
