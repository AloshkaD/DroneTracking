import numpy as np
from operator import itemgetter
import time
import math
import cv2
from pylab import array, uint8 
from PIL import Image


from AirSimClient import *


class myAirSimClient(MultirotorClient):

    def __init__(self):  
        self.drone1_vehicle_name = "Drone1"
        self.target1_vehicle_name = "Target1"
        self.img1 = None
        self.img2 = None

        MultirotorClient.__init__(self)
        MultirotorClient.confirmConnection(self)
        self.enableApiControl(True)
        self.armDisarm(True)
    
        self.home_pos = self.simGetGroundTruthKinematics(self.drone1_vehicle_name).position
 
        self.home_ori = self.simGetGroundTruthKinematics(self.drone1_vehicle_name).orientation
        
        #Define your geofence inside the map
        
        self.minx = -200
        self.maxx = 200
        self.miny = -200
        self.maxy = 200
        self.minz = -2
        self.maxz = -30
        '''
        Small geofence limits
        self.minx = -50
        self.maxx = 180
        self.miny = -90
        self.maxy = 30
        self.minz = -2
        self.maxz = -20
        '''
        self.z = -4
        
    def movement(self, speed_x, speed_y, speed_z, duration,vehicle_name=''):
        
        pitch, roll, yaw  = self.getPitchRollYaw(vehicle_name)
        vel = self.simGetGroundTruthKinematics(vehicle_name).linear_velocity
        drivetrain = DrivetrainType.ForwardOnly
        yaw_mode = YawMode(is_rate= False, yaw_or_rate = 0)

        self.moveByVelocityAsync(vx = vel.x_val + speed_x,
                            vy = vel.y_val + speed_y,
                            vz = vel.z_val + speed_z,
                            duration = duration,
                            drivetrain = drivetrain,
                            yaw_mode = yaw_mode,vehicle_name= vehicle_name)
        

    
    def take_action(self, action,vehicle_name= ''):
        
		 #check if copter is on level cause sometimes he goes up without a reason

        start = time.time()
        duration = 1 
        collided_with = ''
        outside = self.geofence(self.minx, self.maxx, 
                                self.miny, self.maxy,
                                self.minz, self.maxz)
        
        if action == 0:
            
            self.movement(0.5, 0, 0, duration,vehicle_name)
    
        elif action == 1:
         
            self.movement(-0.5, 0, 0, duration,vehicle_name)
                
        elif action == 2:
            
            self.movement(0, 0.5, 0, duration,vehicle_name)
            
                
        elif action == 3:
                    
            self.movement(0, -0.5, 0, duration,vehicle_name)
            
        elif action == 4:
                    
            self.movement(0, 0, 0.5, duration,vehicle_name)
                
        elif action == 5:
                    
            self.movement(0, 0, -0.5, duration,vehicle_name)      
        
        while duration > time.time() - start:
                if self.simGetCollisionInfo(vehicle_name).has_collided == True:
                    print('collision_info',self.simGetCollisionInfo(vehicle_name).object_name)
                    collided_with = self.simGetCollisionInfo(vehicle_name).object_name
                    return True,collided_with 
                       
                if outside == True:
                    print('We have reached the geofence coordinates')
                    collided_with='geofence'
                    return True,collided_with 
                
        return False,collided_with
   
    #def geofence(self, minx, maxx, miny, maxy, minz, maxz):
        
        #outside = False
        
        #if (self.getPosition().x_val < minx) or (self.getPosition().x_val > maxx):
         #           return True
        #f (self.getPosition().y_val < miny) or (self.getPosition().y_val > maxy):
                    #return True
        #if (self.getPosition().z_val > minz) or (self.getPosition().z_val < maxz):
         #           return True
                
        #return outside
    
    def geofence(self, minx, maxx, miny, maxy,minz, maxz,vehicle_name=''):
        position = self.simGetGroundTruthKinematics(vehicle_name).position
        outside = False
        
        if (position.x_val < minx) or (position.x_val > maxx):
                    return True
        if (position.y_val < miny) or (position.y_val > maxy):
                    return True
        if (position.z_val < maxz):
                    return True    
        return outside

    def arrived(self,vehicle_name=''):
        position = self.simGetGroundTruthKinematics(vehicle_name).position
        landed = self.moveToZAsync(0, 1,vehicle_name)
    
        if landed == True:
            return landed
        
        if (position.z_val > -1):
            return True
        
    def goal_direction(self, goal, pos, vehicle_name=''):
        
        pitch, roll, yaw  = self.getPitchRollYaw(vehicle_name)
        yaw = math.degrees(yaw) 
        
        #pos_angle = math.atan2(goal[1] - pos.y_val, goal[0]- pos.x_val)
        pos_angle = math.atan2(goal.y_val - pos.y_val, goal.x_val- pos.x_val)
        pos_angle = math.degrees(pos_angle) % 360

        track = math.radians(pos_angle - yaw)  
        
        return ((math.degrees(track) - 180) % 360) - 180   
    
    '''
    Position data inside environment 
    def mapPosition(self):
        
        xval = self.getPosition().x_val
        yval = self.getPosition().y_val
        
        position = np.array([xval, yval])
        
        return position
    '''
    def mapVelocity(self,vehicle_name=''):
        
        vel = self.simGetGroundTruthKinematics(vehicle_name).linear_velocity
        
        velocity = np.array([vel.x_val, vel.y_val, vel.z_val])
        
        return velocity
    '''
    def mapGeofence(self):
        position = self.simGetGroundTruthKinematics().position
        xpos = position.x_val
        ypos = position.y_val
        zpos = position.z_val
        
        geox1 = self.maxx - xpos
        geox2 = self.minx - xpos
        geoy1 = self.maxy - ypos
        geoy2 = self.miny - ypos
        geoz1 = self.maxz - zpos
        geoz2 = self.minz - zpos
        
        geofence = np.array([geox1, geox2, geoy1, geoy2, geoz1, geoz2])
        
        return geofence
    '''
    def mapGeofence(self,vehicle_name=''):
        position = self.simGetGroundTruthKinematics(vehicle_name).position
        xpos = position.x_val
        ypos = position.y_val
        zpos = position.z_val
        
        geox1 = self.maxx - xpos
        geox2 = self.minx - xpos
        geoy1 = self.maxy - ypos
        geoy2 = self.miny - ypos
        geoz1 = self.maxz - zpos
        geoz2 = self.minz - zpos
        
        geofence = np.array([geox1, geox2, geoy1, geoy2, geoz1,geoz2])
        return geofence
    def mapDistance(self, goal,vehicle_name='' ):
        position = self.simGetGroundTruthKinematics(vehicle_name).position
        x = [0]
        y = [1]
        z = [2]
        goalx = goal.x_val#itemgetter(*x)(goal)
        goaly = goal.y_val#itemgetter(*y)(goal)
        goalz = goal.z_val#itemgetter(*z)(goal)
        xdistance = (goalx - (position.x_val))
        ydistance = (goaly - (position.y_val))
        zdistance = (goalz - (position.z_val))
        meandistance = np.sqrt(np.power((goalx -position.x_val),2) + np.power((goaly - position.y_val),2)+np.power((goalz -position.z_val),2))
        #distances = np.array([xdistance, ydistance,zdistance, meandistance])
        distances = np.array([xdistance, ydistance, meandistance])
        
        return distances
    
    def getScreenDepthVis(self, vehicle_name=''):

        responses = self.simGetImages([ImageRequest(0, ImageType.DepthPerspective, True, False)],vehicle_name)
        if (responses[0] == None):
            print("Camera is not returning image, please check airsim for error messages")
            sys.exit(0)
        img1d = np.array(responses[0].image_data_float, dtype=np.float)
        img1d = 255/np.maximum(np.ones(img1d.size), img1d)
        if (len(img1d) != (144*256)):
            print('The depth camera returned bad data so Im creating zero array to not break the training')
            img1d=np.ones(144*256)
            img2d = np.reshape(img1d,(144, 256))
            image = np.invert(np.array(Image.fromarray(img2d.astype(np.uint8), mode='L')))
        
            factor = 10
            maxIntensity = 255.0 # depends on dtype of image data
        
        
            # Decrease intensity such that dark pixels become much darker, bright pixels become slightly dark 
            newImage1 = (maxIntensity)*(image/maxIntensity)**factor
            newImage1 = array(newImage1,dtype=uint8)
        
        
            scale_percent = 100 # percent of original size, result 230x129 for 90%
            width = int(newImage1.shape[1] * scale_percent / 100)
            height = int(newImage1.shape[0] * scale_percent / 100)
            dim = (width, height)
            small = cv2.resize(newImage1, dim, interpolation = cv2.INTER_AREA)  

            cut = small[20:110,:]
            cut = np.expand_dims(cut, axis=2) # Equivalent to x[:,:,np.newaxis]
            return cut

        img2d = np.reshape(img1d, (responses[0].height, responses[0].width))
        
        
        image = np.invert(np.array(Image.fromarray(img2d.astype(np.uint8), mode='L')))
        
        factor = 10
        maxIntensity = 255.0 # depends on dtype of image data
        
        
        # Decrease intensity such that dark pixels become much darker, bright pixels become slightly dark 
        newImage1 = (maxIntensity)*(image/maxIntensity)**factor
        newImage1 = array(newImage1,dtype=uint8)
        
        
        scale_percent = 100 # percent of original size, result 230x129 for 90%
        width = int(newImage1.shape[1] * scale_percent / 100)
        height = int(newImage1.shape[0] * scale_percent / 100)
        dim = (width, height)
        small = cv2.resize(newImage1, dim, interpolation = cv2.INTER_AREA)
        
       
        cut = small[20:110,:]
        
        #cv2.imshow("Test", total)
        #cv2.waitKey(0)
        cut = np.expand_dims(cut, axis=2) # Equivalent to x[:,:,np.newaxis]
        return cut
    def getScreenSceneVis(self,vehicle_name):
        #reference for processing the images https://stackoverflow.com/questions/49621599/reshaping-an-gym-array-for-tensorflow
        responses = self.simGetImages([ImageRequest("0", ImageType.Scene, False, False)],vehicle_name)
        if (responses == None):
            print("Camera is not returning image, please check airsim for error messages")
            sys.exit(0)
 
         
        img1d = np.array(bytearray(responses[0].image_data_uint8))#.reshape(responses[0].height, responses[0].width, 4)
        if len(img1d) != (144*256*4):
            print('The RGB camera returned bad data so Im creating zero array to not break the training')
            img1d=np.ones(144*256*4)
            img2d = np.reshape(img1d,(144, 256,4))
            newImage1 = array(img2d,dtype=uint8)
            scale_percent = 100 # percent of original size
            width = int(newImage1.shape[1] * scale_percent / 100)
            height = int(newImage1.shape[0] * scale_percent / 100)
            dim = (width, height)
            small = cv2.resize(newImage1, dim, interpolation = cv2.INTER_AREA)  
            cut = small[20:110,:]
            #cv2.imshow("Test", total)
            #cv2.waitKey(0)

            cut=cut[:,:,0:3]
            return cut
        img2d=img1d.reshape(responses[0].height, responses[0].width, 4)   
        #img_gray =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        factor = 10
        maxIntensity = 255.0 # depends on dtype of image data
        
        
        # Decrease intensity such that dark pixels become much darker, bright pixels become slightly dark 
        #newImage1 = (maxIntensity)*(img2d/maxIntensity)**factor
        #newImage1 = array(newImage1,dtype=uint8) 
        scale_percent = 100 # percent of original size
        width = int(img2d.shape[1] * scale_percent / 100)
        height = int(img2d.shape[0] * scale_percent / 100)
        dim = (width, height)
        small = cv2.resize(img2d, dim, interpolation = cv2.INTER_AREA)  
        cut = small[20:110,:]
        #cv2.imshow("Test", total)
        #cv2.waitKey(0)
        #take the 3 channels only
        cut=cut[:,:,0:3]
        
        return cut

    '''
    def AirSim_reset(self):
        
        self.reset()
        time.sleep(0.2)
        self.enableApiControl(True)
        self.armDisarm(True)
        time.sleep(1)
        self.moveToZAsync(self.z, 3) 
        time.sleep(3)
    '''
    
    def takeoff(self,vehicle_name=''):

        self.takeoffAsync(vehicle_name=vehicle_name) 
        #self.moveToZAsync(self.z, 3, vehicle_name)
        #print('moving to coordinates',coordinates)
        #self.straight(duration,speed, vehicle_name=vehicle_name)

    def AirSim_reset(self):        
        self.reset()
        time.sleep(0.2)
        self.enableApiControl(True,self.drone1_vehicle_name)
        self.enableApiControl(True,self.target1_vehicle_name)
        self.armDisarm(True,self.target1_vehicle_name)
        self.armDisarm(True,self.drone1_vehicle_name)
        time.sleep(1)
        #self.moveToZAsync(self.z, 3, self.drone1_vehicle_name)

        self.takeoff(self.drone1_vehicle_name) 
        time.sleep(3)

