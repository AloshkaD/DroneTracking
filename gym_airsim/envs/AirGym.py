import logging
import numpy as np
import time
import gym
import sys
from collections import Counter
from gym import spaces
from gym.utils import seeding
from gym.spaces import Box
from gym.spaces.box import Box

from gym_airsim.envs.myAirSimClient import *
        
from AirSimClient import *

logger = logging.getLogger(__name__)
target1_new_position=[2,-1,-2]

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
offset = {'x_val':2.0,'y_val':0.0,'z_val':-2.0}
offset = dotdict(offset)
class AirSimEnv(gym.Env):

    airgym = None
        
    def __init__(self):
        
        global airgym
        airgym = myAirSimClient()
        self.drone1_vehicle_name = "Drone1"
        self.target1_vehicle_name = "Target1"
        self.cascade_1=10
        self.cascade_2=6
        self.interception=2

        self.simage = np.zeros((90, 256,1), dtype=np.uint8)
        self.rgbimage = np.zeros((90, 256,3), dtype=np.uint8)
        self.svelocity = np.zeros((3,), dtype=np.float32)
        self.sdistance = np.zeros((3,), dtype=np.float32)
        self.sgeofence = np.zeros((6,), dtype=np.float32)
       
        self.stotalvelocity = np.zeros((12,), dtype=np.float32)
        self.stotaldistance = np.zeros((12,), dtype=np.float32)
        self.stotalgeofence = np.zeros((24,), dtype=np.float32)
    
        self.action_space = spaces.Discrete(6)
        #self.drone = trackgym.simGetGroundTruthKinematics(self.drone1_vehicle_name).position
        self.init_goal = airgym.simGetGroundTruthKinematics(self.target1_vehicle_name).position 	#[137.5, -48.7,-4]
        self.goal={'x_val':self.init_goal.x_val+offset.x_val, 'y_val':self.init_goal.y_val+offset.y_val,'z_val':self.init_goal.z_val}
        self.goal = dotdict(self.goal) 
         
        self.distance = np.sqrt(np.power((self.goal.x_val),2) + np.power((self.goal.y_val),2) + np.power((self.goal.z_val),2))
        
        self.episodeN = 0
        self.stepN = 0

        
        #Additional log info
        self.allLogs = { 'reward':[0] }
        self.allLogs['distance'] = [self.distance]
        self.allLogs['track'] = [-2]
        self.allLogs['action'] = [1]
        self.allLogs['svelocity'] = self.svelocity
        self.allLogs['sdistance'] = self.sdistance
        self.allLogs['sgeofence'] = self.sgeofence
         

        self.seed()
        

        
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def computeDistance(self, goal):
        
        distance = np.sqrt(np.power((self.goal.x_val),2) + np.power((self.goal.y_val),2)+ np.power((self.goal.z_val),2))
        
        return distance
        
    
    def state(self, prevVel, prevDst, prevGeo):
        
        totalVel = np.concatenate([self.svelocity, prevVel])
        totalDst = np.concatenate([self.sdistance, prevDst])
        totalGeo = np.concatenate([self.sgeofence, prevGeo])
        
        return self.rgbimage, self.simage, totalVel, totalDst, totalGeo
    
    def computeReward(self, current_position,heading):

        distance_now = np.sqrt(np.power((self.goal.x_val-current_position.x_val),2) + np.power((self.goal.y_val-current_position.y_val),2)+ np.power((self.goal.z_val-current_position.z_val),2))
        distance_before = self.allLogs['distance'][-1]     
        #r = 1
        #r = r - (distance_now**0.4)#(distance_before - distance_now)# - abs(heading_to_target/10)
             
        if  self.cascade_1 >= distance_now >= self.cascade_2:
            r = 1 
            dist_reward=r - (distance_now**0.4)
            heading_reward=1/abs(max(heading,0.1))**0.4
            r=dist_reward+heading_reward
             
        elif self.cascade_2 >= distance_now:
            r = 1 
            z_difference=abs(abs(self.goal.z_val)-abs(current_position.z_val))
            z_reward=1/max(z_difference,0.1)**0.4
            dist_reward=r - (distance_now**0.4)
            heading_reward=1/abs(max(heading,0.1))**0.4
            r=z_reward+dist_reward+heading_reward
        else:
            r = 1
            r = r - (distance_now**0.4)
        return r, distance_now
    def step(self, action):
         
        assert self.action_space.contains(action), "%r (%s) invalid"%(action, type(action))
        
        self.addToLog('action', action)
        
        self.stepN += 1

        collided, collided_with = airgym.take_action(action,self.drone1_vehicle_name)
        self.init_goal = airgym.simGetGroundTruthKinematics(self.target1_vehicle_name).position 	#[137.5, -48.7,-4]
        self.goal={'x_val':self.init_goal.x_val+offset.x_val, 'y_val':self.init_goal.y_val+offset.y_val,'z_val':self.init_goal.z_val}
        self.goal = dotdict(self.goal) 
         
        current_position = airgym.simGetGroundTruthKinematics(self.drone1_vehicle_name).position
        heading = airgym.goal_direction(self.goal, current_position) 
        print('heading',heading)
        if collided == True:
            done = True
            if re.match(r'Target\d',collided_with):
                reward = +100.0
                print('I hit my target drone', collided_with)
            else:
                reward = -100.0  
            distance = np.sqrt(np.power((self.goal.x_val-current_position.x_val),2) + np.power((self.goal.y_val-current_position.y_val),2)+ np.power((self.goal.z_val-current_position.z_val),2))
       
        else: 
            done = False
            reward, distance = self.computeReward(current_position,heading)
             

        if distance < self.interception:
            print('Intercepted target at distance',distance)
            done = True
            reward =+100.0

            with open("reached.txt", "a") as myfile:
                myfile.write(str(self.episodeN) + ", ")
           
            
            '''
            landed = airgym.arrived()
            if landed == True:
                done = True
                reward = 100.0
                with open("reached.txt", "a") as myfile:
                    myfile.write(str(self.episodeN) + ", ")
            '''
                
            
        self.addToLog('reward', reward)
        rewardSum = np.sum(self.allLogs['reward'])
        self.addToLog('distance', distance)
        
        self.addToLog('svelocity', self.svelocity)
        self.addToLog('sdistance', self.sdistance)
        self.addToLog('sgeofence', self.sgeofence)
        
        # Terminate the episode on large cumulative amount penalties, 
        # since drone probably got into an unexpected loop of some sort
        if rewardSum < -300:
            done = True
       
        sys.stdout.write("\r\x1b[K{}/{}==>reward/depth/distance: {:.1f}/{:.1f}/{:.1f}   \t  {:.0f}".format(self.episodeN, self.stepN, reward, rewardSum,distance, action))
        sys.stdout.flush()
        
        info = {"x_pos" : current_position.x_val, "y_pos" : current_position.y_val}
        
        self.simage = airgym.getScreenDepthVis(self.drone1_vehicle_name)
        self.rgbimage = airgym.getScreenSceneVis(self.drone1_vehicle_name)
        self.svelocity = airgym.mapVelocity(self.drone1_vehicle_name)
        self.sdistance = airgym.mapDistance(self.goal,self.drone1_vehicle_name)
        self.sgeofence = airgym.mapGeofence(self.drone1_vehicle_name)
        
        preVel, preDst, preGeo = self.gatherPreviousValues()
        
        state = self.state(preVel, preDst, preGeo) 
        
        return state, reward, done, info

    def addToLog (self, key, value):
        if key not in self.allLogs:
            self.allLogs[key] = []
        self.allLogs[key].append(value)
        
    def gatherPreviousValues(self):
        
        vel_last = self.allLogs['svelocity'][-1]
        vel_twolast = self.allLogs['svelocity'][-2]
        vel_threelast = self.allLogs['svelocity'][-3]
        
        dst_last = self.allLogs['sdistance'][-1]
        dst_twolast = self.allLogs['sdistance'][-2]
        dst_threelast = self.allLogs['sdistance'][-3]
        
        geo_last = self.allLogs['sgeofence'][-1]
        geo_twolast = self.allLogs['sgeofence'][-2]
        geo_threelast = self.allLogs['sgeofence'][-3]
        
        preVel = np.concatenate([vel_last, vel_twolast, vel_threelast])
        prevDst = np.concatenate([dst_last, dst_twolast, dst_threelast])
        prevGeo = np.concatenate([geo_last, geo_twolast, geo_threelast])
        #print(" Shape prevDst ", prevDst.shape)
        return preVel, prevDst, prevGeo
        
    def reset(self):
        """
        Resets the state of the environment and returns an initial observation.
        
        # Returns
            observation (object): The initial observation of the space. Initial reward is assumed to be 0.
        """
        totalrewards = np.sum(self.allLogs['reward'])
        with open("rewards.txt", "a") as myfile:
            myfile.write(str(totalrewards) + ", ")
        airgym.AirSim_reset()
        #########################adding a target drone #############################
        #airgym.AirSim_reset(self.target1_vehicle_name)
        
        #airgym.takeoffAsync(vehicle_name=self.target1_vehicle_name)
        airgym.moveToPositionAsync(target1_new_position[0],target1_new_position[1],target1_new_position[2], 5, vehicle_name=self.target1_vehicle_name)
        ############################################################################
       
        
        #arr = np.array([[137.5, -48.7,-4], [59.1, -15.1,-4], [-62.3, -7.35,-4], [123, 77.3,-4]])
        #probs = [.25, .25, .25, .25]
        #indicies = np.random.choice(len(arr), 1, p=probs)
        #array = (arr[indicies])
        #list = (array.tolist())
        #self.goal = [item for sublist in list for item in sublist]
        
        #Simple goal
        #[137.5, -48.7, -6]
        self.init_goal = airgym.simGetGroundTruthKinematics(self.target1_vehicle_name).position 	#[137.5, -48.7,-4]
        self.goal={'x_val':self.init_goal.x_val+offset.x_val, 'y_val':self.init_goal.y_val+offset.y_val,'z_val':self.init_goal.z_val}
        self.goal = dotdict(self.goal) 
        
        self.stepN = 0
        self.episodeN += 1
        
        distance = np.sqrt(np.power((self.goal.x_val),2) + np.power((self.goal.y_val),2)+ np.power((self.goal.z_val),2))
        self.allLogs = { 'reward': [0] }
        self.allLogs['distance'] = [distance]
        self.allLogs['action'] = [1]
        
        self.simage = airgym.getScreenDepthVis(self.drone1_vehicle_name)
        self.rgbimage = airgym.getScreenSceneVis(self.drone1_vehicle_name)
        self.svelocity = airgym.mapVelocity(self.drone1_vehicle_name)
        self.sdistance = airgym.mapDistance(self.goal,self.drone1_vehicle_name)
        self.sgeofence = airgym.mapGeofence(self.drone1_vehicle_name)

        
        #Testing progressive logs
        self.allLogs['svelocity'] = [0, 0, 0]
        self.addToLog('svelocity', [0, 0, 0])
        self.addToLog('svelocity', [0, 0, 0])
        self.addToLog('svelocity', [0, 0, 0])
        self.allLogs['sdistance'] = [0, 0, 0, 0]
        self.addToLog('sdistance', self.sdistance)
        self.addToLog('sdistance', self.sdistance)
        self.addToLog('sdistance', self.sdistance)
        self.addToLog('sdistance', self.sdistance)
        self.allLogs['sgeofence'] = [0, 0, 0, 0, 0, 0]
        self.addToLog('sgeofence', self.sgeofence)
        self.addToLog('sgeofence', self.sgeofence)
        self.addToLog('sgeofence', self.sgeofence)
        self.addToLog('sgeofence', self.sgeofence)
        
        
        preVel, preDst, preGeo = self.gatherPreviousValues()
       
        state = self.state(preVel, preDst, preGeo)
        
        
        return state