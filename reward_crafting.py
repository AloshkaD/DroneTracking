r = 1
distance =[1,2,3,4,5,6,7,8,9,10,11,12,13,14,100,200]
heading=300
geo=200
velocity =5
z_target=-9
z_drone=-4
z_difference=abs(abs(z_target)-abs(z_drone))
for i in distance:
    print('distance',i)
    dist_reward_new=r - ((i/geo)**0.4)
    dist_reward_old=(-1*r) - (i*0.1)
    heading_reward=1/max(heading**0.4,1)
    z_reward=1/max(z_difference**0.6,1)
    if i <= 10:
        dist_reward_new=dist_reward_new*heading_reward
    elif i <=6:
        dist_reward_new=dist_reward_new*heading_reward*z_reward
    print('final reward_old=', dist_reward_old)
    print('final reward_new=', dist_reward_new)

#dist_reward = 1-(distance**0.4)
#vel_discount=(1-max(velocity,0.1))**(1/max(distance,0.1))
#value=vel_discount*dist_reward

