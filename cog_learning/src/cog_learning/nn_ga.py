#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64
from std_msgs.msg import Bool
from cog_learning.multilayer import *
from cog_learning.hebb_server import *
from cog_learning.skill import *
from detector.msg import Outcome
from motion.msg import DmpAction
from motion.msg import Dmp
from cog_learning.msg import Goal
from sklearn.preprocessing import MinMaxScaler
import copy

class NNGoalAction(object):
    def __init__(self, id_obj):
        self.pub_update_lp = rospy.Publisher('/intrinsic/goal_error', Goal, latch=True, queue_size=1)
        self.pub_new_goal = rospy.Publisher('/intrinsic/new_goal', Goal, latch=True, queue_size=1)
        self.pub_timer = rospy.Publisher('/intrinsic/updating_lp', Float64, latch=True, queue_size=1)
        self.pub_end = rospy.Publisher('/intrinsic/end_action', Bool, queue_size=10)
        self.pub_dmp = rospy.Publisher('/motion_pincher/activate_dmp', Dmp, queue_size=10)
        self.pub_ready = rospy.Publisher('/cog_learning/ready', Bool, queue_size=10)
        self.encoder = MultiLayerEncoder(9,2)#9,6,4,2
        self.encoder.to(device)
        self.decoder = MultiLayerDecoder(2,4,6,9)
        self.decoder.to(device)
        self.memory_size = 20
        self.memory = []
        self.hebbian = HebbServer()
        self.id_nnga =  id_obj
        self.index_skill = -1
        self.skills = []
        self.min_x = 0.18
        self.max_x = 0.67
        self.min_y = -0.35
        self.max_y = 0.35
        self.min_pitch = 0
        self.max_pitch = 1.5
        self.min_vx = -0.2
        self.max_vx = 0.2
        self.min_vy = -0.2
        self.max_vy = 0.2
        self.min_vpitch = -1.2
        self.max_vpitch = 1.2
        self.min_roll = -1.5
        self.max_roll = 1.5
        self.min_grasp = 0
        self.max_grasp = 1
        self.min_angle = -180
        self.max_angle = 180
        torch.manual_seed(32)

    def update_learning_progress(self, data, error):
        update_lp = Goal()
        update_lp.x = data[0]
        update_lp.y = data[1]
        update_lp.value = error
        self.pub_update_lp.publish(update_lp)
        
    def send_new_goal(self, data):
        new_goal = Goal()
        new_goal.x = data[0]
        new_goal.y = data[1]
        new_goal.value = 1.0
        self.pub_new_goal.publish(new_goal)

    def pub_timing(self, value):
        v = Float64()
        v.data = value
        self.pub_timer.publish(v)

    def send_ready(self, value):
        b = Bool()
        b.data = value
        self.pub_ready.publish(b)

    def end_action(self,status):
        v = Bool()
        v.data = status
        self.pub_end.publish(v)

    def create_skill(self):
        new_skill = Skill()
        self.skills.append(new_skill)
        ind = len(self.skills) - 1
        return ind
    
    #scale value from min max to [-1,1]
    def scale_data(self, data, min_, max_):
        n_x = np.array(data)
        n_x = n_x.reshape(-1,1)
        scaler_x = MinMaxScaler(feature_range=(-1,1))
        x_minmax = np.array([min_, max_])
        scaler_x.fit(x_minmax[:, np.newaxis])
        n_x = scaler_x.transform(n_x)
        n_x = n_x.reshape(1,-1)
        n_x = n_x.flatten()
            
        return n_x[0]
    
    #scale latent variable to [0,1] for dft
    def scale_latent_to_dnf(self, data):
        n_x = np.array(data)
        n_x = n_x.reshape(-1,1)
        scaler_x = MinMaxScaler()
        x_minmax = np.array([-1, 1])
        scaler_x.fit(x_minmax[:, np.newaxis])
        n_x = scaler_x.transform(n_x)
        n_x = n_x.reshape(1,-1)
        n_x = n_x.flatten()
            
        return n_x[0]
    
    def scale_dnf_to_latent(self, data):
        n_x = np.array(data)
        n_x = n_x.reshape(-1,1)
        scaler_x = MinMaxScaler(feature_range=(-1,1))
        x_minmax = np.array([0, 1])
        scaler_x.fit(x_minmax[:, np.newaxis])
        n_x = scaler_x.transform(n_x)
        n_x = n_x.reshape(1,-1)
        n_x = n_x.flatten()
                    
        return n_x[0]
    
    def reconstruct_latent(self, data, min_, max_):
        n_x = np.array(data)
        n_x = n_x.reshape(-1,1)
        scaler_x = MinMaxScaler(feature_range=(min_,max_))
        x_minmax = np.array([-1, 1])
        scaler_x.fit(x_minmax[:, np.newaxis])
        n_x = scaler_x.transform(n_x)
        n_x = n_x.reshape(1,-1)
        n_x = n_x.flatten()
                    
        return n_x[0]

    def scale_samples(self, outcome, dmp):
        #scale sample betwen [-1,1] for learning
        new_outcome = Outcome()
        new_outcome.state_x = self.scale_data(outcome.state_x, self.min_x, self.max_x)
        new_outcome.state_y = self.scale_data(outcome.state_y, self.min_y, self.max_y)
        new_outcome.state_angle = self.scale_data(outcome.state_angle, self.min_angle, self.max_angle)
        new_outcome.x = self.scale_data(outcome.x, self.min_vx, self.max_vx)
        new_outcome.y = self.scale_data(outcome.y, self.min_vy, self.max_vy)
        new_outcome.angle = self.scale_data(outcome.angle, self.min_angle, self.max_angle)
        new_outcome.touch = outcome.touch
        new_dmp = DmpAction()
        new_dmp.v_x = self.scale_data(dmp.v_x, self.min_vx, self.max_vx)
        new_dmp.v_y = self.scale_data(dmp.v_y, self.min_vy, self.max_vy)
        new_dmp.v_pitch = self.scale_data(dmp.v_pitch, self.min_vpitch, self.max_vpitch)
        new_dmp.roll = self.scale_data(dmp.roll, self.min_roll, self.max_roll)
        new_dmp.grasp = self.scale_data(dmp.grasp, self.min_grasp, self.max_grasp)
        new_dmp.lpos_x = self.scale_data(dmp.lpos_x, self.min_x, self.max_x)
        new_dmp.lpos_y = self.scale_data(dmp.lpos_y, self.min_y, self.max_y)
        new_dmp.lpos_pitch = self.scale_data(dmp.lpos_pitch, self.min_pitch, self.max_pitch)

        return new_outcome, new_dmp


    #bootstrap learning when we discover first skill during exploration
    def bootstrap_learning(self, outcome_, dmp_):
        outcome, dmp = self.scale_samples(outcome_, dmp_)
        #ouput of decoder
        tmp_sample = [outcome.x,outcome.y,outcome.angle,outcome.touch,dmp.v_x,dmp.v_y,dmp.v_pitch,dmp.roll,dmp.grasp]
        print("scaled dmp : ",tmp_sample)
        tensor_sample_go = torch.tensor(tmp_sample,dtype=torch.float)
        sample_inp_fwd = [outcome.state_x,outcome.state_y,outcome.state_angle,dmp.lpos_x,dmp.lpos_y,dmp.lpos_pitch]
        sample_out_fwd = [outcome.x,outcome.y,outcome.angle,outcome.touch]
        sample_inp_inv = [outcome.state_x,outcome.state_y,outcome.state_angle,outcome.x,outcome.y,outcome.angle,outcome.touch]
        sample_out_inv = [dmp.lpos_x,dmp.lpos_y,dmp.lpos_pitch]
        t_inp_fwd = torch.tensor(sample_inp_fwd,dtype=torch.float)
        t_out_fwd = torch.tensor(sample_out_fwd,dtype=torch.float)
        t_inp_inv = torch.tensor(sample_inp_inv,dtype=torch.float)
        t_out_inv = torch.tensor(sample_out_inv,dtype=torch.float)
        self.add_to_memory(tensor_sample_go)
        ind_skill = self.create_skill()
        sample = []
        sample.append(t_out_fwd)
        sample.append(t_out_inv)
        sample.append(t_inp_fwd)
        sample.append(t_inp_inv)
        #print("Input NNGA ; ",tmp_sample)
        self.skills[ind_skill].add_to_memory(sample)
        err_fwd = self.skills[ind_skill].predictForwardModel(sample[2],sample[0])
        err_inv = self.skills[ind_skill].predictInverseModel(sample[3],sample[1])
        error_fwd = err_fwd.item()
        error_inv = err_inv.item()
        #error_fwd = math.tanh(error_fwd)
        #if error_fwd < 0.15:
        #    error_fwd = error_fwd + 0.4
        #error_inv = math.tanh(error_inv)
        print("ERROR INVERSE : ",error_inv)
        print("ERROR FORWARD : ",error_fwd)
        t_inputs = self.encoder(tensor_sample_go)
        output_l = t_inputs.detach().numpy()
        print("latent space : ",output_l)
        t0 = self.scale_latent_to_dnf(output_l[0])
        t1 = self.scale_latent_to_dnf(output_l[1])
        
        inputs = [round(t0*100),round(t1*100)]
        print("dnf input : ",inputs)
        #publish new goal and fwd error
        self.update_learning_progress(inputs,error_fwd)
        self.send_new_goal(inputs)
        self.pub_timing(1.0)
        rospy.sleep(1)
        self.update_learning_progress(inputs,0)
        self.pub_timing(0.0)
        self.end_action(True)
        rospy.sleep(1)
        self.end_action(False)
        print(inputs)
        print("Hebbian learning, index : ",ind_skill)
        self.hebbian.hebbianLearning(inputs,ind_skill)
        self.skills[ind_skill].train_forward_model()
        self.skills[ind_skill].train_inverse_model()
        self.trainDecoder()
        #self.send_ready(True)

    def trainDecoder(self):
        current_cost = 0
        last_cost = 15
        learning_rate = 5e-3
        epochs = 150
        print("Train NNGA DECODER")
        #self.inverse_model.to(device)
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.decoder.parameters(),lr=learning_rate)        
        current_cost = 0
        for i in range(0,1):
            self.decoder.train()
            optimizer.zero_grad()
            sample = self.memory[-1]
            sample = sample.to(device)
            inputs = self.encoder(sample)
            inputs = inputs.to(device)
            targets = sample
            targets = targets.to(device)
            outputs = self.decoder(inputs)
            cost = criterion(outputs,targets)
            cost.backward()
            optimizer.step()
            #current_cost = current_cost + cost.item()
        while last_cost > 0.0001:
            for j in range(0,len(self.memory)):
                self.decoder.train()
                optimizer.zero_grad()
                sample = self.memory[j]
                sample = sample.to(device)
                inputs = self.encoder(sample)
                inputs = inputs.to(device)
                targets = sample
                targets = targets.to(device)
                outputs = self.decoder(inputs)
                cost = criterion(outputs,targets)
                cost.backward()
                optimizer.step()
                current_cost = current_cost + cost.item()
                last_cost = current_cost
            #print("Epoch: {}/{}...".format(i, epochs),"MSE : ",current_cost)
            current_cost = 0
        print("finish training NNGA")

    def forward_encoder(self, data):
        data = data.to(device)
        output = self.encoder(data)

        return output
    
    def forward_decoder(self, data):
        data = data.to(device)
        output = self.decoder(data)

        return output

    def add_to_memory(self, data):
        self.memory.append(data)

    def get_id(self):
        return self.id_nnga
    
    def activate_dmp(self, goal):
        tmp = [goal.latent_x,goal.latent_y]
        tensor_latent = torch.tensor(tmp,dtype=torch.float)
        output = self.decoder(tensor_latent)
        out = output.detach().numpy()
        dmp = Dmp()
        dmp.v_x = self.reconstruct_latent(out[4],self.min_vx,self.max_vx)
        dmp.v_y = self.reconstruct_latent(out[5],self.min_vy,self.max_vy)
        dmp.v_pitch = self.reconstruct_latent(out[6],self.min_vpitch,self.max_vpitch)
        dmp.roll = self.reconstruct_latent(out[7],self.min_roll,self.max_roll)
        dmp.grasp = self.reconstruct_latent(out[8],self.min_grasp,self.max_vx)
        self.pub_dmp.publish(dmp)

    def activate_hebbian(self, goal):
        tmp = [goal.latent_x,goal.latent_y]
        self.index_skill = self.hebbian.hebbianActivation(tmp)
        print("index models : ",self.index_skill)