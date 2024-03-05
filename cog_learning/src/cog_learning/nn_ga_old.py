#!/usr/bin/env python3
import rospy
from std_msgs.msg import Float64
from std_msgs.msg import Bool
from cog_learning.multilayer import *
from cog_learning.hebb_server import *
from cog_learning.skill import *
from detector.msg import Outcome
from detector.msg import State
from motion.msg import DmpOutcome
from motion.msg import Dmp
from motion.msg import Action
from cog_learning.msg import Goal
from cog_learning.msg import LatentGoalDnf
from cog_learning.msg import LatentNNDNF
from habituation.msg import LatentPos
from sklearn.preprocessing import MinMaxScaler
import copy

class NNGoalAction(object):
    def __init__(self, id_obj):
        self.pub_update_lp = rospy.Publisher('/intrinsic/goal_error', Goal, latch=True, queue_size=1)
        self.pub_new_goal = rospy.Publisher('/intrinsic/new_goal', Goal, latch=True, queue_size=1)
        self.pub_timer = rospy.Publisher('/intrinsic/updating_lp', Float64, latch=True, queue_size=1)
        self.pub_end = rospy.Publisher('/intrinsic/end_action', Bool, queue_size=10)
        self.pub_dmp = rospy.Publisher('/motion_pincher/activate_dmp', Dmp, queue_size=10)
        self.pub_latent_space_display = rospy.Publisher("/display/latent_space", LatentPos, queue_size=1, latch=True)
        self.pub_latent_space_dnf = rospy.Publisher("/intrinsic/latent_space_dnf", LatentNNDNF, queue_size=1, latch=True)
        self.pub_ready = rospy.Publisher("/cog_learning/ready", Bool, queue_size=1, latch=True)
        self.pub_habituation = rospy.Publisher("/habituation/eval_perception", DmpOutcome, queue_size=1, latch=True)
        self.folder_nnga = rospy.get_param("nnga_folder")
        self.mt_field = np.zeros((100,100,1), np.float32)
        self.mt_error = np.zeros((100,100,1), np.float32)
        self.mt_lp = np.zeros((100,100,1), np.float32)
        torch.manual_seed(24)
        self.encoder = MultiLayerEncoder(9,2)#9,6,4,2
        self.encoder.to(device)
        self.decoder = MultiLayerDecoder(2,4,6,9)
        self.decoder.to(device)
        self.memory_size = 20
        self.memory = []
        self.latent_space = []
        self.latent_space_scaled = []
        self.hebbian = HebbServer()
        self.current_dmp = Dmp()
        self.current_goal = LatentGoalDnf()
        self.id_nnga =  id_obj
        self.index_skill = -1
        self.skills = []
        self.min_x = 0.18
        self.max_x = 0.46
        self.min_y = -0.35
        self.max_y = 0.32
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
        self.min_scale = -1.2
        self.max_scale = 1.2
        
    def update_learning_progress(self, data, error):
        update_lp = Goal()
        update_lp.x = data[0]
        update_lp.y = data[1]
        update_lp.value = error
        self.pub_update_lp.publish(update_lp)
        
    def send_new_goal(self, data, val):
        new_goal = Goal()
        new_goal.x = data[0]
        new_goal.y = data[1]
        new_goal.value = val
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
    
    def set_mt_field(self, img):
      self.mt_field = img

    def get_mt_field(self):
      return self.mt_field
    
    def set_mt_error(self, img):
      self.mt_error = img

    def get_mt_error(self):
      return self.mt_error
    
    def set_mt_lp(self, img):
      self.mt_lp = img

    def get_mt_lp(self):
      return self.mt_lp
    
    def plot_latent(self):
        msg_latent = LatentPos()
        for i in self.latent_space:
            msg_latent.x.append(i[0])
            msg_latent.y.append(i[1])
            msg_latent.colors.append("blue")
        self.pub_latent_space_display.publish(msg_latent)
    
    def save_memory(self):
        n_mem = self.folder_nnga + str(self.id_nnga) + "/memory_samples.pkl"
        n_latent = self.folder_nnga + str(self.id_nnga) + "/latent_space.pkl"
        n_latent_scaled = self.folder_nnga + str(self.id_nnga) + "/latent_space_scaled.pkl"
        n_skills = self.folder_nnga + str(self.id_nnga) + "/list_skills.pkl"
        n_hebb = self.folder_nnga + str(self.id_nnga) + "/hebbian_weights.npy"
        n_mtlatent = self.folder_nnga + str(self.id_nnga) + "/latent_space.npy"
        n_mterror = self.folder_nnga + str(self.id_nnga) + "/mt_error.npy"
        n_mtlp = self.folder_nnga + str(self.id_nnga) + "/mt_lp.npy"
        exist = path.exists(n_mem)
        if exist:
            os.remove(n_mem)
            os.remove(n_latent)
            os.remove(n_latent_scaled)
            os.remove(n_skills)
            os.remove(n_hebb)
            os.remove(n_mtlatent)
            os.remove(n_mterror)
            os.remove(n_mtlp)
        filehandler = open(n_mem, 'wb')
        pickle.dump(self.memory, filehandler)
        filehandler_l = open(n_latent, 'wb')
        pickle.dump(self.latent_space, filehandler_l)
        filehandler_ls = open(n_latent_scaled, 'wb')
        pickle.dump(self.latent_space_scaled, filehandler_ls)
        filehandler_s = open(n_skills, 'wb')
        pickle.dump(self.skills, filehandler_s)
        self.hebbian.saveWeights(n_hebb)
        np.save(n_mtlatent,self.mt_field)
        np.save(n_mterror,self.mt_error)
        np.save(n_mtlp,self.mt_lp)

    def save_nn(self):
        name_dir = self.folder_nnga + str(self.id_nnga) 
        n = name_dir + "/nn_ga.pt"
        path = os.path.join(self.folder_nnga, str(self.id_nnga)) 
        access = 0o755
        if os.path.isdir(path):
            os.remove(n)
            torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            }, n)
        else:
            os.makedirs(path,access)  
            torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            }, n)

    def load_memory(self, n):
        n_mem = self.folder_nnga + str(n) + "/memory_samples.pkl"
        n_latent = self.folder_nnga + str(n) + "/latent_space.pkl"
        n_latent_scaled = self.folder_nnga + str(n) + "/latent_space_scaled.pkl"
        n_skills = self.folder_nnga + str(n) + "/list_skills.pkl"
        n_hebb = self.folder_nnga + str(n) + "/hebbian_weights.npy"
        n_mtlatent = self.folder_nnga + str(self.id_nnga) + "/latent_space.npy"
        n_mterror = self.folder_nnga + str(self.id_nnga) + "/mt_error.npy"
        n_mtlp = self.folder_nnga + str(self.id_nnga) + "/mt_lp.npy"
        filehandler = open(n_mem, 'rb') 
        mem = pickle.load(filehandler)
        self.memory = mem
        filehandler_l = open(n_latent, 'rb') 
        nl = pickle.load(filehandler_l)
        self.latent_space = nl
        filehandler_ls = open(n_latent_scaled, 'rb') 
        nls = pickle.load(filehandler_ls)
        self.latent_space_scaled = nls
        filehandler_s = open(n_skills, 'rb') 
        s = pickle.load(filehandler_s)
        self.skills = s
        self.hebbian.loadWeights(n_hebb)
        self.mt_field = np.load(n_mtlatent)
        self.mt_error = np.load(n_mterror)
        self.mt_lp = np.load(n_mtlp)

    def load_nn(self, n):
        nn = self.folder_nnga + str(n) + "/nn_ga.pt"
        checkpoint = torch.load(nn)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])

    def load_skills(self, n):
        name = self.folder_nnga + str(n) + "/"
        for i in range(0,len(self.skills)):
            n = self.skills[i].get_name()
            n_mem = name + n + "_memory.pkl"
            n_fwd = name + n + "_forward.pt"
            n_inv = name + n + "_inverse.pt"
            self.skills[i].load_memory(n_mem)
            self.skills[i].load_fwd_nn(n_fwd)
            self.skills[i].load_inv_nn(n_inv)


    def scale_latent_to_expend(self, data):
        n_x = np.array(data)
        n_x = n_x.reshape(-1,1)
        scaler_x = MinMaxScaler(feature_range=(self.min_scale,self.max_scale))
        x_minmax = np.array([-1, 1])
        scaler_x.fit(x_minmax[:, np.newaxis])
        n_x = scaler_x.transform(n_x)
        n_x = n_x.reshape(1,-1)
        n_x = n_x.flatten()
            
        return n_x[0]
    
    def scale_latent_to_reduce(self, data):
        n_x = np.array(data)
        n_x = n_x.reshape(-1,1)
        scaler_x = MinMaxScaler(feature_range=(-1.0,1.0))
        x_minmax = np.array([self.min_scale, self.max_scale])
        scaler_x.fit(x_minmax[:, np.newaxis])
        n_x = scaler_x.transform(n_x)
        n_x = n_x.reshape(1,-1)
        n_x = n_x.flatten()
            
        return n_x[0]

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
        x_minmax = np.array([self.min_scale, self.max_scale])
        scaler_x.fit(x_minmax[:, np.newaxis])
        n_x = scaler_x.transform(n_x)
        n_x = n_x.reshape(1,-1)
        n_x = n_x.flatten()
            
        return n_x[0]
    
    def scale_dnf_to_latent(self, data):
        n_x = np.array(data)
        n_x = n_x.reshape(-1,1)
        scaler_x = MinMaxScaler(feature_range=(self.min_scale,self.max_scale))
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
    
    def create_skill_sample(self, state, outcome, sample_action):
        sample_inp_fwd = [state.state_x,state.state_y,state.state_angle,sample_action.lpos_x,sample_action.lpos_y,sample_action.lpos_pitch]
        sample_out_fwd = [outcome.x,outcome.y,outcome.angle,outcome.touch]
        sample_inp_inv = [state.state_x,state.state_y,state.state_angle,outcome.x,outcome.y,outcome.angle,outcome.touch]
        sample_out_inv = [sample_action.lpos_x,sample_action.lpos_y,sample_action.lpos_pitch]
        t_inp_fwd = torch.tensor(sample_inp_fwd,dtype=torch.float)
        t_out_fwd = torch.tensor(sample_out_fwd,dtype=torch.float)
        t_inp_inv = torch.tensor(sample_inp_inv,dtype=torch.float)
        t_out_inv = torch.tensor(sample_out_inv,dtype=torch.float)
        sample = []
        sample.append(t_out_fwd)
        sample.append(t_out_inv)
        sample.append(t_inp_fwd)
        sample.append(t_inp_inv)

        return sample

    def scale_samples_new_skill(self, sample):
        #scale sample betwen [-1,1] for learning
        new_state = State()
        new_outcome = Outcome()
        new_dmp = Dmp()
        new_action = Action()
        new_state.state_x = self.scale_data(sample.state_x, self.min_x, self.max_x)
        new_state.state_y = self.scale_data(sample.state_y, self.min_y, self.max_y)
        new_state.state_angle = self.scale_data(sample.state_angle, self.min_angle, self.max_angle)
        new_outcome.x = self.scale_data(sample.outcome_x, self.min_vx, self.max_vx)
        new_outcome.y = self.scale_data(sample.outcome_y, self.min_vy, self.max_vy)
        new_outcome.angle = self.scale_data(sample.outcome_angle, self.min_angle, self.max_angle)
        new_outcome.touch = self.scale_data(sample.outcome_touch, self.min_grasp, self.max_grasp)
        new_dmp.v_x = self.scale_data(sample.v_x, self.min_vx, self.max_vx)
        new_dmp.v_y = self.scale_data(sample.v_y, self.min_vy, self.max_vy)
        new_dmp.v_pitch = self.scale_data(sample.v_pitch, self.min_vpitch, self.max_vpitch)
        new_dmp.roll = self.scale_data(sample.roll, self.min_roll, self.max_roll)
        new_dmp.grasp = self.scale_data(sample.grasp, self.min_grasp, self.max_grasp)
        new_action.lpos_x = self.scale_data(sample.lpos_x, self.min_x, self.max_x)
        new_action.lpos_y = self.scale_data(sample.lpos_y, self.min_y, self.max_y)
        new_action.lpos_pitch = self.scale_data(sample.lpos_pitch, self.min_pitch, self.max_pitch)
        """print("state x before ", sample.state_x, " after : ", new_state.state_x)
        print("state y before ", sample.state_y, " after : ", new_state.state_y)
        print("state angle before ", sample.state_angle, " after : ", new_state.state_angle)
        print("outcome x before ", sample.outcome_x, " after : ", new_outcome.x)
        print("outcome y before ", sample.outcome_y, " after : ", new_outcome.y)
        print("outcome angle before ", sample.outcome_angle, " after : ", new_outcome.angle)
        print("outcome touch before ", sample.outcome_touch, " after : ", new_outcome.touch)
        print("dmp vx before ", sample.v_x, " after : ", new_dmp.v_x)
        print("dmp vy before ", sample.v_y, " after : ", new_dmp.v_y)
        print("dmp vpitch before ", sample.v_pitch, " after : ", new_dmp.v_pitch)
        print("dmp roll before ", sample.roll, " after : ", new_dmp.roll)
        print("dmp grasp before ", sample.grasp, " after : ", new_dmp.grasp)
        print("action lposx before ", sample.lpos_x, " after : ", new_action.lpos_x)
        print("action lposy before ", sample.lpos_y, " after : ", new_action.lpos_y)
        print("action lpospitch before ", sample.lpos_pitch, " after : ", new_action.lpos_pitch)"""

        return new_state, new_dmp, new_outcome, new_action
    
    def scale_samples_existing_skill(self, sample):
        new_state = State()
        new_outcome = Outcome()
        new_action = Action()
        new_state.state_x = self.scale_data(sample.state_x, self.min_x, self.max_x)
        new_state.state_y = self.scale_data(sample.state_y, self.min_y, self.max_y)
        new_state.state_angle = self.scale_data(sample.state_angle, self.min_angle, self.max_angle)
        new_outcome.x = self.scale_data(sample.outcome_x, self.min_vx, self.max_vx)
        new_outcome.y = self.scale_data(sample.outcome_y, self.min_vy, self.max_vy)
        new_outcome.angle = self.scale_data(sample.outcome_angle, self.min_angle, self.max_angle)
        new_outcome.touch = self.scale_data(sample.outcome_touch, self.min_grasp, self.max_grasp)
        new_action.lpos_x = self.scale_data(sample.lpos_x, self.min_x, self.max_x)
        new_action.lpos_y = self.scale_data(sample.lpos_y, self.min_y, self.max_y)
        new_action.lpos_pitch = self.scale_data(sample.lpos_pitch, self.min_pitch, self.max_pitch)

        return new_state, new_outcome, new_action

    def continue_learning(self, data):
        #print("CONTINUAL sample before scaling : ",data)
        state, outcome, sample_action = self.scale_samples_existing_skill(data)
        sample = self.create_skill_sample(state,outcome,sample_action)
        self.skills[self.index_skill].add_to_memory(sample)
        #self.skills[self.index_skill].print_memory()
        err_fwd = self.skills[self.index_skill].predictForwardModel(sample[2],sample[0])
        err_inv = self.skills[self.index_skill].predictInverseModel(sample[3],sample[1])
        error_fwd = err_fwd.item()
        error_inv = err_inv.item()
        #print("ERROR INVERSE : ",error_inv)
        #print("ERROR FORWARD : ",error_fwd)
        inputs = [self.current_goal.latent_x,self.current_goal.latent_y]
        #publish new goal and fwd error
        self.update_learning_progress(inputs,error_fwd)
        #self.send_new_goal(inputs)
        self.pub_timing(1.0)
        #rospy.sleep(10)
        self.update_learning_progress(inputs,0)
        self.pub_timing(0.0)
        self.end_action(True)
        rospy.sleep(1)
        self.end_action(False)
        self.skills[self.index_skill].train_forward_model()
        self.skills[self.index_skill].train_inverse_model()
        pwd = self.folder_nnga + str(self.id_nnga) + "/"
        self.skills[self.index_skill].save_memory(pwd)
        self.skills[self.index_skill].save_fwd_nn(pwd)
        self.skills[self.index_skill].save_inv_nn(pwd)
        self.send_ready(True)

    #bootstrap learning when we discover first skill during exploration
    def bootstrap_learning(self, sample):
        #print("BOOTSTRAP sample before scaling : ",sample)
        state, dmp, outcome, sample_action = self.scale_samples_new_skill(sample)
        #ouput of decoder
        tmp_sample = [outcome.x,outcome.y,outcome.angle,outcome.touch,dmp.v_x,dmp.v_y,dmp.v_pitch,dmp.roll,dmp.grasp]
        #print("scaled dmp : ",tmp_sample)
        tensor_sample_go = torch.tensor(tmp_sample,dtype=torch.float)
        self.add_to_memory(tensor_sample_go)
        ind_skill = self.create_skill()
        sample = self.create_skill_sample(state,outcome,sample_action)
        #print("Input NNGA ; ",sample)
        self.skills[ind_skill].add_to_memory(sample)
        #self.skills[ind_skill].print_memory()
        err_fwd = self.skills[ind_skill].predictForwardModel(sample[2],sample[0])
        err_inv = self.skills[ind_skill].predictInverseModel(sample[3],sample[1])
        error_fwd = err_fwd.item()
        error_inv = err_inv.item()
        #print("ERROR INVERSE : ",error_inv)
        #print("ERROR FORWARD : ",error_fwd)
        t_inputs = self.forward_encoder(tensor_sample_go)
        output_l = t_inputs.detach().numpy()
        #print("latent space original : ",output_l)
        e0 = self.scale_latent_to_expend(output_l[0])
        e1 = self.scale_latent_to_expend(output_l[1])
        t0 = self.scale_latent_to_dnf(e0)
        t1 = self.scale_latent_to_dnf(e1)
        inputs = [round(t0*100),round(t1*100),error_fwd]
        self.skills[ind_skill].set_name(inputs)
        #print("dnf input : ",inputs)
        self.latent_space.append([output_l[0],output_l[1]])
        self.latent_space_scaled.append(inputs)
        self.plot_latent()
        #publish new goal and fwd error
        self.update_learning_progress(inputs,error_fwd)
        self.send_new_goal(inputs,1.0)
        self.pub_timing(1.0)
        rospy.sleep(1.0)
        self.update_learning_progress(inputs,0)
        self.send_new_goal(inputs,0.0)
        self.pub_timing(0.0)
        self.end_action(True)
        rospy.sleep(1)
        self.end_action(False)
        self.send_latent_space()
        #print("Hebbian learning, index : ",ind_skill)
        #print("Inputs Hebbian : ",inputs)
        self.hebbian.hebbianLearning(inputs,ind_skill)
        self.skills[ind_skill].train_forward_model()
        self.skills[ind_skill].train_inverse_model()
        self.trainDecoder()
        #self.test_training()
        self.save_nn()
        self.save_memory()
        print("NNGA latent space DNF : ",self.latent_space_scaled)
        pwd = self.folder_nnga + str(self.id_nnga) + "/"
        self.skills[ind_skill].save_memory(pwd)
        self.skills[ind_skill].save_fwd_nn(pwd)
        self.skills[ind_skill].save_inv_nn(pwd)
        self.send_ready(True)

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
            self.encoder.eval()
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
                self.encoder.eval()
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

    def test_training(self):
        print("EVALUATION")
        for i in range(0,len(self.memory)):
            self.encoder.eval()
            self.decoder.eval()
            sample = self.memory[i]
            print("input sample ",sample)
            sample = sample.to(device)
            inputs = self.encoder(sample)
            inp = inputs.detach().numpy()
            print("Latent values : ",inp)
            inputs = inputs.to(device)
            outputs = self.decoder(inputs)
            out  = outputs.detach().numpy()
            print("reconstruction : ",out)
        print("latent space : ",self.latent_space)
        print("latent space scaled : ",self.latent_space_scaled)

    def forward_encoder(self, data):
        self.encoder.eval()
        data = data.to(device)
        output = self.encoder(data)

        return output
    
    def forward_decoder(self, data):
        self.decoder.eval()
        data = data.to(device)
        output = self.decoder(data)

        return output
    
    def send_latent_space(self):
      ls = self.get_latent_space_dnf()
      msg_latent = LatentNNDNF()
      msg_latent.max_x = 100
      msg_latent.max_y = 100
      for i in ls:
         lg = Goal() 
         lg.x = i[0]
         lg.y = i[1]
         lg.value = i[2]
         msg_latent.list_latent.append(lg)
      #print("Latent space DNF : ",msg_latent)
      self.pub_latent_space_dnf.publish(msg_latent)

    def add_to_memory(self, data):
        self.memory.append(data)

    def get_memory(self):
        return self.memory

    def get_id(self):
        return self.id_nnga
    
    def get_skills(self):
        return self.skills
    
    def get_latent_space_dnf(self):
        return self.latent_space_scaled
    
    def send_habituation(self, goal):
        tmp = [goal.latent_x,goal.latent_y]
        tensor_latent = torch.tensor(tmp,dtype=torch.float)
        output = self.forward_decoder(tensor_latent)
        out = output.detach().numpy()
        print("output sample : ",out)
        print("memory : ",self.memory)
        dmp_out = DmpOutcome()
        dmp_out.x = self.reconstruct_latent(out[0],self.min_vx,self.max_vx)
        dmp_out.y = self.reconstruct_latent(out[1],self.min_vy,self.max_vy)
        dmp_out.angle = self.reconstruct_latent(out[2],self.min_angle,self.max_angle)
        dmp_out.touch = self.reconstruct_latent(out[3],self.min_grasp,self.max_grasp)
        dmp_out.v_x = self.reconstruct_latent(out[4],self.min_vx,self.max_vx)
        dmp_out.v_y = self.reconstruct_latent(out[5],self.min_vy,self.max_vy)
        dmp_out.v_pitch = self.reconstruct_latent(out[6],self.min_vpitch,self.max_vpitch)
        dmp_out.roll = self.reconstruct_latent(out[7],self.min_roll,self.max_roll)
        dmp_out.grasp = self.reconstruct_latent(out[8],self.min_grasp,self.max_vx)
        self.pub_habituation.publish(dmp_out)
    
    def activate_dmp(self, goal):
        tmp = [goal.latent_x,goal.latent_y]
        tensor_latent = torch.tensor(tmp,dtype=torch.float)
        output = self.forward_decoder(tensor_latent)
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
        self.current_goal.latent_x = goal.latent_x
        self.current_goal.latent_y = goal.latent_y
        self.index_skill = self.hebbian.hebbianActivation(tmp)
        print("index models : ",self.index_skill)