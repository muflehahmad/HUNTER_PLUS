import sys
sys.path.append('scheduler/BaGTI/')

from .Scheduler import *
from .BaGTI.train import *
from .BaGTI.src.utils import *
from .BaGTI.src.opt import *
import dgl

class GGCNScheduler(Scheduler):
	def __init__(self, data_type):
		super().__init__()
		dtl = data_type.split('_')
		data_type = '_'.join(dtl[:-2])+'GNN_'+dtl[-2]+'_'+dtl[-1]
		#print("data_type:")
		#print(data_type)
		self.model = eval(data_type+"()")
		self.model, _, _, _ = load_model(data_type, self.model, data_type)
		self.data_type = data_type
		self.hosts = int(data_type.split('_')[-1])
		dtl = data_type.split('_')
		print("load_"+'_'.join(dtl[:-2])+"_data("+dtl[-1]+")")
		_, _, self.max_container_ips = eval("load_"+'_'.join(dtl[:-2])+"_data("+dtl[-1]+")")

	def run_GOBI(self):

		#cpu = [0] * 20
		#for index, host in enumerate(self.env.hostlist):
		#	cpu[index] = host.getCPU()/100
		cpu = [host.getCPU()/100 for host in self.env.hostlist]
		cpu24 = [host.getCPU()/100 for host in self.env.hostlist]
		cpuIDs = [host.id for host in self.env.hostlist]
		#cpu = np.array([cpu]).transpose()

		#print("Number of containers is:")
		#print(len(self.env.containerlist))

		#print("The containers are:")
		#print(self.env.containerlist)
		if 'latency' in self.model.name:
			cpuC = [(c.getApparentIPS()/self.max_container_ips if c else 0) for c in self.env.containerlist]
			cpuCIDs = [(c.getHostID() if c else -1) for c in self.env.containerlist]

			cpuNew = [(cpu[cpuIDs.index(cid)] if cid != -1 else max(cpu)) for cid in cpuCIDs]
			
			for x in cpuC:
				cpu24.append(x)
			cpu24 = np.array([cpu24]).transpose()
			cpu = np.array([cpuNew]).transpose()
			cpuC = np.array([cpuC]).transpose()
			
			

			cpu = np.concatenate((cpu, cpuC), axis=1)

			#print("This is the new cpu with the extra containers:/n", cpu)

		alloc = []; prev_alloc = {}; u, v = [], []
		for c in self.env.containerlist:
			oneHot = [0] * len(self.env.hostlist)
			if c: prev_alloc[c.id] = c.getHostID()
			if c and c.getHostID() != -1: 
				oneHot[c.getHostID()] = 1
				u.append(c.id); v.append(c.getHostID() + self.hosts)
			else: oneHot[np.random.randint(0,len(self.env.hostlist))] = 1
			alloc.append(oneHot)
		#print("The length of u is:\n", len(u))
		#print("\nu is: \n", u)
		#print("\nThe length of v is:\n", len(v))
		#print("\nv is:\n", v)
		g = dgl.graph((u, v), num_nodes = 24)
		#print(u)
		#print("I am here!!!")
		g = dgl.add_self_loop(g)
		data = torch.Tensor(cpu24.reshape(-1, 1))
		#print("data in GGCN:")
		#print(data)
		init = np.concatenate((cpu, alloc), axis=1)
		init = torch.tensor(init, dtype=torch.float, requires_grad=True)
		#print("Init:\n")
		#print(init.data)
		result, iteration, fitness = optGNN(init, g, data, self.model, [], self.data_type)
		decision = []
		for cid in prev_alloc:
			one_hot = result[cid, -self.hosts:].tolist()
			new_host = one_hot.index(max(one_hot))
			if prev_alloc[cid] != new_host: decision.append((cid, new_host))
		return decision

	def selection(self):
		return []

	def placement(self, containerIDs):
		first_alloc = np.all([not (c and c.getHostID() != -1) for c in self.env.containerlist])
		decision = self.run_GOBI()
		return decision