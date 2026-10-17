import torch
import torch.nn as nn
class FitzHughNagumo():
    def __init__(self, ic_pts, bc_pts, data_pts, res_pts):


        ## TODO : Add boundary condition loss for v?
        self.x_data = torch.tensor(data_pts[0], requires_grad=True, dtype=torch.float32)
        self.y_data = torch.tensor(data_pts[1], requires_grad=True, dtype=torch.float32)
        self.t_data = torch.tensor(data_pts[2], requires_grad=True, dtype=torch.float32)

        self.u_gt = torch.tensor(data_pts[3], requires_grad=True, dtype=torch.float32)
        self.v_gt = torch.tensor(data_pts[4], requires_grad=True, dtype=torch.float32)

        self.x_left = torch.tensor(bc_pts['left'][0], requires_grad=True, dtype=torch.float32)
        self.y_left = torch.tensor(bc_pts['left'][1], requires_grad=True, dtype=torch.float32)
        self.t_left = torch.tensor(bc_pts['left'][2], requires_grad=True, dtype=torch.float32)

        self.x_right = torch.tensor(bc_pts['right'][0], requires_grad=True, dtype=torch.float32)
        self.y_right = torch.tensor(bc_pts['right'][1], requires_grad=True, dtype=torch.float32)
        self.t_right = torch.tensor(bc_pts['right'][2], requires_grad=True, dtype=torch.float32)

        self.x_top = torch.tensor(bc_pts['top'][0], requires_grad=True, dtype=torch.float32)
        self.y_top = torch.tensor(bc_pts['top'][1], requires_grad=True, dtype=torch.float32)
        self.t_top = torch.tensor(bc_pts['top'][2], requires_grad=True, dtype=torch.float32)

        self.x_bottom = torch.tensor(bc_pts['bottom'][0], requires_grad=True, dtype=torch.float32)
        self.y_bottom = torch.tensor(bc_pts['bottom'][1], requires_grad=True, dtype=torch.float32)
        self.t_bottom = torch.tensor(bc_pts['bottom'][2], requires_grad=True, dtype=torch.float32)

        self.x_ic = torch.tensor(ic_pts[0], requires_grad=True, dtype=torch.float32)
        self.y_ic = torch.tensor(ic_pts[1], requires_grad=True, dtype=torch.float32)
        self.t_ic = torch.tensor(ic_pts[2], requires_grad=True, dtype=torch.float32)
        self.u_ic = torch.tensor(ic_pts[3], requires_grad=True, dtype=torch.float32)
        self.v_ic = torch.tensor(ic_pts[4], requires_grad=True, dtype=torch.float32)
        
        self.x_res = torch.tensor(res_pts[0], requires_grad=True, dtype=torch.float32)
        self.y_res = torch.tensor(res_pts[1], requires_grad=True, dtype=torch.float32)
        self.t_res = torch.tensor(res_pts[2], requires_grad=True, dtype=torch.float32)
        
        
        
        
        self.data_losses = []
        self.pde_losses = []
        self.bc_losses = []
        self.ic_losses = []
        
        self.losses = [] 




        # initialize network:
        self.network()

        self.optimizer_adam = torch.optim.Adam(self.net.parameters(), lr = 1e-3)

        self.optimizer = torch.optim.LBFGS(self.net.parameters(), lr=0.1, max_iter=200000, max_eval=50000,
                                           history_size=50, tolerance_grad=1e-05, tolerance_change=0.5 * np.finfo(float).eps,
                                           line_search_fn="strong_wolfe")

        self.mse = nn.MSELoss()

        #loss
        self.ls = 0

        #iteration number
        self.iter = 0

    def network(self):
        self.net = nn.Sequential(
        nn.Linear(3, 60), nn.Tanh(),
        nn.Linear(60, 60), nn.Tanh(),
        nn.Linear(60, 60), nn.Tanh(),
        nn.Linear(60, 60), nn.Tanh(),
        nn.Linear(60, 60), nn.Tanh(),
        nn.Linear(60, 2))