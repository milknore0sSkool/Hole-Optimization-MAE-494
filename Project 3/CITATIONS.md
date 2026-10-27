# Sources and citations

## Public data and canonical problem

- Raissi PINNs repository: https://github.com/maziarraissi/PINNs
- Direct public data path used by the downloader: https://raw.githubusercontent.com/maziarraissi/PINNs/master/appendix/Data/burgers_shock.mat
- Alternate public mirror: https://github.com/XinyuanLiao/Adaptive-optimization-of-PINN

The canonical Burgers problem is

`u_t + u u_x - (0.01/pi) u_xx = 0`, `x in [-1,1]`, `t in [0,1]`,
`u(x,0)=-sin(pi x)`, `u(-1,t)=u(1,t)=0`.

## Papers

- Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686–707.
- Krishnapriyan, A. S., Gholami, A., Zhe, S., Kirby, R. M., & Mahoney, M. W. (2021). Characterizing possible failure modes in physics-informed neural networks. NeurIPS 2021.
- Wang, S., Yu, X., & Perdikaris, P. (2022). When and why PINNs fail to train: A neural tangent kernel perspective. Journal of Computational Physics, 449, 110768.
