## BipedalWalker
- [BipedalWalker A2C solo-RNN](BipedalWalker_A2C_solo-RNN.ipynb) - training solo model
- [BipedalWalker A2C multy-RNN](BipedalWalker_A2C_multy-RNN.ipynb) - training multiagent version 

>[MADDPG version of multiwalker](https://github.com/Otnielush/MAS-Machon-Lev/tree/master/MADDPG%20Walker)

[Original version (youtube)](https://youtu.be/l-onY7wZDMI)  
[Hardcore version (youtube)](https://youtu.be/BPrialP6tpY)

##Architecture Actor`s model (Pytorch)
```
Actor(
(rnn): GRU(31, 128, num_layers=2, batch_first=True)
(mu): Linear(in_features=128, out_features=4, bias=True)
(log_std_linear): Linear(in_features=128, out_features=4, bias=True)
)
```

```===================================================================================================================
Layer (type:depth-idx)                   Input Shape               Output Shape              Param #
===================================================================================================================
Actor                                    --                        --                        --
├─GRU: 1-1                               [1, 1, 31]                [1, 1, 128]               160,896
├─Linear: 1-2                            [1, 1, 128]               [1, 1, 4]                 516
├─Linear: 1-3                            [1, 1, 128]               [1, 1, 4]                 516
===================================================================================================================
Total params: 161,928
Trainable params: 161,928
Non-trainable params: 0
Total mult-adds (M): 0.16
===================================================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.00
Params size (MB): 0.65
Estimated Total Size (MB): 0.65
===================================================================================================================
```