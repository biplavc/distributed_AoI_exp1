## Distributed AoI with multiple devices sending information to multiple destinations

1. 5 devices (10,11,12,13,14) generating information in the uplink and 9 pairs (10-11, 10-12, 11-10, 11-13, 12-13, 13-10, 13-12, 14,10, 14-11) in the downlink. Note device 12 in on only 1 DL, so having newer information for 12 in the UL will only support 1 DL. While all others can support 2 DL. This might be reason why DQN better than MAD. ARC folder - d_AOI_1

2. To verify the case above, modify the adjacency matrix so that 12 also has 2 DL and hence the total DL becomes 10. Now each device has same DL. Note that here some devices are receivers more often than others, but that shouldn't affect MAD. If still MAD < DQN, make the change where each device are equally receivers. Here we still see DQN better than MAD. ARC folder - d_AOI_2

3. To make all devices completely identical from both sender and receiver perspective, adjacency matrix has been adjusted accordingly. Also UL capacity = 2, DL capacity =4 (as UL of 2 means 4 DL links can have fresh information in the next slot). MAD should be better as here no memory needed and each step the MAD action should be best. ARC folder - d_AOI_3.

above folders are in old folder of ARC.