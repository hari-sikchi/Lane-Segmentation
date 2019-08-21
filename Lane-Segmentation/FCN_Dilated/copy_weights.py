import torch
from FCN_Dilated import FCN_Dilated
from CAN.CAN import CAN

old_net = CAN()
old_net.load_state_dict(torch.load("../CAN/CAN.wts"))
old_net_state_dict = {name: w for name, w in old_net.named_parameters() if name<'features.33'}

print('Copying...')
for name in old_net_state_dict:
    print(name)

new_net = FCN_Dilated()
new_net.load_state_dict(old_net_state_dict, strict=False)

torch.save(new_net.state_dict(), 'FCN_Dilated.wts')

print('\nDone')
