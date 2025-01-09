import torch
import numpy as np

def evaluate(model, test_dataloader, device):

    qy = []; qy_log = []
    z_list, x_list = [], []

    test_loss = 0

    model.eval()
    with torch.no_grad():
        for data in test_dataloader:
            data = data.to(device)
            y, qy_logit, zl, xl, loss, nent = model(data)
            qy.append(y.detach().cpu().numpy()); qy_log.append(qy_logit.detach().cpu().numpy())
            z_list.append(torch.stack(zl)); x_list.append(torch.stack(xl))

            loss = loss.mean()
            test_loss += loss.item()
    test_loss /= len(test_dataloader)

    qy = np.concatenate(qy)
    qy_log = np.concatenate(qy_log) 
    z_list = torch.cat(z_list, dim=1).detach().cpu().numpy()
    x_list = torch.cat(x_list, dim=1).detach().cpu().numpy()
    
    print(f"Evaluation Loss: {test_loss:.4f} Cross Entropy {nent:.4f}\n")

    return qy, qy_log, z_list, x_list
