import torch
import time

def train(model, train_loader, valid_loader, optimizer, epochs, time_lagged, device):
    train_losses = {"total": [], "nent": []}
    valid_losses = {"total": [], "nent": []}

    start_time = time.time()

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for batch in train_loader:
            if time_lagged:
                x_input, x_output = batch
                x_input = x_input.to(device).float()
                x_output = x_output.to(device).float()
            else:
                x_input = batch.to(device).float()
                x_output = x_input  # For non-time-lagged, output is the same as input

            optimizer.zero_grad()
            _, _, _, _, loss, nent = model(x_input)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses["total"].append(train_loss)
        train_losses["nent"].append(nent.item())
        
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                if time_lagged:
                    x_input, x_output = batch
                    x_input = x_input.to(device).float()
                    x_output = x_output.to(device).float()
                else:
                    x_input = batch.to(device).float()
                    x_output = x_input

                _, _, _, _, loss, nent = model(x_input)
                loss = loss.mean()
                valid_loss += loss.item()
            
            valid_loss /= len(valid_loader)
            valid_losses["total"].append(valid_loss)
            valid_losses["nent"].append(nent.item())
        
        if epoch % 10 == 0 or epoch == epochs-1:
            duration_time = time.time() - start_time
            print(f'Epoch {epoch+1}/{epochs} | Total Train Loss: {train_loss:.4f} | Training Cross Entropy {nent.item():.4f} | Total Valid Loss: {valid_loss:.4f} | Validation Cross Entropy {nent.item():.4f} | Time: {duration_time:.2f}s')
    
    return model, {"train_losses": train_losses, "valid_losses": valid_losses}
