import torch
from tqdm import tqdm


def compute_acc(pred, label):
    pred = pred.argmax(dim=1)
    acc = (pred == label).float().mean()
    
    return acc


def train_step(ep, model, train_loader, criterion, use_mc_loss, optimizer, device):
    model.train()
    total_num, correct, total_loss = 0, 0, 0
    train_bar = tqdm(train_loader, desc=f'Training {ep}')

    for batch_data in train_bar:
        image, label = batch_data['image'], batch_data['label']
        image = image.to(device)
        label = label.to(device)
        
        optimizer.zero_grad()

        if use_mc_loss:
            pred, feature = model(image, use_mc_loss=True)
            loss = criterion(pred, feature, label)

        else:
            pred = model(image)
            loss = criterion(pred, label)

        loss.backward()
        optimizer.step()

        acc = compute_acc(pred, label)
        num = image.shape[0]
        total_num += num
        correct += acc * num
        total_loss += loss.item()

        del image, label, pred
        mean_loss = total_loss / total_num
        mean_acc = correct / total_num
        train_bar.set_postfix({
            'loss': f"{mean_loss:.4f}",
            'acc': f"{mean_acc:.4f}"
        })

    train_bar.close()
    train_record = {
        'loss': f"{mean_loss:.4f}",
        'acc': f"{mean_acc:.4f}"
    }

    return train_record


def val_step(ep, model, val_loader, criterion, use_mc_loss, device):
    model.eval()
    total_num, correct, total_loss = 0, 0, 0
    val_bar = tqdm(val_loader, desc=f'Validation {ep}')

    with torch.no_grad():
        for batch_data in val_bar:
            image, label = batch_data['image'], batch_data['label']
            image = image.to(device)
            label = label.to(device)

            if use_mc_loss:
                pred, feature = model(image, use_mc_loss=True)
                loss = criterion(pred, feature, label)

            else:
                pred = model(image)
                loss = criterion(pred, label)

            acc = compute_acc(pred, label)
            num = image.shape[0]
            total_num += num
            correct += acc * num
            total_loss += loss.item()

            del image, label, pred
            mean_loss = total_loss / total_num
            mean_acc = correct / total_num
            val_bar.set_postfix({
                'loss': f"{mean_loss:.5f}",
                'acc': f"{mean_acc:.5f}"
            })

        val_bar.close()
        val_record = {
        'loss': f"{mean_loss:.4f}",
        'acc': f"{mean_acc:.4f}"
        }

    return val_record
